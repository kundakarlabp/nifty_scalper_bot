# src/backtesting/backtest_engine.py
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import Config
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
# Optional – used if available; we fall back gracefully if API differs
try:
    from src.risk.position_sizing import PositionSizing
except Exception:
    PositionSizing = None  # type: ignore

# --------- dataclasses --------- #

@dataclass
class SimLeg:
    """A single exit leg (TP1, TP2, or SL)"""
    qty: int
    price: float
    tag: str  # 'TP1' | 'TP2' | 'SL' | 'TRAIL_SL' | 'TIMEOUT'

@dataclass
class SimPosition:
    direction: str   # BUY / SELL
    entry: float
    sl: float
    tp1: float
    tp2: float
    qty_tp1: int
    qty_tp2: int
    atr_at_entry: float
    open_idx: int
    open_time: str
    remaining_qty: int
    # dynamic
    active_sl: float
    tp1_done: bool = False
    exit_legs: List[SimLeg] = None

    def __post_init__(self):
        self.exit_legs = [] if self.exit_legs is None else self.exit_legs

@dataclass
class SimTrade:
    idx_in: int
    idx_out: int
    date_in: str
    date_out: str
    symbol: str
    direction: str
    entry: float
    exit_vwap: float
    qty: int
    pnl: float
    r_mult: float
    details: str


# --------- engine --------- #

class BacktestEngine:
    """
    Bar-by-bar backtester that mirrors live behavior:
      - EnhancedScalpingStrategy for entries
      - Risk-based sizing with lot integrity
      - Partial TP + breakeven hop + optional chandelier trailing
      - Touch-first intrabar exit ordering: SL first, then TP
      - CSV trade log + summary metrics
    """

    def __init__(
        self,
        data: pd.DataFrame,
        symbol: str = "NIFTY",
        initial_capital: float = 1_00_000.0,
        log_file: str = "logs/backtest_trades.csv",
    ):
        """
        data: DataFrame with columns: ['open','high','low','close'] (volume optional)
              index can be datetime or any label; will be preserved as 'date'
        """
        assert {"open", "high", "low", "close"}.issubset(data.columns), "Data must have OHLC columns."

        self.symbol = symbol
        self.df = data.copy()
        if "date" not in self.df.columns:
            self.df = self.df.copy()
            self.df["date"] = self.df.index.astype(str)
        self.df = self.df.reset_index(drop=True)

        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.equity_R = 0.0  # equity tracked in R space for DD calc

        # strategy
        self.strategy = EnhancedScalpingStrategy(
            base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
            base_target_points=Config.BASE_TARGET_POINTS,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
            min_score_threshold=int(Config.MIN_SIGNAL_SCORE),
        )

        # sizing
        self.lot_size = int(getattr(Config, "NIFTY_LOT_SIZE", 75))
        self.risk_per_trade = float(getattr(Config, "RISK_PER_TRADE", 0.02))
        self._ps = PositionSizing(self.initial_capital) if PositionSizing else None

        # exits / partials
        self.partial_enable = bool(getattr(Config, "PARTIAL_TP_ENABLE", True))
        self.partial_ratio = float(getattr(Config, "PARTIAL_TP_RATIO", 0.5))
        self.breakeven_after_tp1 = bool(getattr(Config, "BREAKEVEN_AFTER_TP1_ENABLE", True))
        self.breakeven_ticks = int(getattr(Config, "BREAKEVEN_OFFSET_TICKS", 2))
        self.tick = float(getattr(Config, "TICK_SIZE", 0.05))

        # trailing (chandelier-style)
        self.trail_enable = bool(getattr(Config, "TRAILING_ENABLE", True))
        self.chan_n = int(getattr(Config, "CHANDELIER_N", 22))
        self.chan_k = float(getattr(Config, "CHANDELIER_K", 2.5))

        # time-in-trade box exit
        self.max_hold_min = int(getattr(Config, "MAX_HOLD_MIN", 25))
        self.box_hold_pct = float(getattr(Config, "BOX_HOLD_PCT", 0.01))

        # logging
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["date_in","date_out","symbol","dir","qty","entry","exit_vwap","pnl","r_mult","details"]
            )

        # state
        self.pos: Optional[SimPosition] = None
        self.trades: List[SimTrade] = []

        # warmup
        self.warmup = int(getattr(Config, "WARMUP_BARS", 25))

    # ----- helpers ----- #

    def _risk_qty(self, entry: float, sl: float) -> int:
        """Risk-based contracts with lot integrity; safe fallback if PositionSizing differs."""
        stop_pts = abs(entry - sl)
        if stop_pts <= 0:
            return 0

        # Prefer user's PositionSizing if it exposes calculate_position_size(entry, stop)
        if self._ps and hasattr(self._ps, "calculate_position_size"):
            try:
                q = int(self._ps.calculate_position_size(entry, sl))
                if q > 0:
                    # ensure lot integrity
                    if self.lot_size > 0:
                        q = max(self.lot_size, (q // self.lot_size) * self.lot_size)
                    return q
            except Exception:
                pass

        # Fallback: fixed account * risk%
        equity = max(self.current_capital, self.initial_capital)
        risk_amt = self.risk_per_trade * equity
        contracts = int(risk_amt // stop_pts)
        if contracts <= 0:
            return 0
        if self.lot_size > 0:
            lots = max(1, contracts // self.lot_size)
            return lots * self.lot_size
        return contracts

    @staticmethod
    def _touch_first(direction: str, hi: float, lo: float, sl: float, tp: float) -> Tuple[str, Optional[float]]:
        """Return ('SL'|'TP'|'NONE', exit_price) with SL priority on the bar."""
        if direction == "BUY":
            if lo <= sl:
                return "SL", sl
            if hi >= tp:
                return "TP", tp
        else:  # SELL
            if hi >= sl:
                return "SL", sl
            if lo <= tp:
                return "TP", tp
        return "NONE", None

    def _vwap_exit(self, fills: List[Tuple[int, float]]) -> float:
        """Compute VWAP-like blended exit for multi-leg exits."""
        q_sum = sum(q for q, _ in fills)
        if q_sum <= 0:
            return 0.0
        return sum(q * px for q, px in fills) / q_sum

    # ----- main loop ----- #

    def run(self) -> Dict[str, float]:
        for i in range(self.warmup, len(self.df)):
            window = self.df.iloc[: i + 1]
            price_i = float(window["close"].iloc[-1])
            hi = float(window["high"].iloc[-1])
            lo = float(window["low"].iloc[-1])
            tstamp = str(window["date"].iloc[-1])

            # manage open position
            if self.pos:
                self._update_trailing(window, i)
                fills: List[Tuple[int, float]] = []

                # check TP1 first only if not done (but SL always takes absolute priority intrabar)
                tag, px = self._touch_first(self.pos.direction, hi, lo, self.pos.active_sl,
                                            self.pos.tp1 if not self.pos.tp1_done else self.pos.tp2)

                if tag == "SL":
                    # full remaining exits at SL
                    qty = self.pos.remaining_qty
                    fills.append((qty, self.pos.active_sl))
                    self.pos.exit_legs.append(SimLeg(qty=qty, price=self.pos.active_sl, tag="SL"))
                    self._close_trade(i, tstamp, fills, detail="SL")
                    continue

                if tag == "TP":
                    if not self.pos.tp1_done:
                        # TP1 fills qty_tp1
                        q = min(self.pos.qty_tp1, self.pos.remaining_qty)
                        if q > 0:
                            fills.append((q, self.pos.tp1))
                            self.pos.exit_legs.append(SimLeg(qty=q, price=self.pos.tp1, tag="TP1"))
                            self.pos.remaining_qty -= q
                            self.pos.tp1_done = True

                            # Breakeven hop (tighten-only)
                            if self.breakeven_after_tp1:
                                be = self.pos.entry + (self.breakeven_ticks * self.tick if self.pos.direction == "BUY"
                                                       else -self.breakeven_ticks * self.tick)
                                if (self.pos.direction == "BUY" and be > self.pos.active_sl) or \
                                   (self.pos.direction == "SELL" and be < self.pos.active_sl):
                                    self.pos.active_sl = be
                    else:
                        # TP2 (and close position)
                        q = self.pos.remaining_qty
                        if q > 0:
                            fills.append((q, self.pos.tp2))
                            self.pos.exit_legs.append(SimLeg(qty=q, price=self.pos.tp2, tag="TP2"))
                            self._close_trade(i, tstamp, fills, detail="TP2")
                            continue

                # box/timeout exit
                if self._timeout_exit(window, i):
                    q = self.pos.remaining_qty
                    if q > 0:
                        px_to_use = price_i
                        fills.append((q, px_to_use))
                        self.pos.exit_legs.append(SimLeg(qty=q, price=px_to_use, tag="TIMEOUT"))
                        self._close_trade(i, tstamp, fills, detail="TIMEOUT")
                        continue

            # if flat, look for entries
            if not self.pos:
                signal = self.strategy.generate_signal(window.set_index(self.df.index[: i + 1]),
                                                       price_i)
                if not signal:
                    continue

                direction = signal["signal"]
                entry = float(signal["entry_price"])
                sl = float(signal["stop_loss"])
                tp = float(signal["target"])
                conf = float(signal.get("confidence", 5.0))
                atr = float(signal.get("market_volatility", 0.0))

                # size
                qty = self._risk_qty(entry, sl)
                if qty <= 0:
                    continue

                # partials
                if self.partial_enable and qty >= 2:
                    total_lots = qty // max(1, self.lot_size)
                    tp1_lots = max(1, int(round(total_lots * self.partial_ratio)))
                    tp2_lots = max(1, total_lots - tp1_lots)
                    qty_tp1 = tp1_lots * self.lot_size
                    qty_tp2 = tp2_lots * self.lot_size
                    # midpoint TP1
                    tp1 = entry + (tp - entry) / 2.0 if direction == "BUY" else entry - (entry - tp) / 2.0
                    tp2 = tp
                else:
                    qty_tp1 = 0
                    qty_tp2 = qty
                    tp1 = tp  # single take
                    tp2 = tp

                self.pos = SimPosition(
                    direction=direction,
                    entry=entry,
                    sl=sl,
                    tp1=tp1,
                    tp2=tp2,
                    qty_tp1=qty_tp1,
                    qty_tp2=qty_tp2,
                    atr_at_entry=atr,
                    open_idx=i,
                    open_time=tstamp,
                    remaining_qty=qty,
                    active_sl=sl,
                )

        # flush any open position at the last close
        if self.pos:
            last_close = float(self.df["close"].iloc[-1])
            q = self.pos.remaining_qty
            fills = [(q, last_close)]
            self.pos.exit_legs.append(SimLeg(qty=q, price=last_close, tag="EOD"))
            self._close_trade(len(self.df) - 1, str(self.df["date"].iloc[-1]), fills, detail="EOD")

        return self.metrics()

    # ----- exits helpers ----- #

    def _update_trailing(self, window: pd.DataFrame, i: int) -> None:
        """Chandelier-like trailing after trade goes in favor; tighten-only."""
        if not (self.trail_enable and self.pos):
            return
        n = min(self.chan_n, len(window))
        if n < 2:
            return

        if self.pos.direction == "BUY":
            highest = float(window["high"].iloc[-n:].max())
            trail = highest - self.chan_k * float(window["close"].iloc[-n:].std(ddof=0))
            trail = max(trail, self.pos.active_sl)  # tighten only
            if trail > self.pos.active_sl and trail < highest:
                self.pos.active_sl = trail
        else:
            lowest = float(window["low"].iloc[-n:].min())
            trail = lowest + self.chan_k * float(window["close"].iloc[-n:].std(ddof=0))
            trail = min(trail, self.pos.active_sl)
            if trail < self.pos.active_sl and trail > lowest:
                self.pos.active_sl = trail

    def _timeout_exit(self, window: pd.DataFrame, i: int) -> bool:
        """Exit if price stuck near entry beyond MAX_HOLD_MIN."""
        if self.max_hold_min <= 0 or not self.pos:
            return False
        bars_held = i - self.pos.open_idx
        if bars_held < self.max_hold_min:
            return False
        px = float(window["close"].iloc[-1])
        delta = abs(px - self.pos.entry) / max(1e-9, self.pos.entry)
        return delta <= self.box_hold_pct

    def _close_trade(self, i_out: int, t_out: str, fills: List[Tuple[int, float]], detail: str) -> None:
        """Finalize trade, compute PnL & R, write a row, and clear position."""
        assert self.pos, "No position to close."
        qty_total = sum(q for q, _ in fills)
        px_exit = self._vwap_exit(fills)
        direction_mult = 1 if self.pos.direction == "BUY" else -1

        pnl = (px_exit - self.pos.entry) * direction_mult * qty_total
        R = (px_exit - self.pos.entry) / max(1e-9, abs(self.pos.entry - self.pos.sl))
        R *= direction_mult

        # track equity in R for max DD
        self.equity_R += R
        self.current_capital += pnl

        trade = SimTrade(
            idx_in=self.pos.open_idx,
            idx_out=i_out,
            date_in=self.pos.open_time,
            date_out=t_out,
            symbol=self.symbol,
            direction=self.pos.direction,
            entry=self.pos.entry,
            exit_vwap=px_exit,
            qty=qty_total,
            pnl=pnl,
            r_mult=R,
            details=detail,
        )
        self.trades.append(trade)

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [trade.date_in, trade.date_out, trade.symbol, trade.direction, trade.qty,
                 round(trade.entry, 2), round(trade.exit_vwap, 2), round(trade.pnl, 2),
                 round(trade.r_mult, 3), trade.details]
            )
        self.pos = None

    # ----- metrics ----- #

    def metrics(self) -> Dict[str, float]:
        if not self.trades:
            return {
                "trades": 0, "win_rate": 0.0, "net_pnl": 0.0,
                "avg_R": 0.0, "max_dd_R": 0.0, "final_capital": self.current_capital,
            }

        pnl = [t.pnl for t in self.trades]
        rs = [t.r_mult for t in self.trades]
        wins = sum(1 for x in pnl if x > 0)
        net = sum(pnl)
        avg_R = sum(rs) / len(rs)

        # compute max drawdown in R space
        eq = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in rs:
            eq += r
            peak = max(peak, eq)
            dd = eq - peak
            max_dd = min(max_dd, dd)

        return {
            "trades": len(self.trades),
            "win_rate": 100.0 * wins / len(self.trades),
            "net_pnl": net,
            "avg_R": avg_R,
            "max_dd_R": max_dd,
            "final_capital": self.current_capital,
        }


# ---------- convenience runner from CSV ---------- #

def run_backtest_from_csv(csv_path: str, symbol: str = None) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    # normalize column names (case-insensitive)
    lower_map = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close"]
    rename = {lower_map[x]: x for x in need if x in lower_map and lower_map[x] != x}
    if rename:
        df = df.rename(columns=rename)
    # ensure date column for logs
    if "date" not in df.columns:
        df["date"] = df.index.astype(str)
    engine = BacktestEngine(df, symbol or os.path.basename(csv_path))
    return engine.run()