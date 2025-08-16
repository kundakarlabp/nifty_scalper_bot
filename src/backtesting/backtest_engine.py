# src/backtesting/backtest_engine.py
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from src.config import Config
from src.strategies.scalping_strategy import EnhancedScalpingStrategy

# Optional – we’ll use it if available (and handle signature differences safely)
try:
    from src.risk.position_sizing import PositionSizing
except Exception:  # pragma: no cover
    PositionSizing = None  # type: ignore


# ------------------------------- helpers -------------------------------- #

def _round_to_tick(x: float, tick: float) -> float:
    if tick <= 0:
        return float(x)
    # 4 decimals is enough for options ticks like 0.05
    return round(round(float(x) / tick) * tick, 4)


def _slip(direction: str, px: float, bps: float, is_entry: bool) -> float:
    """
    Apply simple bps slippage:
      BUY entry  -> price increases
      BUY exit   -> price decreases
      SELL entry -> price decreases
      SELL exit  -> price increases
    """
    if bps <= 0 or px <= 0:
        return float(px)
    mul = bps / 10_000.0
    if direction == "BUY":
        return float(px * (1.0 + mul) if is_entry else px * (1.0 - mul))
    else:
        return float(px * (1.0 - mul) if is_entry else px * (1.0 + mul))


# ------------------------------- data ----------------------------------- #

@dataclass
class SimLeg:
    qty: int
    price: float
    tag: str  # 'TP1' | 'TP2' | 'SL' | 'TRAIL_SL' | 'TIMEOUT' | 'EOD'


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
    active_sl: float
    tp1_done: bool = False
    exit_legs: List[SimLeg] = field(default_factory=list)


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


# ------------------------------ engine ---------------------------------- #

class BacktestEngine:
    """
    Bar-by-bar backtester that mirrors live behavior:

    - Uses EnhancedScalpingStrategy for entries
    - Risk-based sizing (via PositionSizing if available)
    - Partial TP + breakeven hop + chandelier-style trailing
    - SL touched first on a bar has priority over TP (“touch-first”)
    - Tick rounding, fees and slippage modeling
    - CSV trade log + summary metrics

    DataFrame requirements:
      columns: open, high, low, close  (case-sensitive; volume optional)
      index:   anything; we preserve as 'date' for logs
    """

    def __init__(
        self,
        data: pd.DataFrame,
        symbol: str = "NIFTY",
        initial_capital: float = 100_000.0,
        log_file: str = "logs/backtest_trades.csv",
    ):
        assert {"open", "high", "low", "close"}.issubset(
            map(str, data.columns)
        ), "Data must include columns: open, high, low, close"

        self.symbol = symbol
        self.df = data.copy()

        if "date" not in self.df.columns:
            self.df["date"] = self.df.index.astype(str)
        self.df = self.df.reset_index(drop=True)

        # capital & accounting
        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.equity_R = 0.0  # equity tracked in R for drawdown computation

        # costs
        self.tick = float(getattr(Config, "TICK_SIZE", 0.05))
        self.lot_size = int(getattr(Config, "NIFTY_LOT_SIZE", 75))
        # Treat FEES_PER_LOT as round-trip cost per lot (adjust to your broker if needed)
        self.fees_per_lot = float(getattr(Config, "FEES_PER_LOT", 25.0))
        self.slippage_bps = float(getattr(Config, "SLIPPAGE_BPS", 4.0))

        # strategy
        self.strategy = EnhancedScalpingStrategy(
            base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
            base_target_points=Config.BASE_TARGET_POINTS,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
            min_score_threshold=int(getattr(Config, "MIN_SIGNAL_SCORE", 0)),
        )

        # risk sizing (optional but preferred)
        self.risk_per_trade = float(getattr(Config, "RISK_PER_TRADE", 0.02))
        self._ps = PositionSizing(self.initial_capital) if PositionSizing else None

        # partials / BE
        self.partial_enable = bool(getattr(Config, "PARTIAL_TP_ENABLE", True))
        self.partial_ratio = float(getattr(Config, "PARTIAL_TP_RATIO", 0.5))
        self.breakeven_after_tp1 = bool(getattr(Config, "BREAKEVEN_AFTER_TP1_ENABLE", True))
        self.breakeven_ticks = int(getattr(Config, "BREAKEVEN_OFFSET_TICKS", 1))

        # trailing (chandelier-like; uses close std as simple volatility proxy)
        self.trail_enable = bool(getattr(Config, "TRAILING_ENABLE", True))
        self.chan_n = int(getattr(Config, "CHANDELIER_N", 22))
        self.chan_k = float(getattr(Config, "CHANDELIER_K", 2.5))

        # time-in-trade “box” exit
        self.max_hold_bars = int(getattr(Config, "MAX_HOLD_MIN", 25))
        self.box_hold_pct = float(getattr(Config, "BOX_HOLD_PCT", 0.01))

        # warmup & logging
        self.warmup = int(getattr(Config, "WARMUP_BARS", 25))
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "date_in",
                    "date_out",
                    "symbol",
                    "dir",
                    "qty",
                    "entry",
                    "exit_vwap",
                    "pnl",
                    "r_mult",
                    "details",
                ]
            )

        # state
        self.pos: Optional[SimPosition] = None
        self.trades: List[SimTrade] = []

    # ---------------------------- sizing --------------------------------- #

    def _risk_qty_contracts(
        self, entry: float, sl: float, confidence: float, volatility: float
    ) -> int:
        """
        Contracts (NOT lots). Prefer PositionSizing if available; otherwise
        compute from initial_capital * risk_per_trade / stop distance.
        """
        stop_pts = abs(float(entry) - float(sl))
        if stop_pts <= 0:
            return 0

        # Use your PositionSizing if present
        if self._ps and hasattr(self._ps, "calculate_position_size"):
            try:
                res: Any = self._ps.calculate_position_size(
                    entry_price=float(entry),
                    stop_loss=float(sl),
                    signal_confidence=float(confidence),
                    market_volatility=float(volatility),
                    lot_size=int(self.lot_size),
                )
                # res may be dict {"quantity": lots} or raw int (lots)
                lots = 0
                if isinstance(res, dict):
                    lots = int(res.get("quantity", 0))
                else:
                    lots = int(res)
                return max(0, lots) * max(1, self.lot_size)
            except Exception:
                # fall back below
                pass

        # Simple fallback (contracts)
        risk_amount = float(self.initial_capital) * float(self.risk_per_trade)
        contracts = int(risk_amount // stop_pts)
        if contracts <= 0:
            return 0
        # enforce lot integrity
        if self.lot_size > 0:
            lots = max(1, contracts // self.lot_size)
            return lots * self.lot_size
        return contracts

    # --------------------------- run loop -------------------------------- #

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

    @staticmethod
    def _vwap_exit(fills: List[Tuple[int, float]]) -> float:
        q_sum = sum(q for q, _ in fills)
        if q_sum <= 0:
            return 0.0
        return sum(q * px for q, px in fills) / q_sum

    def _update_trailing(self, window: pd.DataFrame) -> None:
        """Chandelier-like trailing; tighten-only."""
        if not (self.trail_enable and self.pos):
            return
        n = min(self.chan_n, len(window))
        if n < 2:
            return

        if self.pos.direction == "BUY":
            highest = float(window["high"].iloc[-n:].max())
            trail = highest - self.chan_k * float(window["close"].iloc[-n:].std(ddof=0))
            trail = max(trail, self.pos.active_sl)  # tighten-only
            if trail > self.pos.active_sl and trail < highest:
                self.pos.active_sl = _round_to_tick(trail, self.tick)
        else:
            lowest = float(window["low"].iloc[-n:].min())
            trail = lowest + self.chan_k * float(window["close"].iloc[-n:].std(ddof=0))
            trail = min(trail, self.pos.active_sl)
            if trail < self.pos.active_sl and trail > lowest:
                self.pos.active_sl = _round_to_tick(trail, self.tick)

    def _timeout_exit(self, window: pd.DataFrame, i: int) -> bool:
        """Exit if price stuck near entry beyond max_hold_bars."""
        if self.max_hold_bars <= 0 or not self.pos:
            return False
        bars_held = i - self.pos.open_idx
        if bars_held < self.max_hold_bars:
            return False
        px = float(window["close"].iloc[-1])
        delta = abs(px - self.pos.entry) / max(1e-9, self.pos.entry)
        return delta <= self.box_hold_pct

    def run(self) -> Dict[str, float]:
        for i in range(self.warmup, len(self.df)):
            window = self.df.iloc[: i + 1]
            hi = float(window["high"].iloc[-1])
            lo = float(window["low"].iloc[-1])
            close_i = float(window["close"].iloc[-1])
            tstamp = str(window["date"].iloc[-1])

            # ---- manage existing position ----
            if self.pos:
                self._update_trailing(window)
                fills: List[Tuple[int, float]] = []

                # Touch-first logic; SL always has priority
                tag, px = self._touch_first(
                    self.pos.direction, hi, lo, self.pos.active_sl,
                    self.pos.tp1 if not self.pos.tp1_done else self.pos.tp2
                )

                if tag == "SL":
                    q = self.pos.remaining_qty
                    # exit side (opposite to entry) + slippage
                    px_slip = _slip(self.pos.direction, self.pos.active_sl, self.slippage_bps, is_entry=False)
                    fills.append((q, _round_to_tick(px_slip, self.tick)))
                    self.pos.exit_legs.append(SimLeg(qty=q, price=float(px_slip), tag="SL"))
                    self._close_trade(i, tstamp, fills, detail="SL")
                    continue

                if tag == "TP":
                    if not self.pos.tp1_done:
                        # TP1 leg
                        q = min(self.pos.qty_tp1, self.pos.remaining_qty)
                        if q > 0:
                            px_tp1 = _slip(self.pos.direction, self.pos.tp1, self.slippage_bps, is_entry=False)
                            fills.append((q, _round_to_tick(px_tp1, self.tick)))
                            self.pos.exit_legs.append(SimLeg(qty=q, price=float(px_tp1), tag="TP1"))
                            self.pos.remaining_qty -= q
                            self.pos.tp1_done = True

                            # breakeven hop (tighten-only)
                            if self.breakeven_after_tp1:
                                be = self.pos.entry + (
                                    self.breakeven_ticks * self.tick if self.pos.direction == "BUY"
                                    else -self.breakeven_ticks * self.tick
                                )
                                if (self.pos.direction == "BUY" and be > self.pos.active_sl) or \
                                   (self.pos.direction == "SELL" and be < self.pos.active_sl):
                                    self.pos.active_sl = _round_to_tick(be, self.tick)
                    else:
                        # TP2 — close remaining
                        q = self.pos.remaining_qty
                        if q > 0:
                            px_tp2 = _slip(self.pos.direction, self.pos.tp2, self.slippage_bps, is_entry=False)
                            fills.append((q, _round_to_tick(px_tp2, self.tick)))
                            self.pos.exit_legs.append(SimLeg(qty=q, price=float(px_tp2), tag="TP2"))
                            self._close_trade(i, tstamp, fills, detail="TP2")
                            continue

                # timeout/box exit
                if self._timeout_exit(window, i):
                    q = self.pos.remaining_qty
                    if q > 0:
                        px_to_use = _slip(self.pos.direction, close_i, self.slippage_bps, is_entry=False)
                        fills.append((q, _round_to_tick(px_to_use, self.tick)))
                        self.pos.exit_legs.append(SimLeg(qty=q, price=float(px_to_use), tag="TIMEOUT"))
                        self._close_trade(i, tstamp, fills, detail="TIMEOUT")
                        continue

            # ---- look for entries if flat ----
            if not self.pos:
                # Strategy expects OHLC indexed; we can pass window with same columns
                signal = self.strategy.generate_signal(window.set_index(pd.RangeIndex(len(window))), close_i)
                if not signal:
                    continue

                direction = str(signal["signal"]).upper()  # BUY/SELL
                entry = float(signal["entry_price"])
                sl = float(signal["stop_loss"])
                tp = float(signal["target"])
                conf = float(signal.get("confidence", 5.0))
                vol = float(signal.get("market_volatility", 0.0))

                # risk sizing (contracts)
                qty = self._risk_qty_contracts(entry, sl, conf, vol)
                if qty <= 0:
                    continue

                # apply slippage to entry & round to tick
                entry_eff = _round_to_tick(_slip(direction, entry, self.slippage_bps, is_entry=True), self.tick)
                sl_eff = _round_to_tick(sl, self.tick)
                tp_eff = _round_to_tick(tp, self.tick)

                # partials
                if self.partial_enable and qty >= 2:
                    total_lots = qty // max(1, self.lot_size)
                    tp1_lots = max(1, int(round(total_lots * self.partial_ratio)))
                    tp2_lots = max(1, total_lots - tp1_lots)
                    qty_tp1 = tp1_lots * self.lot_size
                    qty_tp2 = tp2_lots * self.lot_size
                    # midpoint TP1
                    tp1 = entry_eff + (tp_eff - entry_eff) / 2.0 if direction == "BUY" else entry_eff - (entry_eff - tp_eff) / 2.0
                    tp2 = tp_eff
                else:
                    qty_tp1 = 0
                    qty_tp2 = qty
                    tp1 = tp_eff
                    tp2 = tp_eff

                self.pos = SimPosition(
                    direction=direction,
                    entry=entry_eff,
                    sl=sl_eff,
                    tp1=_round_to_tick(tp1, self.tick),
                    tp2=_round_to_tick(tp2, self.tick),
                    qty_tp1=int(qty_tp1),
                    qty_tp2=int(qty_tp2),
                    atr_at_entry=vol,
                    open_idx=i,
                    open_time=tstamp,
                    remaining_qty=int(qty),
                    active_sl=sl_eff,
                )

        # EOD flatten (if still open)
        if self.pos:
            last_close = float(self.df["close"].iloc[-1])
            q = self.pos.remaining_qty
            px_eod = _round_to_tick(_slip(self.pos.direction, last_close, self.slippage_bps, is_entry=False), self.tick)
            self.pos.exit_legs.append(SimLeg(qty=q, price=float(px_eod), tag="EOD"))
            self._close_trade(len(self.df) - 1, str(self.df["date"].iloc[-1]), [(q, px_eod)], detail="EOD")

        return self.metrics()

    # --------------------------- finalize -------------------------------- #

    def _close_trade(self, i_out: int, t_out: str, fills: List[Tuple[int, float]], detail: str) -> None:
        assert self.pos, "No open position to close."
        qty_total = sum(q for q, _ in fills)
        px_exit = self._vwap_exit(fills)

        side_mult = 1 if self.pos.direction == "BUY" else -1
        pnl = (px_exit - self.pos.entry) * side_mult * qty_total

        # R multiple (per-contract)
        R = (px_exit - self.pos.entry) / max(1e-9, abs(self.pos.entry - self.pos.sl))
        R *= side_mult

        # deduct fees (round-trip per lot)
        lots = qty_total // max(1, self.lot_size)
        fees = lots * self.fees_per_lot
        pnl_net = pnl - fees

        self.equity_R += R
        self.current_capital += pnl_net

        trade = SimTrade(
            idx_in=self.pos.open_idx,
            idx_out=i_out,
            date_in=self.pos.open_time,
            date_out=t_out,
            symbol=self.symbol,
            direction=self.pos.direction,
            entry=float(self.pos.entry),
            exit_vwap=float(px_exit),
            qty=int(qty_total),
            pnl=float(pnl_net),
            r_mult=float(R),
            details=str(detail),
        )
        self.trades.append(trade)

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    trade.date_in,
                    trade.date_out,
                    trade.symbol,
                    trade.direction,
                    trade.qty,
                    round(trade.entry, 2),
                    round(trade.exit_vwap, 2),
                    round(trade.pnl, 2),
                    round(trade.r_mult, 3),
                    trade.details,
                ]
            )

        self.pos = None

    # ---------------------------- metrics -------------------------------- #

    def metrics(self) -> Dict[str, float]:
        if not self.trades:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "avg_R": 0.0,
                "max_dd_R": 0.0,
                "final_capital": self.current_capital,
                "profit_factor": 0.0,
            }

        pnl = [t.pnl for t in self.trades]
        rs = [t.r_mult for t in self.trades]
        wins = [x for x in pnl if x > 0]
        losses = [x for x in pnl if x <= 0]

        net = sum(pnl)
        avg_R = sum(rs) / len(rs)

        # Drawdown (in R)
        eq = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in rs:
            eq += r
            peak = max(peak, eq)
            dd = eq - peak
            max_dd = min(max_dd, dd)

        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf") if wins else 0.0

        return {
            "trades": len(self.trades),
            "win_rate": 100.0 * len(wins) / len(self.trades),
            "net_pnl": net,
            "avg_R": avg_R,
            "max_dd_R": max_dd,
            "final_capital": self.current_capital,
            "profit_factor": profit_factor,
        }


# ---------------------- convenience: CSV runner ------------------------- #

def run_backtest_from_csv(csv_path: str, symbol: Optional[str] = None) -> Dict[str, float]:
    df = pd.read_csv(csv_path)

    # normalize column names (case-insensitive)
    lower = {c.lower(): c for c in df.columns}
    for need in ("open", "high", "low", "close"):
        if need in lower and lower[need] != need:
            df = df.rename(columns={lower[need]: need})

    if "date" not in df.columns:
        df["date"] = df.index.astype(str)

    engine = BacktestEngine(df, symbol or os.path.basename(csv_path))
    return engine.run()