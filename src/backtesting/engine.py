# src/backtesting/engine.py
"""
Backtesting engine that reuses the live strategy path.

- Reads minute OHLCV (DatetimeIndex).
- Calls EnhancedScalpingStrategy.generate_signal() bar-by-bar.
- Enforces the same risk gates as live via TradingSession + PositionSizer.
- Simulates entries (marketable-limit), TP1 breakeven, ATR trailing, and exits.
- Writes:
  backtests/trades.csv
  backtests/equity_curve.csv
  backtests/summary.json

CLI:
  python -m src.backtesting.engine --from 2024-01-01 --to 2024-01-05 --generate-sample
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import settings
from src.risk.position_sizing import PositionSizer
from src.risk.session import TradingSession, Trade
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.utils.indicators import atr as atr_series

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class SimOrder:
    side: str              # BUY/SELL
    qty: int               # units
    entry_price: float
    stop_loss: float
    target: float
    trail_mult: float      # from strategy signal
    filled: bool = True
    open: bool = True
    tp1_done: bool = False
    tp1_price: Optional[float] = None  # mid target
    breakeven_price: Optional[float] = None


class BacktestEngine:
    def __init__(self, df: pd.DataFrame, initial_capital: float = 100_000.0):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        self.df = df.sort_index().copy()
        self.initial_capital = float(initial_capital)

        # Live-like components
        self.strategy = EnhancedScalpingStrategy(settings.strategy)
        self.session = TradingSession(settings.risk, settings.executor, starting_equity=self.initial_capital)
        self.sizer = PositionSizer(settings.risk, account_size=self.initial_capital)

        # State
        self.orders: List[SimOrder] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        # Pre-compute ATR for trailing proxy
        self.df["ATR"] = atr_series(self.df, period=settings.strategy.atr_period).fillna(method="bfill")

    def _enter(self, ts: pd.Timestamp, signal: Dict) -> None:
        # Risk gates
        deny = self.sizer.check_risk_gates(self.session)
        if deny:
            logger.debug("Denied by risk gate: %s", deny)
            return

        entry = float(signal["entry_price"])
        stop = float(signal["stop_loss"])
        target = float(signal["target"])
        trail_mult = float(signal.get("trail_mult", 1.0))

        sl_points = abs(entry - stop)
        # Size: options are lot-based live, but for backtest we compute units; you can map to lots if you wish.
        qty = self.sizer.calculate_position_size(entry_price=entry, sl_points=sl_points, lot_size=None)
        if qty <= 0:
            return

        side = signal["signal"]
        order = SimOrder(
            side=side, qty=qty, entry_price=entry, stop_loss=stop, target=target, trail_mult=trail_mult
        )
        self.orders.append(order)

        # Session bookkeeping
        t = self.session.on_order_filled_open(
            side=side, symbol="NIFTY-OPT", qty=qty, entry_price=entry, stop_loss=stop, target=target
        )

        logger.info("ENTER %s qty=%d @ %.2f SL=%.2f TP=%.2f", side, qty, entry, stop, target)

        # Save as trade-open record
        self.trades.append({
            "timestamp": ts.isoformat(),
            "action": "ENTER",
            "side": side,
            "qty": qty,
            "price": entry,
            "sl": stop,
            "tp": target,
            "trail_mult": trail_mult,
        })

    def _maybe_tp1_breakeven(self, order: SimOrder, price: float) -> None:
        if order.tp1_done:
            return
        # TP1 halfway to TP
        mid = order.entry_price + (order.target - order.entry_price) * 0.5
        order.tp1_price = mid
        hit = (price >= mid) if order.side == "BUY" else (price <= mid)
        if hit:
            order.tp1_done = True
            # Move SL to breakeven (a tick in our favor)
            if order.side == "BUY":
                order.stop_loss = max(order.stop_loss, order.entry_price + 0.05)
            else:
                order.stop_loss = min(order.stop_loss, order.entry_price - 0.05)
            order.breakeven_price = order.stop_loss

    def _trail(self, order: SimOrder, price: float, atr_val: float) -> None:
        # Simple ATR trail respecting direction and no relaxation
        trail_pts = max(atr_val * max(0.5, order.trail_mult), 0.05)
        if order.side == "BUY":
            new_sl = round(price - trail_pts, 2)
            if new_sl > order.stop_loss:
                order.stop_loss = new_sl
        else:
            new_sl = round(price + trail_pts, 2)
            if new_sl < order.stop_loss:
                order.stop_loss = new_sl

    def _check_exit(self, order: SimOrder, bar_high: float, bar_low: float) -> Optional[float]:
        # Priority: SL first, then TP (conservative fill model)
        if order.side == "BUY":
            if bar_low <= order.stop_loss:
                return float(order.stop_loss)
            if bar_high >= order.target:
                return float(order.target)
        else:
            if bar_high >= order.stop_loss:
                return float(order.stop_loss)
            if bar_low <= order.target:
                return float(order.target)
        return None

    def _close(self, ts: pd.Timestamp, order: SimOrder, exit_price: float, reason: str) -> None:
        if not order.open:
            return
        order.open = False

        # Session trade close
        # Find the last open Trade in session to close
        trade_to_close: Optional[Trade] = next((t for t in self.session.active_trades if t.symbol == "NIFTY-OPT"), None)
        if trade_to_close is None:
            # Fallback: create an ad-hoc trade record to keep PnL consistent
            trade_to_close = Trade(timestamp_open=ts, side=order.side, symbol="NIFTY-OPT", qty=order.qty, entry_price=order.entry_price)
            self.session.active_trades.append(trade_to_close)
        self.session.on_order_filled_close(trade_to_close, exit_price)

        pnl = trade_to_close.pnl
        logger.info("EXIT  %s qty=%d @ %.2f  PnL=%.2f  (%s)", order.side, order.qty, exit_price, pnl, reason)

        self.trades.append({
            "timestamp": ts.isoformat(),
            "action": "EXIT",
            "side": order.side,
            "qty": order.qty,
            "price": exit_price,
            "pnl": pnl,
            "reason": reason,
        })

    def run(self) -> None:
        for ts, row in self.df.iterrows():
            price = float(row["close"])
            atr_val = float(row.get("ATR", 0.5) or 0.5)

            # Generate new signal from strategy
            try:
                sig = self.strategy.generate_signal(self.df.loc[:ts].tail(300), current_price=price)
            except Exception as e:
                logger.error("Strategy error at %s: %s", ts, e, exc_info=True)
                sig = None

            if sig:
                self._enter(ts, sig)

            # Manage & exit open orders using this bar's range
            bar_high = float(row["high"])
            bar_low = float(row["low"])

            for order in list(self.orders):
                if not order.open:
                    continue

                self._maybe_tp1_breakeven(order, price)
                self._trail(order, price, atr_val)

                exit_px = self._check_exit(order, bar_high, bar_low)
                if exit_px is not None:
                    self._close(ts, order, exit_px, reason="TP" if (exit_px == order.target) else "SL")

            # Equity curve point
            self.equity_curve.append({"timestamp": ts.isoformat(), "equity": self.session.equity})

    # ---------- outputs ----------

    def write_outputs(self, outdir: Path) -> Dict[str, float]:
        outdir.mkdir(parents=True, exist_ok=True)
        trades_df = pd.DataFrame(self.trades)
        eq_df = pd.DataFrame(self.equity_curve)

        if not trades_df.empty:
            trades_df.to_csv(outdir / "trades.csv", index=False)
        if not eq_df.empty:
            eq_df.to_csv(outdir / "equity_curve.csv", index=False)

        summary = self._summarize(trades_df, eq_df)
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("Backtest saved to %s", str(outdir))
        return summary

    def _summarize(self, trades_df: pd.DataFrame, eq_df: pd.DataFrame) -> Dict[str, float]:
        if trades_df.empty:
            return {"trades": 0, "profit_factor": 0.0, "win_rate": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0, "final_equity": self.session.equity}

        # PnLs only on EXIT rows
        exits = trades_df[trades_df["action"] == "EXIT"].copy()
        wins = exits[exits["pnl"] > 0]["pnl"].sum()
        losses = exits[exits["pnl"] < 0]["pnl"].sum()
        profit_factor = float(wins / abs(losses)) if losses < 0 else float("inf")
        win_rate = float((exits["pnl"] > 0).mean()) * 100.0

        # Max drawdown on equity
        if not eq_df.empty:
            eq = eq_df["equity"].values.astype(float)
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / (peak + 1e-9)
            max_dd_pct = float(dd.min() * 100.0)
        else:
            max_dd_pct = 0.0

        # Simplified Sharpe (per-bar, assumes 0 RF)
        returns = exits["pnl"] / max(self.initial_capital, 1.0)
        sharpe = float((returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252 * 75))  # 252 days * ~75 1-min bars in a common intraday slice

        return {
            "trades": int(len(exits)),
            "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else float("inf"),
            "win_rate": round(win_rate, 2),
            "max_dd_pct": round(max_dd_pct, 2),
            "sharpe": round(sharpe, 2),
            "final_equity": round(float(self.session.equity), 2),
        }


# ---------- CLI ----------

def _load_input_df(path: Optional[str], date_from: Optional[str], date_to: Optional[str]) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path, parse_dates=["datetime"])
        df = df.rename(columns={"datetime": "datetime"}).set_index("datetime").sort_index()
        return df

    # Otherwise, use synthetic
    from src.backtesting.sample_data_generator import make_synthetic_ohlcv
    if not (date_from and date_to):
        raise ValueError("When no --input is given, both --from and --to are required.")
    return make_synthetic_ohlcv(date_from, date_to)


def main():
    p = argparse.ArgumentParser(description="Backtesting engine")
    p.add_argument("--input", help="CSV with datetime,open,high,low,close,volume", default=None)
    p.add_argument("--from", dest="date_from", help="YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", help="YYYY-MM-DD")
    p.add_argument("--outdir", default="backtests")
    p.add_argument("--generate-sample", action="store_true", help="Ignore --input and generate synthetic OHLCV")
    args = p.parse_args()

    df = _load_input_df(args.input if not args.generate_sample else None, args.date_from, args.date_to)
    engine = BacktestEngine(df, initial_capital=100_000.0)
    engine.run()
    summary = engine.write_outputs(Path(args.outdir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
