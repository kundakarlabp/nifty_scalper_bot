# src/backtesting/backtest_engine.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import settings
from src.strategies.scalping_strategy import DynamicScalpingStrategy
from src.risk.position_sizing import PositionSizing

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, str(settings.log_level).upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class BacktestEngine:
    """
    Simple bar-by-bar backtesting engine using DynamicScalpingStrategy.

    Assumptions:
    - Input `data` is a pandas DataFrame indexed by timestamp with at least a 'close' column.
    - Strategy `generate_signal(df)` returns either:
        None (no trade)
        or a dict with keys like:
            side: "BUY" | "SELL" (optional; default BUY)
            sl: float  (stop distance or stop price context-dependent; we treat as distance)
            tp: float  (optional; not enforced here, but kept for future use)
    - Exit logic: next-bar close (integration smoke test style).
    - Sizing: delegated to PositionSizing(account_size=initial_capital).
    """

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100_000.0) -> None:
        # Defensive copy and basic schema check
        if "close" not in data.columns:
            raise ValueError("Backtest data must contain a 'close' column.")

        self.data = data.copy()
        self.initial_capital = float(initial_capital)
        self.current_capital = float(initial_capital)
        self.positions: List[Dict[str, Any]] = []

        # Strategy & risk
        self.strategy = DynamicScalpingStrategy()
        self.risk_manager = PositionSizing(account_size=self.initial_capital)

        # Config-driven params
        self.min_bars_for_signal: int = int(settings.strategy.min_bars_for_signal or 10)

    # -------------------------------------------------------------------------
    # Core run
    # -------------------------------------------------------------------------
    def run(self) -> None:
        try:
            logger.info("🔁 Starting Backtest...")

            buffer: List[pd.Series] = []
            closes = self.data["close"]

            for idx, (ts, row) in enumerate(self.data.iterrows()):
                # Accumulate rolling window for indicators
                buffer.append(row)
                df_buf = pd.DataFrame(buffer)

                # Warmup
                if len(df_buf) < self.min_bars_for_signal:
                    continue

                # Generate signal
                signal: Optional[Dict[str, Any]] = self.strategy.generate_signal(df_buf)
                if not signal:
                    continue

                # Extract fields with safe defaults
                side = str(signal.get("side", "BUY")).upper()
                entry_price = float(row["close"])

                # Interpret 'sl' as a distance if provided; default small cushion
                sl_dist = float(signal.get("sl", 5.0))
                if sl_dist <= 0:
                    sl_dist = 5.0

                # Convert stop to a price based on direction
                if side == "SELL":
                    stop_loss = entry_price + sl_dist
                else:
                    stop_loss = entry_price - sl_dist

                # Position size via risk manager
                qty = int(self.risk_manager.calculate_position_size(entry_price, stop_loss))
                if qty <= 0:
                    continue

                # Simplified exit: next bar close
                if idx + 1 >= len(closes):
                    break  # no next bar

                exit_price = float(closes.iloc[idx + 1])

                # P&L by direction
                if side == "SELL":
                    pnl = (entry_price - exit_price) * qty
                else:
                    pnl = (exit_price - entry_price) * qty

                self.current_capital += pnl
                self.positions.append(
                    {
                        "timestamp": ts,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "position_size": qty,
                        "pnl": pnl,
                    }
                )

                logger.info(
                    f"📈 {side} @ {ts} | "
                    f"Entry: {entry_price:.2f} | Exit: {exit_price:.2f} | "
                    f"Qty: {qty} | PnL: {pnl:.2f}"
                )

            self._report()

        except Exception as e:
            logger.error(f"❌ Error during backtest: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    def _report(self) -> None:
        total_trades = len(self.positions)
        total_pnl = float(sum(p["pnl"] for p in self.positions))
        win_trades = sum(1 for p in self.positions if p["pnl"] > 0.0)
        win_rate = (win_trades / total_trades * 100.0) if total_trades else 0.0

        logger.info("✅ --- Backtest Summary Report ---")
        logger.info(f"Initial Capital: ₹{self.initial_capital:,.2f}")
        logger.info(f"Final Capital:   ₹{self.current_capital:,.2f}")
        logger.info(f"Total Trades:    {total_trades}")
        logger.info(f"Winning Trades:  {win_trades} ({win_rate:.2f}%)")
        logger.info(f"Net PnL:         ₹{total_pnl:,.2f}")
