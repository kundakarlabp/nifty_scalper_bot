# src/strategies/runner.py
from __future__ import annotations

"""
StrategyRunner — production-grade orchestrator
- Fetches OHLC → generates signal → sizes from LIVE equity (with fallback) → executes
- Enforces trading window, daily drawdown caps, loss-streak halts, and max trades/day
- Caches broker equity with refresh cadence; applies equity floor before sizing
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd

from src.config import settings
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.execution.order_executor import OrderExecutor

# Optional broker SDK (runner can work in paper mode)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore

# Optional data source (gracefully degraded if missing)
try:
    # Expecting a LiveKiteSource or similar with a get_spot_ohlc(...) API
    from src.data.source import LiveKiteSource  # type: ignore
except Exception:  # pragma: no cover
    LiveKiteSource = None  # type: ignore


@dataclass
class RiskState:
    trading_day: datetime
    trades_today: int = 0
    consecutive_losses: int = 0
    day_realized_loss: float = 0.0    # realized (booked) loss for the day (₹)
    day_realized_pnl: float = 0.0     # can be positive or negative


class StrategyRunner:
    def __init__(self, kite: Optional[KiteConnect] = None, telegram_controller: Any = None) -> None:
        self.log = logging.getLogger(self.__class__.__name__)
        self.kite = kite
        self.telegram = telegram_controller

        # Core components
        self.strategy = EnhancedScalpingStrategy()
        self.executor = OrderExecutor(kite=self.kite, telegram_controller=self.telegram)

        # Data source (if available)
        self.data_source = None
        if LiveKiteSource is not None:
            try:
                self.data_source = LiveKiteSource(kite=self.kite)
                self.log.info("Data source initialized: LiveKiteSource")
            except Exception as e:
                self.log.warning(f"Data source init failed; proceeding without: {e}")

        # Risk state
        self.risk = RiskState(trading_day=self._today_ist())

        # Equity cache (live-equity sizing)
        self._equity_last_refresh_ts: float = 0.0
        self._equity_cached_value: float = float(settings.risk_default_equity)

        # Dynamic daily loss cap derived from equity snapshot
        self._max_daily_loss_rupees: float = (
            self._equity_cached_value * float(settings.risk_max_daily_drawdown_pct)
        )

        # Market-time window
        self._start_time = self._parse_hhmm(settings.time_filter_start)
        self._end_time = self._parse_hhmm(settings.time_filter_end)

        self.log.info(
            "StrategyRunner ready "
            f"(live_trading={settings.enable_live_trading}, "
            f"use_live_equity={settings.risk_use_live_equity})"
        )

    # --------------------------
    # Public API (called by main)
    # --------------------------

    def process_tick(self, tick: Dict[str, Any]) -> None:
        """Main entry — called by websocket tick thread."""
        if not self._within_trading_window() and not settings.allow_offhours_testing:
            return

        # Day rollover handling
        self._ensure_day_state()

        # Refresh equity snapshot if due
        self._refresh_equity_if_due()

        # Obtain OHLC data
        df = self._fetch_spot_ohlc()
        if df is None or len(df) < int(settings.strategy_min_bars_for_signal):
            return

        # Generate signal (strategy handles indicators + regime + RR filtering)
        signal = self.strategy.generate_signal(df, current_tick=tick)
        if not signal:
            return

        # Risk gates (drawdown / loss-streak / trades-per-day / equity floor)
        if not self._risk_allows_trade(signal):
            self._notify("🚫 Trade blocked by risk controls")
            return

        # Position sizing (live equity → fallback to default)
        qty = self._calculate_quantity(signal)
        if qty <= 0:
            self._notify("⚠️ Quantity calculated as 0; skipping trade")
            return

        # Prepare execution payload
        exec_payload = {
            "action": signal.action,
            "quantity": qty,
            "entry_price": float(signal.entry_price),
            "stop_loss": float(signal.stop_loss),
            "take_profit": float(signal.take_profit),
            "strike": float(signal.strike),
            "option_type": signal.option_type,  # CE/PE
        }

        # Place the order
        placed = self.executor.place_order(exec_payload)

        # Update risk state on acceptance (count the attempt)
        if placed:
            self.risk.trades_today += 1
            self._notify(
                f"✅ Placed: {exec_payload['action']} {qty} {signal.option_type} "
                f"{int(signal.strike)} @ {signal.entry_price:.2f} "
                f"(SL {signal.stop_loss:.2f}, TP {signal.take_profit:.2f})"
            )
        else:
            self._notify("❌ Order placement failed")

    def health_check(self) -> None:
        """Periodic health tasks from main loop."""
        # Equity refresh (in case no ticks)
        self._refresh_equity_if_due(silent=True)
        # Executor health also updates positions/pnl
        try:
            self.executor.health_check()
        except Exception as e:
            self.log.warning(f"Executor health check warning: {e}")

    def shutdown(self) -> None:
        """Graceful teardown."""
        try:
            self.executor.shutdown()
        except Exception:
            pass

    # --------------------------
    # Equity & Risk management
    # --------------------------

    def _refresh_equity_if_due(self, silent: bool = False) -> None:
        """Refresh live equity snapshot from broker at configured cadence."""
        now = time.time()
        if not settings.risk_use_live_equity:
            # Ensure derived cap still uses current cached value (which may be default)
            self._max_daily_loss_rupees = self._equity_cached_value * float(settings.risk_max_daily_drawdown_pct)
            return

        if (now - self._equity_last_refresh_ts) < int(settings.equity_refresh_seconds):
            return

        new_equity = None
        if self.kite is not None:
            try:
                # Prefer margins() → 'equity' available balance; fallback to funds/profile if needed
                # Zerodha returns a dict; we try typical locations
                margins = self.kite.margins()  # type: ignore[attr-defined]
                if isinstance(margins, dict):
                    # Heuristic: try equity segment first
                    for k in ("equity", "available", "net", "final", "cash"):
                        v = margins.get(k)
                        if isinstance(v, (int, float)):
                            new_equity = float(v)
                            break

                if new_equity is None:
                    # Fallback to profile + default equity if margins struct unexpected
                    new_equity = float(settings.risk_default_equity)
            except Exception as e:
                if not silent:
                    self.log.warning(f"Equity refresh failed; using fallback: {e}")

        # Apply result
        if isinstance(new_equity, (int, float)) and new_equity > 0:
            self._equity_cached_value = float(new_equity)
        else:
            # fallback to configured default
            self._equity_cached_value = float(settings.risk_default_equity)

        # Recompute daily loss cap from current snapshot
        self._max_daily_loss_rupees = self._equity_cached_value * float(settings.risk_max_daily_drawdown_pct)
        self._equity_last_refresh_ts = now

        if not silent:
            self.log.info(
                f"Equity snapshot: ₹{self._equity_cached_value:,.0f} | "
                f"Max daily loss: ₹{self._max_daily_loss_rupees:,.0f}"
            )

    def _active_equity(self) -> float:
        """Equity used for sizing (cached live or fallback)."""
        if settings.risk_use_live_equity:
            return float(self._equity_cached_value)
        return float(settings.risk_default_equity)

    def _risk_allows_trade(self, signal) -> bool:
        """All pre-trade risk gates."""
        # Equity floor
        if settings.risk_use_live_equity and self._active_equity() < float(settings.risk_min_equity_floor):
            self.log.warning("Equity below floor; blocking trade")
            return False

        # Daily drawdown gate
        if self.risk.day_realized_loss >= self._max_daily_loss_rupees:
            self.log.warning("Daily drawdown cap reached; blocking trade")
            return False

        # Loss-streak gate
        if self.risk.consecutive_losses >= int(settings.risk_consecutive_loss_limit):
            self.log.warning("Consecutive loss limit reached; blocking trade")
            return False

        # Trades-per-day gate
        if self.risk.trades_today >= int(settings.risk_max_trades_per_day):
            self.log.warning("Max trades/day reached; blocking trade")
            return False

        # Basic SL sanity (avoid divide-by-zero in sizing)
        sl_points = abs(float(signal.entry_price) - float(signal.stop_loss))
        if sl_points <= 0:
            self.log.warning("Invalid SL distance; blocking trade")
            return False

        return True

    def _calculate_quantity(self, signal) -> int:
        """
        Long options sizing:
        - risk_rupees = equity * risk_per_trade
        - rupee risk per 1 lot = sl_points * lot_size
        - lots = floor(risk_rupees / rupee_risk_per_lot), clipped to [min_lots, max_lots]
        """
        equity = self._active_equity()
        risk_rupees = equity * float(settings.risk_risk_per_trade)

        sl_points = abs(float(signal.entry_price) - float(signal.stop_loss))
        lot_size = int(settings.instruments_nifty_lot_size)
        rupee_risk_per_lot = sl_points * lot_size

        if rupee_risk_per_lot <= 0:
            return 0

        lots = int(risk_rupees // rupee_risk_per_lot)
        lots = max(lots, int(settings.instruments_min_lots))
        lots = min(lots, int(settings.instruments_max_lots))

        # Enforce exposure cap (% of equity) roughly via notional bound:
        # notional ≈ entry_price * lot_size * lots
        notional = float(signal.entry_price) * lot_size * lots
        max_notional = equity * float(settings.risk_max_position_size_pct)
        if max_notional > 0 and notional > max_notional:
            # scale down proportionally (ceil to at least min_lots if possible)
            scaled = int(max(max_notional // (float(signal.entry_price) * lot_size), 0))
            lots = max(min(scaled, lots), 0)

        quantity = lots * lot_size
        return int(quantity)

    # --------------------------
    # Data fetching
    # --------------------------

    def _fetch_spot_ohlc(self) -> Optional[pd.DataFrame]:
        """
        Return minute OHLCV DataFrame (ascending index), or None if unavailable.
        Delegates to LiveKiteSource when available; otherwise returns None.
        """
        if self.data_source is None:
            return None

        try:
            # Expecting data source to accept lookback in minutes and return standardized DF
            df = self.data_source.get_spot_ohlc(lookback_minutes=int(settings.data_lookback_minutes))
            # Basic schema sanitation (strategy expects these columns)
            needed = {"open", "high", "low", "close", "volume"}
            if df is None or not isinstance(df, pd.DataFrame) or not needed.issubset(df.columns):
                return None
            df = df.sort_index()
            return df
        except Exception as e:
            self.log.warning(f"OHLC fetch failed: {e}")
            return None

    # --------------------------
    # Day/session helpers
    # --------------------------

    def _ensure_day_state(self) -> None:
        today = self._today_ist()
        if today.date() != self.risk.trading_day.date():
            # Reset daily counters on new trading day
            self.risk = RiskState(trading_day=today)
            self._notify("🔁 New trading day — risk counters reset")

    def _within_trading_window(self) -> bool:
        now_ist = self._now_ist().time()
        return self._start_time <= now_ist <= self._end_time

    @staticmethod
    def _parse_hhmm(text: str):
        from datetime import datetime as _dt
        return _dt.strptime(text, "%H:%M").time()

    @staticmethod
    def _now_ist():
        # IST = UTC+5:30
        return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

    @staticmethod
    def _today_ist():
        now = StrategyRunner._now_ist()
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    # --------------------------
    # Notifications
    # --------------------------

    def _notify(self, msg: str) -> None:
        logging.info(msg)
        try:
            if self.telegram:
                self.telegram.send_message(msg)
        except Exception:
            pass
