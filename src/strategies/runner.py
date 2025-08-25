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
from typing import Any, Dict, Optional, Tuple

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

        # Controls
        self._paused: bool = False

        # Debug snapshots for Telegram
        self._last_signal_debug: Dict[str, Any] = {"note": "no_evaluation_yet"}
        self._last_flow_debug: Dict[str, Any] = {"note": "no_flow_yet"}

        self.log.info(
            "StrategyRunner ready "
            f"(live_trading={settings.enable_live_trading}, "
            f"use_live_equity={settings.risk_use_live_equity})"
        )

    # --------------------------
    # Public API (called by main)
    # --------------------------

    def process_tick(self, tick: Optional[Dict[str, Any]]) -> None:
        """Main entry — called by websocket tick thread."""
        flow: Dict[str, Any] = {
            "within_window": False,
            "paused": bool(self._paused),
            "data_ok": False,
            "bars": 0,
            "signal_ok": False,
            "rr_ok": True,
            "risk_gates": {},
            "sizing": {},
            "qty": 0,
            "executed": False,
            "reason_block": None,
        }

        try:
            # Trading window
            if not self._within_trading_window() and not settings.allow_offhours_testing:
                flow["reason_block"] = "off_hours"
                self._last_flow_debug = flow
                return
            flow["within_window"] = True

            # Paused?
            if self._paused:
                flow["reason_block"] = "paused"
                self._last_flow_debug = flow
                return

            # Day rollover handling
            self._ensure_day_state()

            # Refresh equity snapshot if due
            self._refresh_equity_if_due()

            # Obtain OHLC data
            df = self._fetch_spot_ohlc()
            flow["bars"] = int(len(df) if isinstance(df, pd.DataFrame) else 0)
            if df is None or len(df) < int(settings.strategy_min_bars_for_signal):
                flow["reason_block"] = "insufficient_data"
                self._last_flow_debug = flow
                return
            flow["data_ok"] = True

            # Generate signal (strategy handles indicators + regime)
            signal = self.strategy.generate_signal(df, current_tick=tick)
            self._last_signal_debug = getattr(self.strategy, "get_debug", lambda: {})()
            if not signal:
                flow["reason_block"] = self._last_signal_debug.get("reason_block", "no_signal")
                self._last_flow_debug = flow
                return
            flow["signal_ok"] = True

            # RR gate (configurable)
            rr_min = None
            try:
                rr_min = float(getattr(settings.strategy, "rr_min", 0.0))
            except Exception:
                rr_min = 0.0
            rr_value = None
            try:
                rr_value = float(getattr(signal, "rr", 0.0))
            except Exception:
                rr_value = 0.0
            if rr_min and rr_value and rr_value < rr_min:
                flow["rr_ok"] = False
                flow["reason_block"] = f"rr<{rr_min}"
                flow["signal"] = {
                    "rr": rr_value,
                    "rr_min": rr_min,
                }
                self._last_flow_debug = flow
                return

            # Summarize signal
            try:
                flow["signal"] = {
                    "action": signal.action,
                    "option_type": signal.option_type,
                    "strike": signal.strike,
                    "entry": signal.entry_price,
                    "sl": signal.stop_loss,
                    "tp": signal.take_profit,
                    "score": getattr(signal, "score", None),
                    "confidence": getattr(signal, "confidence", None),
                    "rr": rr_value if rr_value else self._calc_rr(signal.entry_price, signal.stop_loss, signal.take_profit),
                    "regime": getattr(signal, "regime", None),
                    "reasons": getattr(signal, "reasons", None),
                }
            except Exception:
                pass

            # Risk gates
            gates = self._risk_gates_for(signal)
            flow["risk_gates"] = gates
            if not all(gates.values()):
                flow["reason_block"] = "risk_gate_block"
                self._last_flow_debug = flow
                return

            # Sizing (with diagnostics)
            qty, diag = self._calculate_quantity_diag(
                entry=float(signal.entry_price),
                stop=float(signal.stop_loss),
                lot_size=int(settings.instruments_nifty_lot_size),
                equity=self._active_equity(),
            )
            flow["sizing"] = diag
            flow["qty"] = int(qty)
            if qty <= 0:
                flow["reason_block"] = "qty_zero"
                self._last_flow_debug = flow
                return

            # Execution
            exec_payload = {
                "action": signal.action,
                "quantity": int(qty),
                "entry_price": float(signal.entry_price),
                "stop_loss": float(signal.stop_loss),
                "take_profit": float(signal.take_profit),
                "strike": float(signal.strike),
                "option_type": signal.option_type,  # CE/PE
            }
            placed = self.executor.place_order(exec_payload)
            flow["executed"] = bool(placed)
            if not placed:
                flow["reason_block"] = getattr(self.executor, "last_error", "exec_fail")

            # Update risk state on acceptance (count the attempt)
            if placed:
                self.risk.trades_today += 1
                self._notify(
                    f"✅ Placed: {exec_payload['action']} {qty} {signal.option_type} "
                    f"{int(signal.strike)} @ {signal.entry_price:.2f} "
                    f"(SL {signal.stop_loss:.2f}, TP {signal.take_profit:.2f})"
                )

            self._last_flow_debug = flow

        except Exception as e:
            flow["reason_block"] = f"exception:{e.__class__.__name__}"
            self._last_flow_debug = flow

    # Exposed helper for Telegram: run exactly one decision cycle
    def runner_tick(self, *, dry: bool = False) -> Dict[str, Any]:
        """Run one on-demand tick from Telegram. If dry=True, emulate after-hours allowance."""
        allow_off_before = bool(settings.allow_offhours_testing)
        try:
            if dry:
                setattr(settings, "allow_offhours_testing", True)
            self.process_tick(tick=None)
            return dict(self._last_flow_debug)
        finally:
            setattr(settings, "allow_offhours_testing", allow_off_before)

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
                # Prefer margins() → 'equity' available balance; fallback to default if unknown
                margins = self.kite.margins()  # type: ignore[attr-defined]
                if isinstance(margins, dict):
                    for k in ("equity", "available", "net", "final", "cash"):
                        v = margins.get(k)
                        if isinstance(v, (int, float)):
                            new_equity = float(v)
                            break
                if new_equity is None:
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

    def _risk_gates_for(self, signal) -> Dict[str, bool]:
        """Compute risk gates without side-effects."""
        gates = {
            "equity_floor": True,
            "daily_drawdown": True,
            "loss_streak": True,
            "trades_per_day": True,
            "sl_valid": True,
        }

        # Equity floor
        if settings.risk_use_live_equity and self._active_equity() < float(settings.risk_min_equity_floor):
            gates["equity_floor"] = False

        # Daily drawdown gate
        if self.risk.day_realized_loss >= self._max_daily_loss_rupees:
            gates["daily_drawdown"] = False

        # Loss-streak gate
        if self.risk.consecutive_losses >= int(settings.risk_consecutive_loss_limit):
            gates["loss_streak"] = False

        # Trades-per-day gate
        if self.risk.trades_today >= int(settings.risk_max_trades_per_day):
            gates["trades_per_day"] = False

        # Basic SL sanity
        sl_points = abs(float(signal.entry_price) - float(signal.stop_loss))
        if sl_points <= 0:
            gates["sl_valid"] = False

        return gates

    def _calc_rr(self, entry: float, sl: float, tp: float) -> float:
        try:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            return round((reward / risk) if risk > 0 else 0.0, 2)
        except Exception:
            return 0.0

    def _calculate_quantity_diag(self, *, entry: float, stop: float, lot_size: int, equity: float) -> Tuple[int, Dict]:
        """
        Same logic as _calculate_quantity, but returns a diagnostics dictionary as well.
        """
        risk_rupees = float(equity) * float(settings.risk_risk_per_trade)

        sl_points = abs(float(entry) - float(stop))
        rupee_risk_per_lot = sl_points * int(lot_size)

        if rupee_risk_per_lot <= 0:
            return 0, {
                "entry": entry,
                "stop": stop,
                "equity": equity,
                "risk_per_trade": settings.risk_risk_per_trade,
                "sl_points": sl_points,
                "rupee_risk_per_lot": rupee_risk_per_lot,
                "lots_raw": 0,
                "lots_final": 0,
                "exposure_notional_est": 0.0,
                "max_notional_cap": 0.0,
            }

        lots_raw = int(risk_rupees // rupee_risk_per_lot)
        lots = max(lots_raw, int(settings.instruments_min_lots))
        lots = min(lots, int(settings.instruments_max_lots))

        # Enforce exposure cap
        notional = float(entry) * int(lot_size) * lots
        max_notional = float(equity) * float(settings.risk_max_position_size_pct)
        if max_notional > 0 and notional > max_notional:
            denom = float(entry) * int(lot_size)
            lots_cap = int(max_notional // denom) if denom > 0 else 0
            lots = max(min(lots_cap, lots), 0)

        qty = lots * int(lot_size)
        diag = {
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "equity": round(float(equity), 2),
            "risk_per_trade": float(settings.risk_risk_per_trade),
            "lot_size": int(lot_size),
            "sl_points": round(sl_points, 4),
            "rupee_risk_per_lot": round(rupee_risk_per_lot, 2),
            "lots_raw": int(lots_raw),
            "lots_final": int(lots),
            "exposure_notional_est": round(notional, 2),
            "max_notional_cap": round(max_notional, 2),
        }
        return int(qty), diag

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
    # Telegram providers & controls
    # --------------------------

    def get_last_signal_debug(self) -> Dict[str, Any]:
        return dict(self._last_signal_debug)

    def get_last_flow_debug(self) -> Dict[str, Any]:
        return dict(self._last_flow_debug)

    def get_equity_snapshot(self) -> Dict[str, Any]:
        return {
            "use_live_equity": bool(settings.risk_use_live_equity),
            "equity_cached": round(float(self._equity_cached_value), 2),
            "equity_floor": float(settings.risk_min_equity_floor),
            "max_daily_loss_rupees": round(float(self._max_daily_loss_rupees), 2),
            "refresh_seconds": int(getattr(settings, "equity_refresh_seconds", 60)),
        }

    def get_status_snapshot(self) -> Dict[str, Any]:
        return {
            "time_ist": self._now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            "live_trading": bool(settings.enable_live_trading),
            "broker": "Kite" if self.kite is not None else "Paper",
            "within_window": self._within_trading_window(),
            "paused": self._paused,
            "trades_today": self.risk.trades_today,
            "consecutive_losses": self.risk.consecutive_losses,
            "day_realized_loss": round(self.risk.day_realized_loss, 2),
            "day_realized_pnl": round(self.risk.day_realized_pnl, 2),
            "active_orders": getattr(self.executor, "open_count", 0) if hasattr(self.executor, "open_count") else 0,
        }

    def sizing_test(self, entry: float, sl: float) -> Dict[str, Any]:
        qty, diag = self._calculate_quantity_diag(
            entry=float(entry),
            stop=float(sl),
            lot_size=int(settings.instruments_nifty_lot_size),
            equity=self._active_equity(),
        )
        return {"qty": int(qty), "diag": diag}

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def set_live_mode(self, val: bool) -> None:
        try:
            setattr(settings, "enable_live_trading", bool(val))
        except Exception:
            pass

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