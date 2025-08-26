# Path: src/strategies/runner.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from src.config import settings
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.execution.order_executor import OrderExecutor

# --- Optional broker SDK (we can run in paper mode without it)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore

# --- Data source (optional at import time; runner guards its usage)
try:
    from src.data.source import LiveKiteSource  # type: ignore
except Exception:  # pragma: no cover
    LiveKiteSource = None  # type: ignore


@dataclass
class RiskState:
    trading_day: datetime
    trades_today: int = 0
    consecutive_losses: int = 0
    day_realized_loss: float = 0.0
    day_realized_pnl: float = 0.0


class StrategyRunner:
    """
    Pipeline: data → signal → risk gates → sizing → execution.
    TelegramController is passed in by main.py; we never import telegram here.
    """

    # ---------------- ctor ----------------
    def __init__(self, kite: Optional["KiteConnect"] = None, telegram_controller: Any = None) -> None:
        self.log = logging.getLogger(self.__class__.__name__)

        if telegram_controller is None:
            raise RuntimeError("StrategyRunner requires a TelegramController instance.")
        # keep both names for backward-compat with older controllers
        self.telegram = telegram_controller
        self.telegram_controller = telegram_controller

        # broker + components
        self.kite: Optional["KiteConnect"] = kite
        self.strategy = EnhancedScalpingStrategy()
        self.executor = OrderExecutor(kite=self.kite, telegram_controller=self.telegram)

        self.data_source = None
        if LiveKiteSource is not None:
            try:
                self.data_source = LiveKiteSource(kite=self.kite)
                self.data_source.connect()
                self.log.info("Data source initialized: LiveKiteSource")
            except Exception as e:
                self.log.warning("Data source init failed (continuing in paper): %s", e)

        # risk & equity cache
        self.risk = RiskState(trading_day=self._today_ist())
        self._equity_last_refresh_ts: float = 0.0
        self._equity_cached_value: float = float(settings.risk.default_equity)
        self._max_daily_loss_rupees: float = self._equity_cached_value * float(settings.risk.max_daily_drawdown_pct)

        # trading window
        self._start_time = self._parse_hhmm(settings.data.time_filter_start)
        self._end_time = self._parse_hhmm(settings.data.time_filter_end)

        # runtime
        self._paused: bool = False
        self._last_error: Optional[str] = None
        self._last_signal_at: float = 0.0
        self._last_fetch_ts: float = 0.0
        self._last_signal_debug: Dict[str, Any] = {"note": "no_evaluation_yet"}
        self._last_flow_debug: Dict[str, Any] = {"note": "no_flow_yet"}

        self.log.info(
            "StrategyRunner ready (live_trading=%s, use_live_equity=%s)",
            settings.enable_live_trading, settings.risk.use_live_equity
        )

    # ---------------- public: main work ----------------
    def process_tick(self, tick: Optional[Dict[str, Any]]) -> None:
        flow: Dict[str, Any] = {
            "within_window": False, "paused": self._paused, "data_ok": False, "bars": 0,
            "signal_ok": False, "rr_ok": True, "risk_gates": {}, "sizing": {}, "qty": 0,
            "executed": False, "reason_block": None,
        }

        try:
            # trading hours
            if not self._within_trading_window() and not settings.allow_offhours_testing:
                flow["reason_block"] = "off_hours"; self._last_flow_debug = flow; return
            flow["within_window"] = True

            if self._paused:
                flow["reason_block"] = "paused"; self._last_flow_debug = flow; return

            # day rollover / equity
            self._ensure_day_state()
            self._refresh_equity_if_due(silent=True)

            # data
            df = self._fetch_spot_ohlc()
            flow["bars"] = int(len(df) if isinstance(df, pd.DataFrame) else 0)
            if df is None or len(df) < int(settings.strategy.min_bars_for_signal):
                flow["reason_block"] = "insufficient_data"; self._last_flow_debug = flow; return
            flow["data_ok"] = True

            # signal
            signal = self.strategy.generate_signal(df, current_tick=tick)
            self._last_signal_debug = getattr(self.strategy, "get_debug", lambda: {})()
            if not signal:
                flow["reason_block"] = self._last_signal_debug.get("reason_block", "no_signal")
                self._last_flow_debug = flow; return
            flow["signal_ok"] = True

            # RR threshold
            rr_min = float(getattr(settings.strategy, "rr_min", 0.0) or 0.0)
            rr_val = float(signal.get("rr", 0.0) or 0.0)
            if rr_min and rr_val and rr_val < rr_min:
                flow["rr_ok"] = False
                flow["reason_block"] = f"rr<{rr_min}"
                flow["signal"] = {"rr": rr_val, "rr_min": rr_min}
                self._last_flow_debug = flow; return

            # risk gates
            gates = self._risk_gates_for(signal)
            flow["risk_gates"] = gates
            if not all(gates.values()):
                flow["reason_block"] = "risk_gate_block"; self._last_flow_debug = flow; return

            # sizing
            qty, diag = self._calculate_quantity_diag(
                entry=float(signal["entry_price"]),
                stop=float(signal["stop_loss"]),
                lot_size=int(settings.instruments.nifty_lot_size),
                equity=self._active_equity(),
            )
            flow["sizing"] = diag; flow["qty"] = int(qty)
            if qty <= 0:
                flow["reason_block"] = "qty_zero"; self._last_flow_debug = flow; return

            # execution
            placed_ok = False
            if hasattr(self.executor, "place_order"):
                payload = {
                    "action": signal["action"],
                    "quantity": int(qty),
                    "entry_price": float(signal["entry_price"]),
                    "stop_loss": float(signal["stop_loss"]),
                    "take_profit": float(signal["take_profit"]),
                    "strike": float(signal["strike"]),
                    "option_type": signal["option_type"],
                }
                placed_ok = bool(self.executor.place_order(payload))
            else:
                self.log.error("OrderExecutor missing place_order()")

            flow["executed"] = placed_ok
            if not placed_ok:
                flow["reason_block"] = getattr(self.executor, "last_error", "exec_fail")

            if placed_ok:
                self.risk.trades_today += 1
                self._last_signal_at = time.time()
                self._notify(
                    f"✅ Placed: {signal['action']} {qty} {signal['option_type']} {int(signal['strike'])} "
                    f"@ {float(signal['entry_price']):.2f} (SL {float(signal['stop_loss']):.2f}, "
                    f"TP {float(signal['take_profit']):.2f})"
                )

            self._last_flow_debug = flow

        except Exception as e:
            flow["reason_block"] = f"exception:{e.__class__.__name__}"
            self._last_error = str(e)
            self._last_flow_debug = flow
            self.log.exception("process_tick error: %s", e)

    # one-shot tick for /tick and /tickdry
    def runner_tick(self, *, dry: bool = False) -> Dict[str, Any]:
        prev = bool(settings.allow_offhours_testing)
        try:
            if dry:
                setattr(settings, "allow_offhours_testing", True)
            self.process_tick(tick=None)
            return dict(self._last_flow_debug)
        finally:
            setattr(settings, "allow_offhours_testing", prev)

    def health_check(self) -> None:
        self._refresh_equity_if_due(silent=True)
        try:
            if hasattr(self.executor, "health_check"):
                self.executor.health_check()
        except Exception as e:
            self.log.warning("Executor health check warning: %s", e)

    def shutdown(self) -> None:
        try:
            if hasattr(self.executor, "shutdown"):
                self.executor.shutdown()
        except Exception:
            pass

    # ---------------- live / mode toggling ----------------
    @staticmethod
    def _build_kite_from_settings() -> Optional["KiteConnect"]:
        """Create a KiteConnect instance from env/settings; return None on failure."""
        if not settings.enable_live_trading:
            return None
        if KiteConnect is None:
            raise RuntimeError("ENABLE_LIVE_TRADING=true but kiteconnect is not installed.")
        api_key = getattr(settings.zerodha, "api_key", None)
        access = getattr(settings.zerodha, "access_token", None)
        if not api_key or not access:
            raise RuntimeError("Zerodha credentials missing (API key / access token).")
        kite = KiteConnect(api_key=str(api_key))
        kite.set_access_token(str(access))
        return kite

    def _rewire_components(self, kite: Optional["KiteConnect"]) -> None:
        """Swap broker session on the fly (used by /mode)."""
        self.kite = kite
        # rewire executor
        self.executor = OrderExecutor(kite=self.kite, telegram_controller=self.telegram)
        # rewire datasource
        if LiveKiteSource is not None:
            self.data_source = LiveKiteSource(kite=self.kite)
            try:
                self.data_source.connect()
            except Exception as e:
                self.log.warning("Data source connect failed: %s", e)

    def set_live_mode(self, val: bool) -> None:
        """
        Flip live mode and fully rewire broker/data/executor.
        This is what fixes your 'live but kite=None' situation.
        """
        try:
            setattr(settings, "enable_live_trading", bool(val))
        except Exception:
            pass

        try:
            if val:
                kite = self._build_kite_from_settings()
                self._rewire_components(kite)
                self.log.info("🔓 Live mode ON — broker session initialized.")
            else:
                self._rewire_components(None)
                self.log.info("🔒 Dry mode — paper trading only.")
        except Exception as e:
            # If something failed, fall back to paper so bot continues to run
            self._rewire_components(None)
            self.log.error("Failed to enable live mode (%s). Falling back to paper.", e)
            self._notify(f"⚠️ Live mode failed: {e}")

    # ---------------- equity & risk helpers ----------------
    def _refresh_equity_if_due(self, silent: bool = False) -> None:
        now = time.time()
        if not settings.risk.use_live_equity:
            self._max_daily_loss_rupees = self._equity_cached_value * float(settings.risk.max_daily_drawdown_pct)
            return
        if (now - self._equity_last_refresh_ts) < int(settings.risk.equity_refresh_seconds):
            return

        new_eq = None
        if self.kite is not None:
            try:
                margins = self.kite.margins()  # type: ignore[attr-defined]
                if isinstance(margins, dict):
                    # try a few common shapes
                    for k in ("equity", "available", "net", "final", "cash"):
                        v = margins.get(k)
                        if isinstance(v, (int, float)):
                            new_eq = float(v)
                            break
                if new_eq is None:
                    new_eq = float(settings.risk.default_equity)
            except Exception as e:
                if not silent:
                    self.log.warning("Equity refresh failed; using fallback: %s", e)

        self._equity_cached_value = float(new_eq) if (isinstance(new_eq, (int, float)) and new_eq > 0) \
            else float(settings.risk.default_equity)
        self._max_daily_loss_rupees = self._equity_cached_value * float(settings.risk.max_daily_drawdown_pct)
        self._equity_last_refresh_ts = now

        if not silent:
            self.log.info(
                "Equity snapshot: ₹%s | Max daily loss: ₹%s",
                f"{self._equity_cached_value:,.0f}", f"{self._max_daily_loss_rupees:,.0f}"
            )

    def _active_equity(self) -> float:
        return float(self._equity_cached_value) if settings.risk.use_live_equity else float(settings.risk.default_equity)

    def _risk_gates_for(self, signal: Dict[str, Any]) -> Dict[str, bool]:
        gates = {"equity_floor": True, "daily_drawdown": True, "loss_streak": True,
                 "trades_per_day": True, "sl_valid": True}
        if settings.risk.use_live_equity and self._active_equity() < float(settings.risk.min_equity_floor):
            gates["equity_floor"] = False
        if self.risk.day_realized_loss >= self._max_daily_loss_rupees:
            gates["daily_drawdown"] = False
        if self.risk.consecutive_losses >= int(settings.risk.consecutive_loss_limit):
            gates["loss_streak"] = False
        if self.risk.trades_today >= int(settings.risk.max_trades_per_day):
            gates["trades_per_day"] = False
        if abs(float(signal["entry_price"]) - float(signal["stop_loss"])) <= 0:
            gates["sl_valid"] = False
        return gates

    def _calculate_quantity_diag(self, *, entry: float, stop: float, lot_size: int, equity: float) -> Tuple[int, Dict]:
        risk_rupees = float(equity) * float(settings.risk.risk_per_trade)
        sl_points = abs(float(entry) - float(stop))
        rupee_risk_per_lot = sl_points * int(lot_size)

        if rupee_risk_per_lot <= 0:
            return 0, {
                "entry": entry, "stop": stop, "equity": equity,
                "risk_per_trade": settings.risk.risk_per_trade,
                "sl_points": sl_points, "rupee_risk_per_lot": rupee_risk_per_lot,
                "lots_raw": 0, "lots_final": 0, "exposure_notional_est": 0.0, "max_notional_cap": 0.0,
            }

        lots_raw = int(risk_rupees // rupee_risk_per_lot)
        lots = max(lots_raw, int(settings.instruments.min_lots))
        lots = min(lots, int(settings.instruments.max_lots))

        notional = float(entry) * int(lot_size) * lots
        max_notional = float(equity) * float(settings.risk.max_position_size_pct)
        if max_notional > 0 and notional > max_notional:
            denom = float(entry) * int(lot_size)
            lots_cap = int(max_notional // denom) if denom > 0 else 0
            lots = max(min(lots_cap, lots), 0)

        qty = lots * int(lot_size)
        diag = {
            "entry": round(entry, 4), "stop": round(stop, 4), "equity": round(float(equity), 2),
            "risk_per_trade": float(settings.risk.risk_per_trade), "lot_size": int(lot_size),
            "sl_points": round(sl_points, 4), "rupee_risk_per_lot": round(rupee_risk_per_lot, 2),
            "lots_raw": int(lots_raw), "lots_final": int(lots),
            "exposure_notional_est": round(notional, 2), "max_notional_cap": round(max_notional, 2),
        }
        return int(qty), diag

    # ---------------- data helpers ----------------
    def _fetch_spot_ohlc(self) -> Optional[pd.DataFrame]:
        if self.data_source is None:
            return None
        try:
            lookback = int(settings.data.lookback_minutes)
            end = self._now_ist().replace(second=0, microsecond=0)
            start = end - timedelta(minutes=lookback)
            token = int(getattr(settings.instruments, "instrument_token", 0))
            if token <= 0:
                return None
            df = self.data_source.fetch_ohlc(
                token=token, start=start, end=end, timeframe=str(settings.data.timeframe),
            )
            self._last_fetch_ts = time.time()
            need = {"open", "high", "low", "close", "volume"}
            if df is None or not isinstance(df, pd.DataFrame) or not need.issubset(df.columns):
                return None
            return df.sort_index()
        except Exception as e:
            self.log.warning("OHLC fetch failed: %s", e)
            return None

    # ---------------- session/window ----------------
    def _ensure_day_state(self) -> None:
        today = self._today_ist()
        if today.date() != self.risk.trading_day.date():
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
        return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

    @staticmethod
    def _today_ist():
        now = StrategyRunner._now_ist()
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    # ---------------- diagnostics for Telegram ----------------
    def get_last_signal_debug(self) -> Dict[str, Any]:
        return dict(self._last_signal_debug)

    def _build_diag_bundle(self) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []

        # Controller attached
        checks.append({
            "name": "Telegram wiring",
            "ok": bool(self.telegram is not None),
            "detail": "controller attached" if self.telegram else "missing controller",
        })

        # Broker session
        live = bool(settings.enable_live_trading)
        checks.append({
            "name": "Broker session",
            "ok": (self.kite is not None) if live else True,
            "detail": "live mode with kite" if (live and self.kite) else ("dry mode" if not live else "live but kite=None"),
        })

        # Data feed freshness
        age_s = (time.time() - self._last_fetch_ts) if self._last_fetch_ts else 1e9
        checks.append({
            "name": "Data feed",
            "ok": age_s < 120,
            "detail": "fresh" if age_s < 120 else "stale/never",
            "hint": f"age={int(age_s)}s" if age_s < 1e8 else "no fetch yet",
        })

        # Strategy readiness
        ready = int(self._last_flow_debug.get("bars", 0)) >= int(getattr(settings.strategy, "min_bars_for_signal", 50))
        checks.append({
            "name": "Strategy readiness",
            "ok": bool(ready),
            "detail": f"bars={int(self._last_flow_debug.get('bars', 0))}",
            "hint": f"min_bars={int(getattr(settings.strategy, 'min_bars_for_signal', 50))}",
        })

        # Risk gates last eval
        gates = self._last_flow_debug.get("risk_gates", {}) if isinstance(self._last_flow_debug, dict) else {}
        gates_ok = bool(gates) and all(bool(v) for v in gates.values())
        checks.append({
            "name": "Risk gates",
            "ok": gates_ok,
            "detail": ", ".join([f"{k}={'OK' if v else 'BLOCK'}" for k, v in gates.items()]) if gates else "no-eval",
        })

        # RR status
        rr_ok = bool(self._last_flow_debug.get("rr_ok", True))
        checks.append({"name": "RR threshold", "ok": rr_ok, "detail": str(self._last_flow_debug.get("signal", {}))})

        # Errors
        checks.append({"name": "Errors", "ok": self._last_error is None, "detail": self._last_error or "none"})

        ok = all(c.get("ok", False) for c in checks)
        last_sig = (time.time() - self._last_signal_at) < 900 if self._last_signal_at else False

        return {"ok": ok, "checks": checks, "last_signal": last_sig, "last_flow": dict(self._last_flow_debug)}

    # Expose for Telegram wiring (main wires diag_provider=runner.build_diag)
    def build_diag(self) -> Dict[str, Any]:
        return self._build_diag_bundle()

    # Back-compat name used earlier (Telegram uses this too)
    def get_last_flow_debug(self) -> Dict[str, Any]:
        return self._build_diag_bundle()

    def get_equity_snapshot(self) -> Dict[str, Any]:
        return {
            "use_live_equity": bool(settings.risk.use_live_equity),
            "equity_cached": round(float(self._equity_cached_value), 2),
            "equity_floor": float(settings.risk.min_equity_floor),
            "max_daily_loss_rupees": round(float(self._max_daily_loss_rupees), 2),
            "refresh_seconds": int(settings.risk.equity_refresh_seconds),
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

    # ---------------- small controls used by Telegram tuning ----------------
    def pause(self) -> None: self._paused = True
    def resume(self) -> None: self._paused = False

    # optional strategy tuning hooks (safe if missing)
    def set_min_score(self, n: int) -> None:
        try: setattr(settings.strategy, "min_score", int(n))
        except Exception: pass

    def set_conf_threshold(self, x: float) -> None:
        try: setattr(settings.strategy, "conf_threshold", float(x))
        except Exception: pass

    def set_atr_period(self, n: int) -> None:
        try: setattr(settings.strategy, "atr_period", int(n))
        except Exception: pass

    def set_sl_mult(self, x: float) -> None:
        try: setattr(settings.strategy, "sl_atr_mult", float(x))
        except Exception: pass

    def set_tp_mult(self, x: float) -> None:
        try: setattr(settings.strategy, "tp_atr_mult", float(x))
        except Exception: pass

    def set_trend_boosts(self, tp_boost: float, sl_relax: float) -> None:
        try:
            setattr(settings.strategy, "trend_tp_boost", float(tp_boost))
            setattr(settings.strategy, "trend_sl_relax", float(sl_relax))
        except Exception: pass

    def set_range_tighten(self, tp_tighten: float, sl_tighten: float) -> None:
        try:
            setattr(settings.strategy, "range_tp_tighten", float(tp_tighten))
            setattr(settings.strategy, "range_sl_tighten", float(sl_tighten))
        except Exception: pass

    # ---------------- notify ----------------
    def _notify(self, msg: str) -> None:
        logging.info(msg)
        try:
            if self.telegram:
                self.telegram.send_message(msg)
        except Exception:
            pass