# Path: src/strategies/runner.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config import settings
from src.data.source import LiveKiteSource
from src.execution.order_executor import OrderExecutor
from src.utils.account_info import get_equity_estimate

# Strategy (must exist in your tree)
from src.strategies.scalping_strategy import EnhancedScalpingStrategy

# Optional: bring in PositionSizer if present
try:
    from src.risk.position_sizing import PositionSizer
except Exception:
    PositionSizer = None  # type: ignore

# Optional broker SDK for late-binding in /mode live
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore


@dataclass
class RiskState:
    trading_day: datetime
    trades_today: int = 0
    consecutive_losses: int = 0
    day_realized_loss: float = 0.0
    day_realized_pnl: float = 0.0


class StrategyRunner:
    """
    End-to-end runner:
      data -> signal -> risk gates -> sizing -> execution
    Exposes provider/mutator methods used by TelegramController.
    """

    # ---------------- init ----------------
    def __init__(self, kite: Optional["KiteConnect"] = None, telegram_controller: Any = None) -> None:
        self.log = logging.getLogger(self.__class__.__name__)

        # Runtime wiring
        self.kite = kite
        if telegram_controller is None:
            raise RuntimeError("TelegramController must be provided to StrategyRunner.")
        self.telegram = telegram_controller  # back-compat name expected elsewhere
        self.telegram_controller = telegram_controller

        # Core parts
        self.strategy = EnhancedScalpingStrategy()
        self.executor = OrderExecutor(kite=self.kite, telegram_controller=self.telegram)

        # Data source (shadow-safe)
        self.data_source: Optional[LiveKiteSource] = LiveKiteSource(kite=self.kite)
        self.data_source.connect()
        self.log.info("Data source initialized: LiveKiteSource")

        # Risk & equity caches
        self.risk = RiskState(trading_day=self._today_ist())
        self._equity_last_refresh_ts: float = 0.0
        self._equity_cached_value: float = float(settings.risk.default_equity)
        self._max_daily_loss_rupees: float = self._equity_cached_value * float(settings.risk.max_daily_drawdown_pct)

        # Trading window
        self._start_time = self._parse_hhmm(settings.data.time_filter_start)
        self._end_time = self._parse_hhmm(settings.data.time_filter_end)

        # State / diagnostics
        self._paused: bool = False
        self._last_signal_debug: Dict[str, Any] = {"note": "no_evaluation_yet"}
        self._last_flow_debug: Dict[str, Any] = {"note": "no_flow_yet"}
        self._last_error: Optional[str] = None
        self._last_signal_at: float = 0.0
        self._last_fetch_ts: float = 0.0

        # Build PositionSizer if available, else None
        self._sizer: Optional[PositionSizer] = None
        if PositionSizer is not None:
            try:
                self._sizer = PositionSizer.from_settings(
                    risk_per_trade=float(settings.risk.risk_per_trade),
                    min_lots=int(settings.instruments.min_lots),
                    max_lots=int(settings.instruments.max_lots),
                    max_position_size_pct=float(settings.risk.max_position_size_pct),
                )
            except Exception as e:
                self.log.warning("PositionSizer init failed; falling back to inline sizing. %s", e)
                self._sizer = None

        self.log.info(
            "StrategyRunner ready (live_trading=%s, use_live_equity=%s)",
            settings.enable_live_trading, settings.risk.use_live_equity
        )

    # ---------------- lifecycle ----------------
    def start(self) -> None:
        """Hook for main.py; runner loop is driven externally via health loop and /tick."""
        return

    def shutdown(self) -> None:
        try:
            if hasattr(self.executor, "shutdown"):
                self.executor.shutdown()
        except Exception:
            pass

    # ---------------- main tick ----------------
    def process_tick(self, tick: Optional[Dict[str, Any]]) -> None:
        flow: Dict[str, Any] = {
            "within_window": False, "paused": self._paused, "data_ok": False, "bars": 0,
            "signal_ok": False, "rr_ok": True, "risk_gates": {}, "sizing": {}, "qty": 0,
            "executed": False, "reason_block": None,
        }

        try:
            # trading window gate
            if not self._within_trading_window() and not settings.allow_offhours_testing:
                flow["reason_block"] = "off_hours"
                self._last_flow_debug = flow
                return
            flow["within_window"] = True

            # pause gate
            if self._paused:
                flow["reason_block"] = "paused"
                self._last_flow_debug = flow
                return

            # day rollover & equity refresh
            self._ensure_day_state()
            self._refresh_equity_if_due()

            # data fetch
            df = self._fetch_spot_ohlc()
            flow["bars"] = int(len(df) if isinstance(df, pd.DataFrame) else 0)
            if df is None or len(df) < int(settings.strategy.min_bars_for_signal):
                flow["reason_block"] = "insufficient_data"
                self._last_flow_debug = flow
                return
            flow["data_ok"] = True

            # signal
            signal = self._safe_signal(df, current_tick=tick)
            self._last_signal_debug = self._safe_strategy_debug()
            if not signal:
                flow["reason_block"] = self._last_signal_debug.get("reason_block", "no_signal")
                self._last_flow_debug = flow
                return
            flow["signal_ok"] = True

            # RR minimum (if provided by strategy)
            rr_min = float(getattr(settings.strategy, "rr_min", 0.0) or 0.0)
            rr_val = float(signal.get("rr", 0.0) or 0.0)
            if rr_min and rr_val and rr_val < rr_min:
                flow["rr_ok"] = False
                flow["reason_block"] = f"rr<{rr_min}"
                flow["signal"] = {"rr": rr_val, "rr_min": rr_min}
                self._last_flow_debug = flow
                return

            # risk gates
            gates = self._risk_gates_for(signal)
            flow["risk_gates"] = gates
            if not all(gates.values()):
                flow["reason_block"] = "risk_gate_block"
                self._last_flow_debug = flow
                return

            # sizing
            qty, diag = self._calculate_quantity_diag(
                entry=float(signal["entry_price"]),
                stop=float(signal["stop_loss"]),
                lot_size=int(settings.instruments.nifty_lot_size),
                equity=self._active_equity(),
            )
            flow["sizing"] = diag
            flow["qty"] = int(qty)
            if qty <= 0:
                flow["reason_block"] = "qty_zero"
                self._last_flow_debug = flow
                return

            # execution
            exec_payload = {
                "action": str(signal.get("action", "BUY")).upper(),
                "quantity": int(qty),
                "entry_price": float(signal["entry_price"]),
                "stop_loss": float(signal["stop_loss"]),
                "take_profit": float(signal["take_profit"]),
                "strike": float(signal.get("strike", 0.0)),
                "option_type": str(signal.get("option_type", "CE")).upper(),
            }
            placed_id = self.executor.place_order(exec_payload)
            placed_ok = bool(placed_id)
            flow["executed"] = placed_ok
            if not placed_ok:
                flow["reason_block"] = getattr(self.executor, "last_error", "exec_fail")
            else:
                self.risk.trades_today += 1
                self._last_signal_at = time.time()
                self._notify(
                    f"✅ Placed: {exec_payload['action']} x{qty} "
                    f"{exec_payload['option_type']} {int(exec_payload['strike'])} @ {exec_payload['entry_price']:.2f}  "
                    f"(SL {exec_payload['stop_loss']:.2f} · TP {exec_payload['take_profit']:.2f})"
                )

            self._last_flow_debug = flow

        except Exception as e:
            flow["reason_block"] = f"exception:{e.__class__.__name__}"
            self._last_error = str(e)
            self._last_flow_debug = flow
            self.log.exception("process_tick error: %s", e)

    # Public tick for Telegram (/tick, /tickdry)
    def runner_tick(self, *, dry: bool = False) -> Dict[str, Any]:
        prev = bool(settings.allow_offhours_testing)
        try:
            if dry:
                try:
                    setattr(settings, "allow_offhours_testing", True)
                except Exception:
                    pass
            self.process_tick(tick=None)
            return dict(self._last_flow_debug)
        finally:
            try:
                setattr(settings, "allow_offhours_testing", prev)
            except Exception:
                pass

    # ---------------- health & equity ----------------
    def health_check(self) -> None:
        self._refresh_equity_if_due(silent=True)
        try:
            if hasattr(self.executor, "health_check"):
                self.executor.health_check()
        except Exception as e:
            self.log.warning("Executor health check warning: %s", e)

    def _refresh_equity_if_due(self, silent: bool = False) -> None:
        now = time.time()
        # If not using live equity, just keep caps in sync with cached value
        if not settings.risk.use_live_equity:
            self._max_daily_loss_rupees = self._equity_cached_value * float(settings.risk.max_daily_drawdown_pct)
            return
        if (now - self._equity_last_refresh_ts) < int(settings.risk.equity_refresh_seconds):
            return

        # use helper (reads margins() safely if kite present)
        new_eq = get_equity_estimate(self.kite)
        if not isinstance(new_eq, (int, float)) or new_eq <= 0:
            new_eq = float(settings.risk.default_equity)

        self._equity_cached_value = float(new_eq)
        self._max_daily_loss_rupees = self._equity_cached_value * float(settings.risk.max_daily_drawdown_pct)
        self._equity_last_refresh_ts = now

        if not silent:
            self.log.info(
                "Equity snapshot: ₹%s | Max daily loss: ₹%s",
                f"{self._equity_cached_value:,.0f}", f"{self._max_daily_loss_rupees:,.0f}"
            )

    def _active_equity(self) -> float:
        return float(self._equity_cached_value) if settings.risk.use_live_equity else float(settings.risk.default_equity)

    # ---------------- sizing ----------------
    def _calculate_quantity_diag(self, *, entry: float, stop: float, lot_size: int, equity: float) -> Tuple[int, Dict]:
        """
        Use PositionSizer if available; else inline math with caps.
        """
        if PositionSizer is not None and self._sizer is not None:
            qty, lots, diag = self._sizer.size_from_signal(
                entry_price=float(entry),
                stop_loss=float(stop),
                lot_size=int(lot_size),
                equity=float(equity),
            )
            # Enrich for continuity with older diag keys
            diag = dict(diag)
            diag.update({"lots_final": lots, "lot_size": int(lot_size)})
            return int(qty), diag

        # Fallback inline math
        risk_rupees = float(equity) * float(settings.risk.risk_per_trade)
        sl_points = abs(float(entry) - float(stop))
        rupee_risk_per_lot = sl_points * int(lot_size)

        if rupee_risk_per_lot <= 0:
            return 0, {
                "entry": entry, "stop": stop, "equity": equity,
                "risk_per_trade": settings.risk.risk_per_trade,
                "sl_points": sl_points, "rupee_risk_per_lot": rupee_risk_per_lot,
                "lots_raw": 0, "lots_final": 0, "exposure_notional_est": 0.0, "max_notional_cap": 0.0,
                "lot_size": int(lot_size),
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
            notional = float(entry) * int(lot_size) * lots

        qty = lots * int(lot_size)
        diag = {
            "entry": round(entry, 4), "stop": round(stop, 4), "equity": round(float(equity), 2),
            "risk_per_trade": float(settings.risk.risk_per_trade), "lot_size": int(lot_size),
            "sl_points": round(sl_points, 4), "rupee_risk_per_lot": round(rupee_risk_per_lot, 2),
            "lots_raw": int(lots_raw), "lots_final": int(lots),
            "exposure_notional_est": round(notional, 2), "max_notional_cap": round(max_notional, 2),
        }
        return int(qty), diag

    # ---------------- data ----------------
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
                token=token,
                start=start,
                end=end,
                timeframe=str(settings.data.timeframe),
            )
            self._last_fetch_ts = time.time()
            need = {"open", "high", "low", "close", "volume"}
            if df is None or not isinstance(df, pd.DataFrame) or not need.issubset(df.columns):
                return None
            return df.sort_index()
        except Exception as e:
            self.log.warning("OHLC fetch failed: %s", e)
            return None

    # ---------------- diag & status providers ----------------
    def get_last_signal_debug(self) -> Dict[str, Any]:
        return dict(self._last_signal_debug or {})

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

    def build_diag(self) -> Dict[str, Any]:
        """
        Health cards bundle used by /diag and /check (names align with TelegramController).
        """
        checks: List[Dict[str, Any]] = []

        # Data source / broker
        checks.append({
            "name": "Data source (Kite)",
            "ok": self.data_source is not None,
            "detail": "wired" if self.data_source else "missing",
        })

        # Trading window
        checks.append({
            "name": "Trading window",
            "ok": self._within_trading_window() or bool(settings.allow_offhours_testing),
            "hint": f"{settings.data.time_filter_start}-{settings.data.time_filter_end}",
            "detail": "IST ok" if self._within_trading_window() else ("offhours-ok" if settings.allow_offhours_testing else "offhours"),
        })

        # OHLC fetch freshness
        age_s = (time.time() - self._last_fetch_ts) if self._last_fetch_ts else 1e9
        checks.append({
            "name": "OHLC fetch",
            "ok": age_s < 120,  # < 2 min fresh
            "hint": f"age_s={int(age_s)}" if age_s < 1e8 else "never",
        })

        # Equity snapshot
        checks.append({
            "name": "Equity snapshot",
            "ok": self._active_equity() > 0,
            "detail": f"₹{self._active_equity():,.0f}",
            "hint": f"max_loss=₹{self._max_daily_loss_rupees:,.0f}",
        })

        # Risk gates view (last evaluation)
        gates = self._last_flow_debug.get("risk_gates", {}) if isinstance(self._last_flow_debug, dict) else {}
        gates_ok = bool(gates) and all(bool(v) for v in gates.values())
        checks.append({
            "name": "Risk gates",
            "ok": gates_ok,
            "detail": ", ".join([f"{k}={'OK' if v else 'BLOCK'}" for k, v in gates.items()]) if gates else "no-eval",
        })

        # Strategy readiness (bars)
        min_bars = int(getattr(settings.strategy, "min_bars_for_signal", 50))
        bars = int(self._last_flow_debug.get("bars", 0)) if isinstance(self._last_flow_debug, dict) else 0
        checks.append({
            "name": "Strategy readiness",
            "ok": bars >= min_bars,
            "detail": f"bars={bars}",
            "hint": f"min_bars={min_bars}",
        })

        # RR threshold (from last flow)
        rr_ok = bool(self._last_flow_debug.get("rr_ok", True)) if isinstance(self._last_flow_debug, dict) else True
        checks.append({
            "name": "RR threshold",
            "ok": rr_ok,
            "detail": str(self._last_flow_debug.get("signal", {})),
        })

        # Errors
        checks.append({
            "name": "Errors",
            "ok": self._last_error is None,
            "detail": "none" if self._last_error is None else self._last_error,
        })

        # Broker session (if live requested, ensure kite exists)
        live = bool(settings.enable_live_trading)
        checks.append({
            "name": "Broker session",
            "ok": (self.kite is not None) if live else True,
            "detail": "live+kite" if (live and self.kite) else ("dry" if not live else "live but kite=None"),
        })

        ok = all(c.get("ok", False) for c in checks)
        last_sig = (time.time() - self._last_signal_at) < 900 if self._last_signal_at else False  # 15m
        bundle = {
            "ok": ok,
            "checks": checks,
            "last_signal": last_sig,
            "last_flow": dict(self._last_flow_debug),
        }
        return bundle

    # ---------------- mutators used by Telegram ----------------
    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def set_live_mode(self, val: bool) -> None:
        """
        /mode live|dry → flip setting and (if turning ON) late-bind Kite with settings.
        """
        try:
            setattr(settings, "enable_live_trading", bool(val))
        except Exception:
            pass

        if val:
            # If already wired, just reconnect datasource
            if self.kite is not None:
                try:
                    if self.data_source:
                        self.data_source.kite = self.kite
                        self.data_source.connect()
                    self.log.info("🔓 Live mode ON — broker session initialized.")
                except Exception as e:
                    self.log.warning("Data source connect failed: %s", e)
                return

            # Late-bind Kite (only if SDK and creds available)
            if KiteConnect is None:
                self.log.warning("Live requested but kiteconnect package not available.")
                return
            try:
                api_key = str(settings.zerodha.api_key)
                access_token = str(settings.zerodha.access_token)
                if not api_key or not access_token:
                    self.log.warning("Live requested but Zerodha credentials missing.")
                    return
                self.kite = KiteConnect(api_key=api_key)  # type: ignore[call-arg]
                self.kite.set_access_token(access_token)  # type: ignore[call-arg]
                # Rewire into components
                if self.data_source:
                    self.data_source.kite = self.kite
                    self.data_source.connect()
                if hasattr(self.executor, "kite"):
                    self.executor.kite = self.kite
                self.log.info("🔓 Live mode ON — broker session initialized.")
            except Exception as e:
                self.log.warning("Late-bind Kite failed: %s", e)
        else:
            self.log.info("🔒 Dry mode — paper trading only.")

    # Strategy tuning hooks (no-ops if your strategy lacks these attrs)
    def set_min_score(self, n: int) -> None:
        try:
            setattr(settings.strategy, "min_signal_score", int(n))
            if hasattr(self.strategy, "set_min_score"):
                self.strategy.set_min_score(int(n))
        except Exception:
            pass

    def set_conf_threshold(self, x: float) -> None:
        try:
            setattr(settings.strategy, "confidence_threshold", float(x))
            if hasattr(self.strategy, "set_conf_threshold"):
                self.strategy.set_conf_threshold(float(x))
        except Exception:
            pass

    def set_atr_period(self, n: int) -> None:
        try:
            setattr(settings.strategy, "atr_period", int(n))
            if hasattr(self.strategy, "set_atr_period"):
                self.strategy.set_atr_period(int(n))
        except Exception:
            pass

    def set_sl_mult(self, x: float) -> None:
        try:
            setattr(settings.strategy, "atr_sl_multiplier", float(x))
            if hasattr(self.strategy, "set_sl_mult"):
                self.strategy.set_sl_mult(float(x))
        except Exception:
            pass

    def set_tp_mult(self, x: float) -> None:
        try:
            setattr(settings.strategy, "atr_tp_multiplier", float(x))
            if hasattr(self.strategy, "set_tp_mult"):
                self.strategy.set_tp_mult(float(x))
        except Exception:
            pass

    def set_trend_boosts(self, a: float, b: float) -> None:
        try:
            if hasattr(self.strategy, "set_trend_boosts"):
                self.strategy.set_trend_boosts(float(a), float(b))
        except Exception:
            pass

    def set_range_tighten(self, a: float, b: float) -> None:
        try:
            if hasattr(self.strategy, "set_range_tighten"):
                self.strategy.set_range_tighten(float(a), float(b))
        except Exception:
            pass

    # ---------------- misc helpers ----------------
    def _safe_signal(self, df: pd.DataFrame, *, current_tick: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Call strategy.generate_signal defensively; normalize expected keys.
        """
        try:
            sig = self.strategy.generate_signal(df, current_tick=current_tick)
        except TypeError:
            # older signature without current_tick
            sig = self.strategy.generate_signal(df)
        except Exception as e:
            self._last_error = f"signal:{e}"
            return None

        if not sig or not isinstance(sig, dict):
            return None

        # Normalise mandatory fields if present
        need = ("entry_price", "stop_loss", "take_profit", "action")
        if not all(k in sig for k in need):
            return None
        return sig

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
        # IST = UTC+5:30; keep tz-aware here, only format to string for display
        return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

    @staticmethod
    def _today_ist():
        now = StrategyRunner._now_ist()
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

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

    def _safe_strategy_debug(self) -> Dict[str, Any]:
        try:
            if hasattr(self.strategy, "get_debug"):
                x = self.strategy.get_debug()
                if isinstance(x, dict):
                    return x
        except Exception:
            pass
        return {}

    def _notify(self, msg: str) -> None:
        logging.info(msg)
        try:
            if self.telegram:
                self.telegram.send_message(msg)
        except Exception:
            pass