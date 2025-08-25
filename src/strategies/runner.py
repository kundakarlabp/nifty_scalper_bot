# Path: src/strategies/runner.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.config import settings
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.execution.order_executor import OrderExecutor

# Optional broker SDK
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

# Optional data source
try:
    from src.data.source import LiveKiteSource  # type: ignore
except Exception:
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
    Orchestrates: data → signal → risk gates → sizing → execution.
    A TelegramController is REQUIRED and is injected from main.py.
    """
    def __init__(self, kite: Optional[KiteConnect] = None, telegram_controller: Any = None) -> None:
        self.log = logging.getLogger(self.__class__.__name__)
        self.kite = kite

        # Telegram is mandatory for runtime control
        if telegram_controller is None:
            raise RuntimeError("TelegramController must be provided to StrategyRunner.")
        # Keep both names for compatibility across your codebase
        self.telegram = telegram_controller
        self.telegram_controller = telegram_controller

        # Core components
        self.strategy = EnhancedScalpingStrategy()
        self.executor = OrderExecutor(kite=self.kite, telegram_controller=self.telegram)

        # Data source (live or shadow)
        self.data_source = None
        if LiveKiteSource is not None:
            try:
                self.data_source = LiveKiteSource(kite=self.kite)
                self.log.info("Data source initialized: LiveKiteSource")
            except Exception as e:
                self.log.warning(f"Data source init failed; proceeding without: {e}")

        # Risk + equity cache
        self.risk = RiskState(trading_day=self._today_ist())
        self._equity_last_refresh_ts: float = 0.0
        self._equity_cached_value: float = float(settings.risk.default_equity)
        self._max_daily_loss_rupees: float = self._equity_cached_value * float(settings.risk.max_daily_drawdown_pct)

        # Trading window
        self._start_time = self._parse_hhmm(settings.data.time_filter_start)
        self._end_time = self._parse_hhmm(settings.data.time_filter_end)

        self._paused: bool = False
        self._last_signal_debug: Dict[str, Any] = {"note": "no_evaluation_yet"}
        self._last_flow_debug: Dict[str, Any] = {"note": "no_flow_yet"}

        self.log.info(
            "StrategyRunner ready (live_trading=%s, use_live_equity=%s)",
            settings.enable_live_trading, settings.risk.use_live_equity
        )

    # ---------------- Main loop entry ----------------

    def process_tick(self, tick: Optional[Dict[str, Any]]) -> None:
        flow: Dict[str, Any] = {
            "within_window": False, "paused": self._paused, "data_ok": False, "bars": 0,
            "signal_ok": False, "rr_ok": True, "risk_gates": {}, "sizing": {}, "qty": 0,
            "executed": False, "reason_block": None,
        }

        try:
            if not self._within_trading_window() and not settings.allow_offhours_testing:
                flow["reason_block"] = "off_hours"; self._last_flow_debug = flow; return
            flow["within_window"] = True

            if self._paused:
                flow["reason_block"] = "paused"; self._last_flow_debug = flow; return

            self._ensure_day_state()
            self._refresh_equity_if_due()

            # ---- data
            df = self._fetch_spot_ohlc()
            flow["bars"] = int(len(df) if isinstance(df, pd.DataFrame) else 0)
            if df is None or len(df) < int(settings.strategy.min_bars_for_signal):
                flow["reason_block"] = "insufficient_data"; self._last_flow_debug = flow; return
            flow["data_ok"] = True

            # ---- signal
            signal = self.strategy.generate_signal(df, current_tick=tick)
            self._last_signal_debug = getattr(self.strategy, "get_debug", lambda: {})()
            if not signal:
                flow["reason_block"] = self._last_signal_debug.get("reason_block", "no_signal")
                self._last_flow_debug = flow; return
            flow["signal_ok"] = True

            # ---- RR minimum
            rr_min = float(getattr(settings.strategy, "rr_min", 0.0) or 0.0)
            rr_val = float(getattr(signal, "risk_reward_ratio", getattr(signal, "rr", 0.0)) or 0.0)
            if rr_min and rr_val and rr_val < rr_min:
                flow["rr_ok"] = False; flow["reason_block"] = f"rr<{rr_min}"
                flow["signal"] = {"rr": rr_val, "rr_min": rr_min}
                self._last_flow_debug = flow; return

            # ---- risk gates
            gates = self._risk_gates_for(signal)
            flow["risk_gates"] = gates
            if not all(gates.values()):
                flow["reason_block"] = "risk_gate_block"; self._last_flow_debug = flow; return

            # ---- sizing
            qty, diag = self._calculate_quantity_diag(
                entry=float(signal["entry_price"]),
                stop=float(signal["stop_loss"]),
                lot_size=int(settings.instruments.nifty_lot_size),
                equity=self._active_equity(),
            )
            flow["sizing"] = diag; flow["qty"] = int(qty)
            if qty <= 0:
                flow["reason_block"] = "qty_zero"; self._last_flow_debug = flow; return

            # ---- execution (supports both old/new executors)
            placed_ok = False
            if hasattr(self.executor, "place_order"):
                placed_ok = bool(self.executor.place_order({
                    "action": signal["action"],
                    "quantity": int(qty),
                    "entry_price": float(signal["entry_price"]),
                    "stop_loss": float(signal["stop_loss"]),
                    "take_profit": float(signal["take_profit"]),
                    "strike": float(signal.get("strike", 0)),
                    "option_type": signal.get("option_type"),
                }))
            elif hasattr(self.executor, "place_entry_order"):
                side = "BUY" if str(signal["action"]).upper() == "BUY" else "SELL"
                symbol = getattr(settings.instruments, "trade_symbol", "NIFTY")
                token = int(getattr(settings.instruments, "instrument_token", 0))
                oid = self.executor.place_entry_order(
                    token=token, symbol=symbol, side=side,
                    quantity=int(qty), price=float(signal["entry_price"])
                )
                placed_ok = bool(oid)
                if placed_ok and hasattr(self.executor, "setup_gtt_orders"):
                    try:
                        self.executor.setup_gtt_orders(
                            record_id=oid,
                            sl_price=float(signal["stop_loss"]),
                            tp_price=float(signal["take_profit"]),
                        )
                    except Exception as e:
                        self.log.warning("setup_gtt_orders failed: %s", e)
            else:
                self.log.error("No known execution method found on OrderExecutor")

            flow["executed"] = placed_ok
            if not placed_ok:
                flow["reason_block"] = getattr(self.executor, "last_error", "exec_fail")

            if placed_ok:
                self.risk.trades_today += 1
                self._notify(
                    f"✅ Placed: {signal['action']} {qty} {signal.get('option_type','')} "
                    f"{int(signal.get('strike', 0))} @ {float(signal['entry_price']):.2f} "
                    f"(SL {float(signal['stop_loss']):.2f}, TP {float(signal['take_profit']):.2f})"
                )

            self._last_flow_debug = flow

        except Exception as e:
            flow["reason_block"] = f"exception:{e.__class__.__name__}"
            self._last_flow_debug = flow
            self.log.exception("process_tick error: %s", e)

    # ----- one‑shot tick used by Telegram
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

    # ---------------- diagnostics for Telegram (/diag, /check) ----------------
    def get_system_diag(self) -> Dict[str, Any]:
        """
        Structured health report with per-file/component checks.
        TelegramController renders this for /diag (compact) and /check (detailed).
        """
        checks: list[Dict[str, Any]] = []

        # a) config
        try:
            ok_conf = bool(settings and settings.telegram.bot_token and settings.telegram.chat_id)
            checks.append({
                "name": "config.py",
                "ok": ok_conf,
                "hint": None if ok_conf else "Telegram envs missing",
            })
        except Exception as e:
            checks.append({"name": "config.py", "ok": False, "hint": f"{e.__class__.__name__}"})

        # b) telegram controller wiring
        try:
            tg = getattr(self, "telegram", None) or getattr(self, "telegram_controller", None)
            ok_tg = (tg is not None) and hasattr(tg, "send_message")
            checks.append({
                "name": "notifications/telegram_controller.py",
                "ok": ok_tg,
                "hint": None if ok_tg else "not wired",
            })
        except Exception as e:
            checks.append({"name": "notifications/telegram_controller.py", "ok": False, "hint": f"{e.__class__.__name__}"})

        # c) data source
        try:
            ok_data = self.data_source is not None
            if ok_data and hasattr(self.data_source, "is_ready"):
                ok_data = bool(self.data_source.is_ready())
            checks.append({
                "name": "data/source.py",
                "ok": ok_data,
                "hint": None if ok_data else "LiveKiteSource not ready",
            })
        except Exception as e:
            checks.append({"name": "data/source.py", "ok": False, "hint": f"{e.__class__.__name__}"})

        # d) strategy
        try:
            ok_strat = self.strategy is not None and hasattr(self.strategy, "generate_signal")
            checks.append({
                "name": "strategies/scalping_strategy.py",
                "ok": ok_strat,
                "hint": None if ok_strat else "generate_signal missing",
            })
        except Exception as e:
            checks.append({"name": "strategies/scalping_strategy.py", "ok": False, "hint": f"{e.__class__.__name__}"})

        # e) executor
        try:
            ok_exec = self.executor is not None and (
                hasattr(self.executor, "place_order") or hasattr(self.executor, "place_entry_order")
            )
            checks.append({
                "name": "execution/order_executor.py",
                "ok": ok_exec,
                "hint": None if ok_exec else "place_order/entry not found",
            })
        except Exception as e:
            checks.append({"name": "execution/order_executor.py", "ok": False, "hint": f"{e.__class__.__name__}"})

        # f) kite connectivity (when live)
        try:
            ok_kite = True
            if settings.enable_live_trading:
                ok_kite = self.kite is not None
                if ok_kite and hasattr(self.kite, "profile"):
                    try:
                        self.kite.profile()  # soft ping; ignore details
                    except Exception:
                        ok_kite = False
            checks.append({
                "name": "broker/kiteconnect",
                "ok": ok_kite,
                "hint": None if ok_kite else "not connected (check access token)",
            })
        except Exception as e:
            checks.append({"name": "broker/kiteconnect", "ok": False, "hint": f"{e.__class__.__name__}"})

        # g) trading window / flow readiness
        try:
            within = self._within_trading_window() or bool(settings.allow_offhours_testing)
            df = self._fetch_spot_ohlc()
            enough = isinstance(df, pd.DataFrame) and len(df) >= int(settings.strategy.min_bars_for_signal)
            checks.append({
                "name": "flow/window & data",
                "ok": within and enough and (not self._paused),
                "hint": (
                    "paused" if self._paused else
                    ("off-hours" if not within else ("insufficient bars" if not enough else None))
                ),
            })
        except Exception as e:
            checks.append({"name": "flow/window & data", "ok": False, "hint": f"{e.__class__.__name__}"})

        # h) risk snapshot
        try:
            self._refresh_equity_if_due(silent=True)
            ok_risk = self._active_equity() >= float(settings.risk.min_equity_floor)
            checks.append({
                "name": "risk/limits",
                "ok": ok_risk,
                "hint": None if ok_risk else "equity below floor",
            })
        except Exception as e:
            checks.append({"name": "risk/limits", "ok": False, "hint": f"{e.__class__.__name__}"})

        report_ok = all(c.get("ok", False) for c in checks)
        return {
            "ok": report_ok,
            "checks": checks,
            "last_signal": getattr(self, "_last_signal_debug", None),
            "last_flow": getattr(self, "_last_flow_debug", None),
        }

    # ---------------- Equity & risk ----------------

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
                    for k in ("equity", "available", "net", "final", "cash"):
                        v = margins.get(k)
                        if isinstance(v, (int, float)):
                            new_eq = float(v); break
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

    def _risk_gates_for(self, signal) -> Dict[str, bool]:
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
        """
        Build a spot OHLC frame using LiveKiteSource.fetch_ohlc with the configured lookback.
        """
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

    # ---------------- Telegram helpers ----------------

    def get_last_signal_debug(self) -> Dict[str, Any]:
        return dict(self._last_signal_debug)

    def get_last_flow_debug(self) -> Dict[str, Any]:
        return dict(self._last_flow_debug)

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

    def sizing_test(self, entry: float, sl: float) -> Dict[str, Any]:
        qty, diag = self._calculate_quantity_diag(
            entry=float(entry), stop=float(sl),
            lot_size=int(settings.instruments.nifty_lot_size),
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

    def _notify(self, msg: str) -> None:
        logging.info(msg)
        try:
            if self.telegram:
                self.telegram.send_message(msg)
        except Exception:
            pass