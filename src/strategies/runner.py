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

# Optional broker SDK (graceful if not installed)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

# Optional live data source (graceful if not present)
try:
    from src.data.source import LiveKiteSource  # type: ignore
except Exception:
    LiveKiteSource = None  # type: ignore


# ================================ Models ================================

@dataclass
class RiskState:
    trading_day: datetime
    trades_today: int = 0
    consecutive_losses: int = 0
    day_realized_loss: float = 0.0
    day_realized_pnl: float = 0.0


# Sentinel used when risk gates are intentionally skipped
RISK_GATES_SKIPPED = object()


# ============================== Runner =================================

class StrategyRunner:
    """
    Pipeline: data → signal → risk gates → sizing → execution.
    TelegramController is provided by main.py; here we only consume it.
    """

    # ---------------- init ----------------
    def __init__(self, kite: Optional[KiteConnect] = None, telegram_controller: Any = None) -> None:
        self.log = logging.getLogger(self.__class__.__name__)
        self.kite = kite

        if telegram_controller is None:
            raise RuntimeError("TelegramController must be provided to StrategyRunner.")
        # keep both names for old controllers
        self.telegram = telegram_controller
        self.telegram_controller = telegram_controller

        # Core components
        self.strategy = EnhancedScalpingStrategy()
        self.executor = OrderExecutor(kite=self.kite, telegram_controller=self.telegram)

        # Data source
        self.data_source = None
        self._last_fetch_ts: float = 0.0
        if LiveKiteSource is not None:
            try:
                self.data_source = LiveKiteSource(kite=self.kite)
                self.data_source.connect()
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

        # State + debug
        self._paused: bool = False
        self._last_signal_debug: Dict[str, Any] = {"note": "no_evaluation_yet"}
        self._last_flow_debug: Dict[str, Any] = {"note": "no_flow_yet"}

        # Runtime flags
        self._last_error: Optional[str] = None
        self._last_signal_at: float = 0.0

        self.log.info(
            "StrategyRunner ready (live_trading=%s, use_live_equity=%s)",
            settings.enable_live_trading, settings.risk.use_live_equity
        )

    # Optional start hook (main calls it if present)
    def start(self) -> None:
        return

    # ---------------- main loop entry ----------------
    def process_tick(self, tick: Optional[Dict[str, Any]]) -> None:
        flow: Dict[str, Any] = {
            "within_window": False, "paused": self._paused, "data_ok": False, "bars": 0,
            "signal_ok": False, "rr_ok": True, "risk_gates": {}, "sizing": {}, "qty": 0,
            "executed": False, "reason_block": None,
        }

        try:
            # window
            if not self._within_trading_window() and not settings.allow_offhours_testing:
                flow["risk_gates"] = {"skipped": True}
                flow["reason_block"] = "off_hours"; self._last_flow_debug = flow; return
            flow["within_window"] = True

            # pause
            if self._paused:
                flow["reason_block"] = "paused"; self._last_flow_debug = flow; return

            # new day / equity
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
            flow["signal"] = dict(signal)

            # ---- RR minimum
            rr_min = float(getattr(settings.strategy, "rr_min", 0.0) or 0.0)
            rr_val = float(signal.get("rr", 0.0) or 0.0)
            if rr_min and rr_val and rr_val < rr_min:
                flow["rr_ok"] = False
                flow["reason_block"] = f"rr<{rr_min}"
                flow["signal"] = {**signal, "rr_min": rr_min}
                self._last_flow_debug = flow
                return

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

            # ---- execution (support both executors)
            placed_ok = False
            if hasattr(self.executor, "place_order"):
                exec_payload = {
                    "action": signal["action"],
                    "quantity": int(qty),
                    "entry_price": float(signal["entry_price"]),
                    "stop_loss": float(signal["stop_loss"]),
                    "take_profit": float(signal["take_profit"]),
                    "strike": float(signal["strike"]),
                    "option_type": signal["option_type"],
                }
                placed_ok = bool(self.executor.place_order(exec_payload))
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
                err = getattr(self.executor, "last_error", None)
                if err:
                    self._notify(f"⚠️ Execution error: {err}")

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

    # one-shot tick used by Telegram
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
        # refresh equity and executor heartbeat
        self._refresh_equity_if_due(silent=True)
        # try a passive data refresh so "Data feed" check gets updated
        try:
            _ = self._fetch_spot_ohlc()
        except Exception as e:
            self.log.debug("Passive data refresh warn: %s", e)
        try:
            if hasattr(self.executor, "health_check"):
                self.executor.health_check()
        except Exception as e:
            self.log.warning("Executor health check warning: %s", e)

    def shutdown(self) -> None:
        """Graceful shutdown used by /stop or process exit."""
        try:
            # IMPORTANT: use cancel_all_orders (compat) instead of nonexistent close_all_positions
            if hasattr(self.executor, "cancel_all_orders"):
                self.executor.cancel_all_orders()
            if hasattr(self.executor, "shutdown"):
                self.executor.shutdown()
        except Exception:
            self.log.warning("Executor shutdown encountered an error", exc_info=True)

    # ---------------- equity & risk ----------------
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
        if abs(float(signal["entry_price"]) - float(signal["stop_loss"])) <= float(getattr(settings.executor, "tick_size", 0.0)):
            # stop loss must differ from entry by at least one tick
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
        Build SPOT OHLC frame using LiveKiteSource with configured lookback.
        If no valid token is configured or broker returns empty data, synthesize a 1-bar DF from LTP.
        """
        if self.data_source is None:
            return None

        try:
            lookback = int(settings.data.lookback_minutes)
            end = self._now_ist().replace(second=0, microsecond=0)
            start = end - timedelta(minutes=lookback)

            # Resolve token with fallbacks
            token = int(getattr(settings.instruments, "instrument_token", 0) or 0)
            if token <= 0:
                token = int(getattr(settings.instruments, "spot_token", 0) or 0)

            timeframe = str(getattr(settings.data, "timeframe", "minute"))
            self._last_fetch_ts = time.time()  # mark an attempt (diag shows freshness)

            if token > 0:
                df = self.data_source.fetch_ohlc(
                    token=token,
                    start=start,
                    end=end,
                    timeframe=timeframe,
                )
                need = {"open", "high", "low", "close", "volume"}
                min_bars = int(getattr(settings.strategy, "min_bars_for_signal", 0))
                valid = isinstance(df, pd.DataFrame) and need.issubset(df.columns)
                rows = len(df) if isinstance(df, pd.DataFrame) else 0

                if not valid or rows < min_bars:
                    # First attempt yielded insufficient data; try again with expanded window
                    if rows < min_bars:
                        self.log.warning(
                            "historical_data short %s<%s; refetching with expanded lookback",
                            rows,
                            min_bars,
                        )
                    else:
                        self.log.warning(
                            "historical_data empty for token=%s interval=%s window=%s..%s; refetching with expanded lookback.",
                            token,
                            timeframe,
                            start.isoformat(),
                            end.isoformat(),
                        )

                    start2 = end - timedelta(minutes=lookback * 2)
                    df2 = self.data_source.fetch_ohlc(
                        token=token,
                        start=start2,
                        end=end,
                        timeframe=timeframe,
                    )
                    if (
                        isinstance(df2, pd.DataFrame)
                        and not df2.empty
                        and need.issubset(df2.columns)
                    ):
                        df = df2.sort_index()
                        valid = True
                        rows = len(df)

                if valid and rows >= min_bars:
                    return df.sort_index()

                self.log.warning(
                    "Insufficient historical_data (%s<%s) after expanded fetch; using LTP fallback.",
                    rows,
                    min_bars,
                )

            # Fallback: synthesize a single bar from trade symbol/token LTP
            sym = getattr(settings.instruments, "trade_symbol", None)
            ltp = self.data_source.get_last_price(sym if sym else token)
            if isinstance(ltp, (int, float)) and ltp > 0:
                ts = end
                df = pd.DataFrame(
                    {"open": [ltp], "high": [ltp], "low": [ltp], "close": [ltp], "volume": [0]},
                    index=[ts],
                )
                return df

            # If we get here, we truly have nothing
            return None

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

    # ---------------- Telegram helpers & diagnostics ----------------
    def get_last_signal_debug(self) -> Dict[str, Any]:
        return dict(self._last_signal_debug)

    # detailed bundle used by /check
    def build_diag(self) -> Dict[str, Any]:
        return self._build_diag_bundle()

    def get_last_flow_debug(self) -> Dict[str, Any]:
        return dict(self._last_flow_debug)

    def _build_diag_bundle(self) -> Dict[str, Any]:
        """Health cards for /diag (compact) and /check (detailed)."""
        checks: List[Dict[str, Any]] = []

        # Telegram wiring
        checks.append({
            "name": "Telegram wiring",
            "ok": bool(self.telegram is not None),
            "detail": "controller attached" if self.telegram else "missing controller",
        })

        # Broker session (live flag + kite object)
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
            "ok": age_s < 120,  # < 2 minutes considered fresh
            "detail": "fresh" if age_s < 120 else "stale/never",
            "hint": (
                f"age={int(age_s)}s "
                f"token={int(getattr(settings.instruments,'instrument_token',0) or getattr(settings.instruments,'spot_token',0) or 0)} "
                f"tf={getattr(settings.data,'timeframe','minute')} lookback={int(getattr(settings.data,'lookback_minutes',15))}m"
            ),
        })

        # Strategy readiness (min bars)
        ready = isinstance(self._last_flow_debug, dict) and int(self._last_flow_debug.get("bars", 0)) >= int(getattr(settings.strategy, "min_bars_for_signal", 50))
        checks.append({
            "name": "Strategy readiness",
            "ok": ready,
            "detail": f"bars={int(self._last_flow_debug.get('bars', 0))}",
            "hint": f"min_bars={int(getattr(settings.strategy, 'min_bars_for_signal', 50))}",
        })

        # Risk gates last view
        gates = (
            self._last_flow_debug.get("risk_gates", RISK_GATES_SKIPPED)
            if isinstance(self._last_flow_debug, dict)
            else RISK_GATES_SKIPPED
        )
        skipped = gates is RISK_GATES_SKIPPED
        gates_ok = True if skipped else (bool(gates) and all(bool(v) for v in gates.values()))
        checks.append({
            "name": "Risk gates",
            "ok": gates_ok,
            "detail": (
                "skipped"
                if skipped
                else ", ".join([f"{k}={'OK' if v else 'BLOCK'}" for k, v in gates.items()]) if gates else "no-eval"
            ),
        })

        # RR check
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

        ok = all(c.get("ok", False) for c in checks)
        last_sig = (time.time() - self._last_signal_at) < 900 if self._last_signal_at else False  # 15min
        bundle = {
            "ok": ok,
            "checks": checks,
            "last_signal": last_sig,
            "last_flow": dict(self._last_flow_debug),
        }
        return bundle

    # compact one-line summary for /diag
    def get_compact_diag_summary(self) -> Dict[str, Any]:
        """Concise status for /diag without building the full multiline text."""
        bundle = self._build_diag_bundle()
        flow = bundle.get("last_flow", {}) if isinstance(bundle, dict) else {}

        telegram_ok = bool(self.telegram is not None)
        live = bool(settings.enable_live_trading)
        broker_ok = (self.kite is not None) if live else True
        data_fresh = (time.time() - getattr(self, "_last_fetch_ts", 0.0)) < 120
        bars = int(flow.get("bars", 0) or 0)
        min_bars = int(getattr(settings.strategy, "min_bars_for_signal", 50))
        strat_ready = bars >= min_bars
        gates = (
            flow.get("risk_gates", RISK_GATES_SKIPPED)
            if isinstance(flow, dict)
            else RISK_GATES_SKIPPED
        )
        skipped = gates is RISK_GATES_SKIPPED
        gates_ok = isinstance(gates, dict) and all(bool(v) for v in gates.values())
        rr_ok = bool(flow.get("rr_ok", True))
        no_errors = (self._last_error is None)

        return {
            "ok": bool(bundle.get("ok", False)),
            "status_messages": {
                "telegram_wiring": "ok" if telegram_ok else "missing",
                "broker_session": "ok" if broker_ok else ("dry mode" if not live else "missing"),
                "data_feed": "ok" if data_fresh else "stale",
                "strategy_readiness": "ok" if strat_ready else "not ready",
                "risk_gates": "skipped" if skipped else ("ok" if gates_ok else "blocked" if gates else "no-eval"),
                "rr_threshold": "ok" if rr_ok else "blocked",
                "errors": "ok" if no_errors else "present",
            },
        }

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

    # ---------------- live-mode wiring ----------------
    def _create_kite_from_settings(self):
        """Create a KiteConnect session from settings.

        Raises:
            RuntimeError: If the broker SDK is missing or credentials are not
                provided.
        """
        if KiteConnect is None:
            msg = "kiteconnect not installed; cannot enter live."
            self.log.warning(msg)
            raise RuntimeError(msg)

        api_key = getattr(settings.zerodha, "api_key", None)
        access_token = getattr(settings.zerodha, "access_token", None)
        if not api_key or not access_token:
            msg = "Zerodha credentials missing; cannot enter live."
            self.log.error(msg)
            raise RuntimeError(msg)

        try:
            k = KiteConnect(api_key=str(api_key))
            k.set_access_token(str(access_token))
            return k
        except Exception as e:
            msg = f"Failed to create KiteConnect session: {e}"
            self.log.warning(msg)
            raise RuntimeError(msg)

    def set_live_mode(self, val: bool) -> None:
        """
        Flip live mode and (if enabling) ensure a live broker session is present, then rewire executor and data source safely.
        """
        try:
            setattr(settings, "enable_live_trading", bool(val))
        except Exception:
            self.log.debug("Unable to set enable_live_trading flag", exc_info=True)

        if not val:
            self.log.info("🔒 Dry mode — paper trading only.")
            return

        # Enabling LIVE: ensure we have a Kite session
        if not self.kite:
            try:
                self.kite = self._create_kite_from_settings()
            except Exception as e:
                msg = f"Broker init failed: {e}"
                self.log.error(msg)
                try:
                    if self.telegram:
                        self.telegram.send_message(msg)
                except Exception:
                    self.log.warning("Failed to notify Telegram about broker init failure", exc_info=True)
                # Re-raise so callers know live mode failed
                raise

        # Rewire executor
        try:
            if hasattr(self.executor, "set_live_broker"):
                self.executor.set_live_broker(self.kite)
            elif hasattr(self.executor, "set_kite"):
                self.executor.set_kite(self.kite)
            else:
                self.executor.kite = self.kite  # best-effort
        except Exception as e:
            self.log.warning("Executor rewire failed: %s", e)

        # Rewire data source
        if self.data_source is not None:
            try:
                if hasattr(self.data_source, "set_kite"):
                    self.data_source.set_kite(self.kite)
                else:
                    setattr(self.data_source, "kite", self.kite)
                self.data_source.connect()
            except Exception as e:
                self.log.warning("Data source connect failed: %s", e)

        self.log.info("🔓 Live mode ON — broker session initialized.")

    # ---------------- notify ----------------
    def _notify(self, msg: str) -> None:
        logging.info(msg)
        try:
            if self.telegram:
                self.telegram.send_message(msg)
        except Exception:
            self.log.debug("Failed to send Telegram notification", exc_info=True)
