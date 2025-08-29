# Path: src/strategies/runner.py
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from src.config import settings
from src.utils.time_windows import now_ist, TZ
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
    # If set, trading is halted until this timestamp after hitting loss streak
    loss_cooldown_until: Optional[datetime] = None


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

        # Trading window
        self._start_time = self._parse_hhmm(settings.data.time_filter_start)
        self._end_time = self._parse_hhmm(settings.data.time_filter_end)

        # Data source
        self.data_source = None
        self._last_fetch_ts: float = 0.0
        if LiveKiteSource is not None:
            try:
                self.data_source = LiveKiteSource(kite=self.kite)
                self.data_source.connect()

                if self.kite is not None:
                    # Validate configured instrument token with a tiny fetch
                    token = int(getattr(settings.instruments, "instrument_token", 0) or 0)
                    if token > 0:
                        end = self._now_ist().replace(second=0, microsecond=0)
                        start = end - timedelta(minutes=1)
                        df = self.data_source.fetch_ohlc(
                            token=token, start=start, end=end, timeframe="minute"
                        )
                        if not isinstance(df, pd.DataFrame) or df.empty:
                            self.log.warning(
                                "instrument_token %s returned no historical data; "
                                "falling back to symbol lookup if available.",
                                token,
                            )
                if self.data_source is not None:
                    self.log.info("Data source initialized: LiveKiteSource")
                    try:
                        self._fetch_spot_ohlc()
                    except Exception as e:
                        self.log.debug("Initial OHLC fetch failed: %s", e)
            except Exception as e:
                self.log.warning(f"Data source init failed; proceeding without: {e}")

        # Risk + equity cache
        self.risk = RiskState(trading_day=self._today_ist())
        self._equity_last_refresh_ts: float = 0.0
        self._equity_cached_value: float = float(settings.risk.default_equity)
        self._max_daily_loss_rupees: float = self._equity_cached_value * float(settings.risk.max_daily_drawdown_pct)

        # State + debug
        self._paused: bool = False
        self._last_signal_debug: Dict[str, Any] = {"note": "no_evaluation_yet"}
        self._last_flow_debug: Dict[str, Any] = {"note": "no_flow_yet"}
        self.last_plan: Optional[Dict[str, Any]] = None
        self._log_signal_changes_only = os.getenv("LOG_SIGNAL_CHANGES_ONLY", "true").lower() != "false"
        self._last_reason_block: Optional[str] = None
        self._last_has_signal: Optional[bool] = None
        self.eval_count: int = 0
        self.last_eval_ts: Optional[str] = None
        self._trace_remaining: int = 0

        # Runtime flags
        self._last_error: Optional[str] = None
        self._last_signal_at: float = 0.0
        # ensure off-hours notification is not spammed
        self._offhours_notified: bool = False
        # track last notification to avoid spamming identical messages
        self._last_notification: Tuple[str, float] = ("", 0.0)

        self.settings = settings
        self._last_diag_emit_ts: float = 0.0
        self._last_signal_hash: tuple | None = None
        self._last_hb: float = 0.0

        self.log.info(
            "StrategyRunner ready (live_trading=%s, use_live_equity=%s)",
            settings.enable_live_trading, settings.risk.use_live_equity
        )
        # Log initial equity snapshot
        self._refresh_equity_if_due(silent=False)

    # Optional start hook (main calls it if present)
    def start(self) -> None:
        if self.data_source is None:
            return
        try:
            df = self._fetch_spot_ohlc()
            bars = int(len(df)) if isinstance(df, pd.DataFrame) else 0
            self._last_flow_debug["bars"] = bars
        except Exception as e:
            self.log.warning("Initial data fetch failed: %s", e)
            self._last_flow_debug["bars"] = 0

    def _emit_diag(self, plan: dict, micro: dict | None = None):
        msg = (f"diag | within_window={getattr(self, 'within_window', None)} "
               f"regime={plan.get('regime')} score={plan.get('score')} atr%={plan.get('atr_pct')} "
               f"rr={plan.get('rr')} opt={plan.get('option_type')} strike={plan.get('strike')} "
               f"reason_block={plan.get('reason_block')} "
               f"reasons={','.join(plan.get('reasons', []))}")
        if micro:
            msg += f" micro={{spread%:{micro.get('spread_pct')}, depth_ok:{micro.get('depth_ok')}}}"
        self.log.info(msg)
        if getattr(self.settings, "TELEGRAM__PRETRADE_ALERTS", False):
            try:
                self.telegram.send(msg)
            except Exception:
                pass

    def _maybe_emit_minute_diag(self, plan: dict):
        import time
        if not getattr(self.settings, "ENABLE_SIGNAL_DEBUG", False):
            return
        interval = int(getattr(self.settings, "DIAG_INTERVAL_SECONDS", 60))
        now = time.time()
        if now - self._last_diag_emit_ts >= interval:
            self._last_diag_emit_ts = now
            self._emit_diag(plan)

    def _preview_candidate(self, plan: dict, micro: dict | None):
        min_preview = float(getattr(self.settings, "MIN_PREVIEW_SCORE", 8))
        score = float(plan.get("score") or 0.0)
        rb = plan.get("reason_block") or ""
        hard_block = rb in {"outside_window","warmup","cooloff","daily_dd","regime_no_trade"}
        if hard_block or score < min_preview:
            return
        sig = (plan.get("regime"), plan.get("option_type"), plan.get("strike"),
               round(score,1), round(float(plan.get("rr") or 0.0),2))
        if sig == self._last_signal_hash:
            return
        self._last_signal_hash = sig
        text = (f"\U0001F7E1 Candidate | {plan.get('regime')} {plan.get('option_type')} {plan.get('strike')} "
                f"score={score:.1f} rr={plan.get('rr')} entry\u2248{plan.get('entry')} "
                f"sl={plan.get('sl')} tp1={plan.get('tp1')} tp2={plan.get('tp2')} "
                f"reason_block={rb}")
        self.log.info(text)
        if getattr(self.settings, "TELEGRAM__PRETRADE_ALERTS", False):
            try:
                self.telegram.send(text)
            except Exception:
                pass

    def _record_plan(self, plan: Dict[str, Any]) -> None:
        micro = plan.get("micro") or {"spread_pct": 0.0, "depth_ok": False}
        changed = (
            plan.get("has_signal") != self._last_has_signal
            or plan.get("reason_block") != self._last_reason_block
        )
        if (not self._log_signal_changes_only) or changed:
            self.log.info(
                "Signal plan: action=%s %s strike=%s qty=%s regime=%s score=%s atr%%=%.2f spread%%=%.2f depth=%s rr=%.2f sl=%s tp1=%s tp2=%s reason_block=%s",
                plan.get("action"), plan.get("option_type"), plan.get("strike"), plan.get("qty_lots"),
                plan.get("regime"), plan.get("score"), float(plan.get("atr_pct") or 0.0),
                float(micro.get("spread_pct") or 0.0), micro.get("depth_ok"),
                float(plan.get("rr") or 0.0), plan.get("sl"), plan.get("tp1"), plan.get("tp2"),
                plan.get("reason_block"),
            )
        plan["eval_count"] = self.eval_count
        plan["last_eval_ts"] = self.last_eval_ts
        self._last_reason_block = plan.get("reason_block")
        self._last_has_signal = plan.get("has_signal")
        self.last_plan = dict(plan)
        if self._trace_remaining > 0:
            self._trace_remaining -= 1
            micro = plan.get("micro") or {}
            self.log.info(
                "TRACE regime=%s score=%s atr%%=%s rr=%s micro_spread=%s micro_depth=%s entry=%s sl=%s tp1=%s tp2=%s reason_block=%s reasons=%s",
                plan.get("regime"),
                plan.get("score"),
                plan.get("atr_pct"),
                plan.get("rr"),
                micro.get("spread_pct"),
                micro.get("depth_ok"),
                plan.get("entry"),
                plan.get("sl"),
                plan.get("tp1"),
                plan.get("tp2"),
                plan.get("reason_block"),
                ",".join(plan.get("reasons", [])),
            )

        if time.time() - getattr(self, "_last_hb", 0.0) >= 60.0:
            micro = plan.get("micro") or {}
            self.log.info(
                "HB eval=%s regime=%s atr%%=%.2f score=%s spread%%=%s depth=%s block=%s",
                self.eval_count,
                plan.get("regime"),
                float(plan.get("atr_pct") or 0.0),
                int(plan.get("score") or 0),
                micro.get("spread_pct"),
                micro.get("depth_ok"),
                plan.get("reason_block"),
            )
            self._last_hb = time.time()

    # ---------------- main loop entry ----------------
    def process_tick(self, tick: Optional[Dict[str, Any]]) -> None:
        self.eval_count += 1
        self.last_eval_ts = datetime.utcnow().isoformat()
        flow: Dict[str, Any] = {
            "within_window": False, "paused": self._paused, "data_ok": False, "bars": 0,
            "signal_ok": False, "rr_ok": True, "risk_gates": {}, "sizing": {}, "qty": 0,
            "executed": False, "reason_block": None,
        }
        self._last_error = None
        try:
            # fetch data first to allow ADX‑based window override
            df = self._fetch_spot_ohlc()
            flow["bars"] = int(len(df) if isinstance(df, pd.DataFrame) else 0)
            adx_val = None
            try:
                adx_series = df.get("adx")
                if adx_series is None:
                    adx_cols = [c for c in df.columns if c.startswith("adx_")]
                    if adx_cols:
                        adx_series = df[sorted(adx_cols)[-1]]
                if adx_series is not None and len(adx_series):
                    adx_val = float(adx_series.iloc[-1])
            except Exception:
                adx_val = None

            try:
                window_ok = self._within_trading_window(adx_val)
            except TypeError:
                window_ok = self._within_trading_window()
            enable_windows = getattr(settings, "enable_time_windows", True)
            within = (
                (not enable_windows)
                or window_ok
                or bool(settings.allow_offhours_testing)
            )
            if not within:
                flow["risk_gates"] = {"skipped": True}
                flow["reason_block"] = "outside_window"
                self._last_flow_debug = flow
                if not self._offhours_notified:
                    now = self._now_ist().strftime("%H:%M:%S")
                    tz_name = getattr(settings, "tz", "IST")
                    self._notify(
                        f"⏰ Tick blocked outside trading window at {now} {tz_name}"
                    )
                    self._offhours_notified = True
                self.log.debug("Skipping tick: outside trading window")
                return
            flow["within_window"] = True
            self._offhours_notified = False

            # pause
            if self._paused:
                flow["reason_block"] = "paused"
                self._last_flow_debug = flow
                self.log.debug("Skipping tick: runner paused")
                return

            # new day / equity
            self._ensure_day_state()
            self._refresh_equity_if_due()

            # we already fetched df above; validate sufficiency
            self.log.debug("Fetched %s bars", flow["bars"])
            if df is None or len(df) < int(settings.strategy.min_bars_for_signal):
                flow["reason_block"] = "insufficient_data"
                self._last_flow_debug = flow
                self.log.debug(
                    "Signal evaluation skipped: insufficient data (bars=%s, need=%s)",
                    flow["bars"], int(settings.strategy.min_bars_for_signal)
                )
                return
            flow["data_ok"] = True

            # ---- plan
            plan = self.strategy.generate_signal(df, current_tick=tick)
            self._last_signal_debug = getattr(self.strategy, "get_debug", lambda: {})()
            self._maybe_emit_minute_diag(plan)

            last_bar_iso = plan.get("last_bar_ts")
            if last_bar_iso:
                try:
                    lb = datetime.fromisoformat(str(last_bar_iso))
                    if lb.tzinfo is None:
                        lb = lb.replace(tzinfo=TZ)
                    lag_sec = abs((now_ist() - lb).total_seconds())
                    if lag_sec > 90:
                        plan["reason_block"] = "data_stale"
                    elif lb > now_ist() + timedelta(seconds=5):
                        plan["reason_block"] = "clock_skew"
                except Exception:
                    pass

            if plan.get("reason_block"):
                self._record_plan(plan)
                flow["reason_block"] = plan["reason_block"]
                self._last_flow_debug = flow
                self.log.debug("No tradable plan: %s", flow["reason_block"])
                return
            flow["signal_ok"] = True
            flow["plan"] = dict(plan)

            # ---- RR minimum
            rr_min = float(getattr(settings.strategy, "rr_min", 0.0) or 0.0)
            rr_val = float(plan.get("rr", 0.0) or 0.0)
            if rr_min and rr_val and rr_val < rr_min:
                plan["reason_block"] = f"rr<{rr_min}"
                flow["rr_ok"] = False
                flow["reason_block"] = plan["reason_block"]
                flow["plan"] = {**plan, "rr_min": rr_min}
                self._last_flow_debug = flow
                self._record_plan(plan)
                self.log.info("Signal skipped: rr %.2f below minimum %.2f", rr_val, rr_min)
                return

            # store some diagnostics from signal
            flow.update(
                {
                    "regime": plan.get("regime"),
                    "score": plan.get("score"),
                    "rr": plan.get("rr"),
                    "sl": plan.get("stop_loss"),
                    "tp1": plan.get("tp1"),
                    "tp2": plan.get("tp2"),
                }
            )

            # ---- risk gates
            gates = self._risk_gates_for(plan)
            flow["risk_gates"] = gates
            if not all(gates.values()):
                blocked = [k for k, v in gates.items() if not v]
                if "daily_drawdown" in blocked:
                    plan["reason_block"] = "daily_dd_hit"
                elif "loss_streak" in blocked:
                    plan["reason_block"] = "loss_cooloff"
                else:
                    plan["reason_block"] = "risk_gate_block"
                flow["reason_block"] = plan["reason_block"]
                self._last_flow_debug = flow
                self._record_plan(plan)
                self.log.info("Signal blocked by risk gates: %s", blocked)
                return

            # ---- sizing
            qty, diag = self._calculate_quantity_diag(
                entry=float(plan.get("entry")),
                stop=float(plan.get("sl")),
                lot_size=int(settings.instruments.nifty_lot_size),
                equity=self._active_equity(),
            )
            flow["sizing"] = diag
            flow["qty"] = int(qty)
            flow["equity"] = self._active_equity()
            flow["risk_rupees"] = round(
                float(diag.get("rupee_risk_per_lot", 0.0)) * float(diag.get("lots_final", 0)),
                2,
            )
            flow["trades_today"] = self.risk.trades_today
            flow["consecutive_losses"] = self.risk.consecutive_losses
            if qty <= 0:
                plan["reason_block"] = "qty_zero"
                flow["reason_block"] = plan["reason_block"]
                self._last_flow_debug = flow
                self._record_plan(plan)
                self.log.debug("Signal skipped: quantity %s <= 0", qty)
                return

            planned_lots = int(qty / int(settings.instruments.nifty_lot_size))
            plan["qty_lots"] = planned_lots
            quote = {
                "bid": tick.get("bid") if tick else 0.0,
                "ask": tick.get("ask") if tick else 0.0,
                "bid_qty": tick.get("bid_qty") if tick else 0,
                "ask_qty": tick.get("ask_qty") if tick else 0,
                "bid_qty_top5": tick.get("bid_qty_top5") if tick else 0,
                "ask_qty_top5": tick.get("ask_qty_top5") if tick else 0,
            }
            ok_micro, micro = self.executor.micro_ok(
                quote=quote,
                qty_lots=planned_lots,
                lot_size=int(settings.instruments.nifty_lot_size),
                max_spread_pct=float(getattr(settings.executor, "max_spread_pct", 0.35)),
                depth_mult=int(getattr(settings.executor, "depth_multiplier", 5)),
            )
            plan["micro"] = micro
            self._preview_candidate(plan, micro)
            if plan.get("reason_block") in (None, "") and ok_micro and plan.get("score", 0) >= int(settings.strategy.min_signal_score):
                self._emit_diag(plan, micro)
            else:
                if plan.get("reason_block") in ("", None) and not ok_micro:
                    plan["reason_block"] = "microstructure"
                flow["reason_block"] = flow.get("reason_block") or plan.get("reason_block")
                self._last_flow_debug = flow
                self._record_plan(plan)
                return

            self._record_plan(plan)

            # ---- execution (support both executors)
            placed_ok = False
            if hasattr(self.executor, "place_order"):
                exec_payload = {
                    "action": plan["action"],
                    "quantity": int(qty),
                    "entry_price": float(plan.get("entry")),
                    "stop_loss": float(plan.get("sl")),
                    "take_profit": float(plan.get("tp2")),
                    "strike": float(plan["strike"]),
                    "option_type": plan["option_type"],
                }
                placed_ok = bool(self.executor.place_order(exec_payload))
            elif hasattr(self.executor, "place_entry_order"):
                side = "BUY" if str(plan["action"]).upper() == "BUY" else "SELL"
                symbol = getattr(settings.instruments, "trade_symbol", "NIFTY")
                token = int(getattr(settings.instruments, "instrument_token", 0))
                oid = self.executor.place_entry_order(
                    token=token, symbol=symbol, side=side,
                    quantity=int(qty), price=float(plan.get("entry"))
                )
                placed_ok = bool(oid)
                if placed_ok and hasattr(self.executor, "setup_gtt_orders"):
                    try:
                        self.executor.setup_gtt_orders(
                            record_id=oid,
                            sl_price=float(plan.get("sl")),
                            tp_price=float(plan.get("tp2")),
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
                    f"✅ Placed: {plan['action']} {qty} {plan['option_type']} {int(plan['strike'])} "
                    f"@ {float(plan.get('entry')):.2f} (SL {float(plan.get('sl')):.2f}, "
                    f"TP {float(plan.get('tp2')):.2f})"
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
        # clear stale errors if health check completed
        self._last_error = None

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
                    # Typical structure: {'equity': {'net': ..., 'available': {'cash': ...}}}
                    segment = margins.get("equity") if isinstance(margins.get("equity"), dict) else margins
                    if isinstance(segment, dict):
                        # First try direct numeric fields (net/cash/final)
                        for k in ("net", "cash", "final", "equity"):
                            v = segment.get(k)
                            if isinstance(v, (int, float)):
                                new_eq = float(v)
                                break
                        # Then drill into nested 'available' dicts
                        if new_eq is None:
                            avail = segment.get("available")
                            if isinstance(avail, dict):
                                for k in ("cash", "net", "equity", "final"):
                                    v = avail.get(k)
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
        # loss streak cooldown logic
        now = self._now_ist()
        if self.risk.loss_cooldown_until:
            if now < self.risk.loss_cooldown_until:
                gates["loss_streak"] = False
            else:
                # Cool-off finished; reset counters
                self.risk.loss_cooldown_until = None
                self.risk.consecutive_losses = 0
        if gates.get("loss_streak", True) and self.risk.consecutive_losses >= 3:
            gates["loss_streak"] = False
            cutoff = self._parse_hhmm("14:30")
            if now.time() >= cutoff:
                # stop for the rest of the day
                tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                self.risk.loss_cooldown_until = tomorrow
            else:
                self.risk.loss_cooldown_until = now + timedelta(minutes=45)
        if self.risk.trades_today >= int(settings.risk.max_trades_per_day):
            gates["trades_per_day"] = False
        entry = signal.get("entry") or signal.get("entry_price")
        stop = signal.get("sl") or signal.get("stop_loss")
        try:
            entry_f = float(entry)
        except (TypeError, ValueError):
            gates["sl_valid"] = False
            self._last_error = f"invalid entry_price: {entry}"
            self.log.warning("Invalid entry_price: %r", entry)
            return {k: bool(v) for k, v in gates.items()}
        try:
            stop_f = float(stop)
        except (TypeError, ValueError):
            gates["sl_valid"] = False
            self._last_error = f"invalid stop_loss: {stop}"
            self.log.warning("Invalid stop_loss: %r", stop)
            return {k: bool(v) for k, v in gates.items()}
        if abs(entry_f - stop_f) <= float(getattr(settings.executor, "tick_size", 0.0)):
            # stop loss must differ from entry by at least one tick
            gates["sl_valid"] = False
        return {k: bool(v) for k, v in gates.items()}

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
            # Use the larger of configured lookback or required bars for signal.
            lookback = max(
                int(settings.data.lookback_minutes),
                int(settings.strategy.min_bars_for_signal),
            )
            # Add a small buffer (10%) to account for any missing candles.
            lookback = int(lookback * 1.1)

            now = self._now_ist().replace(second=0, microsecond=0)

            # Derive session bounds using configured start/end times.
            session_start = now.replace(
                hour=self._start_time.hour,
                minute=self._start_time.minute,
                second=0,
                microsecond=0,
            )
            session_end = now.replace(
                hour=self._end_time.hour,
                minute=self._end_time.minute,
                second=0,
                microsecond=0,
            )

            if session_end <= session_start:
                session_end += timedelta(days=1)

            if now < session_start:
                # Before today's session start: shift to the previous session
                session_start -= timedelta(days=1)
                session_end -= timedelta(days=1)
                end = session_end
            elif now > session_end:
                # After today's session end: anchor to today's close
                end = session_end
            else:
                # Within the session: end at current time
                end = now

            start = end - timedelta(minutes=lookback)
            if start < session_start:
                start = session_start
            if start >= end:
                self.log.warning(
                    "Adjusted OHLC window %s..%s collapses; aborting",
                    start.isoformat(),
                    end.isoformat(),
                )
                return None

            # Resolve token with fallbacks
            token = int(getattr(settings.instruments, "instrument_token", 0) or 0)
            if token <= 0:
                token = int(getattr(settings.instruments, "spot_token", 0) or 0)

            timeframe = str(getattr(settings.data, "timeframe", "minute"))

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
                    self._last_fetch_ts = time.time()
                    return df.sort_index()

                self.log.error(
                    "Insufficient historical_data (%s<%s) after expanded fetch.",
                    rows,
                    min_bars,
                )
                self._last_error = "no_historical_data"
                # Avoid spamming Telegram with automatic notifications for missing historical data
                self.log.warning(
                    "⚠️ Historical data unavailable from broker — check credentials or subscription."
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
                self._last_fetch_ts = time.time()
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

    def _within_trading_window(self, adx_val: Optional[float] = None) -> bool:
        """Return ``True`` if current IST time falls within the configured window.

        Start and end times are sourced from the environment via ``settings``
        so trading hours can be tuned without modifying code.
        """

        _ = adx_val  # legacy arg ignored; window no longer depends on ADX
        now = self._now_ist().time()
        start = getattr(self, "_start_time", self._parse_hhmm(settings.data.time_filter_start))
        end = getattr(self, "_end_time", self._parse_hhmm(settings.data.time_filter_end))
        return start <= now <= end

    @staticmethod
    def _parse_hhmm(text: str):
        from datetime import datetime as _dt
        return _dt.strptime(text, "%H:%M").time()

    @staticmethod
    def _now_ist():
        """Current time in configured timezone as a timezone-aware datetime."""
        return now_ist()

    @staticmethod
    def _today_ist():
        now = StrategyRunner._now_ist()
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    # ---------------- Telegram helpers & diagnostics ----------------
    def get_last_signal_debug(self) -> Dict[str, Any]:
        return dict(self.last_plan or {})

    def get_recent_bars(self, n: int = 5) -> str:
        if not self.data_source:
            return "data_source_unavailable"
        from src.data.source import render_last_bars
        return render_last_bars(self.data_source, n)

    def enable_trace(self, n: int) -> None:
        self._trace_remaining = int(max(0, n))

    def disable_trace(self) -> None:
        self._trace_remaining = 0

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
                f"tf={getattr(settings.data,'timeframe','minute')} lookback={int(getattr(settings.data,'lookback_minutes',20))}m"
            ),
        })

        # Strategy readiness (min bars)
        ready = isinstance(self._last_flow_debug, dict) and int(self._last_flow_debug.get("bars", 0)) >= int(getattr(settings.strategy, "min_bars_for_signal", 20))
        checks.append({
            "name": "Strategy readiness",
            "ok": ready,
            "detail": f"bars={int(self._last_flow_debug.get('bars', 0))}",
            "hint": f"min_bars={int(getattr(settings.strategy, 'min_bars_for_signal', 20))}",
        })

        # Risk gates last view
        gates = (
            self._last_flow_debug.get("risk_gates", RISK_GATES_SKIPPED)
            if isinstance(self._last_flow_debug, dict)
            else RISK_GATES_SKIPPED
        )
        skipped = (
            gates is RISK_GATES_SKIPPED
            or (isinstance(gates, dict) and bool(gates.get("skipped")))
        )
        gates_dict = gates if isinstance(gates, dict) else {}
        gates_ok = True
        if not skipped and gates_dict:
            gates_ok = all(bool(v) for v in gates_dict.values())
        checks.append({
            "name": "Risk gates",
            "ok": gates_ok,
            "detail": (
                "skipped"
                if skipped
                else ", ".join([f"{k}={'OK' if v else 'BLOCK'}" for k, v in gates_dict.items()]) if gates_dict else "no-eval"
            ),
        })

        # RR check
        rr_ok = bool(self._last_flow_debug.get("rr_ok", True)) if isinstance(self._last_flow_debug, dict) else True
        checks.append({
            "name": "RR threshold",
            "ok": rr_ok,
            "detail": str(self._last_flow_debug.get("plan", {})),
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

        telegram_obj = getattr(self, "telegram", None)
        telegram_ok = bool(
            telegram_obj and telegram_obj.__class__.__name__ != "_NoopTelegram"
        )
        live = bool(settings.enable_live_trading)
        broker_ok = (self.kite is not None) if live else True
        data_fresh = (time.time() - getattr(self, "_last_fetch_ts", 0.0)) < 120
        bars = int(flow.get("bars", 0) or 0)
        min_bars = int(getattr(settings.strategy, "min_bars_for_signal", 20))
        strat_ready = bars >= min_bars
        gates = (
            flow.get("risk_gates", RISK_GATES_SKIPPED)
            if isinstance(flow, dict)
            else RISK_GATES_SKIPPED
        )
        skipped = (
            gates is RISK_GATES_SKIPPED
            or (isinstance(gates, dict) and bool(gates.get("skipped")))
        )
        gates_dict = gates if isinstance(gates, dict) else {}
        gates_ok = True
        if not skipped and gates_dict:
            gates_ok = all(bool(v) for v in gates_dict.values())
        rr_ok = bool(flow.get("rr_ok", True))
        no_errors = (self._last_error is None)

        if skipped:
            gate_status = "skipped"
        elif not gates_dict:
            gate_status = "no-eval"
        else:
            gate_status = "ok" if gates_ok else "blocked"

        return {
            "ok": bool(bundle.get("ok", False)),
            "status_messages": {
                "telegram_wiring": "ok" if telegram_ok else "missing",
                "broker_session": "ok" if broker_ok else ("dry mode" if not live else "missing"),
                "data_feed": "ok" if data_fresh else "stale",
                "strategy_readiness": "ok" if strat_ready else "not ready",
                "risk_gates": gate_status,
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
        market_open = self._within_trading_window(None)
        within_window = (
            (not getattr(settings, "enable_time_windows", True))
            or market_open
            or bool(settings.allow_offhours_testing)
        )
        return {
            "time_ist": self._now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            "live_trading": bool(settings.enable_live_trading),
            "broker": "Kite" if self.kite is not None else "Paper",
            "market_open": market_open,
            "within_window": within_window,
            "daily_dd_hit": self.risk.day_realized_loss >= self._max_daily_loss_rupees,
            "cooloff_until": self.risk.loss_cooldown_until.isoformat() if self.risk.loss_cooldown_until else "-",
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

        # Rewire or initialize data source
        if self.data_source is not None:
            try:
                if hasattr(self.data_source, "set_kite"):
                    self.data_source.set_kite(self.kite)
                else:
                    setattr(self.data_source, "kite", self.kite)
                self.data_source.connect()
            except Exception as e:
                self.log.warning("Data source connect failed: %s", e)
        elif LiveKiteSource is not None:
            try:
                self.data_source = LiveKiteSource(kite=self.kite)
                self.data_source.connect()
                try:
                    self._fetch_spot_ohlc()
                except Exception as e:
                    self.log.debug("Initial OHLC fetch failed: %s", e)
            except Exception as e:
                self.log.warning("Data source init failed: %s", e)
                self.data_source = None

        self.log.info("🔓 Live mode ON — broker session initialized.")

    # ---------------- notify ----------------
    def _notify(self, msg: str) -> None:
        now = time.time()
        last_msg, last_ts = self._last_notification
        # Skip duplicate messages within a short time window
        if msg == last_msg and (now - last_ts) < 300:
            return
        self._last_notification = (msg, now)
        logging.info(msg)
        try:
            if self.telegram:
                self.telegram.send_message(msg)
        except Exception:
            self.log.debug("Failed to send Telegram notification", exc_info=True)
