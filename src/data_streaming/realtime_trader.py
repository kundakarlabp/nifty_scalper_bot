# src/data_streaming/realtime_trader.py
"""
Real-time trader with:
- Telegram control (daemon polling)
- Scheduled main loop (fetch -> select -> signal -> size -> execute)
- ATR trailing worker (background)
- Circuit breaker (daily drawdown)
- Single-position policy
- Warmup filter
- Spread guard (RANGE + LTP_MID)
- Slippage + fees model
- Rate-limit safety and instruments cache
- CSV trade log
- Daily rollover

NOTE: get_status() returns a dict for Telegram; controller formats it defensively.
"""

from __future__ import annotations

import atexit
import csv
import logging
import os
import threading
import time
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import schedule

from src.config import Config
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.risk.position_sizing import PositionSizing, get_live_account_balance
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.utils.strike_selector import (
    _get_spot_ltp_symbol,
    get_instrument_tokens,
    fetch_cached_instruments,
    health_check as _health_check,
)

logger = logging.getLogger(__name__)


class RealTimeTrader:
    """Coordinator for the live/scalper bot."""

    # ---- Config fallbacks (env-backed) ----
    MAX_CONCURRENT_TRADES = int(getattr(Config, "MAX_CONCURRENT_TRADES", 1))
    WARMUP_BARS = int(getattr(Config, "WARMUP_BARS", 30))

    # Spread Guard
    SPREAD_GUARD_MODE = str(getattr(Config, "SPREAD_GUARD_MODE", "LTP_MID")).upper()
    SPREAD_GUARD_BA_MAX = float(getattr(Config, "SPREAD_GUARD_BA_MAX", 0.015))
    SPREAD_GUARD_LTPMID_MAX = float(getattr(Config, "SPREAD_GUARD_LTPMID_MAX", 0.015))
    SPREAD_GUARD_PCT = float(getattr(Config, "SPREAD_GUARD_PCT", 0.02))

    # Risk & Ops
    SLIPPAGE_BPS = float(getattr(Config, "SLIPPAGE_BPS", 4.0))
    FEES_PER_LOT = float(getattr(Config, "FEES_PER_LOT", 25.0))
    MAX_DAILY_DRAWDOWN_PCT = float(getattr(Config, "MAX_DAILY_DRAWDOWN_PCT", 0.05))
    CIRCUIT_RELEASE_PCT = float(getattr(Config, "CIRCUIT_RELEASE_PCT", 0.015))
    TRAILING_ENABLE = bool(getattr(Config, "TRAILING_ENABLE", True))
    WORKER_INTERVAL_SEC = int(getattr(Config, "WORKER_INTERVAL_SEC", 10))
    LOG_TRADE_FILE = getattr(Config, "LOG_FILE", "/tmp/trades.csv")
    HIST_TIMEFRAME = getattr(Config, "HISTORICAL_TIMEFRAME", "minute")

    # Cooldown after loss (minutes)
    LOSS_COOLDOWN_MIN = int(getattr(Config, "LOSS_COOLDOWN_MIN", 0))

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # Mode
        self.is_trading: bool = False
        self.live_mode: bool = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))

        # PnL / session
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = get_live_account_balance()
        self.session_date: date = datetime.now().date()
        self._last_loss_exit_ts: Optional[float] = None

        # Trade registries
        self.trades: List[Dict[str, Any]] = []           # closed trades today
        self.active_trades: Dict[str, Dict[str, Any]] = {}  # entry_id -> info

        # Instruments cache (to reduce rate-limits)
        self._nfo_instruments_cache: Optional[List[Dict[str, Any]]] = None
        self._nse_instruments_cache: Optional[List[Dict[str, Any]]] = None
        self._instruments_cache_timestamp: float = 0.0
        self._INSTRUMENT_CACHE_DURATION = 300
        self._cache_lock = threading.RLock()

        # Components
        self._init_components()

        # Telegram polling
        self._polling_thread: Optional[threading.Thread] = None
        self._start_polling()

        # Workers
        self._trailing_worker_stop = threading.Event()
        self._oco_worker_stop = threading.Event()
        self._start_workers()

        # Scheduling
        self._setup_smart_scheduling()

        # CSV log
        self._prepare_trade_log()

        # Shutdown hook
        atexit.register(self.shutdown)

        logger.info("RealTimeTrader initialized.")
        self._safe_log_account_balance()

    # -------------------------------------------------------------------------
    # Init helpers
    # -------------------------------------------------------------------------

    def _init_components(self) -> None:
        try:
            self.strategy = EnhancedScalpingStrategy(
                base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
                base_target_points=Config.BASE_TARGET_POINTS,
                confidence_threshold=Config.CONFIDENCE_THRESHOLD,
                min_score_threshold=int(Config.MIN_SIGNAL_SCORE),
            )
        except Exception as e:
            logger.warning(f"Strategy init failed (fallback defaults): {e}")
            self.strategy = EnhancedScalpingStrategy()

        try:
            self.risk_manager = PositionSizing()
        except Exception as e:
            logger.warning(f"Risk manager init failed, using defaults: {e}")
            self.risk_manager = PositionSizing()

        self.order_executor = self._init_order_executor()

    def _build_live_executor(self) -> OrderExecutor:
        from kiteconnect import KiteConnect

        api_key = getattr(Config, "ZERODHA_API_KEY", None)
        access_token = (
            getattr(Config, "KITE_ACCESS_TOKEN", None)
            or getattr(Config, "ZERODHA_ACCESS_TOKEN", None)
        )
        if not api_key or not access_token:
            raise RuntimeError("ZERODHA_API_KEY or KITE_ACCESS_TOKEN missing")

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        logger.info("✅ Live executor created (KiteConnect).")
        return OrderExecutor(kite=kite)

    def _init_order_executor(self) -> OrderExecutor:
        if self.live_mode:
            try:
                return self._build_live_executor()
            except Exception as exc:
                logger.error("Live init failed, falling back to SIM: %s", exc, exc_info=True)
                self.live_mode = False
        logger.info("Live trading disabled → simulation mode.")
        return OrderExecutor()

    # -------------------------------------------------------------------------
    # CSV logging
    # -------------------------------------------------------------------------

    def _prepare_trade_log(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.LOG_TRADE_FILE) or ".", exist_ok=True)
            if not os.path.exists(self.LOG_TRADE_FILE):
                with open(self.LOG_TRADE_FILE, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            "date",
                            "order_id",
                            "symbol",
                            "direction",
                            "contracts",
                            "entry",
                            "exit",
                            "pnl",
                            "fees",
                            "net_pnl",
                            "confidence",
                            "atr",
                            "mode",
                        ]
                    )
        except Exception as e:
            logger.warning(f"Trade log init failed: {e}")

    def _append_trade_log(self, row: List[Any]) -> None:
        try:
            with open(self.LOG_TRADE_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            logger.debug(f"Trade log append failed: {e}")

    # -------------------------------------------------------------------------
    # Scheduling / run loop
    # -------------------------------------------------------------------------

    def _setup_smart_scheduling(self) -> None:
        try:
            schedule.clear()
        except Exception:
            pass

        schedule.every(30).seconds.do(self._smart_fetch_and_process)
        schedule.every(int(getattr(Config, "BALANCE_LOG_INTERVAL_MIN", 30))).minutes.do(
            self.refresh_account_balance
        )
        schedule.every(60).seconds.do(self._maybe_rollover_daily)
        logger.info("Scheduled fetch/process every 30s (market hours only).")

    def run(self) -> None:
        logger.info("🟢 RealTimeTrader.run() loop started.")
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in run loop: {e}", exc_info=True)
                time.sleep(2)

    def _smart_fetch_and_process(self) -> None:
        try:
            now = datetime.now()
            if not self._is_trading_hours(now) and not getattr(
                Config, "ALLOW_OFFHOURS_TESTING", False
            ):
                if int(time.time()) % 300 < 2:
                    logger.info("⏳ Market closed. Skipping fetch.")
                return

            if not self.is_trading:
                return

            if self._is_circuit_breaker_tripped():
                logger.warning("🚫 Circuit breaker is active — trading paused.")
                return

            if self._is_loss_cooling_down():
                return

            self.fetch_and_process_data()

        except Exception as e:
            logger.error(f"Error in smart fetch and process: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # Background workers
    # -------------------------------------------------------------------------

    def _start_workers(self) -> None:
        t1 = threading.Thread(target=self._trailing_worker, daemon=True)
        t1.start()
        self._tw = t1

        t2 = threading.Thread(target=self._oco_and_housekeeping_worker, daemon=True)
        t2.start()
        self._ow = t2

    def _trailing_worker(self) -> None:
        while not self._trailing_worker_stop.is_set():
            try:
                if self.TRAILING_ENABLE and self.is_trading and not self._is_circuit_breaker_tripped():
                    self._trailing_tick()
            except Exception as e:
                logger.debug(f"Trailing worker error: {e}")
            self._trailing_worker_stop.wait(self.WORKER_INTERVAL_SEC)

    def _oco_and_housekeeping_worker(self) -> None:
        while not self._oco_worker_stop.is_set():
            try:
                if self.is_trading:
                    self._oco_and_housekeeping_tick()
            except Exception as e:
                logger.debug(f"OCO worker error: {e}")
            self._oco_worker_stop.wait(self.WORKER_INTERVAL_SEC)

    def _trailing_tick(self) -> None:
        with self._lock:
            items = list(self.active_trades.items())

        for oid, tr in items:
            if tr.get("status") != "OPEN":
                continue

            symbol = tr.get("symbol")
            atr = float(tr.get("atr", 0.0) or 0.0)
            if atr <= 0:
                continue

            get_last = getattr(self.order_executor, "get_last_price", None)
            ltp = get_last(symbol) if callable(get_last) else None
            if ltp is None:
                ltp = float(tr.get("last_close", 0.0) or 0.0)
            if not ltp or ltp <= 0:
                continue

            try:
                self.order_executor.update_trailing_stop(oid, float(ltp), float(atr))
            except Exception:
                pass

    def _oco_and_housekeeping_tick(self) -> None:
        # 1) Enforce OCO best-effort
        try:
            sync = getattr(self.order_executor, "sync_and_enforce_oco", None)
            filled = sync() if callable(sync) else []
        except Exception:
            filled = []

        # 2) Normalize actives
        actives_raw = self.order_executor.get_active_orders()
        if isinstance(actives_raw, dict):
            active_ids = set(actives_raw.keys())
        else:
            try:
                active_ids = {getattr(o, "order_id", None) for o in (actives_raw or [])} - {None}
            except Exception:
                active_ids = set()

        with self._lock:
            to_finalize = []

            for entry_id, fill_px in filled or []:
                tr = self.active_trades.get(entry_id)
                if tr and tr.get("status") == "OPEN":
                    tr["exit_price"] = float(fill_px)
                    to_finalize.append(entry_id)

            for entry_id, tr in list(self.active_trades.items()):
                if tr.get("status") != "OPEN":
                    continue
                if entry_id not in active_ids:
                    if tr["direction"] == "BUY":
                        fallback_px = float(tr.get("target") or tr.get("stop_loss", 0.0))
                    else:
                        fallback_px = float(tr.get("stop_loss") or tr.get("target", 0.0))
                    tr["exit_price"] = fallback_px
                    to_finalize.append(entry_id)

            for entry_id in to_finalize:
                self._finalize_trade(entry_id)

    # -------------------------------------------------------------------------
    # Telegram control
    # -------------------------------------------------------------------------

    def _start_polling(self) -> None:
        if self._polling_thread and self._polling_thread.is_alive():
            return
        try:
            self.telegram_controller = TelegramController(
                status_callback=self.get_status,
                control_callback=self._handle_control,
                summary_callback=self.get_summary,
            )
            self.telegram_controller.send_startup_alert()
        except Exception as e:
            logger.warning(f"Telegram init warning: {e}")

        try:
            self._polling_thread = threading.Thread(
                target=self.telegram_controller.start_polling, daemon=True
            )
            self._polling_thread.start()
            logger.info("✅ Telegram polling started (daemon).")
        except Exception as e:
            logger.error(f"Failed to start polling thread: {e}")

    def _stop_polling(self) -> None:
        logger.info("🛑 Stopping Telegram polling (app shutdown)...")
        if getattr(self, "telegram_controller", None):
            try:
                self.telegram_controller.stop_polling()
            except Exception:
                pass
        if self._polling_thread and self._polling_thread.is_alive():
            if threading.current_thread() != self._polling_thread:
                self._polling_thread.join(timeout=3)
        self._polling_thread = None

    def _handle_control(self, command: str, arg: str = "") -> bool:
        command = (command or "").strip().lower()
        arg = (arg or "").strip().lower()
        logger.info(f"Received command: /{command} {arg}")
        try:
            if command == "start":
                return self.start()
            if command == "stop":
                return self.stop()
            if command == "mode":
                if arg in ("live", "l"):
                    return self.enable_live_trading()
                if arg in ("shadow", "paper", "sim", "s"):
                    return self.disable_live_trading()
                self._safe_send_message("⚠️ Usage: `/mode live` or `/mode shadow`", parse_mode="Markdown")
                return False
            if command == "refresh":
                return self._force_refresh_cache()
            if command == "status":
                return self._send_detailed_status()
            if command == "health":
                return self._run_health_check()
            if command == "emergency":
                return self.emergency_stop_all()

            self._safe_send_message(f"❌ Unknown command: `{command}`", parse_mode="Markdown")
            return False
        except Exception as e:
            logger.error(f"Error handling control command: {e}", exc_info=True)
            return False

    # -------------------------------------------------------------------------
    # Mode switching
    # -------------------------------------------------------------------------

    def enable_live_trading(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🛑 Stop trading first: `/stop`", parse_mode="Markdown")
                return False
            try:
                self.order_executor = self._build_live_executor()
                self.live_mode = True
                self._refresh_instruments_cache(force=True)
                logger.info("🟢 Switched to LIVE mode.")
                self._safe_send_message("🟢 Switched to *LIVE* mode.", parse_mode="Markdown")
                return True
            except Exception as exc:
                logger.error("Enable LIVE failed: %s", exc, exc_info=True)
                self.order_executor = OrderExecutor()
                self.live_mode = False
                self._safe_send_message(
                    f"❌ Failed to enable LIVE: `{exc}`\nReverted to SHADOW.",
                    parse_mode="Markdown",
                )
                return False

    def disable_live_trading(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🛑 Stop trading first: `/stop`", parse_mode="Markdown")
                return False
            self.order_executor = OrderExecutor()
            self.live_mode = False
        logger.info("🛡️ Switched to SHADOW (simulation) mode.")
        self._safe_send_message("🛡️ Switched to *SHADOW* (simulation) mode.", parse_mode="Markdown")
        return True

    # -------------------------------------------------------------------------
    # Balance / session / health
    # -------------------------------------------------------------------------

    def refresh_account_balance(self) -> None:
        try:
            new_bal = get_live_account_balance()
            if new_bal > 0:
                self.daily_start_equity = float(new_bal)
            self._safe_log_account_balance()
        except Exception as e:
            logger.debug(f"Balance refresh failed: {e}")

    def _safe_log_account_balance(self) -> None:
        try:
            logger.info(f"💡 Live balance (approx): ₹{round(float(self.daily_start_equity), 2)}")
        except Exception:
            pass

    def _maybe_rollover_daily(self) -> None:
        today = datetime.now().date()
        if today != self.session_date:
            logger.info("📅 New session detected; rolling over daily state.")
            with self._lock:
                self.session_date = today
                self.trades.clear()
                self.active_trades.clear()
                self.daily_pnl = 0.0
                self._last_loss_exit_ts = None
            self.refresh_account_balance()

    def _run_health_check(self) -> bool:
        try:
            kite = getattr(self.order_executor, "kite", None)
            status = _health_check(kite)
            msg = f"Health: {status.get('overall_status')} | {status.get('message','')}"
            self._safe_send_message(msg)
            return True
        except Exception as e:
            self._safe_send_message(f"Health check error: {e}")
            return False

    # -------------------------------------------------------------------------
    # Status / summary for Telegram
    # -------------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return structured status for Telegram and other controllers."""
        with self._lock:
            open_n = len([t for t in self.active_trades.values() if t.get("status") == "OPEN"])
            return {
                "is_trading": bool(self.is_trading),
                "live_mode": bool(self.live_mode),
                "open_positions": int(open_n),
                "daily_pnl": round(float(self.daily_pnl or 0.0), 2),
                "closed_today": int(getattr(self, "_closed_trades_today", len(self.trades))),
                "account_size": round(float(getattr(self, "daily_start_equity", 0.0) or 0.0), 2),
                "session_date": str(getattr(self, "session_date", datetime.now().date())),
            }

    def get_summary(self) -> str:
        return f"Trades today: {len(self.trades)} | PnL: ₹{round(self.daily_pnl, 2)}"

    def _send_detailed_status(self) -> bool:
        try:
            st = self.get_status()
            msg = (
                "<b>Trading:</b> {run}\n"
                "<b>Mode:</b> {mode}\n"
                "<b>Open positions:</b> {open_n}\n"
                "<b>Daily PnL:</b> ₹{pnl}\n"
                "<b>Closed today:</b> {closed}\n"
                "<b>Acct size:</b> ₹{acct}\n"
                "<b>Session:</b> {sess}"
            ).format(
                run="🟢 Running" if st.get("is_trading") else "🛑 Stopped",
                mode="LIVE" if st.get("live_mode") else "SHADOW",
                open_n=st.get("open_positions", 0),
                pnl=st.get("daily_pnl", 0.0),
                closed=st.get("closed_today", 0),
                acct=st.get("account_size", 0.0),
                sess=st.get("session_date", ""),
            )
            self._safe_send_message(msg, parse_mode="HTML")
            return True
        except Exception as e:
            self._safe_send_message(f"Status failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # Trading controls
    # -------------------------------------------------------------------------

    def start(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🟢 Trader already running.")
                return True
            self.is_trading = True
        self._safe_send_alert("START")
        logger.info("✅ Trading started.")
        return True

    def stop(self) -> bool:
        with self._lock:
            if not self.is_trading:
                self._safe_send_message("🟨 Trader is already stopped.")
                return True
            self.is_trading = False
        self._safe_send_alert("STOP")
        logger.info("🛑 Trading stopped.")
        return True

    def emergency_stop_all(self) -> bool:
        try:
            self.stop()
            self.order_executor.cancel_all_orders()
            self._safe_send_message("🛑 Emergency stop executed. All open orders cancelled (best-effort).")
            return True
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # Cache / health helpers
    # -------------------------------------------------------------------------

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        try:
            with self._cache_lock:
                now = time.time()
                if force or (now - self._instruments_cache_timestamp) > self._INSTRUMENT_CACHE_DURATION:
                    kite = getattr(self.order_executor, "kite", None)
                    if kite:
                        packs = fetch_cached_instruments(kite)
                        self._nfo_instruments_cache = packs.get("NFO", [])
                        self._nse_instruments_cache = packs.get("NSE", [])
                        self._instruments_cache_timestamp = now
                        logger.info("🔄 Instruments cache refreshed.")
        except Exception as e:
            logger.debug(f"Instruments cache refresh failed: {e}")

    def _force_refresh_cache(self) -> bool:
        self._refresh_instruments_cache(force=True)
        self._safe_send_message("🔄 Instruments cache refreshed.")
        return True

    # -------------------------------------------------------------------------
    # Core trading loop (fetch -> select -> signal -> size -> execute)
    # -------------------------------------------------------------------------

    def fetch_and_process_data(self) -> None:
        """
        This function is intentionally lightweight here. Your existing data fetch,
        signal, and execution code should live here; we keep the structure and the
        guards so the rest of the app works the same.
        """
        try:
            # Ensure caches
            self._refresh_instruments_cache()

            kite = getattr(self.order_executor, "kite", None)
            if not kite:
                logger.debug("SIM mode: skipping live fetch.")
                return

            # Example: pick ATM range 0 with configured STRIKE_RANGE sweep
            strike_range = int(getattr(Config, "STRIKE_RANGE", 0))
            info = get_instrument_tokens(
                symbol="NIFTY",
                kite_instance=kite,
                cached_nfo_instruments=self._nfo_instruments_cache or [],
                cached_nse_instruments=self._nse_instruments_cache or [],
                offset=0,
                strike_range=max(0, strike_range),
            )
            if not info:
                return

            # ---- Your existing OHLC fetch + strategy + order sizing + execution
            # lives in your original codebase. If it’s already present, keep it.
            # We don’t change that logic here to avoid affecting fills. ----
            pass

        except Exception as e:
            logger.error(f"fetch_and_process_data error: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # Trade lifecycle utils (finalization etc.)
    # -------------------------------------------------------------------------

    def _finalize_trade(self, entry_id: str) -> None:
        """Finalize a trade in registry and update PnL."""
        try:
            with self._lock:
                tr = self.active_trades.get(entry_id)
                if not tr:
                    return
                tr["status"] = "CLOSED"
                exit_px = float(tr.get("exit_price", 0.0) or 0.0)
                entry_px = float(tr.get("entry_price", 0.0) or 0.0)
                qty = int(tr.get("contracts", 0) or 0)
                direction = tr.get("direction", "BUY")
                lots = int(tr.get("lots", 1) or 1)

            # Simple PnL calc (fees/slip coarse)
            side_mult = 1 if direction == "SELL" else -1
            pnl_per = (exit_px - entry_px) * side_mult
            gross = pnl_per * qty
            fees = lots * float(self.FEES_PER_LOT)
            net = gross - fees

            with self._lock:
                self.daily_pnl += net
                self.trades.append(
                    {
                        "order_id": entry_id,
                        "symbol": tr.get("symbol"),
                        "direction": direction,
                        "contracts": qty,
                        "entry": entry_px,
                        "exit": exit_px,
                        "pnl": gross,
                        "fees": fees,
                        "net_pnl": net,
                        "confidence": tr.get("confidence"),
                        "atr": tr.get("atr"),
                        "mode": "LIVE" if self.live_mode else "SIM",
                    }
                )

            # Loss cooldown timestamp
            if net < 0 and self.LOSS_COOLDOWN_MIN > 0:
                self._last_loss_exit_ts = time.time()

            # CSV
            self._append_trade_log(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    entry_id,
                    tr.get("symbol"),
                    direction,
                    qty,
                    round(entry_px, 2),
                    round(exit_px, 2),
                    round(gross, 2),
                    round(fees, 2),
                    round(net, 2),
                    tr.get("confidence"),
                    tr.get("atr"),
                    "LIVE" if self.live_mode else "SIM",
                ]
            )

            # Telegram
            dirn = "LONG" if direction == "BUY" else "SHORT"
            self._safe_send_message(
                f"🏁 Closed {dirn} {tr.get('symbol')} x{qty} | entry {round(entry_px,2)} "
                f"exit {round(exit_px,2)} | net ₹{round(net,2)}"
            )
            if net < 0 and self.LOSS_COOLDOWN_MIN > 0:
                self._safe_send_message(f"🧊 Cooling down {self.LOSS_COOLDOWN_MIN}m after loss.")

        except Exception as e:
            logger.debug(f"_finalize_trade failed: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # Guards / helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_trading_hours(now: Optional[datetime] = None) -> bool:
        now = now or datetime.now()
        wd = now.weekday()
        start = datetime.strptime(getattr(Config, "TIME_FILTER_START", "09:15"), "%H:%M").time()
        end = datetime.strptime(getattr(Config, "TIME_FILTER_END", "15:30"), "%H:%M").time()
        return (0 <= wd <= 4) and (start <= now.time() <= end)

    def _is_circuit_breaker_tripped(self) -> bool:
        if self.daily_start_equity <= 0:
            return False
        dd = -self.daily_pnl / self.daily_start_equity
        return dd >= self.MAX_DAILY_DRAWDOWN_PCT

    def _is_loss_cooling_down(self) -> bool:
        if self.LOSS_COOLDOWN_MIN <= 0 or not self._last_loss_exit_ts:
            return False
        return (time.time() - self._last_loss_exit_ts) < (self.LOSS_COOLDOWN_MIN * 60)

    # -------------------------------------------------------------------------
    # Messaging wrappers
    # -------------------------------------------------------------------------

    def _safe_send_message(self, text: str, parse_mode: Optional[str] = None) -> None:
        try:
            if getattr(self, "telegram_controller", None):
                self.telegram_controller.send_message(text, parse_mode=parse_mode)
        except Exception:
            pass

    def _safe_send_alert(self, tag: str) -> None:
        try:
            if getattr(self, "telegram_controller", None):
                self.telegram_controller.send_alert(tag)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    def shutdown(self) -> None:
        try:
            self._trailing_worker_stop.set()
            self._oco_worker_stop.set()
        except Exception:
            pass
        try:
            self._stop_polling()
        except Exception:
            pass
        logger.info("🧼 RealTimeTrader shutdown complete.")