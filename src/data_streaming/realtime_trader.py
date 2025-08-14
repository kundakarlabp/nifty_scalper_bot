# src/data_streaming/realtime_trader.py
"""
Real-time trader with:
- Telegram control (daemon polling)
- Adaptive main loop (peak/off-peak cadence)
- Risk-based lot sizing
- Warmup & trading-hours filters (IST aware)
- Options strike resolution (ATM ± range via cached instruments)
- Strategy signals (spot/option aware)
- Spread guard (RANGE or LTP_MID)
- Partial TP, breakeven hop, trailing SL (via OrderExecutor)
- Daily circuit breaker + loss cooldown
- CSV trade log + daily rollover
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import atexit
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import schedule
from datetime import datetime, timedelta, date, time as dtime
import time

from src.config import Config
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import PositionSizing, get_live_account_balance
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.utils.strike_selector import (
    get_instrument_tokens,
    fetch_cached_instruments,
)
from src.utils.atr_helper import compute_atr_df

logger = logging.getLogger(__name__)


# ----------------------------- helpers ----------------------------- #

def _ist_now() -> datetime:
    """IST clock without pytz (UTC+5:30)."""
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _between_ist(start_hm: str, end_hm: str) -> bool:
    """True if IST now is between HH:MM ranges inclusive."""
    now = _ist_now().time()
    s = datetime.strptime(start_hm, "%H:%M").time()
    e = datetime.strptime(end_hm, "%H:%M").time()
    return s <= now <= e


def _safe_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


# ============================== Trader ============================= #

class RealTimeTrader:
    # ---- knobs from .env (with safe fallbacks) ----
    MAX_CONCURRENT_TRADES = int(getattr(Config, "MAX_CONCURRENT_POSITIONS", 1))
    WARMUP_BARS = int(getattr(Config, "WARMUP_BARS", 30))
    DATA_LOOKBACK_MINUTES = int(getattr(Config, "DATA_LOOKBACK_MINUTES", 60))
    HIST_TIMEFRAME = str(getattr(Config, "HISTORICAL_TIMEFRAME", "minute"))

    SPREAD_GUARD_MODE = str(getattr(Config, "SPREAD_GUARD_MODE", "RANGE")).upper()  # RANGE|LTP_MID
    SPREAD_GUARD_BA_MAX = float(getattr(Config, "SPREAD_GUARD_BA_MAX", 0.012))
    SPREAD_GUARD_LTPMID_MAX = float(getattr(Config, "SPREAD_GUARD_LTPMID_MAX", 0.015))
    SPREAD_GUARD_PCT = float(getattr(Config, "SPREAD_GUARD_PCT", 0.02))

    SLIPPAGE_BPS = float(getattr(Config, "SLIPPAGE_BPS", 4.0))
    FEES_PER_LOT = float(getattr(Config, "FEES_PER_LOT", 25.0))
    MAX_DAILY_DRAWDOWN_PCT = float(getattr(Config, "MAX_DAILY_DRAWDOWN_PCT", 0.05))
    CIRCUIT_RELEASE_PCT = float(getattr(Config, "CIRCUIT_RELEASE_PCT", 0.015))
    TRAILING_ENABLE = bool(getattr(Config, "TRAILING_ENABLE", True))
    TRAIL_ATR_MULTIPLIER = float(getattr(Config, "ATR_SL_MULTIPLIER", 1.5))
    WORKER_INTERVAL_SEC = int(getattr(Config, "WORKER_INTERVAL_SEC", 5))
    LOG_TRADE_FILE = str(getattr(Config, "LOG_FILE", "logs/trades.csv"))

    LOT_SIZE = int(getattr(Config, "NIFTY_LOT_SIZE", 75))
    MIN_LOTS = int(getattr(Config, "MIN_LOTS", 1))
    MAX_LOTS = int(getattr(Config, "MAX_LOTS", 10))

    RISK_PER_TRADE = float(getattr(Config, "RISK_PER_TRADE", 0.02))
    MAX_TRADES_PER_DAY = int(getattr(Config, "MAX_TRADES_PER_DAY", 20))
    LOSS_COOLDOWN_MIN = int(getattr(Config, "LOSS_COOLDOWN_MIN", 2))
    PEAK_POLL_SEC = int(getattr(Config, "PEAK_POLL_SEC", 15))
    OFFPEAK_POLL_SEC = int(getattr(Config, "OFFPEAK_POLL_SEC", 30))
    PREFERRED_TIE_RULE = str(getattr(Config, "PREFERRED_TIE_RULE", "TREND")).upper()

    USE_IST_CLOCK = bool(getattr(Config, "USE_IST_CLOCK", True))
    TIME_FILTER_START = str(getattr(Config, "TIME_FILTER_START", "09:20"))
    TIME_FILTER_END = str(getattr(Config, "TIME_FILTER_END", "15:20"))
    SKIP_FIRST_MIN = int(getattr(Config, "SKIP_FIRST_MIN", 5))

    STRIKE_RANGE = int(getattr(Config, "STRIKE_RANGE", 2))  # ± range around ATM

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.is_trading: bool = False
        self.live_mode: bool = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))

        # PnL / session
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = float(get_live_account_balance() or 0.0)
        self.session_date: date = _ist_now().date()
        self._closed_trades_today = 0
        self._last_closed_was_loss = False
        self._cooldown_until_ts: float = 0.0

        # Trades
        self.trades: List[Dict[str, Any]] = []                    # closed trades (for the day)
        self.active_trades: Dict[str, Dict[str, Any]] = {}        # entry_order_id → info

        # Instruments cache
        self._nfo_cache: List[Dict[str, Any]] = []
        self._nse_cache: List[Dict[str, Any]] = []
        self._cache_ts: float = 0.0
        self._CACHE_TTL = 300.0
        self._cache_lock = threading.RLock()

        # Components
        self._init_components()

        # Telegram
        self._polling_thread: Optional[threading.Thread] = None
        self._start_polling()

        # Background workers
        self._trailing_evt = threading.Event()
        self._oco_evt = threading.Event()
        self._start_workers()

        # Adaptive scheduler
        self._data_job = None
        self._setup_adaptive_scheduler()

        # CSV log
        self._prepare_trade_log()

        # Shutdown hook
        atexit.register(self.shutdown)

        logger.info("RealTimeTrader initialized.")
        self._safe_log_account_balance()

    # -------------------- init -------------------- #

    def _init_components(self) -> None:
        try:
            self.strategy = EnhancedScalpingStrategy(
                base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
                base_target_points=Config.BASE_TARGET_POINTS,
                confidence_threshold=Config.CONFIDENCE_THRESHOLD,
                min_score_threshold=int(Config.MIN_SIGNAL_SCORE),
            )
        except Exception as e:
            logger.warning(f"Strategy init failed: {e}")
            self.strategy = None

        try:
            self.risk = PositionSizing()
            try:
                self.risk.set_equity(float(self.daily_start_equity or 0.0))
            except Exception:
                setattr(self.risk, "equity", float(self.daily_start_equity or 0.0))
        except Exception as e:
            logger.warning(f"Risk manager init failed: {e}")
            self.risk = PositionSizing()

        self.executor = self._init_executor()

        try:
            self.tg = TelegramController(
                status_callback=self.get_status,
                control_callback=self._handle_control,
                summary_callback=self.get_summary,
            )
        except Exception as e:
            logger.warning(f"Telegram init failed: {e}")
            self.tg = None

    def _build_live_executor(self) -> OrderExecutor:
        from kiteconnect import KiteConnect
        api_key = getattr(Config, "ZERODHA_API_KEY", None)
        access_token = getattr(Config, "KITE_ACCESS_TOKEN", None) or getattr(
            Config, "ZERODHA_ACCESS_TOKEN", None
        )
        if not api_key or not access_token:
            raise RuntimeError("ZERODHA_API_KEY or KITE_ACCESS_TOKEN missing")

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        logger.info("✅ Live executor created (KiteConnect).")
        return OrderExecutor(kite=kite)

    def _init_executor(self) -> OrderExecutor:
        if self.live_mode:
            try:
                return self._build_live_executor()
            except Exception as exc:
                logger.error("Live init failed, using simulation: %s", exc, exc_info=True)
                self.live_mode = False
        logger.info("Live trading disabled → simulation.")
        return OrderExecutor()

    # -------------------- csv log -------------------- #

    def _prepare_trade_log(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.LOG_TRADE_FILE) or ".", exist_ok=True)
            if not os.path.exists(self.LOG_TRADE_FILE):
                with open(self.LOG_TRADE_FILE, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(
                        ["date", "order_id", "symbol", "direction", "contracts",
                         "entry", "exit", "pnl", "fees", "net_pnl", "confidence", "atr", "mode"]
                    )
        except Exception as e:
            logger.warning(f"Trade log init failed: {e}")

    def _append_trade_log(self, row: List[Any]) -> None:
        try:
            with open(self.LOG_TRADE_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            logger.debug(f"Trade log append failed: {e}")

    # -------------------- scheduler -------------------- #

    def _setup_adaptive_scheduler(self) -> None:
        try:
            schedule.clear()
        except Exception:
            pass
        schedule.every(5).seconds.do(self._ensure_cadence)
        logger.info("Adaptive scheduler primed.")

    def _ensure_cadence(self) -> None:
        try:
            sec = self._current_poll_seconds()
            if self._data_job and self._data_job.interval.seconds == sec:
                return
            if self._data_job:
                schedule.cancel_job(self._data_job)
            self._data_job = schedule.every(sec).seconds.do(self._smart_fetch_and_process)
            logger.info("Data loop cadence → every %ds.", sec)
        except Exception as e:
            logger.debug(f"Cadence ensure error: {e}")

    def _current_poll_seconds(self) -> int:
        try:
            now = _ist_now().time()
            in_peak = (dtime(9, 20) <= now <= dtime(11, 30)) or (dtime(13, 30) <= now <= dtime(15, 10))
            return self.PEAK_POLL_SEC if in_peak else self.OFFPEAK_POLL_SEC
        except Exception:
            return self.OFFPEAK_POLL_SEC

    def run(self) -> None:
        logger.info("🟢 RealTimeTrader.run() started.")
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Run loop error: {e}", exc_info=True)
                time.sleep(2)

    # -------------------- workers -------------------- #

    def _start_workers(self) -> None:
        t1 = threading.Thread(target=self._trailing_worker, daemon=True)
        t1.start()
        t2 = threading.Thread(target=self._oco_worker, daemon=True)
        t2.start()

    def _trailing_worker(self) -> None:
        while not self._trailing_evt.is_set():
            try:
                if self.TRAILING_ENABLE and self.is_trading and not self._is_circuit_breaker_tripped():
                    self._trailing_tick()
            except Exception as e:
                logger.debug(f"Trailing worker error: {e}")
            self._trailing_evt.wait(self.WORKER_INTERVAL_SEC)

    def _oco_worker(self) -> None:
        while not self._oco_evt.is_set():
            try:
                if self.is_trading:
                    self._oco_tick()
            except Exception as e:
                logger.debug(f"OCO worker error: {e}")
            self._oco_evt.wait(self.WORKER_INTERVAL_SEC)

    def _trailing_tick(self) -> None:
        with self._lock:
            items = list(self.active_trades.items())

        for oid, tr in items:
            if tr.get("status") != "OPEN":
                continue
            atr = float(tr.get("atr") or 0.0)
            if atr <= 0:
                continue
            get_last = getattr(self.executor, "get_last_price", None)
            ltp = get_last(tr.get("symbol")) if callable(get_last) else None
            if ltp is None:
                ltp = float(tr.get("last_close") or 0.0)
            if not ltp or ltp <= 0:
                continue
            try:
                self.executor.update_trailing_stop(oid, float(ltp), float(atr))
            except Exception:
                pass

    def _oco_tick(self) -> None:
        try:
            sync = getattr(self.executor, "sync_and_enforce_oco", None)
            filled = sync() if callable(sync) else []
        except Exception:
            filled = []

        actives_raw = self.executor.get_active_orders()
        if isinstance(actives_raw, dict):
            active_ids = set(actives_raw.keys())
        else:
            try:
                active_ids = {getattr(o, "order_id", None) for o in (actives_raw or [])} - {None}
            except Exception:
                active_ids = set()

        with self._lock:
            to_finalize: List[str] = []

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
                        px = float(tr.get("target") or tr.get("stop_loss") or 0.0)
                    else:
                        px = float(tr.get("stop_loss") or tr.get("target") or 0.0)
                    tr["exit_price"] = px
                    to_finalize.append(entry_id)

            for entry_id in to_finalize:
                self._finalize_trade(entry_id)

    # -------------------- telegram -------------------- #

    def _start_polling(self) -> None:
        if self._polling_thread and self._polling_thread.is_alive():
            return
        try:
            if self.tg:
                self.tg.send_startup_alert()
        except Exception:
            pass
        try:
            if self.tg:
                self._polling_thread = threading.Thread(target=self.tg.start_polling, daemon=True)
                self._polling_thread.start()
                logger.info("✅ Telegram polling started (daemon).")
        except Exception as e:
            logger.error(f"Polling thread error: {e}")

    def _stop_polling(self) -> None:
        logger.info("🛑 Stopping Telegram polling…")
        if self.tg:
            try:
                self.tg.stop_polling()
            except Exception:
                pass
        if self._polling_thread and self._polling_thread.is_alive():
            if threading.current_thread() != self._polling_thread:
                self._polling_thread.join(timeout=3)
        self._polling_thread = None

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
            self.executor.cancel_all_orders()
            self._safe_send_message("🛑 Emergency stop executed. All open orders cancelled (best-effort).")
            return True
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False

    def _handle_control(self, command: str, arg: str = "") -> bool:
        command = (command or "").strip().lower()
        arg = (arg or "").strip().lower()
        logger.info(f"Command: /{command} {arg}")
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
            logger.error(f"Control error: {e}")
            return False

    # -------------------- mode switching -------------------- #

    def enable_live_trading(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🛑 Stop trading first: `/stop`", parse_mode="Markdown")
                return False
            try:
                self.executor = self._build_live_executor()
                self.live_mode = True
                self._refresh_instruments_cache(force=True)
                logger.info("🟢 Switched to LIVE mode.")
                self._safe_send_message("🟢 Switched to *LIVE* mode.", parse_mode="Markdown")
                return True
            except Exception as exc:
                logger.error("Enable LIVE failed: %s", exc, exc_info=True)
                self.executor = OrderExecutor()
                self.live_mode = False
                self._safe_send_message(
                    f"❌ Failed to enable LIVE: `{exc}`\nReverted to SHADOW.", parse_mode="Markdown"
                )
                return False

    def disable_live_trading(self) -> bool:
        with self._lock:
            if self.is_trading:
                self._safe_send_message("🛑 Stop trading first: `/stop`", parse_mode="Markdown")
                return False
            self.executor = OrderExecutor()
            self.live_mode = False
        logger.info("🛡️ Switched to SHADOW (simulation) mode.")
        self._safe_send_message("🛡️ Switched to *SHADOW* (simulation) mode.", parse_mode="Markdown")
        return True

    # -------------------- balance & guards -------------------- #

    def refresh_account_balance(self) -> None:
        try:
            new_bal = float(get_live_account_balance() or 0.0)
            if new_bal > 0:
                self.daily_start_equity = new_bal if _ist_now().date() != self.session_date else self.daily_start_equity
                try:
                    self.risk.set_equity(new_bal)
                except Exception:
                    setattr(self.risk, "equity", new_bal)
                logger.info("💰 Live balance (approx): ₹%.2f", new_bal)
        except Exception as e:
            logger.debug(f"Balance refresh failed: {e}")

    def _safe_log_account_balance(self) -> None:
        try:
            bal = float(get_live_account_balance() or 0.0)
            if bal > 0:
                logger.info("💰 Live balance (approx): ₹%.2f", bal)
        except Exception:
            pass

    def _is_trading_hours(self, now: Optional[datetime] = None) -> bool:
        if not self.USE_IST_CLOCK:
            return True
        now = now or _ist_now()
        if not _between_ist(self.TIME_FILTER_START, self.TIME_FILTER_END):
            return False
        # skip opening churn for SKIP_FIRST_MIN
        open_floor = dtime(9, 15 + int(self.SKIP_FIRST_MIN))
        return now.time() >= open_floor

    def _is_circuit_breaker_tripped(self) -> bool:
        if self.MAX_DAILY_DRAWDOWN_PCT <= 0:
            return False
        pnl_drop = (self.daily_pnl or 0.0) / max(1.0, self.daily_start_equity or 1.0)
        return pnl_drop <= -abs(self.MAX_DAILY_DRAWDOWN_PCT)

    def _in_loss_cooldown(self) -> bool:
        return time.time() < self._cooldown_until_ts

    # -------------------- cache & health -------------------- #

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        if not self.live_mode or not getattr(self.executor, "kite", None):
            return
        try:
            with self._cache_lock:
                if (not force) and (time.time() - self._cache_ts < self._CACHE_TTL):
                    return
                bundles = fetch_cached_instruments(self.executor.kite)
                self._nfo_cache = bundles.get("NFO", []) or []
                self._nse_cache = bundles.get("NSE", []) or []
                self._cache_ts = time.time()
                logger.info("📦 Instruments cache refreshed. NFO=%d, NSE=%d", len(self._nfo_cache), len(self._nse_cache))
        except Exception as e:
            logger.warning(f"Instrument cache refresh failed: {e}")

    def _force_refresh_cache(self) -> bool:
        self._refresh_instruments_cache(force=True)
        self._safe_send_message("🔄 Instruments cache refreshed.")
        return True

    def _run_health_check(self) -> bool:
        ok = bool(self.strategy and self.executor and self.risk)
        self._safe_send_message("✅ Health OK." if ok else "❌ Health check failed.")
        return ok

    # -------------------- core loop -------------------- #

    def _smart_fetch_and_process(self) -> None:
        try:
            now = _ist_now()
            if not self._is_trading_hours(now) and not getattr(Config, "ALLOW_OFFHOURS_TESTING", False):
                if int(time.time()) % 300 < 2:
                    logger.info("⏳ Market closed. Skipping fetch.")
                return

            if not self.is_trading:
                return

            if self._is_circuit_breaker_tripped():
                logger.warning("🚫 Circuit breaker is active — trading paused.")
                return

            if self._in_loss_cooldown():
                return

            # **Your repo’s existing implementation should fetch data and drive trades here**
            self.fetch_and_process_data()

        except Exception as e:
            logger.error(f"Error in smart fetch and process: {e}", exc_info=True)

    # NOTE: fetch_and_process_data() is expected to be implemented in your repo
    # to perform: data fetch → strike selection → signal → sizing → executor calls.

    # -------------------- finalize trades & logging -------------------- #

    def _finalize_trade(self, entry_id: str) -> None:
        with self._lock:
            tr = self.active_trades.get(entry_id)
            if not tr:
                return
            tr["status"] = "CLOSED"
            exit_px = float(tr.get("exit_price") or 0.0)
            entry_px = float(tr.get("entry") or 0.0)
            qty = int(tr.get("contracts") or 0)
            side = tr.get("direction", "BUY")

            # PnL calc (simple; your repo may have fees/slippage elsewhere)
            gross = (exit_px - entry_px) * qty if side == "BUY" else (entry_px - exit_px) * qty
            lots = qty // max(1, self.LOT_SIZE)
            fees = lots * float(self.FEES_PER_LOT or 0.0)
            net = gross - fees
            self.daily_pnl += net
            self._closed_trades_today += 1
            self._last_closed_was_loss = net < 0

            if self._last_closed_was_loss and self.LOSS_COOLDOWN_MIN > 0:
                self._cooldown_until_ts = time.time() + self.LOSS_COOLDOWN_MIN * 60

            # write CSV
            row = [
                _ist_now().strftime("%Y-%m-%d %H:%M:%S"),
                entry_id,
                tr.get("symbol"),
                side,
                qty,
                f"{entry_px:.2f}",
                f"{exit_px:.2f}",
                f"{gross:.2f}",
                f"{fees:.2f}",
                f"{net:.2f}",
                f"{float(tr.get('confidence') or 0.0):.2f}",
                f"{float(tr.get('atr') or 0.0):.2f}",
                "LIVE" if self.live_mode else "SHADOW",
            ]
            self._append_trade_log(row)

            # cleanup
            self.active_trades.pop(entry_id, None)

            # circuit breaker release logic (optional)
            if self._is_circuit_breaker_tripped():
                self._safe_send_message("⚠️ Drawdown limit hit. Trading paused.")
            elif self._cooldown_until_ts:
                self._safe_send_message(f"🟡 Cooldown for {self.LOSS_COOLDOWN_MIN} min after loss.")

    # -------------------- status / summary -------------------- #

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            open_orders = len(self.executor.get_active_orders() or {})
            return {
                "is_trading": self.is_trading,
                "live_mode": self.live_mode,
                "open_orders": open_orders,
                "trades_today": self._closed_trades_today,
                "daily_pnl": round(self.daily_pnl, 2),
                "risk_level": f"{self.RISK_PER_TRADE*100:.1f}%",
            }

    def get_summary(self) -> str:
        s = self.get_status()
        return (
            "📊 <b>Summary</b>\n"
            f"Trading: {'🟢' if s['is_trading'] else '🔴'} | Mode: {'LIVE' if s['live_mode'] else 'SHADOW'}\n"
            f"Open Orders: {s['open_orders']} | Trades Today: {s['trades_today']}\n"
            f"Daily P&L: ₹{s['daily_pnl']:.2f} | Risk/Trade: {s['risk_level']}"
        )

    # -------------------- telegram send wrappers -------------------- #

    def _safe_send_message(self, text: str, parse_mode: Optional[str] = None) -> None:
        try:
            if self.tg:
                self.tg.send_message(text, parse_mode=parse_mode)
        except Exception:
            pass

    def _safe_send_alert(self, action: str) -> None:
        try:
            if self.tg:
                self.tg.send_realtime_session_alert(action)
        except Exception:
            pass

    # -------------------- shutdown -------------------- #

    def shutdown(self) -> None:
        try:
            self._trailing_evt.set()
            self._oco_evt.set()
            self._stop_polling()
        except Exception:
            pass
        logger.info("👋 RealTimeTrader shutdown complete.")