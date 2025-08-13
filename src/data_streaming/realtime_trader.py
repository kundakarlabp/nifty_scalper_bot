# src/data_streaming/realtime_trader.py
"""
Real-time trader with:
- Telegram control (daemon polling)
- Scheduled main loop (fetch -> select -> signal -> size -> execute)
- ATR trailing worker (background)
- Circuit breaker (daily drawdown)
- Single-position policy
- Warmup filter
- Spread guard:
    • RANGE mode (candle-range proxy)
    • LTP_MID mode (bid/ask depth via quote())
- Slippage + fees model
- Rate limit safety (bulk quote)
- CSV trade log persistence
- Daily session rollover

Enhancements:
- Adaptive polling cadence (peak/off-peak)
- Risk-based lot sizing (RISK_PER_TRADE × equity)
- CE/PE tie-breaking (trend → confidence → spread → volume impulse)
- Daily trade cap and loss cooldown
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
    _get_spot_ltp_symbol,
    get_instrument_tokens,
    fetch_cached_instruments,
    is_trading_hours,
)

logger = logging.getLogger(__name__)


class RealTimeTrader:
    # --- Core config with fallbacks ---
    MAX_CONCURRENT_TRADES = int(getattr(Config, "MAX_CONCURRENT_POSITIONS", 1))
    WARMUP_BARS = int(getattr(Config, "WARMUP_BARS", 30))
    DATA_LOOKBACK_MINUTES = int(getattr(Config, "DATA_LOOKBACK_MINUTES", 60))
    HIST_TIMEFRAME = getattr(Config, "HISTORICAL_TIMEFRAME", "minute")

    # Spread guard config
    SPREAD_GUARD_MODE = str(getattr(Config, "SPREAD_GUARD_MODE", "LTP_MID")).upper()
    SPREAD_GUARD_BA_MAX = float(getattr(Config, "SPREAD_GUARD_BA_MAX", 0.012))
    SPREAD_GUARD_LTPMID_MAX = float(getattr(Config, "SPREAD_GUARD_LTPMID_MAX", 0.015))
    SPREAD_GUARD_PCT = float(getattr(Config, "SPREAD_GUARD_PCT", 0.02))

    # Costs / risk / trailing
    SLIPPAGE_BPS = float(getattr(Config, "SLIPPAGE_BPS", 4.0))
    FEES_PER_LOT = float(getattr(Config, "FEES_PER_LOT", 25.0))
    MAX_DAILY_DRAWDOWN_PCT = float(getattr(Config, "MAX_DAILY_DRAWDOWN_PCT", 0.05))  # use your .env
    CIRCUIT_RELEASE_PCT = float(getattr(Config, "CIRCUIT_RELEASE_PCT", 0.015))
    TRAILING_ENABLE = bool(getattr(Config, "TRAILING_ENABLE", True))
    TRAIL_ATR_MULTIPLIER = float(getattr(Config, "ATR_SL_MULTIPLIER", 1.5))
    WORKER_INTERVAL_SEC = int(getattr(Config, "WORKER_INTERVAL_SEC", 10))
    LOG_TRADE_FILE = getattr(Config, "LOG_FILE", "logs/trades.csv")

    # Lots / lot size
    LOT_SIZE = int(getattr(Config, "NIFTY_LOT_SIZE", 75))
    MIN_LOTS = int(getattr(Config, "MIN_LOTS", 1))
    MAX_LOTS = int(getattr(Config, "MAX_LOTS", 5))

    # --- New knobs (optional in .env) ---
    RISK_PER_TRADE = float(getattr(Config, "RISK_PER_TRADE", 0.02))  # fraction of equity
    MAX_TRADES_PER_DAY = int(getattr(Config, "MAX_TRADES_PER_DAY", 20))
    LOSS_COOLDOWN_MIN = int(getattr(Config, "LOSS_COOLDOWN_MIN", 2))  # pause after a losing close
    PEAK_POLL_SEC = int(getattr(Config, "PEAK_POLL_SEC", 15))
    OFFPEAK_POLL_SEC = int(getattr(Config, "OFFPEAK_POLL_SEC", 30))
    # Tie rule: TREND (default) or CONFIDENCE_ONLY
    PREFERRED_TIE_RULE = str(getattr(Config, "PREFERRED_TIE_RULE", "TREND")).upper()

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.is_trading: bool = False
        self.live_mode: bool = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))

        # PnL / session
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = float(get_live_account_balance() or 0.0)
        self.session_date: date = datetime.now().date()
        self._closed_trades_today = 0
        self._last_closed_was_loss = False
        self._cooldown_until_ts: float = 0.0

        # Trade registry
        self.trades: List[Dict[str, Any]] = []  # closed trades for the day
        self.active_trades: Dict[str, Dict[str, Any]] = {}  # entry_order_id → info

        # Instrument cache
        self._nfo_instruments_cache: Optional[List[Dict]] = None
        self._nse_instruments_cache: Optional[List[Dict]] = None
        self._instruments_cache_timestamp: float = 0
        self._INSTRUMENT_CACHE_DURATION: int = 300
        self._cache_lock = threading.RLock()

        # Strategy / Risk / Executor / Telegram
        self._init_components()

        # Telegram polling
        self._polling_thread: Optional[threading.Thread] = None
        self._start_polling()

        # Background workers
        self._trailing_worker_stop = threading.Event()
        self._oco_worker_stop = threading.Event()
        self._start_workers()

        # Scheduling
        self._schedule_job = None
        self._setup_smart_scheduling()

        # CSV log setup
        self._prepare_trade_log()

        # Shutdown
        atexit.register(self.shutdown)

        logger.info("RealTimeTrader initialized.")
        self._safe_log_account_balance()

    # ---------- Init helpers ----------

    def _init_components(self) -> None:
        try:
            self.strategy = EnhancedScalpingStrategy(
                base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
                base_target_points=Config.BASE_TARGET_POINTS,
                confidence_threshold=Config.CONFIDENCE_THRESHOLD,
                min_score_threshold=int(Config.MIN_SIGNAL_SCORE),
            )
        except Exception as e:
            logger.warning(f"Failed to initialize strategy: {e}")
            self.strategy = None

        try:
            self.risk_manager = PositionSizing()
            try:
                self.risk_manager.set_equity(float(self.daily_start_equity or 0.0))
            except Exception:
                setattr(self.risk_manager, "equity", float(self.daily_start_equity or 0.0))
        except Exception as e:
            logger.warning(f"Failed to initialize risk manager: {e}")
            self.risk_manager = PositionSizing()

        self.order_executor = self._init_order_executor()

        try:
            self.telegram_controller = TelegramController(
                status_callback=self.get_status,
                control_callback=self._handle_control,
                summary_callback=self.get_summary,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Telegram controller: {e}")
            self.telegram_controller = None

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

    def _init_order_executor(self) -> OrderExecutor:
        if self.live_mode:
            try:
                return self._build_live_executor()
            except Exception as exc:
                logger.error("Live init failed, falling back to simulation: %s", exc, exc_info=True)
                self.live_mode = False
        logger.info("Live trading disabled → simulation mode.")
        return OrderExecutor()

    # ---------- CSV log ----------

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
                w = csv.writer(f)
                w.writerow(row)
        except Exception as e:
            logger.debug(f"Trade log append failed: {e}")

    # ---------- Scheduling / main loop ----------

    def _setup_smart_scheduling(self) -> None:
        try:
            schedule.clear()
        except Exception:
            pass
        # set up adaptive cadence tick
        schedule.every(5).seconds.do(self._ensure_adaptive_job)
        logger.info("Adaptive scheduler primed (ensures peak/off-peak cadence).")

    def _ensure_adaptive_job(self) -> None:
        """Recreates the data loop job if cadence should change."""
        try:
            sec = self._current_poll_seconds()
            if self._schedule_job and self._schedule_job.interval.seconds == sec:
                return
            if self._schedule_job:
                schedule.cancel_job(self._schedule_job)
                self._schedule_job = None
            self._schedule_job = schedule.every(sec).seconds.do(self._smart_fetch_and_process)
            logger.info("Data loop cadence set to every %ds.", sec)
        except Exception as e:
            logger.debug(f"adaptive cadence error: {e}")

    def _current_poll_seconds(self) -> int:
        # Peak windows: 09:20–11:30, 13:30–15:10 IST
        try:
            now = datetime.now().time()
            in_peak = (
                dtime(9, 20) <= now <= dtime(11, 30)
                or dtime(13, 30) <= now <= dtime(15, 10)
            )
            return self.PEAK_POLL_SEC if in_peak else self.OFFPEAK_POLL_SEC
        except Exception:
            return self.OFFPEAK_POLL_SEC

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
                # heartbeat occasionally
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

            self.fetch_and_process_data()

        except Exception as e:
            logger.error(f"Error in smart fetch and process: {e}", exc_info=True)

    # ---------- Background workers ----------

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
                if (
                    self.TRAILING_ENABLE
                    and self.is_trading
                    and not self._is_circuit_breaker_tripped()
                ):
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
            atr = float(tr.get("atr", 0.0)) or 0.0
            if atr <= 0:
                continue
            get_last = getattr(self.order_executor, "get_last_price", None)
            ltp = get_last(tr.get("symbol")) if callable(get_last) else None
            if ltp is None:
                ltp = float(tr.get("last_close", 0.0) or 0.0)
            if not ltp or ltp <= 0:
                continue
            try:
                self.order_executor.update_trailing_stop(oid, float(ltp), float(atr))
            except Exception:
                pass

    def _oco_and_housekeeping_tick(self) -> None:
        try:
            sync = getattr(self.order_executor, "sync_and_enforce_oco", None)
            filled = sync() if callable(sync) else []
        except Exception:
            filled = []

        actives_raw = self.order_executor.get_active_orders()
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
                        fallback_px = float(tr.get("target") or tr.get("stop_loss", 0.0))
                    else:
                        fallback_px = float(tr.get("stop_loss") or tr.get("target", 0.0))
                    tr["exit_price"] = fallback_px
                    to_finalize.append(entry_id)

            for entry_id in to_finalize:
                self._finalize_trade(entry_id)

    # ---------- Telegram control ----------

    def _start_polling(self) -> None:
        if self._polling_thread and self._polling_thread.is_alive():
            return
        try:
            if self.telegram_controller:
                self.telegram_controller.send_startup_alert()
        except Exception:
            pass
        try:
            if self.telegram_controller:
                self._polling_thread = threading.Thread(
                    target=self.telegram_controller.start_polling, daemon=True
                )
                self._polling_thread.start()
                logger.info("✅ Telegram polling started (daemon).")
        except Exception as e:
            logger.error(f"Failed to start polling thread: {e}")

    def _stop_polling(self) -> None:
        logger.info("🛑 Stopping Telegram polling (app shutdown)...")
        if self.telegram_controller:
            try:
                self.telegram_controller.stop_polling()
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
            self.order_executor.cancel_all_orders()
            self._safe_send_message(
                "🛑 Emergency stop executed. All open orders cancelled (best-effort)."
            )
            return True
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False

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
                self._safe_send_message(
                    "⚠️ Usage: `/mode live` or `/mode shadow`", parse_mode="Markdown"
                )
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
            logger.error(f"Error handling control command: {e}")
            return False

    # ---------- Mode switching ----------

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

    # ---------- Balance/session helpers ----------

    def _safe_log_account_balance(self) -> None:
        try:
            bal = float(get_live_account_balance()) or 0.0
            logger.info("💰 Live balance (approx): ₹%.2f", bal)
        except Exception as e:
            logger.debug("Balance fetch failed: %s", e)

    def refresh_account_balance(self) -> None:
        try:
            bal = float(get_live_account_balance()) or 0.0
            try:
                self.risk_manager.set_equity(bal)
            except Exception:
                setattr(self.risk_manager, "equity", bal)
            logger.info("↻ Balance refreshed: ₹%.2f", bal)
        except Exception as e:
            logger.debug("Balance refresh failed: %s", e)

    def _maybe_rollover_daily(self) -> None:
        try:
            today = datetime.now().date()
            if today != self.session_date:
                logger.info("📅 New session — resetting counters.")
                self.session_date = today
                self.daily_pnl = 0.0
                self.trades.clear()
                self._closed_trades_today = 0
                self._last_closed_was_loss = False
                self._cooldown_until_ts = 0.0
                try:
                    self.order_executor.cancel_all_orders()
                except Exception:
                    pass
        except Exception:
            pass

    def _is_trading_hours(self, _now: Optional[datetime] = None) -> bool:
        if getattr(Config, "ALLOW_OFFHOURS_TESTING", False):
            return True
        try:
            return is_trading_hours()
        except Exception:
            return True

    def _in_loss_cooldown(self) -> bool:
        return time.time() < float(self._cooldown_until_ts or 0.0)

    # ---------- Instruments cache ----------

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        with self._cache_lock:
            now = time.time()
            if (not force) and (now - self._instruments_cache_timestamp < self._INSTRUMENT_CACHE_DURATION):
                return
            try:
                kite = getattr(self.order_executor, "kite", None)
                if not kite:
                    return
                caches = fetch_cached_instruments(kite)
                self._nfo_instruments_cache = caches.get("NFO") or []
                self._nse_instruments_cache = caches.get("NSE") or []
                self._instruments_cache_timestamp = now
                logger.info("🔄 Instruments cache refreshed: NFO=%d, NSE=%d",
                            len(self._nfo_instruments_cache or []),
                            len(self._nse_instruments_cache or []))
            except Exception as e:
                logger.debug("Instruments cache refresh failed: %s", e)

    def _force_refresh_cache(self) -> bool:
        try:
            self._refresh_instruments_cache(force=True)
            self._safe_send_message("🔄 Instruments cache refreshed.")
            return True
        except Exception as e:
            self._safe_send_message(f"❌ Refresh failed: {e}")
            return False

    # ---------- Fetch / process ----------

    def _kite_historical_df(self, token: int, minutes: int) -> pd.DataFrame:
        kite = getattr(self.order_executor, "kite", None)
        if not kite:
            return pd.DataFrame()
        try:
            end = datetime.now()
            start = end - timedelta(minutes=max(int(minutes), self.DATA_LOOKBACK_MINUTES))
            candles = kite.historical_data(int(token), start, end, self.HIST_TIMEFRAME, oi=False) or []
            df = pd.DataFrame(candles)
            if df.empty:
                return df
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            return df[keep].copy()
        except Exception as e:
            logger.debug(f"historical_data failed for token {token}: {e}")
            return pd.DataFrame()

    def _passes_spread_guard(self, tsym: str, last_close: float) -> bool:
        mode = self.SPREAD_GUARD_MODE
        kite = getattr(self.order_executor, "kite", None)
        if mode == "LTP_MID" and kite:
            try:
                q = kite.quote([f"{getattr(Config,'TRADE_EXCHANGE','NFO')}:{tsym}"]) or {}
                qd = q.get(f"{getattr(Config,'TRADE_EXCHANGE','NFO')}:{tsym}", {})
                depth = (qd.get("depth") or {})
                bids = depth.get("buy") or []
                asks = depth.get("sell") or []
                best_bid = float(bids[0]["price"]) if bids else 0.0
                best_ask = float(asks[0]["price"]) if asks else 0.0
                ltp = float(qd.get("last_price") or 0.0)
                if best_bid <= 0 or best_ask <= 0:
                    return False
                mid = 0.5 * (best_bid + best_ask)
                ba = (best_ask - best_bid) / mid
                lm = abs(ltp - mid) / mid if mid > 0 else 1.0
                return (ba <= self.SPREAD_GUARD_BA_MAX) and (lm <= self.SPREAD_GUARD_LTPMID_MAX)
            except Exception:
                return False
        else:
            return bool(last_close > 0)

    def _spot_trend_bias(self, spot_df: pd.DataFrame) -> str:
        """Return 'UP', 'DOWN', or 'FLAT' via EMA20/EMA50 or 20/5 close change."""
        try:
            if spot_df is None or spot_df.empty or "close" not in spot_df.columns:
                return "FLAT"
            close = spot_df["close"]
            if len(close) < 60:
                ret = (close.iloc[-1] / close.iloc[max(0, len(close)-6)] - 1.0)
                if ret > 0.0015:
                    return "UP"
                if ret < -0.0015:
                    return "DOWN"
                return "FLAT"
            ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            if ema20 > ema50 * 1.0005:
                return "UP"
            if ema20 < ema50 * 0.9995:
                return "DOWN"
            return "FLAT"
        except Exception:
            return "FLAT"

    def _size_contracts(self, entry: float, sl: float) -> int:
        """
        Risk-based sizing: risk_cap = equity * RISK_PER_TRADE
        per_contract_risk = |entry - sl| * LOT_SIZE
        contracts = floor(risk_cap / per_contract_risk), clamped to MIN..MAX lots
        """
        L = max(1, int(self.LOT_SIZE))
        min_c = int(max(1, self.MIN_LOTS)) * L
        max_c = int(max(self.MIN_LOTS, self.MAX_LOTS)) * L

        try:
            equity = float(getattr(self.risk_manager, "equity", self.daily_start_equity) or 0.0)
            risk_cap = max(0.0, equity * float(self.RISK_PER_TRADE))
            per_c_risk = max(0.01, abs(float(entry) - float(sl)) * L)
            raw_contracts = int(risk_cap // per_c_risk)
            if raw_contracts <= 0:
                raw_contracts = min_c
            # clamp to lot multiples
            raw_contracts = (raw_contracts // L) * L
            contracts = max(min_c, min(raw_contracts, max_c))
            return int(contracts)
        except Exception:
            return min_c

    def fetch_and_process_data(self) -> None:
        kite = getattr(self.order_executor, "kite", None)
        if not kite:
            logger.info("Shadow mode: no live broker — skipping live fetch this tick.")
            return

        # Maintain instrument cache
        self._refresh_instruments_cache()

        # Resolve strikes
        try:
            strike_range = int(getattr(Config, "STRIKE_RANGE", 0))
        except Exception:
            strike_range = 0

        try:
            spot_cfg_symbol = _get_spot_ltp_symbol()
            info = get_instrument_tokens(
                symbol=spot_cfg_symbol,
                kite_instance=kite,
                cached_nfo_instruments=self._nfo_instruments_cache or [],
                cached_nse_instruments=self._nse_instruments_cache or [],
                offset=0,
                strike_range=max(0, strike_range),
            )
            if not info:
                return
        except Exception as e:
            logger.debug(f"get_instrument_tokens error: {e}")
            return

        ce_token, pe_token = info.get("ce_token"), info.get("pe_token")
        if not ce_token and not pe_token:
            return

        # Fetch data
        spot_token = info.get("spot_token")
        spot_df = self._kite_historical_df(spot_token, self.DATA_LOOKBACK_MINUTES) if spot_token else pd.DataFrame()

        ce_df = self._kite_historical_df(ce_token, self.DATA_LOOKBACK_MINUTES) if ce_token else pd.DataFrame()
        pe_df = self._kite_historical_df(pe_token, self.DATA_LOOKBACK_MINUTES) if pe_token else pd.DataFrame()

        # Warmup
        if ce_df.empty and pe_df.empty:
            return
        if max(len(ce_df), len(pe_df)) < self.WARMUP_BARS:
            return

        # Signals
        candidates: List[Tuple[str, Dict[str, Any], pd.DataFrame, str]] = []  # (side, sig, df, tsym)
        try:
            if ce_token and not ce_df.empty:
                ce_sym = info.get("ce_symbol")
                ce_price = float(ce_df["close"].iloc[-1])
                sig = self.strategy.generate_options_signal(ce_df, spot_df, {"type": "CE"}, ce_price)
                if sig:
                    candidates.append(("BUY", sig, ce_df, ce_sym))
            if pe_token and not pe_df.empty:
                pe_sym = info.get("pe_symbol")
                pe_price = float(pe_df["close"].iloc[-1])
                sig = self.strategy.generate_options_signal(pe_df, spot_df, {"type": "PE"}, pe_price)
                if sig:
                    candidates.append(("BUY", sig, pe_df, pe_sym))
        except Exception as e:
            logger.debug(f"Signal generation failed: {e}")
            return

        if not candidates:
            return

        # Tie-breaking:
        chosen = None
        if len(candidates) == 1 or self.PREFERRED_TIE_RULE == "CONFIDENCE_ONLY":
            candidates.sort(key=lambda x: float(x[1].get("confidence", 0.0)), reverse=True)
            chosen = candidates[0]
        else:
            # 1) trend bias
            bias = self._spot_trend_bias(spot_df)  # 'UP', 'DOWN', 'FLAT'
            ce_cand = next((c for c in candidates if c[3] and c[3].endswith("CE")), None)
            pe_cand = next((c for c in candidates if c[3] and c[3].endswith("PE")), None)
            if bias == "UP" and ce_cand:
                chosen = ce_cand
            elif bias == "DOWN" and pe_cand:
                chosen = pe_cand
            else:
                # 2) confidence
                candidates.sort(key=lambda x: float(x[1].get("confidence", 0.0)), reverse=True)
                top = [c for c in candidates if c[1].get("confidence") == candidates[0][1].get("confidence")]
                if len(top) == 1:
                    chosen = top[0]
                else:
                    # 3) tighter spread via quote
                    try:
                        kite = getattr(self.order_executor, "kite", None)
                        best = None
                        best_ba = 999.0
                        for cand in top:
                            tsym = cand[3]
                            q = kite.quote([f"{getattr(Config,'TRADE_EXCHANGE','NFO')}:{tsym}"]) or {}
                            qd = q.get(f"{getattr(Config,'TRADE_EXCHANGE','NFO')}:{tsym}", {})
                            depth = (qd.get("depth") or {})
                            bids = depth.get("buy") or []
                            asks = depth.get("sell") or []
                            if not bids or not asks:
                                continue
                            mid = 0.5 * (float(bids[0]["price"]) + float(asks[0]["price"]))
                            ba = (float(asks[0]["price"]) - float(bids[0]["price"])) / max(mid, 1e-9)
                            if ba < best_ba:
                                best_ba = ba
                                best = cand
                        chosen = best or top[0]
                    except Exception:
                        # 4) stronger recent volume impulse
                        def vol_impulse(c):
                            df = c[2]
                            if "volume" not in df.columns or len(df) < 6:
                                return 0.0
                            return float(df["volume"].iloc[-1]) / max(1.0, float(df["volume"].iloc[-5:-1].mean() or 1.0))
                        top.sort(key=vol_impulse, reverse=True)
                        chosen = top[0]

        direction, signal, df_sel, tsym = chosen
        entry = float(signal["entry_price"])
        sl = float(signal["stop_loss"])
        tp = float(signal["target"])
        conf = float(signal.get("confidence", 0.0))
        atr = float(signal.get("market_volatility", 0.0))
        last_close = float(df_sel["close"].iloc[-1])

        if not self._passes_spread_guard(tsym, last_close):
            logger.info("⛔ Spread guard blocked entry for %s", tsym)
            return

        with self._lock:
            if len([t for t in self.active_trades.values() if t.get("status") == "OPEN"]) >= self.MAX_CONCURRENT_TRADES:
                logger.info("Max concurrent positions reached; skip new entry.")
                return
            if self._closed_trades_today >= self.MAX_TRADES_PER_DAY:
                logger.info("Daily trade cap reached; skip new entry.")
                return

        contracts = self._size_contracts(entry, sl)
        if contracts <= 0:
            return

        try:
            entry_oid = self.order_executor.place_entry_order(
                symbol=tsym,
                exchange=getattr(Config, "TRADE_EXCHANGE", "NFO"),
                transaction_type=direction,
                quantity=int(contracts),
                product=getattr(Config, "DEFAULT_PRODUCT", "MIS"),
                order_type=getattr(Config, "DEFAULT_ORDER_TYPE", "MARKET"),
                validity=getattr(Config, "DEFAULT_VALIDITY", "DAY"),
            )
            if not entry_oid:
                logger.info("Entry order not placed.")
                return

            ok = self.order_executor.setup_gtt_orders(
                entry_order_id=entry_oid,
                entry_price=entry,
                stop_loss_price=sl,
                target_price=tp,
                symbol=tsym,
                exchange=getattr(Config, "TRADE_EXCHANGE", "NFO"),
                quantity=int(contracts),
                transaction_type=direction,
            )
            if not ok:
                logger.info("Failed to setup exits; cancelling entry.")
                try:
                    self.order_executor.exit_order(entry_oid, "setup_exits_failed")
                except Exception:
                    pass
                return

            with self._lock:
                self.active_trades[entry_oid] = {
                    "status": "OPEN",
                    "time": datetime.now(),
                    "symbol": tsym,
                    "direction": direction,
                    "contracts": int(contracts),
                    "entry": float(entry),
                    "stop_loss": float(sl),
                    "target": float(tp),
                    "confidence": float(conf),
                    "atr": float(atr),
                    "last_close": float(last_close),
                    "mode": "LIVE" if self.live_mode else "SIM",
                    "exit_price": None,
                }
            self._safe_send_message(
                f"✅ Entry {direction} {tsym} x{contracts} @~{round(entry,2)} | SL {round(sl,2)} TP {round(tp,2)} | conf {round(conf,2)}"
            )

        except Exception as e:
            logger.error(f"Entry placement failed: {e}", exc_info=True)

    # ---------- Finalization / circuit / status ----------

    def _finalize_trade(self, entry_id: str) -> None:
        tr = self.active_trades.get(entry_id)
        if not tr or tr.get("status") != "OPEN":
            return
        tr["status"] = "CLOSED"
        exit_px = float(tr.get("exit_price") or tr.get("target") or tr.get("stop_loss") or tr.get("entry"))
        entry_px = float(tr.get("entry", 0.0))
        dirn = tr.get("direction", "BUY")
        qty = int(tr.get("contracts", 0))
        L = max(1, int(self.LOT_SIZE))
        lots = qty // L if L > 0 else qty

        pnl_per = (exit_px - entry_px) if dirn == "BUY" else (entry_px - exit_px)
        gross = pnl_per * qty
        fees = float(self.FEES_PER_LOT) * lots
        net = gross - fees

        self.daily_pnl += net
        self.trades.append(
            {
                "id": entry_id,
                "symbol": tr.get("symbol"),
                "dir": dirn,
                "qty": qty,
                "entry": entry_px,
                "exit": exit_px,
                "pnl": gross,
                "fees": fees,
                "net": net,
                "time": datetime.now(),
            }
        )
        self._append_trade_log(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                entry_id,
                tr.get("symbol"),
                dirn,
                qty,
                round(entry_px, 2),
                round(exit_px, 2),
                round(gross, 2),
                round(fees, 2),
                round(net, 2),
                round(float(tr.get("confidence", 0.0)), 2),
                round(float(tr.get("atr", 0.0)), 4),
                tr.get("mode"),
            ]
        )
        self._closed_trades_today += 1
        self._last_closed_was_loss = net < 0
        if self._last_closed_was_loss and self.LOSS_COOLDOWN_MIN > 0:
            self._cooldown_until_ts = time.time() + 60 * int(self.LOSS_COOLDOWN_MIN)
            self._safe_send_message(f"⏸ Cooling down {self.LOSS_COOLDOWN_MIN}m after loss.")
        self._safe_send_message(
            f"🏁 Closed {dirn} {tr.get('symbol')} x{qty} | entry {round(entry_px,2)} exit {round(exit_px,2)} | net ₹{round(net,2)}"
        )

    def _is_circuit_breaker_tripped(self) -> bool:
        if self.daily_start_equity <= 0:
            return False
        dd = -self.daily_pnl / self.daily_start_equity
        return dd >= self.MAX_DAILY_DRAWDOWN_PCT

    def get_status(self) -> dict:
    """Return structured status for Telegram and other controllers."""
    with self._lock:
        open_n = len([t for t in self.active_trades.values() if t.get("status") == "OPEN"])
        return {
            "is_trading": self.is_trading,
            "live_mode": self.live_mode,
            "open_positions": open_n,
            "daily_pnl": round(self.daily_pnl, 2),
            "closed_today": self._closed_trades_today,
            "account_size": round(self.daily_start_equity, 2),
        }

    def get_summary(self) -> str:
        return f"Trades today: {len(self.trades)} | PnL: ₹{round(self.daily_pnl,2)}"

    def _send_detailed_status(self) -> bool:
        try:
            msg = self.get_status()
            if self.active_trades:
                with self._lock:
                    for k, v in self.active_trades.items():
                        msg += f"\n- {k[:8]} {v.get('symbol')} {v.get('direction')} x{v.get('contracts')} @ {round(v.get('entry',0.0),2)}"
            self._safe_send_message(msg)
            return True
        except Exception:
            return False

    def _run_health_check(self) -> bool:
        try:
            self._refresh_instruments_cache(force=True)
            self._safe_send_message("✅ Health OK (instruments cache refreshed).")
            return True
        except Exception as e:
            self._safe_send_message(f"❌ Health check failed: {e}")
            return False

    # ---------- TG helpers ----------

    def _safe_send_message(self, text: str, parse_mode: Optional[str] = None) -> None:
        try:
            if self.telegram_controller:
                self.telegram_controller.send_message(text, parse_mode=parse_mode)
            else:
                logger.info("[TG] %s", text)
        except Exception:
            logger.info("[TG?] %s", text)

    def _safe_send_alert(self, tag: str) -> None:
        try:
            if self.telegram_controller:
                self.telegram_controller.send_message(f"⚡ {tag} | live={self.live_mode}")
            else:
                logger.info("[ALERT] %s", tag)
        except Exception:
            logger.info("[ALERT] %s", tag)

    # ---------- Shutdown ----------

    def shutdown(self) -> None:
        try:
            self._trailing_worker_stop.set()
            self._oco_worker_stop.set()
        except Exception:
            pass
        self._stop_polling()
        try:
            self.order_executor.cancel_all_orders()
        except Exception:
            pass
        logger.info("👋 RealTimeTrader shut down.")