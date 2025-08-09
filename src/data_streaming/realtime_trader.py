# src/data_streaming/realtime_trader.py
from __future__ import annotations

import atexit
import csv
import logging
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import schedule

# --- Config & components -----------------------------------------------------
try:
    from src.config import Config
except Exception:  # pragma: no cover
    class Config:  # tiny fallback for local import issues
        ENABLE_LIVE_TRADING = False
        ENABLE_TELEGRAM = True
        BASE_STOP_LOSS_POINTS = 50
        BASE_TARGET_POINTS = 100
        CONFIDENCE_THRESHOLD = 7.0
        MIN_SIGNAL_SCORE = 6
        SPOT_SYMBOL = "NSE:NIFTY 50"
        NIFTY_LOT_SIZE = 75
        DATA_LOOKBACK_MINUTES = 30
        STRIKE_RANGE = 4
        TIME_FILTER_START = "09:15"
        TIME_FILTER_END = "15:25"
        STRIKE_SELECTION_TYPE = "OTM"
        OPTIONS_STOP_LOSS_PCT = 20.0
        OPTIONS_TARGET_PCT = 50.0
        MAX_DAILY_OPTIONS_TRADES = 5
        MAX_POSITION_VALUE = 50000
        MIN_SIGNAL_CONFIDENCE = 6.0
        ZERODHA_API_KEY = ""
        KITE_ACCESS_TOKEN = ""

try:
    from src.strategies.scalping_strategy import EnhancedScalpingStrategy
except Exception:  # pragma: no cover
    class EnhancedScalpingStrategy:
        def __init__(self, **kwargs): ...
        def generate_signal(self, *args, **kwargs): return None

try:
    from src.risk.position_sizing import PositionSizing
except Exception:  # pragma: no cover
    class PositionSizing:
        def calculate_position_size(self, **kwargs): return {"quantity": 1}
        def get_risk_status(self): return {}

try:
    from src.execution.order_executor import OrderExecutor
except Exception:  # pragma: no cover
    class OrderExecutor:
        def __init__(self, kite=None): self.kite = kite
        def place_entry_order(self, **kw): return "SIM-ORDER"
        def setup_gtt_orders(self, **kw): return True
        def get_active_orders(self): return {}
        def get_positions(self): return []
        def cancel_all_orders(self): return 0

try:
    from src.notifications.telegram_controller import TelegramController
except Exception:  # pragma: no cover
    class TelegramController:
        def __init__(self, **kwargs): ...
        def send_message(self, *a, **kw): ...
        def send_realtime_session_alert(self, *a, **kw): ...
        def send_startup_alert(self): ...
        def send_signal_alert(self, *a, **kw): ...
        def start_polling(self): ...
        def stop_polling(self): ...

try:
    from src.utils.strike_selector import (
        _get_spot_ltp_symbol,
        get_instrument_tokens,
        get_next_expiry_date,
        health_check,  # your helper; safe if no-op
    )
except Exception:  # pragma: no cover
    def _get_spot_ltp_symbol(): return "NSE:NIFTY 50"
    def get_instrument_tokens(**kw): return None
    def get_next_expiry_date(kite_instance): return ""
    def health_check(k): return {"overall_status": "ERROR", "message": "no kite"}

logger = logging.getLogger(__name__)


# =============================================================================
# RealTimeTrader
# =============================================================================
class RealTimeTrader:
    """
    Orchestrates:
      - data fetch (spot/options)
      - signal generation (strategy)
      - sizing + execution
      - Telegram I/O via callbacks
      - scheduling & graceful shutdown

    Threading model:
      - Telegram long-poll runs in a background *daemon* thread.
      - Main thread executes schedule.run_pending().
    """

    def __init__(self) -> None:
        self.is_trading: bool = False
        self.live_mode: bool = bool(getattr(Config, "ENABLE_LIVE_TRADING", False))
        self.daily_pnl: float = 0.0
        self.trades: List[Dict[str, Any]] = []

        # Instrument cache (thread-safe)
        self._nfo_instruments_cache: Optional[List[Dict]] = None
        self._nse_instruments_cache: Optional[List[Dict]] = None
        self._instruments_cache_timestamp: float = 0
        self._INSTRUMENT_CACHE_DURATION: int = 300  # seconds
        self._cache_lock = threading.RLock()

        # ATM cache (small TTL)
        self._atm_cache: Dict[str, Tuple[int, float]] = {}
        self._ATM_CACHE_DURATION: int = 30  # seconds

        # Worker pool
        self._executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="RTT")

        # Components
        self.strategy: Optional[EnhancedScalpingStrategy] = None
        self.risk_manager: Optional[PositionSizing] = None
        self.order_executor: OrderExecutor = self._init_order_executor()
        self.telegram_controller: Optional[TelegramController] = None
        self._polling_thread: Optional[threading.Thread] = None

        self._init_components()

        # Scheduling
        self._setup_smart_scheduling()

        # Start Telegram (optional)
        self._start_polling()

        # cleanup on exit
        atexit.register(self.shutdown)
        logger.info("RealTimeTrader initialized.")

    # ----- init helpers -------------------------------------------------------
    def _init_components(self) -> None:
        try:
            self.strategy = EnhancedScalpingStrategy(
                base_stop_loss_points=getattr(Config, "BASE_STOP_LOSS_POINTS", 50),
                base_target_points=getattr(Config, "BASE_TARGET_POINTS", 100),
                confidence_threshold=getattr(Config, "CONFIDENCE_THRESHOLD", 7.0),
                min_score_threshold=int(getattr(Config, "MIN_SIGNAL_SCORE", 6)),
            )
        except Exception as e:
            logger.warning("Strategy init failed: %s", e)
            self.strategy = None

        try:
            self.risk_manager = PositionSizing()
        except Exception as e:
            logger.warning("Risk manager init failed: %s", e)
            self.risk_manager = PositionSizing()  # basic fallback

        # Telegram is optional
        if getattr(Config, "ENABLE_TELEGRAM", True):
            try:
                self.telegram_controller = TelegramController(
                    status_callback=self.get_status,
                    control_callback=self._handle_control,
                    summary_callback=self.get_summary,
                )
            except Exception as e:
                logger.error("Telegram init failed: %s", e)
                self.telegram_controller = None

    def _init_order_executor(self) -> OrderExecutor:
        if not getattr(Config, "ENABLE_LIVE_TRADING", False):
            logger.info("Live trading disabled → simulation mode.")
            return OrderExecutor()

        try:
            from kiteconnect import KiteConnect
            api_key = getattr(Config, "ZERODHA_API_KEY", "")
            access_token = getattr(Config, "KITE_ACCESS_TOKEN", "")
            if not api_key or not access_token:
                raise ValueError("Missing ZERODHA_API_KEY or KITE_ACCESS_TOKEN")
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            logger.info("Live trading enabled with Kite.")
            return OrderExecutor(kite=kite)
        except Exception as e:
            logger.error("Live init failed → simulation mode: %s", e)
            return OrderExecutor()

    def _setup_smart_scheduling(self) -> None:
        try:
            schedule.every(30).seconds.do(self._smart_fetch_and_process)
            logger.info("Scheduled fetch/process every 30s (market hours only).")
            # End-of-day tidy
            schedule.every().day.at("15:35").do(self._end_of_day_tasks)
        except Exception as e:
            logger.error("Scheduling error: %s", e)

    # ----- telegram thread ----------------------------------------------------
    def _start_polling(self) -> None:
        if not self.telegram_controller:
            logger.info("Telegram disabled or not initialized.")
            return
        if self._polling_thread and self._polling_thread.is_alive():
            return
        try:
            self.telegram_controller.send_startup_alert()
        except Exception:
            pass

        def _runner():
            try:
                self.telegram_controller.start_polling()
            except Exception as e:
                logger.error("Telegram polling crashed: %s", e)

        self._polling_thread = threading.Thread(target=_runner, daemon=True)
        self._polling_thread.start()
        logger.info("✅ Telegram polling started (daemon).")

    def _stop_polling(self) -> None:
        if self.telegram_controller:
            try:
                self.telegram_controller.stop_polling()
            except Exception:
                pass
        if self._polling_thread and self._polling_thread.is_alive():
            if threading.current_thread() is not self._polling_thread:
                self._polling_thread.join(timeout=3)
        self._polling_thread = None

    # ----- public controls (callbacks) ---------------------------------------
    def start(self) -> bool:
        if self.is_trading:
            self._safe_send_message("🟢 Trader already running.")
            return True
        self.is_trading = True
        self._safe_send_alert("START")
        logger.info("Trading started.")
        return True

    def stop(self) -> bool:
        if not self.is_trading:
            self._safe_send_message("ℹ️ Trader already stopped.")
            return True
        self.is_trading = False
        self._safe_send_alert("STOP")
        logger.info("Trading stopped.")
        return True

    def _handle_control(self, command: str, arg: str = "") -> bool:
        cmd = (command or "").strip().lower()
        arg = (arg or "").strip().lower()
        try:
            if cmd == "start":
                return self.start()
            if cmd == "stop":
                return self.stop()
            if cmd == "mode":
                return self._set_live_mode(arg)
            if cmd == "refresh":
                return self._force_refresh_cache()
            if cmd == "status":
                return self._send_detailed_status()
            if cmd == "health":
                return self._run_health_check()
            if cmd == "emergency":
                return self.emergency_stop_all()
            self._safe_send_message(f"❌ Unknown command: `{cmd}`", parse_mode="Markdown")
            return False
        except Exception as e:
            logger.error("Control error: %s", e)
            return False

    def _set_live_mode(self, mode: str) -> bool:
        desired_live = (mode == "live")
        if desired_live == self.live_mode:
            self._safe_send_message(f"Already in *{'LIVE' if self.live_mode else 'SHADOW'}*.", parse_mode="Markdown")
            return True
        if self.is_trading:
            self._safe_send_message("🛑 Stop trading first with /stop.", parse_mode="Markdown")
            return False

        if desired_live:
            try:
                from kiteconnect import KiteConnect
                api_key = getattr(Config, "ZERODHA_API_KEY", "")
                access_token = getattr(Config, "KITE_ACCESS_TOKEN", "")
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(access_token)
                self.order_executor = OrderExecutor(kite=kite)
                self.live_mode = True
                self._refresh_instruments_cache(force=True)
                self._safe_send_message("🟢 Switched to *LIVE*.", parse_mode="Markdown")
                return True
            except Exception as e:
                logger.error("Switch to LIVE failed: %s", e)
                self._safe_send_message("❌ Could not switch to LIVE. Staying in SHADOW.", parse_mode="Markdown")
                return False
        else:
            self.order_executor = OrderExecutor()
            self.live_mode = False
            self._safe_send_message("🛡️ Switched to *SHADOW*.", parse_mode="Markdown")
            return True

    # ----- cache --------------------------------------------------------------
    def _force_refresh_cache(self) -> bool:
        try:
            self._refresh_instruments_cache(force=True)
            self._safe_send_message("🔄 Instruments cache refreshed.")
            return True
        except Exception as e:
            logger.error("Refresh cache error: %s", e)
            return False

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        with self._cache_lock:
            kite = getattr(self.order_executor, "kite", None)
            if not kite:
                self._nfo_instruments_cache = []
                self._nse_instruments_cache = []
                return

            now = time.time()
            if not force and (now - self._instruments_cache_timestamp) < self._INSTRUMENT_CACHE_DURATION:
                return

            try:
                with ThreadPoolExecutor(max_workers=2) as ex:
                    nf = ex.submit(kite.instruments, "NFO")
                    ns = ex.submit(kite.instruments, "NSE")
                    self._nfo_instruments_cache = nf.result(timeout=15) or []
                    self._nse_instruments_cache = ns.result(timeout=15) or []
                self._instruments_cache_timestamp = now
                logger.debug("Instrument cache refreshed: NFO=%d, NSE=%d",
                             len(self._nfo_instruments_cache), len(self._nse_instruments_cache))
            except Exception as e:
                logger.error("Instrument cache refresh failed: %s", e)
                self._nfo_instruments_cache = self._nfo_instruments_cache or []
                self._nse_instruments_cache = self._nse_instruments_cache or []

    def _get_cached_instruments(self) -> Tuple[List[Dict], List[Dict]]:
        with self._cache_lock:
            return self._nfo_instruments_cache or [], self._nse_instruments_cache or []

    @lru_cache(maxsize=256)
    def _atm_from_spot(self, spot: float) -> int:
        return int(round(float(spot) / 50.0) * 50)

    # ----- scheduling entry ---------------------------------------------------
    def _smart_fetch_and_process(self) -> None:
        now = datetime.now()
        if now.hour < 9 or (now.hour == 9 and now.minute < 10) or now.hour > 15:
            return
        try:
            self.fetch_and_process_data()
        except Exception as e:
            logger.error("smart_fetch error: %s", e)

    # ----- core fetch/process -------------------------------------------------
    def fetch_and_process_data(self) -> None:
        if not self.is_trading:
            return
        kite = getattr(self.order_executor, "kite", None)
        if not kite:
            logger.error("No Kite instance (shadow/live not enabled).")
            return

        t0 = time.time()
        self._refresh_instruments_cache()
        cached_nfo, cached_nse = self._get_cached_instruments()
        if not cached_nfo:
            logger.error("Empty NFO cache; skipping cycle.")
            return

        # spot & instruments lookup in parallel
        with ThreadPoolExecutor(max_workers=3) as ex:
            fut_spot = ex.submit(self._fetch_spot_price, kite)
            fut_instr = ex.submit(
                get_instrument_tokens,
                symbol=getattr(Config, "SPOT_SYMBOL", "NIFTY"),
                kite_instance=kite,
                cached_nfo_instruments=cached_nfo,
                cached_nse_instruments=cached_nse,
            )

            spot_price = fut_spot.result(timeout=10)
            instruments = fut_instr.result(timeout=10)

        if not spot_price or not instruments:
            logger.warning("Missing spot or instruments; skipping.")
            return

        atm_strike = instruments.get("atm_strike")
        spot_token = instruments.get("spot_token")
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=int(getattr(Config, "DATA_LOOKBACK_MINUTES", 30)))

        # fetch historical: spot + selected options around ATM
        spot_df, options_data = self._fetch_all_data_parallel(
            kite, spot_token, instruments, start_time, end_time, cached_nfo, cached_nse
        )

        # Use your scalping strategy for spot/futures if you want (left intact)
        # Here we focus on options path that your trader uses.
        selected = self._select_strikes_from_data(spot_price, options_data, atm_strike)
        if not selected:
            selected = self._fallback_strikes(atm_strike, cached_nfo, cached_nse, kite)

        self._process_selected_strikes(selected, options_data, spot_df)

        dt = time.time() - t0
        logger.debug("Cycle done in %.2fs", dt)

    def _fetch_spot_price(self, kite) -> Optional[float]:
        try:
            sym = _get_spot_ltp_symbol()
            ltp = kite.ltp([sym]).get(sym, {}).get("last_price")
            return float(ltp) if ltp is not None else None
        except Exception as e:
            logger.error("Spot LTP error: %s", e)
            return None

    def _fetch_all_data_parallel(
        self,
        kite,
        spot_token: Optional[int],
        instruments: Dict[str, Any],
        start_time: datetime,
        end_time: datetime,
        cached_nfo: List[Dict],
        cached_nse: List[Dict],
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        spot_df = pd.DataFrame()
        options: Dict[str, pd.DataFrame] = {}

        tasks: List[Tuple[str, int]] = []
        if spot_token:
            tasks.append(("__SPOT__", spot_token))

        strike_range = int(getattr(Config, "STRIKE_RANGE", 4))
        for side in ("ce_token", "pe_token"):
            tok = instruments.get(side)
            sym = instruments.get(side.replace("_token", "_symbol"))
            if tok and sym:
                tasks.append((sym, tok))

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(self._fetch_hist, kite, tok, start_time, end_time): name for name, tok in tasks}
            for fut in as_completed(futures, timeout=25):
                name = futures[fut]
                try:
                    df = fut.result()
                    if df is None or df.empty:
                        continue
                    if name == "__SPOT__":
                        spot_df = df
                    else:
                        options[name] = df
                except Exception as e:
                    logger.warning("Hist fetch failed for %s: %s", name, e)

        return spot_df, options

    def _fetch_hist(self, kite, token: int, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        try:
            data = kite.historical_data(instrument_token=token, from_date=start_time, to_date=end_time, interval="minute")
            if not data:
                return None
            df = pd.DataFrame(data)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            return df
        except Exception as e:
            logger.error("historical_data error token %s: %s", token, e)
            return None

    # ----- selection & processing --------------------------------------------
    def _select_strikes_from_data(self, spot_price: float, options_data: Dict[str, pd.DataFrame], atm_strike: int) -> List[Dict[str, Any]]:
        if not options_data:
            return []
        selected: List[Dict[str, Any]] = []
        for opt_type in ("CE", "PE"):
            bucket: List[Dict[str, Any]] = []
            for symbol, df in options_data.items():
                if opt_type not in symbol or df.empty:
                    continue
                try:
                    ix = symbol.find(opt_type)
                    strike = int(symbol[ix - 5: ix])
                except Exception:
                    continue
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else last
                oi_change = float(last.get("oi", 0)) - float(prev.get("oi", 0))
                is_atm = (strike == atm_strike)
                info = {
                    "symbol": symbol,
                    "strike": strike,
                    "type": opt_type,
                    "ltp": float(last.get("last_price", 0) or last.get("close", 0) or 0),
                    "oi": int(last.get("oi", 0) or 0),
                    "oi_change": int(oi_change),
                    "is_atm": is_atm,
                    "is_otm": (strike > atm_strike) if opt_type == "CE" else (strike < atm_strike),
                    "is_itm": (strike < atm_strike) if opt_type == "CE" else (strike > atm_strike),
                }
                bucket.append(info)
            # sort by OI change desc, prefer ATM first
            bucket.sort(key=lambda x: (not x["is_atm"], -x["oi_change"]))
            selected.extend(bucket[:2])  # pick top 2 per side (ATM + best OI change)
        return selected

    def _fallback_strikes(self, atm_strike: int, cached_nfo: List[Dict], cached_nse: List[Dict], kite) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for off in (0, 1, -1):
            instr = get_instrument_tokens(
                symbol=getattr(Config, "SPOT_SYMBOL", "NIFTY"),
                kite_instance=kite,
                cached_nfo_instruments=cached_nfo,
                cached_nse_instruments=cached_nse,
                offset=off,
            )
            if not instr:
                continue
            for side, key in (("CE", "ce_symbol"), ("PE", "pe_symbol")):
                sym = instr.get(key)
                if sym:
                    out.append({
                        "symbol": sym,
                        "strike": atm_strike + 50 * off,
                        "type": side,
                        "ltp": 0,
                        "oi": 0,
                        "oi_change": 0,
                        "is_atm": (off == 0),
                        "is_otm": (off > 0) if side == "CE" else (off < 0),
                        "is_itm": (off < 0) if side == "CE" else (off > 0),
                    })
        if out:
            logger.warning("Using fallback strike selection.")
        return out

    def _process_selected_strikes(self, selected: List[Dict[str, Any]], options_data: Dict[str, pd.DataFrame], spot_df: pd.DataFrame) -> None:
        if not selected:
            return
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = []
            for s in selected:
                df = options_data.get(s["symbol"])
                if df is not None and not df.empty:
                    futs.append(ex.submit(self._process_option_bar, s["symbol"], df, spot_df, s))
                else:
                    futs.append(ex.submit(self._process_ltp_only, s))
            for fut in as_completed(futs, timeout=20):
                try:
                    fut.result()
                except Exception as e:
                    logger.error("process strike error: %s", e)

    def _process_ltp_only(self, s: Dict[str, Any]) -> None:
        kite = getattr(self.order_executor, "kite", None)
        if not kite:
            return
        try:
            data = kite.ltp([f"NFO:{s['symbol']}"])
            ltp = float(data.get(f"NFO:{s['symbol']}", {}).get("last_price", 0) or 0)
            if ltp <= 0:
                return
            df = pd.DataFrame([{"date": pd.Timestamp.now(), "last_price": ltp, "oi": s.get("oi", 0), "volume": 0}]).set_index("date")
            self._process_option_bar(s["symbol"], df, pd.DataFrame(), s)
        except Exception as e:
            logger.error("ltp-only error for %s: %s", s["symbol"], e)

    def _process_option_bar(self, symbol: str, ohlc: pd.DataFrame, spot_ohlc: pd.DataFrame, s: Dict[str, Any]) -> None:
        if not self.is_trading:
            return

        now = datetime.now()
        if now.hour < 9 or (now.hour == 9 and now.minute < 15) or now.hour > 15 or (now.hour == 15 and now.minute > 30):
            return

        # current price
        if not ohlc.empty:
            current_price = float(ohlc.iloc[-1].get("last_price", ohlc.iloc[-1].get("close", 0)) or 0)
        else:
            current_price = float(s.get("ltp", 0))
        if current_price <= 0:
            return

        # time window filter (optional)
        start = getattr(Config, "TIME_FILTER_START", None)
        end = getattr(Config, "TIME_FILTER_END", None)
        if start and end:
            ts = (ohlc.index[-1] if not ohlc.empty else pd.Timestamp.now()).strftime("%H:%M")
            if not (start <= ts <= end):
                return

        # simple multi-factor: momentum + OI tilt
        conf = 0.0
        direction = None
        if len(ohlc) >= 3:
            last = ohlc["last_price"].iloc[-1]
            first = ohlc["last_price"].iloc[-3]
            chg = (last - first) / max(first, 1e-9) * 100.0
            if chg > 2:
                direction, conf = "BUY", min(5.0, chg)
            elif chg < -1.5 and s.get("is_otm"):
                direction, conf = "BUY", min(4.0, abs(chg) * 0.8)

        if s.get("oi_change", 0) > 0:
            conf *= 1.2

        if not direction:
            return

        min_conf = float(getattr(Config, "MIN_SIGNAL_CONFIDENCE", 6.0))
        if conf < min_conf:
            return

        # position sizing
        position = (self.risk_manager or PositionSizing()).calculate_position_size(
            entry_price=current_price,
            stop_loss=current_price * (1 - float(getattr(Config, "OPTION_SL_PERCENT", 0.05))),
            signal_confidence=conf,
            market_volatility=0.0,
        )
        if not position or position.get("quantity", 0) <= 0:
            return

        # risk guard: cap notional
        max_pos_val = float(getattr(Config, "MAX_POSITION_VALUE", 50000))
        notional = position["quantity"] * current_price * int(getattr(Config, "NIFTY_LOT_SIZE", 75))
        if notional > max_pos_val:
            return

        # place order
        try:
            if self.telegram_controller:
                self.telegram_controller.send_signal_alert(len(self.trades) + 1, {
                    "signal": direction,
                    "entry_price": current_price,
                    "stop_loss": current_price * (1 - float(getattr(Config, "OPTION_SL_PERCENT", 0.05))),
                    "target": current_price * (1 + float(getattr(Config, "OPTION_TP_PERCENT", 0.15))),
                    "confidence": conf,
                    "strategy_type": "options_momentum",
                }, position)
        except Exception:
            pass

        order_id = self.order_executor.place_entry_order(
            symbol=symbol,
            exchange="NFO",
            transaction_type=direction,
            quantity=position["quantity"],
        )
        if not order_id:
            return

        self.order_executor.setup_gtt_orders(
            entry_order_id=order_id,
            entry_price=current_price,
            stop_loss_price=current_price * (1 - float(getattr(Config, "OPTION_SL_PERCENT", 0.05))),
            target_price=current_price * (1 + float(getattr(Config, "OPTION_TP_PERCENT", 0.15))),
            symbol=symbol,
            exchange="NFO",
            quantity=position["quantity"],
            transaction_type=direction,
        )

        self.trades.append({
            "order_id": order_id,
            "symbol": symbol,
            "direction": direction,
            "quantity": position["quantity"],
            "entry_price": current_price,
            "stop_loss": current_price * (1 - float(getattr(Config, "OPTION_SL_PERCENT", 0.05))),
            "target": current_price * (1 + float(getattr(Config, "OPTION_TP_PERCENT", 0.15))),
            "confidence": conf,
            "strike_info": s,
            "timestamp": datetime.now(),
            "strategy_type": "options_momentum",
        })
        logger.info("✅ Trade: %s %sx %s @ %.2f", direction, position["quantity"], symbol, current_price)

    # ----- admin/status -------------------------------------------------------
    def _run_health_check(self) -> bool:
        try:
            kite = getattr(self.order_executor, "kite", None)
            res = health_check(kite) if kite else {"overall_status": "ERROR", "message": "no kite"}
            self._safe_send_message(
                f"🏥 Health: {res.get('overall_status')} | Trading: {'ON' if self.is_trading else 'OFF'} | Mode: {'LIVE' if self.live_mode else 'SHADOW'}",
                parse_mode="Markdown",
            )
            return True
        except Exception as e:
            logger.error("health check error: %s", e)
            return False

    def get_status(self) -> Dict[str, Any]:
        try:
            active = 0
            try:
                active = len(self.order_executor.get_active_orders())
            except Exception:
                pass
            status = {
                "is_trading": self.is_trading,
                "open_orders": active,
                "trades_today": len(self.trades),
                "live_mode": self.live_mode,
                "daily_pnl": self.daily_pnl,
                "last_update": datetime.now().strftime("%H:%M:%S"),
            }
            try:
                if self.risk_manager:
                    status.update(self.risk_manager.get_risk_status())
            except Exception:
                pass
            return status
        except Exception as e:
            logger.error("status error: %s", e)
            return {"error": str(e)}

    def _send_detailed_status(self) -> bool:
        try:
            st = self.get_status()
            msg = (
                "📊 <b>Detailed Status</b>\n"
                f"🔄 Trading: {'✅ Active' if st['is_trading'] else '❌ Stopped'}\n"
                f"🎯 Mode: {'🟢 LIVE' if self.live_mode else '🛡️ SHADOW'}\n"
                f"📈 Open Orders: {st.get('open_orders', 0)}\n"
                f"💼 Trades Today: {st['trades_today']}\n"
                f"💰 Daily PnL: ₹{st.get('daily_pnl', 0.0):.2f}\n"
            )
            self._safe_send_message(msg, parse_mode="HTML")
            return True
        except Exception as e:
            logger.error("send status error: %s", e)
            return False

    def get_summary(self) -> str:
        try:
            lines = [
                "📊 <b>Daily Options Trading Summary</b>",
                f"🔁 <b>Total trades:</b> {len(self.trades)}",
                f"💰 <b>PnL:</b> ₹{self.daily_pnl:.2f}",
                f"📈 <b>Mode:</b> {'🟢 LIVE' if self.live_mode else '🛡️ SHADOW'}",
                "────────────────────",
            ]
            recent = self.trades[-5:]
            for t in recent:
                s = t.get("strike_info", {})
                lines.append(
                    f"{s.get('type','?')} {s.get('strike','?')} {t['direction']} {t['quantity']} @ ₹{t['entry_price']:.2f} "
                    f"(SL ₹{t['stop_loss']:.2f}, TP ₹{t['target']:.2f})"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.error("summary error: %s", e)
            return "Summary unavailable."

    def _end_of_day_tasks(self) -> None:
        try:
            if self.is_trading:
                self.stop()
            if self.trades:
                self.export_trades_to_csv()
            self._safe_send_message("🕟 EOD tasks done.")
        except Exception as e:
            logger.error("EOD error: %s", e)

    # ----- exports & emergency ------------------------------------------------
    def export_trades_to_csv(self, filename: Optional[str] = None) -> str:
        try:
            filename = filename or f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
            fields = [
                "timestamp", "order_id", "symbol", "direction", "quantity",
                "entry_price", "stop_loss", "target", "confidence",
                "strategy_type", "option_type", "strike", "expiry",
            ]
            with open(filename, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for t in self.trades:
                    s = t.get("strike_info", {})
                    w.writerow({
                        "timestamp": t.get("timestamp", ""),
                        "order_id": t.get("order_id", ""),
                        "symbol": t.get("symbol", ""),
                        "direction": t.get("direction", ""),
                        "quantity": t.get("quantity", 0),
                        "entry_price": t.get("entry_price", 0),
                        "stop_loss": t.get("stop_loss", 0),
                        "target": t.get("target", 0),
                        "confidence": t.get("confidence", 0.0),
                        "strategy_type": t.get("strategy_type", "unknown"),
                        "option_type": s.get("type", ""),
                        "strike": s.get("strike", ""),
                        "expiry": s.get("expiry", ""),
                    })
            logger.info("Trades exported to %s", filename)
            return filename
        except Exception as e:
            logger.error("export error: %s", e)
            return ""

    def emergency_stop_all(self) -> bool:
        try:
            self.is_trading = False
            canceled = 0
            try:
                canceled = self.order_executor.cancel_all_orders()
            except Exception:
                pass
            self._safe_send_message(f"🚨 EMERGENCY STOP. Canceled orders: {canceled}")
            return True
        except Exception as e:
            logger.error("emergency stop error: %s", e)
            return False

    # ----- util comms --------------------------------------------------------
    def _safe_send_message(self, msg: str, **kw) -> None:
        try:
            if self.telegram_controller:
                self.telegram_controller.send_message(msg, **kw)
        except Exception:
            pass

    def _safe_send_alert(self, action: str) -> None:
        try:
            if self.telegram_controller:
                self.telegram_controller.send_realtime_session_alert(action)
        except Exception:
            pass

    # ----- shutdown -----------------------------------------------------------
    def shutdown(self) -> None:
        logger.info("Shutting down RealTimeTrader...")
        try:
            self.stop()
        except Exception:
            pass
        self._stop_polling()
        try:
            self._executor.shutdown(wait=True, timeout=5)
        except Exception:
            pass
        logger.info("✅ Shutdown complete.")

    def __repr__(self) -> str:
        return f"<RealTimeTrader trading={self.is_trading} live={self.live_mode} trades={len(self.trades)}>"
