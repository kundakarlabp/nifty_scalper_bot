# src/data_streaming/realtime_trader.py
import logging
import threading
import atexit
import signal
import sys
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import schedule
from datetime import datetime, timedelta
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

from src.config import Config
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import PositionSizing, get_live_account_balance
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.utils.strike_selector import (
    _get_spot_ltp_symbol,
    get_instrument_tokens,
    get_next_expiry_date,
    health_check as _health_check,  # your helper (keep name distinct)
)

logger = logging.getLogger(__name__)


class RealTimeTrader:
    """Real-time options trader. Telegram polling runs in its own daemon thread from controller."""

    def __init__(self) -> None:
        self.is_trading: bool = False
        self.daily_pnl: float = 0.0
        self.trades: List[Dict[str, Any]] = []
        self.live_mode: bool = getattr(Config, "ENABLE_LIVE_TRADING", False)

        # Instrument cache (for Kite rate-limits)
        self._nfo_instruments_cache: Optional[List[Dict]] = None
        self._nse_instruments_cache: Optional[List[Dict]] = None
        self._instruments_cache_timestamp: float = 0
        self._INSTRUMENT_CACHE_DURATION: int = 300
        self._cache_lock = threading.RLock()

        # ATM cache
        self._atm_cache: Dict[str, Tuple[int, float]] = {}
        self._ATM_CACHE_DURATION: int = 30

        # Thread pool
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="OptionsDataWorker")

        # Components
        self._init_components()
        self._polling_thread: Optional[threading.Thread] = None

        # Schedule & polling
        self._start_polling()
        self._setup_smart_scheduling()

        atexit.register(self.shutdown)
        logger.info("RealTimeTrader initialized.")

        # One-time lightweight balance log
        self._log_account_balance()

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
            # Hard fail here is better so user fixes token/chat id
            raise

    def _init_order_executor(self) -> OrderExecutor:
        if not self.live_mode:
            logger.info("Live trading disabled → simulation mode.")
            return OrderExecutor()
        try:
            from kiteconnect import KiteConnect
            api_key = Config.ZERODHA_API_KEY
            access_token = Config.KITE_ACCESS_TOKEN
            if not api_key or not access_token:
                raise ValueError("API key or access token missing")
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            logger.info("✅ Live order executor initialized.")
            return OrderExecutor(kite=kite)
        except Exception as exc:
            logger.error(f"Failed live init, falling back to simulation: {exc}")
            self.live_mode = False
            return OrderExecutor()

    # ---------- Scheduling ----------
    def _setup_smart_scheduling(self) -> None:
        try:
            schedule.every(30).seconds.do(self._smart_fetch_and_process)
            schedule.every(getattr(Config, "BALANCE_LOG_INTERVAL_MIN", 30)).minutes.do(
                self.refresh_account_balance
            )
            logger.info("Scheduled fetch/process every 30s (market hours only).")
        except Exception as e:
            logger.error(f"Error setting up scheduling: {e}")

    def _smart_fetch_and_process(self) -> None:
        try:
            now = datetime.now()
            if not self._is_trading_hours(now):
                # heartbeat every ~5 minutes
                if int(time.time()) % 300 < 2:
                    logger.info("⏳ Market closed. Skipping fetch.")
                return
            self.fetch_and_process_data()
        except Exception as e:
            logger.error(f"Error in smart fetch and process: {e}")

    # ---------- Telegram control ----------
    def _start_polling(self) -> None:
        if self._polling_thread and self._polling_thread.is_alive():
            return
        try:
            self.telegram_controller.send_startup_alert()
        except Exception as e:
            logger.warning(f"Failed to send startup alert: {e}")
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
        if self.telegram_controller:
            try:
                self.telegram_controller.stop_polling()
            except Exception as e:
                logger.warning(f"Error stopping telegram polling: {e}")
        if self._polling_thread and self._polling_thread.is_alive():
            if threading.current_thread() != self._polling_thread:
                self._polling_thread.join(timeout=3)
        self._polling_thread = None

    def start(self) -> bool:
        if self.is_trading:
            self._safe_send_message("🛑 Trader already running.")
            return True
        self.is_trading = True
        self._safe_send_alert("START")
        logger.info("✅ Trading started.")
        return True

    def stop(self) -> bool:
        if not self.is_trading:
            self._safe_send_message("🛑 Trader is already stopped.")
            return True
        self.is_trading = False
        self._safe_send_alert("STOP")
        logger.info("🛑 Trading stopped.")
        return True

    def _handle_control(self, command: str, arg: str = "") -> bool:
        command = command.strip().lower()
        arg = arg.strip().lower() if arg else ""
        logger.info(f"Received command: /{command} {arg}")
        try:
            if command == "start":
                return self.start()
            elif command == "stop":
                return self.stop()
            elif command == "mode":
                if arg not in ["live", "shadow"]:
                    self._safe_send_message("⚠️ Usage: `/mode live` or `/mode shadow`", parse_mode="Markdown")
                    return False
                return self._set_live_mode(arg)
            elif command == "refresh":
                return self._force_refresh_cache()
            elif command == "status":
                return self._send_detailed_status()
            elif command == "health":
                return self._run_health_check()
            elif command == "emergency":
                return self.emergency_stop_all()
            else:
                self._safe_send_message(f"❌ Unknown command: `{command}`", parse_mode="Markdown")
                return False
        except Exception as e:
            logger.error(f"Error handling control command: {e}")
            return False

    # ---------- Balance helpers ----------
    def refresh_account_balance(self) -> None:
        """Pull live balance and push into risk manager (logs result)."""
        try:
            new_bal = get_live_account_balance()
            self.risk_manager.account_size = new_bal
            self.risk_manager.equity = new_bal
            self.risk_manager.equity_peak = max(self.risk_manager.equity_peak, new_bal)
            logger.info(f"💳 Refreshed account balance: ₹{new_bal:.2f}")
        except Exception as e:
            logger.warning(f"Balance refresh failed: {e}")

    def _log_account_balance(self) -> None:
        try:
            logger.info(f"💰 Account size (cached): ₹{self.risk_manager.account_size:.2f}")
        except Exception as e:
            logger.debug(f"Could not log account balance: {e}")

    # ---------- Market hours ----------
    def _is_trading_hours(self, current_time: datetime) -> bool:
        if getattr(Config, "ALLOW_OFFHOURS_TESTING", False):
            return True
        hour = current_time.hour
        minute = current_time.minute
        if hour < 9 or (hour == 9 and minute < 15):
            return False
        if hour > 15 or (hour == 15 and minute > 30):
            return False
        return True

    # ---------- Cache handling ----------
    def _force_refresh_cache(self) -> bool:
        try:
            self._refresh_instruments_cache(force=True)
            self._safe_send_message("🔄 Instrument cache refreshed successfully.")
            return True
        except Exception as e:
            logger.error(f"Error refreshing cache: {e}")
            return False

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        with self._cache_lock:
            if not self.order_executor or not getattr(self.order_executor, "kite", None):
                self._nfo_instruments_cache = []
                self._nse_instruments_cache = []
                return

            current_time = time.time()
            needs_refresh = (
                force
                or self._nfo_instruments_cache is None
                or self._nse_instruments_cache is None
                or (current_time - self._instruments_cache_timestamp) > self._INSTRUMENT_CACHE_DURATION
            )

            if needs_refresh:
                try:
                    with ThreadPoolExecutor(max_workers=2) as ex:
                        nfo_f = ex.submit(self.order_executor.kite.instruments, "NFO")
                        nse_f = ex.submit(self.order_executor.kite.instruments, "NSE")
                        self._nfo_instruments_cache = nfo_f.result(timeout=10)
                        self._nse_instruments_cache = nse_f.result(timeout=10)
                    self._instruments_cache_timestamp = current_time
                    logger.info("✅ Instruments cache refreshed.")
                except Exception as e:
                    logger.error(f"Failed to refresh instruments cache: {e}")
                    self._nfo_instruments_cache = self._nfo_instruments_cache or []
                    self._nse_instruments_cache = self._nse_instruments_cache or []

    def _get_cached_instruments(self) -> Tuple[List[Dict], List[Dict]]:
        with self._cache_lock:
            return self._nfo_instruments_cache or [], self._nse_instruments_cache or []

    @lru_cache(maxsize=128)
    def _get_cached_atm_strike(self, spot_price: float, timestamp_bucket: int) -> int:
        return round(spot_price / 50) * 50

    # ---------- Health/Mode ----------
    def _send_detailed_status(self) -> bool:
        try:
            status = self.get_status()
            cache_age = (time.time() - self._instruments_cache_timestamp) / 60
            status_msg = f"""
📊 **Detailed Status**
🔄 Trading: {'✅ Active' if status['is_trading'] else '❌ Stopped'}
🎯 Mode: {'🟢 LIVE' if self.live_mode else '🛡️ SHADOW'}
📈 Open Orders: {status.get('open_orders', 0)}
💼 Trades Today: {status['trades_today']}
🕐 Cache Age: {cache_age:.1f} min
📊 Daily PnL: ₹{self.daily_pnl:.2f}
"""
            self._safe_send_message(status_msg, parse_mode="Markdown")
            return True
        except Exception as e:
            logger.error(f"Error sending detailed status: {e}")
            return False

    def _run_health_check(self) -> bool:
        try:
            if getattr(self.order_executor, "kite", None):
                health_result = _health_check(self.order_executor.kite)
            else:
                health_result = {"overall_status": "ERROR", "message": "No Kite instance available"}
            health_msg = f"""
🏥 **System Health Check**
📊 Overall: {health_result.get('overall_status', 'UNKNOWN')}
🔄 Trading: {'✅ ACTIVE' if self.is_trading else '⏹️ STOPPED'}
🎯 Mode: {'🟢 LIVE' if self.live_mode else '🛡️ SHADOW'}
💼 Trades: {len(self.trades)}
📱 Telegram: {'✅ ACTIVE' if self._polling_thread and self._polling_thread.is_alive() else '❌ INACTIVE'}
"""
            self._safe_send_message(health_msg, parse_mode="Markdown")
            return True
        except Exception as e:
            logger.error(f"Error running health check: {e}")
            return False

    def _set_live_mode(self, mode: str) -> bool:
        desired_live = (mode == "live")
        if desired_live == self.live_mode:
            current_mode = "LIVE" if self.live_mode else "SHADOW"
            self._safe_send_message(f"🟢 Already in *{current_mode}* mode.", parse_mode="Markdown")
            return True
        if self.is_trading:
            self._safe_send_message("🛑 Cannot change mode while trading. Use `/stop` first.", parse_mode="Markdown")
            return False
        if desired_live:
            try:
                from kiteconnect import KiteConnect
                kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
                kite.set_access_token(Config.KITE_ACCESS_TOKEN)
                self.order_executor = OrderExecutor(kite=kite)
                self.live_mode = True
                self._refresh_instruments_cache(force=True)
                self._safe_send_message("🚀 Switched to *LIVE* trading mode.", parse_mode="Markdown")
                return True
            except Exception as exc:
                logger.error(f"Failed to switch to LIVE mode: {exc}")
                self._safe_send_message(
                    f"❌ Failed to switch to LIVE mode: `{str(exc)[:100]}...` Reverted to SHADOW mode.",
                    parse_mode="Markdown",
                )
                self.live_mode = False
                self.order_executor = OrderExecutor()
                return False
        else:
            self.order_executor = OrderExecutor()
            self.live_mode = False
            self._safe_send_message("🛡️ Switched to *SHADOW* (simulation) mode.", parse_mode="Markdown")
            return True

    # ---------- Fetch & process (same structure; trimmed only for readability) ----------
    def fetch_and_process_data(self) -> None:
        if not self.is_trading:
            return
        t0 = time.time()
        try:
            if not getattr(self.order_executor, "kite", None):
                logger.error("KiteConnect instance not found. Is live mode enabled?")
                return

            self._refresh_instruments_cache()
            cached_nfo, cached_nse = self._get_cached_instruments()
            if not cached_nfo and not cached_nse:
                logger.error("Instrument cache is empty. Cannot proceed.")
                return

            with ThreadPoolExecutor(max_workers=3) as ex:
                spot_future = ex.submit(self._fetch_spot_price)
                instr_future = ex.submit(self._get_instruments_data, cached_nfo, cached_nse)
                spot_price = spot_future.result(timeout=5)
                instruments_data = instr_future.result(timeout=5)

            if not spot_price or not instruments_data:
                logger.error("Failed to fetch essential data (spot price or instruments)")
                return

            atm_strike = instruments_data['atm_strike']
            spot_token = instruments_data.get('spot_token')
            end_time = datetime.now()
            start_time_data = end_time - timedelta(minutes=Config.DATA_LOOKBACK_MINUTES)

            spot_df, options_data = self._fetch_all_data_parallel(
                spot_token, atm_strike, start_time_data, end_time, cached_nfo, cached_nse
            )

            if options_data:
                selected_strikes_info = self._analyze_options_data_optimized(spot_price, options_data)
            else:
                selected_strikes_info = self._get_fallback_strikes(atm_strike, cached_nfo, cached_nse)

            if not selected_strikes_info:
                logger.warning("No strikes selected for processing")
                return

            self._process_selected_strikes(selected_strikes_info, options_data, spot_df)
            logger.debug(f"Data fetch+process in {time.time() - t0:.2f}s")

        except Exception as e:
            logger.error(f"Error in fetch_and_process_data: {e}", exc_info=True)

    def _fetch_spot_price(self) -> Optional[float]:
        try:
            sym = _get_spot_ltp_symbol()
            ltp = self.order_executor.kite.ltp([sym])
            price = ltp.get(sym, {}).get("last_price")
            return float(price) if price is not None else None
        except Exception as e:
            logger.error(f"Exception fetching spot price: {e}")
            return None

    def _get_instruments_data(self, cached_nfo: List[Dict], cached_nse: List[Dict]) -> Optional[Dict]:
        try:
            return get_instrument_tokens(
                symbol=Config.SPOT_SYMBOL,
                kite_instance=self.order_executor.kite,
                cached_nfo_instruments=cached_nfo,
                cached_nse_instruments=cached_nse,
            )
        except Exception as e:
            logger.error(f"Error getting instruments data: {e}")
            return None

    def _fetch_all_data_parallel(
        self,
        spot_token: Optional[int],
        atm_strike: int,
        start_time: datetime,
        end_time: datetime,
        cached_nfo: List[Dict],
        cached_nse: List[Dict],
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        spot_df = pd.DataFrame()
        options_data: Dict[str, pd.DataFrame] = {}
        tasks = []
        if spot_token:
            tasks.append(('SPOT', spot_token))

        try:
            for offset in range(-Config.STRIKE_RANGE, Config.STRIKE_RANGE + 1):
                temp = get_instrument_tokens(
                    symbol=Config.SPOT_SYMBOL,
                    offset=offset,
                    kite_instance=self.order_executor.kite,
                    cached_nfo_instruments=cached_nfo,
                    cached_nse_instruments=cached_nse,
                )
                if temp:
                    for opt_type, token_key, symbol_key in [('CE', 'ce_token', 'ce_symbol'), ('PE', 'pe_token', 'pe_symbol')]:
                        token = temp.get(token_key)
                        symbol = temp.get(symbol_key)
                        if token and symbol:
                            tasks.append((symbol, token))
        except Exception as e:
            logger.error(f"Error preparing fetch tasks: {e}")

        with ThreadPoolExecutor(max_workers=8) as ex:
            fut2sym = {
                ex.submit(self._fetch_historical_data, token, start_time, end_time): symbol
                for symbol, token in tasks
            }
            for fut in as_completed(fut2sym, timeout=15):
                sym = fut2sym[fut]
                try:
                    df = fut.result()
                    if df is not None:
                        if sym == 'SPOT':
                            spot_df = df
                        else:
                            options_data[sym] = df
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {sym}: {e}")

        return spot_df, options_data

    def _fetch_historical_data(self, token: int, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        try:
            data = self.order_executor.kite.historical_data(
                instrument_token=token, from_date=start_time, to_date=end_time, interval="minute"
            )
            if data:
                df = pd.DataFrame(data)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                return df
            return None
        except Exception as e:
            logger.error(f"Error fetching historical data for token {token}: {e}")
            return None

    # ---------- (The rest: selection/processing — unchanged from your version) ----------
    # Keep your analyze/selection/process methods exactly as in your latest working script.
    # If you need me to paste the entire long block again here, I can — but no changes were required
    # except calling _is_trading_hours / scheduling above.

    # ---------- Status / Summary / Shutdown ----------
    def get_status(self) -> Dict[str, Any]:
        try:
            active_orders = 0
            try:
                active_orders = len(self.order_executor.get_active_orders())
            except Exception as e:
                logger.debug(f"Could not get active orders: {e}")
            status: Dict[str, Any] = {
                "is_trading": self.is_trading,
                "open_orders": active_orders,
                "trades_today": len(self.trades),
                "live_mode": self.live_mode,
                "cache_age_minutes": (time.time() - self._instruments_cache_timestamp) / 60,
                "total_pnl": self.daily_pnl,
                "last_update": datetime.now().strftime("%H:%M:%S"),
            }
            try:
                status.update(self.risk_manager.get_risk_status())
            except Exception as e:
                logger.debug(f"Could not get risk status: {e}")
            return status
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}

    def get_summary(self) -> str:
        try:
            lines = [
                f"📊 <b>Daily Options Trading Summary</b>",
                f"🔁 <b>Total trades:</b> {len(self.trades)}",
                f"💰 <b>PnL:</b> ₹{self.daily_pnl:.2f}",
                f"📈 <b>Mode:</b> {'🟢 LIVE' if self.live_mode else '🛡️ SHADOW'}",
                "────────────────────",
            ]
            recent = self.trades[-5:] if len(self.trades) > 5 else self.trades
            for t in recent:
                lines.append(
                    f"{t.get('symbol','N/A')} {t.get('direction','?')} {t.get('quantity',0)} "
                    f"@ ₹{t.get('entry_price',0):.2f} (SL ₹{t.get('stop_loss',0):.2f}, TP ₹{t.get('target',0):.2f})"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"📊 Summary Error: {str(e)}"

    def emergency_stop_all(self) -> bool:
        try:
            self.is_trading = False
            cancelled = 0
            try:
                cancelled = self.order_executor.cancel_all_orders()
            except Exception as e:
                logger.error(f"Error cancelling orders: {e}")
            self._safe_send_message(
                f"🚨 EMERGENCY STOP\nOrders Cancelled: {cancelled}\nTrades Today: {len(self.trades)}\nPnL: ₹{self.daily_pnl:.2f}",
                parse_mode="Markdown",
            )
            return True
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
            return False

    def shutdown(self) -> None:
        if not self.is_trading and (not self._polling_thread or not self._polling_thread.is_alive()):
            return
        logger.info("👋 Shutting down RealTimeTrader...")
        self.stop()
        self._stop_polling()
        if self._executor:
            try:
                self._executor.shutdown(wait=True, timeout=5)
            except Exception as e:
                logger.warning(f"Error shutting down executor: {e}")
        logger.info("✅ RealTimeTrader shutdown complete.")

    # ---------- Safe Telegram helpers ----------
    def _safe_send_message(self, message: str, **kwargs) -> None:
        try:
            if self.telegram_controller:
                self.telegram_controller.send_message(message, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to send telegram message: {e}")

    def _safe_send_alert(self, action: str) -> None:
        try:
            if self.telegram_controller:
                self.telegram_controller.send_realtime_session_alert(action)
        except Exception as e:
            logger.warning(f"Failed to send telegram alert: {e}")


# Optional: standalone entry for debugging this module
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("🚀 Starting RealTime Options Trader...")
    trader = RealTimeTrader()

    def _sig(signum, _):
        logger.info(f"Signal {signum}: shutting down...")
        trader.emergency_stop_all()
        trader.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    trader._run_health_check()

    startup_msg = f"""🚀 **REALTIME TRADER STARTED**
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 Mode: {'🟢 LIVE' if trader.live_mode else '🛡️ SHADOW'}
🔄 Status: {'✅ ACTIVE' if trader.is_trading else '⏹️ STANDBY'}
📱 Telegram: ✅ CONNECTED
"""
    trader._safe_send_message(startup_msg, parse_mode="Markdown")

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()