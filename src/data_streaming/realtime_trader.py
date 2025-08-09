# src/data_streaming/realtime_trader.py
"""
Complete Real-Time Options Trading System
A comprehensive automated trading system for the Indian options market.
"""
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

# --- Configuration Import ---
try:
    from src.config import Config
except ImportError:
    logging.warning("Could not import Config, using default configuration.")
    class Config:
        ENABLE_LIVE_TRADING = False
        SPOT_SYMBOL = "NSE:NIFTY 50"
        CONFIDENCE_THRESHOLD = 7.0
        MIN_SIGNAL_CONFIDENCE = 6.0
        STRIKE_SELECTION_TYPE = "OTM"
        OPTIONS_STOP_LOSS_PCT = 20.0
        OPTIONS_TARGET_PCT = 40.0
        MAX_DAILY_OPTIONS_TRADES = 5
        MAX_POSITION_VALUE = 50000
        NIFTY_LOT_SIZE = 50
        DATA_LOOKBACK_MINUTES = 30

# --- Component Imports ---
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import PositionSizing
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController

# --- Utility Imports with Fallbacks ---
try:
    from src.utils.strike_selector import (
        _get_spot_ltp_symbol, 
        get_instrument_tokens, 
        get_next_expiry_date,
        is_trading_hours,
        health_check
    )
except ImportError:
    logging.error("Could not import strike_selector utilities. Using fallbacks.")
    def _get_spot_ltp_symbol(): return getattr(Config, 'SPOT_SYMBOL', 'NSE:NIFTY 50')
    def get_instrument_tokens(*args, **kwargs): return None
    def get_next_expiry_date(*args, **kwargs): return ""
    def health_check(*args, **kwargs): return {"overall_status": "ERROR", "message": "Strike selector not available"}

logger = logging.getLogger(__name__)


class RealTimeTrader:
    """
    Manages the entire real-time trading lifecycle, including data fetching,
    strategy execution, order management, and notifications.
    """
    def __init__(self) -> None:
        """Initializes the trading system components and state."""
        self._start_time = time.time()
        self._session_start_time = datetime.now().isoformat()
        self._last_activity_time = None
        self._error_count = 0
        
        self.is_trading: bool = False
        self.daily_pnl: float = 0.0
        self.trades: List[Dict[str, Any]] = []
        self.positions: Dict[str, Any] = {}
        self.live_mode: bool = getattr(Config, 'ENABLE_LIVE_TRADING', False)
        
        # --- Caching Setup ---
        self._nfo_instruments_cache: Optional[List[Dict]] = None
        self._nse_instruments_cache: Optional[List[Dict]] = None
        self._instruments_cache_timestamp: float = 0
        self._INSTRUMENT_CACHE_DURATION: int = 300  # 5 minutes
        self._cache_lock = threading.Lock()
        
        # --- Core Components ---
        self.strategy = EnhancedScalpingStrategy()
        self.position_sizer = PositionSizing()
        self.order_executor = OrderExecutor()
        
        self.telegram_controller = TelegramController(
            status_callback=self.get_status,
            control_callback=self._handle_control,
            summary_callback=self.get_summary,
        )
        
        # --- Threading and Scheduling ---
        self._executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix='TraderWorker')
        self._polling_thread: Optional[threading.Thread] = None
        self._start_polling()
        self._setup_smart_scheduling()
        
        # --- Graceful Shutdown ---
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, lambda s, f: self.shutdown())
        signal.signal(signal.SIGINT, lambda s, f: self.shutdown())
        
        logger.info("✅ RealTimeTrader initialized. Live Mode: %s", self.live_mode)

    def _start_polling(self) -> None:
        """Starts the Telegram polling in a separate daemon thread."""
        if self.telegram_controller:
            self._polling_thread = threading.Thread(target=self.telegram_controller.start_polling, daemon=True)
            self._polling_thread.start()
            logger.info("📞 Telegram polling thread started.")

    def _setup_smart_scheduling(self) -> None:
        """Sets up intelligent scheduling for the main trading loop."""
        try:
            schedule.every().minute.at(":05").do(self._smart_fetch_and_process)
            logger.info("📅 Smart scheduling setup complete (runs every minute at :05).")
        except Exception as e:
            logger.error(f"Error setting up scheduling: {e}", exc_info=True)

    def _smart_fetch_and_process(self) -> None:
        """Wrapper to run the main trading logic, checking the trading state first."""
        if self.is_trading and is_trading_hours():
            self._executor.submit(self.fetch_and_process_data)
        elif not self.is_trading:
            logger.debug("Trading is stopped. Skipping processing cycle.")
        else:
            logger.debug("Outside trading hours. Skipping processing cycle.")

    def start(self) -> None:
        """Starts the real-time trading process."""
        self.is_trading = True
        logger.info("🚀 Real-time trading started.")
        self.telegram_controller.send_realtime_session_alert("STARTED")

    def stop(self) -> None:
        """Stops the real-time trading process."""
        self.is_trading = False
        logger.info("🛑 Real-time trading stopped.")
        self.telegram_controller.send_realtime_session_alert("STOPPED")

    def shutdown(self) -> None:
        """Gracefully shuts down the trading system."""
        logger.info("🔌 Shutting down RealTimeTrader...")
        self.stop()
        if self._executor:
            self._executor.shutdown(wait=True)
        if self.telegram_controller:
            self.telegram_controller.stop_polling()
        logger.info("✅ Shutdown complete.")

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        """Thread-safe instrument cache refresh."""
        with self._cache_lock:
            if not self.order_executor or not self.order_executor.kite:
                logger.warning("Cannot refresh cache: Kite instance not available.")
                return

            current_time = time.time()
            is_cache_stale = (current_time - self._instruments_cache_timestamp) > self._INSTRUMENT_CACHE_DURATION
            
            if not force and self._nfo_instruments_cache and not is_cache_stale:
                logger.debug("Instrument cache is still valid.")
                return
                
            try:
                logger.info("🔄 Refreshing instruments cache...")
                self._nfo_instruments_cache = self.order_executor.kite.instruments("NFO")
                self._nse_instruments_cache = self.order_executor.kite.instruments("NSE")
                self._instruments_cache_timestamp = current_time
                logger.info("✅ Instruments cache refreshed with %d NFO and %d NSE instruments.",
                            len(self._nfo_instruments_cache), len(self._nse_instruments_cache))
            except Exception as e:
                logger.error(f"Failed to refresh instruments cache: {e}", exc_info=True)
                if self._nfo_instruments_cache is None: self._nfo_instruments_cache = []
                if self._nse_instruments_cache is None: self._nse_instruments_cache = []

    def _get_cached_instruments(self) -> Tuple[List[Dict], List[Dict]]:
        """Safely gets copies of cached instruments."""
        with self._cache_lock:
            return (self._nfo_instruments_cache or []).copy(), (self._nse_instruments_cache or []).copy()

    @lru_cache(maxsize=32)
    def _get_cached_atm_strike(self, spot_price: float) -> int:
        """Calculates and caches the ATM strike."""
        return round(spot_price / 50) * 50

    def _fetch_spot_price(self) -> Optional[float]:
        """Fetches the current spot price of the underlying asset."""
        try:
            spot_symbol = _get_spot_ltp_symbol()
            spot_data = self.order_executor.kite.ltp([spot_symbol])
            return spot_data.get(spot_symbol, {}).get('last_price')
        except Exception as e:
            logger.error(f"Error fetching spot price: {e}")
            return None

    def _fetch_historical_data(self, instrument_token: int, from_time: datetime, to_time: datetime) -> pd.DataFrame:
        """Fetches historical data for a given instrument token."""
        try:
            hist_data = self.order_executor.kite.historical_data(
                instrument_token=instrument_token, from_date=from_time, to_date=to_time, interval="minute"
            )
            df = pd.DataFrame(hist_data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for token {instrument_token}: {e}")
            return pd.DataFrame()

    def _fetch_all_data_parallel(self, spot_token: int, start_time: datetime, end_time: datetime,
                                cached_nfo: List[Dict], cached_nse: List[Dict]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Fetches spot and relevant options data in parallel."""
        spot_df = pd.DataFrame()
        options_data = {}
        
        try:
            instruments_data = get_instrument_tokens(
                symbol=Config.SPOT_SYMBOL, offset=0, kite_instance=self.order_executor.kite,
                cached_nfo_instruments=cached_nfo, cached_nse_instruments=cached_nse
            )
            
            # *** THIS IS THE CORRECTED BLOCK ***
            if not instruments_data:
                logger.error("Failed to get instrument tokens, cannot fetch data.")
                return spot_df, options_data
                
            tokens_to_fetch = {
                'spot': spot_token,
                'ce': instruments_data.get('ce_token'),
                'pe': instruments_data.get('pe_token')
            }
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_name = {
                    executor.submit(self._fetch_historical_data, token, start_time, end_time): name
                    for name, token in tokens_to_fetch.items() if token
                }
                
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    try:
                        df = future.result(timeout=10)
                        if name == 'spot':
                            spot_df = df
                        else:
                            symbol_key = 'ce_symbol' if name == 'ce' else 'pe_symbol'
                            symbol = instruments_data.get(symbol_key)
                            if symbol and not df.empty:
                                options_data[symbol] = df
                    except Exception as e:
                        logger.error(f"Error fetching data for {name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error in parallel data fetching: {e}", exc_info=True)
            
        return spot_df, options_data

    def fetch_and_process_data(self) -> None:
        """Main data fetching and processing workflow."""
        cycle_start_time = time.time()
        logger.debug("🔄 Starting data fetch and process cycle...")
        
        try:
            if not self.order_executor.kite:
                logger.error("KiteConnect instance not found. Cannot proceed.")
                return
                
            self._refresh_instruments_cache()
            cached_nfo, cached_nse = self._get_cached_instruments()
            
            if not cached_nfo:
                logger.error("Instrument cache is empty. Cannot proceed.")
                return
                
            spot_price = self._fetch_spot_price()
            if not spot_price:
                logger.error("Failed to fetch spot price. Aborting cycle.")
                return
            
            spot_token_info = get_instrument_tokens(Config.SPOT_SYMBOL, 0, self.order_executor.kite, cached_nfo, cached_nse)
            spot_token = spot_token_info.get('spot_token') if spot_token_info else None

            if not spot_token:
                logger.error("Failed to resolve spot instrument token. Aborting cycle.")
                return

            logger.info(f"Spot Price: {spot_price:.2f}, ATM Strike: {self._get_cached_atm_strike(spot_price)}")
            
            end_time = datetime.now()
            start_time_data = end_time - timedelta(minutes=getattr(Config, 'DATA_LOOKBACK_MINUTES', 30))
            
            spot_df, options_data = self._fetch_all_data_parallel(
                spot_token, start_time_data, end_time, cached_nfo, cached_nse
            )
            
            if not options_data:
                logger.warning("No options data fetched. Using fallback.")
                selected_strikes_info = self._get_fallback_strikes(self._get_cached_atm_strike(spot_price), cached_nfo)
            else:
                selected_strikes_info = self._analyze_options_data_optimized(spot_price, options_data)
                
            if not selected_strikes_info:
                logger.warning("No strikes selected for processing.")
                return
                
            self._process_selected_strikes(selected_strikes_info, options_data, spot_df)
            
            logger.debug("Cycle completed in %.2fs", time.time() - cycle_start_time)
            
        except Exception as e:
            logger.error(f"Critical error in fetch_and_process_data: {e}", exc_info=True)

    def _analyze_options_data_optimized(self, spot_price: float, options_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Analyzes fetched options data to select the best strikes for trading."""
        if not options_data:
            return []
            
        atm_strike = self._get_cached_atm_strike(spot_price)
        
        def analyze_option_type(opt_type: str) -> List[Dict[str, Any]]:
            type_strikes = []
            relevant_data = {k: v for k, v in options_data.items() if opt_type in k and not v.empty}
            
            if not relevant_data:
                return type_strikes
                
            for symbol, df in relevant_data.items():
                try:
                    strike_str = ''.join(filter(str.isdigit, symbol.split(opt_type)[0][-5:]))
                    strike = int(strike_str)
                    
                    last_row = df.iloc[-1]
                    oi_change = last_row.get('oi', 0) - (df.iloc[-2].get('oi', 0) if len(df) > 1 else 0)
                    
                    type_strikes.append({
                        'symbol': symbol, 'strike': strike, 'type': opt_type,
                        'ltp': last_row.get('last_price', 0), 'oi': last_row.get('oi', 0),
                        'oi_change': oi_change, 'is_atm': strike == atm_strike
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse strike from symbol '{symbol}': {e}")
                    continue
                    
            return self._select_best_strikes(type_strikes, opt_type)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(analyze_option_type, 'CE'), executor.submit(analyze_option_type, 'PE')]
            results = [future.result() for future in as_completed(futures)]
        
        return [item for sublist in results for item in sublist]

    def _select_best_strikes(self, strikes: List[Dict[str, Any]], opt_type: str) -> List[Dict[str, Any]]:
        """Selects the best strike based on strategy rules (e.g., OI change)."""
        if not strikes:
            return []
        # Example: select the one with the highest OI change
        strikes.sort(key=lambda x: x['oi_change'], reverse=True)
        return strikes[:1] # Return the top candidate

    def _get_fallback_strikes(self, atm_strike: int, cached_nfo: List[Dict]) -> List[Dict[str, Any]]:
        """Provides fallback strike information if primary data fetching fails."""
        logger.warning("Using fallback strike selection.")
        # This is a placeholder. Implement logic to find CE/PE symbols for the ATM strike from the cache.
        return []

    def _process_selected_strikes(self, strikes_info: List[Dict], options_data: Dict[str, pd.DataFrame], spot_df: pd.DataFrame) -> None:
        """Processes each selected strike, generates signals, and executes trades."""
        for strike_info in strikes_info:
            symbol = strike_info['symbol']
            option_df = options_data.get(symbol)
            
            if option_df is None or option_df.empty:
                logger.warning(f"No data available for selected strike {symbol}. Skipping.")
                continue
            
            # Here you would call your strategy to generate a signal
            # signal = self.strategy.generate_signal(spot_df, option_df, strike_info)
            # if signal:
            #     self.execute_trade(signal)
            logger.info(f"Processing strike: {symbol}")

    # --- Placeholder methods for callbacks and further implementation ---
    def get_status(self) -> str:
        """Returns the current status of the bot."""
        # Implement status logic
        return f"Status: {'Trading' if self.is_trading else 'Stopped'}, PnL: {self.daily_pnl}"

    def _handle_control(self, command: str) -> str:
        """Handles control commands from Telegram."""
        if command == 'start':
            self.start()
            return "Trading started."
        elif command == 'stop':
            self.stop()
            return "Trading stopped."
        return "Unknown command."

    def get_summary(self) -> str:
        """Returns a summary of the day's trading activity."""
        # Implement summary logic
        return f"Trades: {len(self.trades)}, PnL: {self.daily_pnl}"

