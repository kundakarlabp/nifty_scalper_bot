# src/data_streaming/realtime_trader.py
"""
Complete Real-Time Options Trading System
A comprehensive automated trading system for Indian options market
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
import time # Import for caching
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

# Safe imports with fallbacks
try:
    from src.config import Config
except ImportError:
    logging.warning("Could not import Config, using default configuration")
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

# Assuming you'll have an OptionsStrategy or modify the existing one
# from src.strategies.options_strategy import OptionsStrategy
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import PositionSizing
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController

# Import helper for consistent spot LTP symbol usage and strike selector
# Handle potential import issues gracefully
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
    def _get_spot_ltp_symbol():
        return getattr(Config, 'SPOT_SYMBOL', 'NSE:NIFTY 50')
    def get_instrument_tokens(*args, **kwargs):
        return None
    def get_next_expiry_date(*args, **kwargs):
        return ""
    def health_check(*args, **kwargs):
        return {"overall_status": "ERROR", "message": "Strike selector not available"}

logger = logging.getLogger(__name__)


class RealTimeTrader:
    def __init__(self) -> None:
        """Enhanced initialization with additional tracking"""
        # Store start time for uptime calculation
        self._start_time = time.time()
        self._session_start_time = datetime.now().isoformat()
        self._last_activity_time = None
        self._error_count = 0
        self._last_spot_prices = [] # For trend tracking
        
        # Initialize all the original attributes
        self.is_trading: bool = False
        self.daily_pnl: float = 0.0
        self.trades: List[Dict[str, Any]] = []
        self.positions: Dict[str, Any] = {}
        self.live_mode: bool = getattr(Config, 'ENABLE_LIVE_TRADING', False)
        
        # Cache for instruments to reduce API calls
        self._nfo_instruments_cache: Optional[List[Dict]] = None
        self._nse_instruments_cache: Optional[List[Dict]] = None
        self._instruments_cache_timestamp: float = 0
        self._INSTRUMENT_CACHE_DURATION: int = 300  # 5 minutes
        self._cache_lock = threading.Lock()
        
        # Initialize components
        self.strategy = EnhancedScalpingStrategy()
        self.position_sizer = PositionSizing()
        self.order_executor = OrderExecutor()
        
        # Initialize TelegramController with proper callbacks
        # This ensures Telegram polling starts correctly
        self.telegram_controller = TelegramController(
            status_callback=self.get_status,
            control_callback=self._handle_control,
            summary_callback=self.get_summary,
        )
        self._polling_thread: Optional[threading.Thread] = None
        # Start Telegram polling in a daemon thread
        self._start_polling()
        
        # Setup threading
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize Kite connection
        self._set_live_mode()
        
        # Setup scheduling
        self._setup_smart_scheduling()
        
        # Register shutdown handlers
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, lambda signum, frame: self.shutdown())
        signal.signal(signal.SIGINT, lambda signum, frame: self.shutdown())
        
        logger.info("✅ Enhanced RealTimeTrader initialized and ready")
        logger.info(f"🔧 Features: Live={self.live_mode}, Cache=Enabled, Threads={self._executor._max_workers}")

    def _set_live_mode(self) -> None:
        """Initialize components based on live mode setting"""
        try:
            if self.live_mode:
                logger.info("🟢 Live trading mode enabled")
                # Components are already initialized for live mode
            else:
                logger.info("🛡️ Shadow/paper trading mode enabled")
                # Reinitialize order executor for simulation
                # This would depend on your simulation implementation
        except Exception as e:
            logger.error(f"Error setting trading mode: {e}")

    def _setup_smart_scheduling(self) -> None:
        """Setup intelligent scheduling for data fetching"""
        try:
            # Schedule the smart fetch and process every minute
            schedule.every().minute.at(":00").do(self._smart_fetch_and_process)
            logger.info("📅 Smart scheduling setup complete")
        except Exception as e:
            logger.error(f"Error setting up scheduling: {e}")

    def _smart_fetch_and_process(self) -> None:
        """Smart fetch and process with trading state check"""
        if self.is_trading:
            self.fetch_and_process_data()

    def start(self) -> None:
        """Start the real-time trading process"""
        try:
            self.is_trading = True
            logger.info("🚀 Real-time trading started")
            self.telegram_controller.send_realtime_session_alert("STARTED")
        except Exception as e:
            logger.error(f"Error starting trading: {e}")

    def stop(self) -> None:
        """Stop the real-time trading process"""
        try:
            self.is_trading = False
            logger.info("🛑 Real-time trading stopped")
            self.telegram_controller.send_realtime_session_alert("STOPPED")
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        """Thread-safe instrument cache refresh with enhanced error handling"""
        with self._cache_lock:
            if not self.order_executor or not self.order_executor.kite:
                logger.warning("[_refresh_instruments_cache] Cannot refresh cache, Kite instance not available.")
                self._nfo_instruments_cache = []
                self._nse_instruments_cache = []
                return
                
            current_time = time.time()
            needs_refresh = (
                force or
                self._nfo_instruments_cache is None or
                self._nse_instruments_cache is None or
                (current_time - self._instruments_cache_timestamp) > self._INSTRUMENT_CACHE_DURATION
            )
            
            if not needs_refresh:
                logger.debug("[_refresh_instruments_cache] Cache is still valid")
                return
                
            try:
                logger.info("[_refresh_instruments_cache] 🔄 Refreshing instruments cache...")
                self._nfo_instruments_cache = self.order_executor.kite.instruments("NFO")
                logger.debug(f"[_refresh_instruments_cache] Cached {len(self._nfo_instruments_cache)} NFO instruments.")
                self._nse_instruments_cache = self.order_executor.kite.instruments("NSE")
                logger.debug(f"[_refresh_instruments_cache] Cached {len(self._nse_instruments_cache)} NSE instruments.")
                self._instruments_cache_timestamp = current_time
                logger.info("[_refresh_instruments_cache] ✅ Instruments cache refreshed.")
            except Exception as e:
                logger.error(f"[_refresh_instruments_cache] Failed to refresh instruments cache: {e}")
                # Keep old cache if it exists, otherwise use empty list
                if self._nfo_instruments_cache is None:
                    self._nfo_instruments_cache = []
                if self._nse_instruments_cache is None:
                    self._nse_instruments_cache = []

    def _get_cached_instruments(self) -> Tuple[List[Dict], List[Dict]]:
        """Get cached instruments safely"""
        with self._cache_lock:
            # Return copies to prevent external modification
            return (self._nfo_instruments_cache or []).copy(), (self._nse_instruments_cache or []).copy()

    @lru_cache(maxsize=32)
    def _get_cached_atm_strike(self, spot_price: float, timestamp_bucket: int) -> int:
        """Get cached ATM strike calculation"""
        return round(spot_price / 50) * 50

    def _fetch_spot_price(self) -> Optional[float]:
        """Fetch current spot price"""
        try:
            spot_symbol = _get_spot_ltp_symbol()
            spot_data = self.order_executor.kite.ltp([spot_symbol])
            return spot_data.get(spot_symbol, {}).get('last_price')
        except Exception as e:
            logger.error(f"[RT] Error fetching spot price: {e}")
            return None

    def _fetch_historical_data(self, instrument_token: int, from_time: datetime, to_time: datetime) -> pd.DataFrame:
        """Fetch historical data for an instrument"""
        try:
            # Fetch historical data
            hist_data = self.order_executor.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_time,
                to_date=to_time,
                interval="minute"
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(hist_data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
            return df
        except Exception as e:
            logger.error(f"[RT] Error fetching historical data for {instrument_token}: {e}")
            return pd.DataFrame()

    def _fetch_all_data_parallel(self, spot_token: int, atm_strike: int, start_time: datetime, end_time: datetime,
                                cached_nfo: List[Dict], cached_nse: List[Dict]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Fetch all required data in parallel"""
        spot_df = pd.DataFrame()
        options_data = {}
        
        try:
            # --- CRITICAL CHANGE: Pass cached data to get_instrument_tokens ---
            instruments_data = get_instrument_tokens(
                symbol=Config.SPOT_SYMBOL,
                offset=0, # ATM
                kite_instance=self.order_executor.kite,
                cached_nfo_instruments=cached_nfo, # Pass cached data
                cached_nse_instruments=cached_nse  # Pass cached data
            )
            
            if not instruments_
                logger.error("Failed to get instrument tokens")
                return spot_df, options_data
                
            # Prepare tokens to fetch
            tokens_to_fetch = {
                'spot': spot_token,
                'ce': instruments_data.get('ce_token'),
                'pe': instruments_data.get('pe_token')
            }
            
            # Fetch data in parallel
            futures = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                for name, token in tokens_to_fetch.items():
                    if token:
                        futures[name] = executor.submit(
                            self._fetch_historical_data, 
                            token, 
                            start_time, 
                            end_time
                        )
                
                # Collect results
                for name, future in futures.items():
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
                        logger.error(f"Error fetching {name} data: {e}")
                        
        except Exception as e:
            logger.error(f"Error in parallel data fetching: {e}")
            
        return spot_df, options_data

    def fetch_and_process_data(self) -> None:
        """Main data fetching and processing workflow
        
        Optimized data fetching with parallel processing and better error handling
        """
        if not self.is_trading:
            logger.debug("fetch_and_process_ Trading not active, skipping.")
            return
            
        start_time = time.time()
        logger.debug("🔄 Starting data fetch and process cycle...")
        
        try:
            if not hasattr(self.order_executor, 'kite') or not self.order_executor.kite:
                logger.error("KiteConnect instance not found. Is live mode enabled?")
                return
                
            # Refresh instruments cache with timeout
            self._refresh_instruments_cache()
            
            # --- CRITICAL CHANGE: Get cached instruments ---
            cached_nfo, cached_nse = self._get_cached_instruments()
            
            if not cached_nfo and not cached_nse:
                logger.error("Instrument cache is empty. Cannot proceed.")
                return
                
            # Parallel spot price and instrument token fetching
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit spot price fetch
                spot_future = executor.submit(self._fetch_spot_price)
                
                # --- CRITICAL CHANGE: Submit instrument token fetch with cached data ---
                instruments_future = executor.submit(
                    get_instrument_tokens,
                    Config.SPOT_SYMBOL,
                    0, # ATM offset
                    self.order_executor.kite,
                    cached_nfo, # Pass cached data
                    cached_nse  # Pass cached data
                )
                
                # Get results with timeout
                spot_price = spot_future.result(timeout=5)
                instruments_data = instruments_future.result(timeout=5)
                
                if not spot_price or not instruments_
                    logger.error("Failed to fetch essential data (spot price or instruments)")
                    return
                    
            logger.info(f"[RT] Successfully fetched current spot price: {spot_price}")
            
            atm_strike = instruments_data['atm_strike']
            expiry = instruments_data['expiry']
            spot_token = instruments_data.get('spot_token')
            
            logger.info(f"ATM Strike: {atm_strike}, Expiry: {expiry}, Spot: {spot_price}")
            
            # Optimized timeframe calculation
            end_time = datetime.now()
            lookback_minutes = getattr(Config, 'DATA_LOOKBACK_MINUTES', 30)
            start_time_data = end_time - timedelta(minutes=lookback_minutes)
            
            # Parallel data fetching for spot and options
            spot_df, options_data = self._fetch_all_data_parallel(
                spot_token, atm_strike, start_time_data, end_time, cached_nfo, cached_nse # Pass cached data
            )
            
            # Optimized analysis
            if options_
                selected_strikes_info = self._analyze_options_data_optimized(spot_price, options_data)
            else:
                selected_strikes_info = self._get_fallback_strikes(atm_strike, cached_nfo, cached_nse)
                
            if not selected_strikes_info:
                logger.warning("No strikes selected for processing")
                return
                
            # Process selected strikes
            self._process_selected_strikes(selected_strikes_info, options_data, spot_df)
            
            processing_time = time.time() - start_time
            logger.debug(f"Data fetch and process completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in fetch_and_process_data: {e}", exc_info=True)

    def _analyze_options_data_optimized(self, spot_price: float, options_ Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Optimized options data analysis for strike selection"""
        if not options_
            return []
            
        # Use cached ATM calculation
        timestamp_bucket = int(time.time() // 30)  # 30-second buckets
        atm_strike = self._get_cached_atm_strike(spot_price, timestamp_bucket)
        
        selected_strikes = []
        
        # Process CE and PE in parallel
        def analyze_option_type(opt_type: str) -> List[Dict[str, Any]]:
            type_strikes = []
            relevant_data = {k: v for k, v in options_data.items() if opt_type in k and not v.empty}
            
            if not relevant_
                return type_strikes
                
            for symbol, df in relevant_data.items():
                try:
                    # Optimized strike extraction
                    opt_index = symbol.find(opt_type)
                    if opt_index == -1:
                        continue
                        
                    strike_str = symbol[opt_index-5:opt_index]
                    strike = int(strike_str)
                    
                    # Quick data quality check
                    if len(df) < 1:
                        continue
                        
                    last_row = df.iloc[-1]
                    prev_row = df.iloc[-2] if len(df) > 1 else last_row
                    
                    # Optimized calculations
                    oi_change = last_row.get('oi', 0) - prev_row.get('oi', 0)
                    delta_approx = self._calculate_delta_approximation(opt_type, strike, atm_strike, spot_price)
                    
                    type_strikes.append({
                        'symbol': symbol,
                        'strike': strike,
                        'type': opt_type,
                        'ltp': last_row.get('last_price', 0),
                        'oi': last_row.get('oi', 0),
                        'oi_change': oi_change,
                        'delta': delta_approx,
                        'is_atm': strike == atm_strike,
                        'is_otm': (strike > atm_strike) if opt_type == 'CE' else (strike < atm_strike),
                        'is_itm': (strike < atm_strike) if opt_type == 'CE' else (strike > atm_strike)
                    })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not extract strike from symbol {symbol}: {e}")
                    continue
                    
            # Sort by OI change and select best candidates
            type_strikes.sort(key=lambda x: x['oi_change'], reverse=True)
            return self._select_best_strikes(type_strikes, opt_type)
        
        # Process both types in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            ce_future = executor.submit(analyze_option_type, 'CE')
          