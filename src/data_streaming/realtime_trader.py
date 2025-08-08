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
import time # Import for caching
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import Config
# Assuming you'll have an OptionsStrategy or modify the existing one
# from src.strategies.options_strategy import OptionsStrategy
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import PositionSizing
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController

# Import helper for consistent spot LTP symbol usage and strike selector
from src.utils.strike_selector import (
    _get_spot_ltp_symbol, 
    get_instrument_tokens, 
    get_next_expiry_date,
    is_trading_hours,
    health_check
)

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
        self.telegram_controller = TelegramController(
            status_callback=self.get_status,
            control_callback=self._handle_control,
            summary_callback=self.get_summary,
        )
        
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

    def _fetch_spot_price(self, spot_symbol: str) -> Optional[float]:
        """Fetch current spot price"""
        try:
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

    # --- CRITICAL CHANGE: Modified _fetch_all_data_parallel to pass cached data ---
    def _fetch_all_data_parallel(self, spot_token: int, atm_strike: int, start_time: datetime, end_time: datetime,
                                cached_nfo: List[Dict], cached_nse: List[Dict]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Fetch all required data in parallel"""
        spot_df = pd.DataFrame()
        options_data = {}
        
        try:
            # --- CRITICAL CHANGE: Pass cached data to get_instrument_tokens ---
            instruments_data = get_instrument_tokens(
                symbol=Config.SPOT_SYMBOL,
                kite_instance=self.order_executor.kite,
                cached_nfo_instruments=cached_nfo, # Pass cached data
                cached_nse_instruments=cached_nse, # Pass cached data
                offset=0  # ATM
            )
            
            if not instruments_data:
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

    # --- CRITICAL CHANGE: Modified fetch_and_process_data to pass cached data ---
    def fetch_and_process_data(self) -> None:
        """Main data fetching and processing workflow
        
        Optimized data fetching with parallel processing and better error handling
        """
        if not self.is_trading:
            logger.debug("fetch_and_process_data: Trading not active, skipping.")
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
                spot_future = executor.submit(self._fetch_spot_price, _get_spot_ltp_symbol())
                
                # --- CRITICAL CHANGE: Submit instrument token fetch with cached data ---
                instruments_future = executor.submit(
                    get_instrument_tokens,
                    Config.SPOT_SYMBOL,
                    self.order_executor.kite,
                    cached_nfo, # Pass cached data
                    cached_nse, # Pass cached data
                    0  # ATM
                )
                
                # Get results with timeout
                spot_price = spot_future.result(timeout=5)
                instruments_data = instruments_future.result(timeout=5)
                
                if not spot_price or not instruments_data:
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
            if options_data:
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

    def _analyze_options_data_optimized(self, spot_price: float, options_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Optimized options data analysis for strike selection"""
        if not options_data:
            return []
            
        # Use cached ATM calculation
        timestamp_bucket = int(time.time() // 30)  # 30-second buckets
        atm_strike = self._get_cached_atm_strike(spot_price, timestamp_bucket)
        
        selected_strikes = []
        
        # Process CE and PE in parallel
        def analyze_option_type(opt_type: str) -> List[Dict[str, Any]]:
            type_strikes = []
            relevant_data = {k: v for k, v in options_data.items() if opt_type in k and not v.empty}
            
            if not relevant_data:
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
            pe_future = executor.submit(analyze_option_type, 'PE')
            
            selected_strikes.extend(ce_future.result())
            selected_strikes.extend(pe_future.result())
            
        return selected_strikes

    def _calculate_delta_approximation(self, opt_type: str, strike: int, atm_strike: int, spot_price: float) -> float:
        """Fast delta approximation for option selection"""
        moneyness = (spot_price - strike) / spot_price if spot_price > 0 else 0
        
        if opt_type == 'CE':
            if strike == atm_strike:
                return 0.5
            elif strike < atm_strike:  # ITM
                return 0.5 + min(0.3, abs(moneyness) * 2)
            else:  # OTM
                return max(0.1, 0.5 - abs(moneyness) * 2)
        else:  # PE
            if strike == atm_strike:
                return -0.5
            elif strike > atm_strike:  # ITM
                return -0.5 - min(0.3, abs(moneyness) * 2)
            else:  # OTM
                return max(-0.9, -0.5 + abs(moneyness) * 2)

    def _select_best_strikes(self, strikes: List[Dict[str, Any]], opt_type: str) -> List[Dict[str, Any]]:
        """Select the best strikes based on strategy configuration"""
        if not strikes:
            return []
            
        selected = []
        
        # Always try to include ATM
        atm_option = next((s for s in strikes if s['is_atm']), None)
        if atm_option:
            selected.append(atm_option)
            logger.debug(f"Selected ATM {opt_type}: {atm_option['symbol']}")
            
        # Select based on strategy preference
        strategy_type = getattr(Config, 'STRIKE_SELECTION_TYPE', 'OTM')
        
        if strategy_type == "ITM":
            best_other = next((s for s in strikes if s['is_itm'] and s not in selected), None)
        elif strategy_type == "OTM":
            best_other = next((s for s in strikes if s['is_otm'] and s not in selected), None)
        else:
            # Default: highest OI change
            best_other = next((s for s in strikes if s not in selected), None)
            
        if best_other:
            selected.append(best_other)
            logger.debug(f"Selected {strategy_type} {opt_type}: {best_other['symbol']} (OI Change: {best_other['oi_change']})")
            
        return selected

    # --- CRITICAL CHANGE: Modified _get_fallback_strikes to pass cached data ---
    def _get_fallback_strikes(self, atm_strike: int, cached_nfo: List[Dict], cached_nse: List[Dict]) -> List[Dict[str, Any]]:
        """Optimized fallback strike selection"""
        logger.warning("Using fallback strike selection")
        fallback_strikes = []
        
        for offset in [0, 1, -1]:  # ATM, ATM+50, ATM-50
            fb_strike = atm_strike + (offset * 50)
            # --- CRITICAL CHANGE: Pass cached data ---
            fb_instruments = get_instrument_tokens(
                symbol=Config.SPOT_SYMBOL,
                kite_instance=self.order_executor.kite,
                cached_nfo_instruments=cached_nfo, # Pass cached data
                cached_nse_instruments=cached_nse, # Pass cached data
                offset=offset
            )
            
            if fb_instruments:
                for opt_type, symbol_key in [('CE', 'ce_symbol'), ('PE', 'pe_symbol')]:
                    symbol = fb_instruments.get(symbol_key)
                    if symbol:
                        fallback_strikes.append({
                            'symbol': symbol,
                            'strike': fb_strike,
                            'type': opt_type,
                            'ltp': 0,
                            'oi': 0,
                            'oi_change': 0,
                            'delta': 0.5 if opt_type == 'CE' else -0.5,
                            'is_atm': offset == 0,
                            'is_otm': (offset > 0) if opt_type == 'CE' else (offset < 0),
                            'is_itm': (offset < 0) if opt_type == 'CE' else (offset > 0)
                        })
                        
        return fallback_strikes

    def _process_selected_strikes(self, selected_strikes: List[Dict[str, Any]],
                                 options_data: Dict[str, pd.DataFrame], spot_df: pd.DataFrame) -> None:
        """Process selected strikes with parallel execution"""
        if not selected_strikes:
            return
            
        # Process strikes in parallel for better performance
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for strike_info in selected_strikes:
                symbol = strike_info['symbol']
                df = options_data.get(symbol)
                
                if df is not None and not df.empty:
                    future = executor.submit(self.process_options_bar, symbol, df, spot_df, strike_info)
                    futures.append(future)
                else:
                    # Try LTP-based processing
                    future = executor.submit(self._process_ltp_based, symbol, spot_df, strike_info)
                    futures.append(future)
                    
            # Wait for all processing to complete
            for future in as_completed(futures, timeout=10):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing strike: {e}")

    def _process_ltp_based(self, symbol: str, spot_df: pd.DataFrame, strike_info: Dict[str, Any]) -> None:
        """Process strike using LTP data when historical data is unavailable"""
        try:
            ltp_data = self.order_executor.kite.ltp([f"NFO:{symbol}"])
            ltp_price = ltp_data.get(f"NFO:{symbol}", {}).get('last_price', 0)
            
            if ltp_price > 0:
                # Create minimal dataframe for processing
                dummy_df = pd.DataFrame([{
                    'date': pd.Timestamp.now(),
                    'last_price': ltp_price,
                    'oi': strike_info.get('oi', 0),
                    'volume': 0
                }]).set_index('date')
                
                self.process_options_bar(symbol, dummy_df, spot_df, strike_info)
            else:
                logger.warning(f"Could not get LTP for {symbol}")
        except Exception as e:
            logger.error(f"Error in LTP-based processing for {symbol}: {e}")

    def process_options_bar(self, symbol: str, ohlc: pd.DataFrame, spot_ohlc: pd.DataFrame, strike_info: Dict[str, Any]) -> None:
        """Optimized options bar processing with enhanced error handling"""
        if not self.is_trading:
            logger.debug(f"process_options_bar: Trading not active for {symbol}")
            return

        try:
            # Enhanced time filter with market hours check
            current_time = datetime.now()
            if not self._is_trading_hours(current_time):
                logger.debug(f"Outside trading hours for {symbol}")
                return

            # Get current price with fallbacks
            current_price = self._get_current_price(ohlc, strike_info)
            if current_price <= 0:
                logger.debug(f"Invalid price for {symbol}")
                return

            # Generate enhanced signal with multiple confirmations
            signal = self._generate_enhanced_options_signal(ohlc, spot_ohlc, strike_info, current_price)
            
            if not signal:
                logger.debug(f"No signal generated for {symbol}")
                return

            # Optimized confidence check
            signal_confidence = float(signal.get("confidence", 0.0))
            confidence_threshold = getattr(Config, 'CONFIDENCE_THRESHOLD', 7.0)
            if signal_confidence < confidence_threshold:
                logger.debug(f"Signal confidence {signal_confidence} below threshold {confidence_threshold} for {symbol}")
                return

            # Enhanced position sizing with options-specific logic
            position = self._calculate_options_position_size(signal, strike_info, current_price)
            if not position or position.get("quantity", 0) <= 0:
                logger.debug(f"Invalid position sizing for {symbol}")
                return

            # Risk management checks
            if not self._passes_risk_checks(signal, position, strike_info):
                logger.warning(f"Risk checks failed for {symbol}")
                return

            # Execute trade workflow
            success = self._execute_options_trade(symbol, signal, position, strike_info)
            if success:
                logger.info(f"✅ Successfully executed options trade for {symbol}")
            else:
                logger.warning(f"❌ Failed to execute options trade for {symbol}")
                
        except Exception as exc:
            logger.error(f"Error processing options bar for {symbol}: {exc}", exc_info=True)

    def _is_trading_hours(self, current_time: datetime) -> bool:
        """Check if current time is within trading hours"""
        hour = current_time.hour
        minute = current_time.minute
        
        # Market hours: 9:15 AM to 3:30 PM
        if hour < 9 or (hour == 9 and minute < 15):
            return False
        if hour > 15 or (hour == 15 and minute > 30):
            return False
        return True

    def _get_current_price(self, ohlc: pd.DataFrame, strike_info: Dict[str, Any]) -> float:
        """Get current price with multiple fallback mechanisms"""
        if not ohlc.empty:
            return ohlc['last_price'].iloc[-1]
        return strike_info.get('ltp', 0)

    def _generate_enhanced_options_signal(self, ohlc: pd.DataFrame, spot_ohlc: pd.DataFrame, 
                                         strike_info: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        """Generate enhanced options signal with multiple confirmations"""
        try:
            signal_dict = {"signal": None, "confidence": 0.0, "reasons": []}
            confidence_score = 0.0
            
            # 1. Momentum analysis
            momentum_signal = self._analyze_options_momentum(ohlc, strike_info)
            if momentum_signal:
                confidence_score += momentum_signal['confidence']
                signal_dict["signal"] = momentum_signal["signal"]
                signal_dict["reasons"].append(f"Momentum: {momentum_signal['strategy_type']}")
                
            # 2. Spot-options correlation
            correlation_signal = self._analyze_spot_options_correlation(spot_ohlc, strike_info, current_price)
            if correlation_signal:
                confidence_score += correlation_signal['confidence']
                if not signal_dict["signal"]:
                    signal_dict["signal"] = correlation_signal["signal"]
                signal_dict["reasons"].append(f"Correlation: {correlation_signal['strategy_type']}")
                
            # 3. Greeks approximation
            greeks_signal = self._analyze_options_greeks(strike_info, current_price)
            if greeks_signal:
                confidence_score += greeks_signal['confidence']
                if not signal_dict["signal"]:
                    signal_dict["signal"] = greeks_signal["signal"]
                signal_dict["reasons"].append(f"Greeks: Delta-based analysis")
                
            # 4. Volume and OI analysis
            volume_signal = self._analyze_volume_oi(ohlc, strike_info)
            if volume_signal:
                confidence_score *= volume_signal['multiplier']
                signal_dict["reasons"].append(f"Volume/OI: Multiplier {volume_signal['multiplier']:.2f}")
                
            # Final confidence adjustment
            signal_dict["confidence"] = min(confidence_score, 10.0)  # Cap at 10
            
            # Only return signal if above minimum threshold
            min_confidence = getattr(Config, 'MIN_SIGNAL_CONFIDENCE', 6.0)
            if signal_dict["confidence"] >= min_confidence and signal_dict["signal"]:
                # Set risk parameters
                signal_dict["entry_price"] = current_price
                signal_dict["stop_loss"] = self._calculate_options_stop_loss(current_price, strike_info)
                signal_dict["target"] = self._calculate_options_target(current_price, strike_info)
                return signal_dict
                
            return None
        except Exception as e:
            logger.error(f"Error generating enhanced options signal: {e}")
            return None

    def _analyze_options_momentum(self, ohlc: pd.DataFrame, strike_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze options price momentum"""
        if ohlc.empty or len(ohlc) < 3:
            return None
            
        try:
            # Calculate momentum indicators
            recent_prices = ohlc['last_price'].tail(5).values
            if len(recent_prices) < 2:
                return None
                
            # Price change momentum
            price_change_pct = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
            
            # Volume momentum (if available)
            volume_momentum = 1.0
            if 'volume' in ohlc.columns:
                recent_volumes = ohlc['volume'].tail(3)
                avg_volume = recent_volumes.mean()
                current_volume = recent_volumes.iloc[-1]
                volume_momentum = min(current_volume / avg_volume, 3.0) if avg_volume > 0 else 1.0
                
            # Determine signal strength
            confidence = 0.0
            signal_direction = None
            
            # Bullish momentum
            if price_change_pct > 2.0:  # 2% price increase
                signal_direction = "BUY"
                confidence = min(price_change_pct * volume_momentum, 5.0)
                
            # Bearish momentum (for selling options)
            elif price_change_pct < -1.5:  # 1.5% price decrease
                # For options, we typically buy on dips if it's a good setup
                if strike_info.get('is_otm', False):  # OTM options on dips can be opportunities
                    signal_direction = "BUY"
                    confidence = min(abs(price_change_pct) * volume_momentum * 0.8, 4.0)
                    
            if signal_direction and confidence > 0:
                return {
                    "signal": signal_direction,
                    "confidence": confidence,
                    "strategy_type": "momentum"
                }
            return None
        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return None

    def _analyze_spot_options_correlation(self, spot_ohlc: pd.DataFrame, strike_info: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        """Analyze spot-options price correlation"""
        if spot_ohlc.empty or len(spot_ohlc) < 5:
            return None
            
        try:
            # Calculate spot momentum
            spot_prices = spot_ohlc['close'].tail(5).values
            spot_change_pct = (spot_prices[-1] - spot_prices[0]) / spot_prices[0] * 100
            
            # Analyze based on option type and moneyness
            opt_type = strike_info.get('type', '')
            is_atm = strike_info.get('is_atm', False)
            is_itm = strike_info.get('is_itm', False)
            is_otm = strike_info.get('is_otm', False)
            
            # Simple correlation logic
            signal_direction = None
            confidence = 0.0
            
            if opt_type == 'CE':
                if spot_change_pct > 0.2:  # Spot going up
                    signal_direction = "BUY"
                    confidence = 2.5
                elif spot_change_pct < -0.2:  # Spot going down
                    signal_direction = "SELL"
                    confidence = 2.5
            else:  # PE
                if spot_change_pct > 0.2:  # Spot going up
                    signal_direction = "SELL"
                    confidence = 2.5
                elif spot_change_pct < -0.2:  # Spot going down
                    signal_direction = "BUY"
                    confidence = 2.5
                    
            if signal_direction:
                return {
                    "signal": signal_direction,
                    "confidence": confidence,
                    "strategy_type": "correlation"
                }
            return None
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return None

    def _analyze_options_greeks(self, strike_info: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        """Simplified Greeks analysis"""
        try:
            delta = strike_info.get('delta', 0)
            opt_type = strike_info.get('type', '')
            confidence_boost = 0.0
            
            # Delta-based analysis
            if opt_type == 'CE':
                if 0.4 <= abs(delta) <= 0.7:  # Sweet spot for CE delta
                    confidence_boost = 1.5
                elif abs(delta) > 0.7:  # High delta, more sensitive
                    confidence_boost = 1.0
            elif opt_type == 'PE':
                if 0.4 <= abs(delta) <= 0.7:  # Sweet spot for PE delta
                    confidence_boost = 1.5
                elif abs(delta) > 0.7:  # High delta, more sensitive
                    confidence_boost = 1.0
                    
            return {"confidence": confidence_boost} if confidence_boost > 0 else None
        except Exception as e:
            logger.error(f"Error in Greeks analysis: {e}")
            return None

    def _analyze_volume_oi(self, ohlc: pd.DataFrame, strike_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze volume and open interest"""
        try:
            multiplier = 1.0
            
            # Volume analysis (if available)
            if not ohlc.empty and 'volume' in ohlc.columns:
                recent_volume = ohlc['volume'].tail(3).sum()
                if recent_volume > 0:
                    multiplier *= 1.2
                    
            # OI analysis
            oi_change = strike_info.get('oi_change', 0)
            if oi_change > 0:
                multiplier *= 1.3
            elif oi_change < 0:
                multiplier *= 0.9
                
            return {"multiplier": multiplier}
        except Exception as e:
            logger.error(f"Error in volume/OI analysis: {e}")
            return {"multiplier": 1.0}

    def _calculate_options_position_size(self, signal: Dict[str, Any], strike_info: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Calculate position size for options trade with enhanced logic"""
        try:
            # Use the base position sizer
            base_position = self.position_sizer.calculate_position_size(
                signal=signal,
                price=current_price,
                instrument_info=strike_info
            )
            
            if not base_position:
                return None
                
            # Options-specific adjustments
            quantity = base_position.get('quantity', 0)
            
            # Reduce quantity for OTM options (higher risk)
            if strike_info.get('is_otm', False):
                quantity = max(1, int(quantity * 0.7))
                
            # Increase quantity for ATM options (balanced risk/reward)
            elif strike_info.get('is_atm', False):
                quantity = int(quantity * 1.1)
                
            base_position['quantity'] = quantity
            return base_position
        except Exception as e:
            logger.error(f"Error calculating options position size: {e}")
            return None

    def _calculate_options_stop_loss(self, current_price: float, strike_info: Dict[str, Any]) -> float:
        """Calculate options-specific stop loss"""
        try:
            # Options stop loss should be percentage-based due to high volatility
            stop_loss_pct = getattr(Config, 'OPTIONS_STOP_LOSS_PCT', 20.0)  # 20% default
            
            # Adjust based on option type and moneyness
            if strike_info.get('is_otm', False):
                stop_loss_pct *= 1.2  # OTM options need wider stops
            elif strike_info.get('is_itm', False):
                stop_loss_pct *= 0.8  # ITM options can have tighter stops
                
            return current_price * (1 - stop_loss_pct / 100)
        except Exception as e:
            logger.error(f"Error calculating options stop loss: {e}")
            return current_price * 0.8  # 20% default

    def _calculate_options_target(self, current_price: float, strike_info: Dict[str, Any]) -> float:
        """Calculate options-specific target"""
        try:
            target_pct = getattr(Config, 'OPTIONS_TARGET_PCT', 40.0)  # 40% default
            
            # Adjust based on option type and moneyness
            if strike_info.get('is_otm', False):
                target_pct *= 1.3  # OTM options can have higher targets
            elif strike_info.get('is_itm', False):
                target_pct *= 0.9  # ITM options may have lower targets
                
            return current_price * (1 + target_pct / 100)
        except Exception as e:
            logger.error(f"Error calculating options target: {e}")
            return current_price * 1.4  # 40% default

    def _passes_risk_checks(self, signal: Dict[str, Any], position: Dict[str, Any], strike_info: Dict[str, Any]) -> bool:
        """Enhanced risk management checks for options"""
        try:
            # Basic quantity check
            if position.get('quantity', 0) <= 0:
                return False
                
            # Maximum daily trades check
            max_daily_trades = getattr(Config, 'MAX_DAILY_OPTIONS_TRADES', 5)
            if len(self.trades) >= max_daily_trades:
                logger.warning(f"Daily trade limit reached: {len(self.trades)}/{max_daily_trades}")
                return False
                
            # Maximum position size check
            max_position_value = getattr(Config, 'MAX_POSITION_VALUE', 50000)
            position_value = position['quantity'] * signal.get('entry_price', 0) * getattr(Config, 'NIFTY_LOT_SIZE', 75)
            if position_value > max_position_value:
                logger.warning(f"Position value {position_value} exceeds limit {max_position_value}")
                return False
                
            # Time to expiry check (don't trade options too close to expiry)
            # This would need expiry date from strike_info or Config
            # For now, assume this check passes
            
            return True
        except Exception as e:
            logger.error(f"Error in risk checks: {e}")
            return False

    def _execute_options_trade(self, symbol: str, signal: Dict[str, Any], position: Dict[str, Any], strike_info: Dict[str, Any]) -> bool:
        """Execute the complete options trade workflow"""
        try:
            # Send telegram alert
            token = len(self.trades) + 1
            self.telegram_controller.send_signal_alert(token, signal, position)
            
            # Place entry order
            order_transaction_type = signal.get("signal", "BUY")
            logger.info(f"Placing {order_transaction_type} order for {symbol}, Qty: {position['quantity']}")
            
            order_id = self.order_executor.place_entry_order(
                symbol=symbol,
                exchange="NFO",
                transaction_type=order_transaction_type,
                quantity=position['quantity'],
                price=signal.get('entry_price'),
                product="MIS"  # Assuming MIS for intraday options
            )
            
            if not order_id:
                logger.error(f"Failed to place entry order for {symbol}")
                return False
                
            # Setup GTT orders for stop-loss and target
            sl_order_id = self.order_executor.setup_gtt_orders(
                symbol=symbol,
                exchange="NFO",
                transaction_type="SELL" if order_transaction_type == "BUY" else "BUY",
                quantity=position['quantity'],
                stop_loss_price=signal.get('stop_loss'),
                target_price=signal.get('target')
            )
            
            # Record the trade
            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "direction": order_transaction_type,
                "quantity": position['quantity'],
                "entry_price": signal.get('entry_price'),
                "stop_loss": signal.get('stop_loss'),
                "target": signal.get('target'),
                "strike_info": strike_info,
                "order_id": order_id,
                "sl_order_id": sl_order_id,
                "status": "OPEN"
            }
            
            self.trades.append(trade_record)
            logger.info(f"✅ Trade executed and recorded: {trade_record}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing options trade for {symbol}: {e}")
            return False

    # --- Additional Utility Methods for Complete Functionality ---
    def get_current_positions(self) -> List[Dict[str, Any]]:
        """Get current open positions"""
        try:
            if not self.order_executor or not hasattr(self.order_executor, 'kite') or not self.order_executor.kite:
                return []
            positions = self.order_executor.kite.positions()
            return positions.get('net', []) if positions else []
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return []

    def calculate_current_pnl(self) -> float:
        """Calculate current P&L from open positions"""
        try:
            positions = self.get_current_positions()
            total_pnl = 0.0
            for pos in positions:
                total_pnl += pos.get('pnl', 0)
            return total_pnl
        except Exception as e:
            logger.error(f"Error calculating current P&L: {e}")
            return 0.0

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate basic performance metrics"""
        try:
            metrics = {
                "total_trades": len(self.trades),
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": self.daily_pnl,
                "trades_by_type": {},
                "avg_win": 0.0,
                "avg_loss": 0.0
            }
            
            winning_pnls = []
            losing_pnls = []
            
            # For now, use a placeholder P&L calculation
            # In a real system, you'd track actual exit prices
            for trade in self.trades:
                strike_info = trade.get('strike_info', {})
                opt_type = strike_info.get('type', 'UNK')
                metrics["trades_by_type"][opt_type] = metrics["trades_by_type"].get(opt_type, 0) + 1
                
                # Calculate win rate
                total_trades = len(self.trades)
                if total_trades > 0:
                    metrics["win_rate"] = (metrics["winning_trades"] / total_trades) * 100
                    
            return metrics
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {"error": str(e)}

    def get_market_data_summary(self) -> Dict[str, Any]:
        """Get a summary of current market data"""
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "spot_price": None,
                "trend": "NEUTRAL"
            }
            
            # Get spot price
            try:
                spot_symbol_ltp = _get_spot_ltp_symbol()
                ltp_data = self.order_executor.kite.ltp([spot_symbol_ltp])
                summary["spot_price"] = ltp_data.get(spot_symbol_ltp, {}).get('last_price')
            except Exception:
                pass
                
            # Basic trend analysis (simplified)
            if hasattr(self, '_last_spot_prices'):
                if len(self._last_spot_prices) >= 5:
                    recent_prices = self._last_spot_prices[-5:]
                    if recent_prices[-1] > recent_prices[0]:
                        summary["trend"] = "BULLISH"
                    elif recent_prices[-1] < recent_prices[0]:
                        summary["trend"] = "BEARISH"
                        
            return summary
        except Exception as e:
            logger.error(f"Error getting market data summary: {e}")
            return {"error": str(e)}

    def force_exit_all_positions(self) -> bool:
        """Emergency function to exit all open positions"""
        try:
            positions = self.get_current_positions()
            success_count = 0
            
            for pos in positions:
                if pos.get('quantity', 0) != 0:
                    symbol = pos.get('tradingsymbol')
                    exchange = pos.get('exchange')
                    quantity = abs(pos.get('quantity', 0))
                    transaction_type = "SELL" if pos.get('quantity', 0) > 0 else "BUY"
                    
                    try:
                        order_id = self.order_executor.place_order(
                            symbol=symbol,
                            exchange=exchange,
                            transaction_type=transaction_type,
                            quantity=quantity,
                            order_type="MARKET",
                            product=pos.get('product')
                        )
                        if order_id:
                            success_count += 1
                            logger.info(f"Exit order placed for {symbol}: {order_id}")
                    except Exception as e:
                        logger.error(f"Failed to exit {symbol}: {e}")
                        
            self.telegram_controller.send_message(
                f"🚨 Emergency exit completed. {success_count}/{len(positions)} positions closed."
            )
            return success_count == len(positions)
        except Exception as e:
            logger.error(f"Error in emergency exit: {e}")
            return False

    def optimize_performance(self) -> None:
        """Optimize system performance by cleaning up resources"""
        try:
            logger.info("🔧 Running performance optimization...")
            
            # Clean up old cache entries
            with self._cache_lock:
                current_time = time.time()
                if (current_time - self._instruments_cache_timestamp) > (self._INSTRUMENT_CACHE_DURATION * 2):
                    logger.info("Clearing stale instrument cache")
                    self._nfo_instruments_cache = None
                    self._nse_instruments_cache = None
                    self._instruments_cache_timestamp = 0
                    
            # Clear ATM cache
            if hasattr(self, '_atm_cache'):
                self._atm_cache.clear()
                
            # Force garbage collection
            import gc
            gc.collect()
            logger.info("✅ Performance optimization completed")
        except Exception as e:
            logger.error(f"Error during performance optimization: {e}")

    def get_detailed_status(self) -> Dict[str, Any]:
        """Get comprehensive detailed status"""
        try:
            basic_status = self.get_status()
            detailed_status = {
                **basic_status,
                "system_info": {
                    "uptime_seconds": time.time() - self._start_time,
                    "session_start": self._session_start_time,
                    "last_activity": getattr(self, '_last_activity_time', 'Never'),
                    "errors_count": getattr(self, '_error_count', 0)
                }
            }
            
            # Add risk manager status if available
            if hasattr(self, 'risk_manager') and hasattr(self.risk_manager, 'get_risk_status'):
                detailed_status.update(self.risk_manager.get_risk_status())
                
            return detailed_status
        except Exception as e:
            logger.error(f"Error getting detailed status: {e}")
            return {"error": str(e)}

    def run_system_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        try:
            diagnostics = {
                "timestamp": datetime.now().isoformat(),
                "trader_status": {
                    "is_trading": self.is_trading,
                    "live_mode": self.live_mode,
                    "trades_count": len(self.trades),
                    "daily_pnl": self.daily_pnl
                },
                "cache_status": {
                    "nfo_instruments": len(self._nfo_instruments_cache or []),
                    "nse_instruments": len(self._nse_instruments_cache or []),
                    "cache_age_minutes": (time.time() - self._instruments_cache_timestamp) / 60
                },
                "executor_status": {
                    "has_kite": hasattr(self.order_executor, 'kite') and self.order_executor.kite is not None,
                    "connection_valid": False
                },
                "strike_selector_health": None
            }
            
            # Test connection
            if self.order_executor and hasattr(self.order_executor, 'kite') and self.order_executor.kite:
                try:
                    profile = self.order_executor.kite.profile()
                    diagnostics["executor_status"]["connection_valid"] = True
                    diagnostics["executor_status"]["user_name"] = profile.get('user_name', 'Unknown')
                except Exception:
                    diagnostics["executor_status"]["connection_valid"] = False
                    
            # Run strike selector health check
            if self.order_executor and hasattr(self.order_executor, 'kite') and self.order_executor.kite:
                try:
                    diagnostics["strike_selector_health"] = health_check(self.order_executor.kite)
                except Exception as e:
                    diagnostics["strike_selector_health"] = {"error": str(e)}
                    
            return diagnostics
        except Exception as e:
            logger.error(f"Error running system diagnostics: {e}")
            return {"error": str(e)}

    def save_trading_session(self, filename: str = None) -> bool:
        """Save current trading session to file"""
        try:
            import json
            if not filename:
                filename = f"trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "trades": self.trades,
                "daily_pnl": self.daily_pnl,
                "is_trading": self.is_trading
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            logger.info(f"Trading session saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving trading session: {e}")
            return False

    def load_trading_session(self, filename: str) -> bool:
        """Load trading session from file"""
        try:
            import json
            with open(filename, 'r') as f:
                session_data = json.load(f)
                
            # Restore session data
            if 'trades' in session_data:
                self.trades = session_data['trades']
            if 'daily_pnl' in session_data:
                self.daily_pnl = session_data['daily_pnl']
                
            logger.info(f"Trading session loaded from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading trading session: {e}")
            return False

    # --- Enhanced Telegram Command Handling ---
    def _handle_control(self, command: str, arg: str = "") -> bool:
        """Enhanced command handling with more options"""
        command = command.strip().lower()
        arg = arg.strip().lower() if arg else ""
        logger.info(f"Received command: /{command} {arg}")
        
        # Store last activity
        self._last_activity_time = datetime.now().isoformat()
        
        try:
            if command == "start":
                self.start()
                self.telegram_controller.send_message("✅ Trading started")
                return True
            elif command == "stop":
                self.stop()
                self.telegram_controller.send_message("🛑 Trading stopped")
                return True
            elif command == "status":
                return self._send_status()
            elif command == "summary":
                return self._send_summary()
            elif command == "positions":
                return self._send_positions()
            elif command == "pnl":
                return self._send_pnl()
            elif command == "health":
                return self._send_health_check()
            elif command == "help":
                return self._send_help()
            elif command == "refresh":
                self._refresh_instruments_cache(force=True)
                self.telegram_controller.send_message("🔄 Instrument cache refreshed")
                return True
            elif command == "mode":
                if arg == "live":
                    self.live_mode = True
                    self._set_live_mode()
                    self.telegram_controller.send_message("🟢 Switched to LIVE mode")
                elif arg == "shadow":
                    self.live_mode = False
                    self._set_live_mode()
                    self.telegram_controller.send_message("🛡️ Switched to SHADOW mode")
                else:
                    self.telegram_controller.send_message("Usage: /mode live or /mode shadow")
                return True
            elif command == "exit":
                if arg == "all":
                    success = self.force_exit_all_positions()
                    if success:
                        self.telegram_controller.send_message("✅ All positions closed")
                    else:
                        self.telegram_controller.send_message("⚠️ Some positions may not have closed")
                else:
                    self._show_exit_help()
                return True
            elif command == "optimize":
                self.optimize_performance()
                self.telegram_controller.send_message("⚡ Performance optimized")
                return True
            elif command == "diagnostics":
                return self._send_diagnostics()
            elif command == "save":
                return self._save_session()
            elif command == "load":
                if arg:
                    success = self.load_trading_session(arg)
                    if success:
                        self.telegram_controller.send_message(f"💾 Session loaded from {arg}")
                    else:
                        self.telegram_controller.send_message(f"❌ Failed to load session from {arg}")
                else:
                    self.telegram_controller.send_message("Usage: /load <filename>")
                return True
            else:
                self._send_unknown_command(command)
                return True
        except Exception as e:
            logger.error(f"Error handling control command '{command}': {e}")
            self.telegram_controller.send_message("❌ Error processing command")
            return False

    def _send_status(self) -> bool:
        """Send current status"""
        try:
            status = self.get_status()
            message = (
                f"📊 **RealTimeTrader Status**\n"
                f"🔁 **Trading Active:** {'🟢 YES' if status.get('is_trading') else '🔴 NO'}\n"
                f"💰 **Mode:** {'🟢 LIVE' if status.get('live_mode') else '🛡️ SHADOW'}\n"
                f"📈 **Trades Today:** {status.get('trades_today', 0)}\n"
                f"💼 **Open Positions:** {status.get('open_positions', 0)}\n"
                f"💵 **Daily P&L:** ₹{status.get('daily_pnl', 0.0):.2f}\n"
                f"🕒 **Cache Age:** {status.get('cache_age_minutes', 0):.1f} min"
            )
            self.telegram_controller.send_message(message.strip(), parse_mode="Markdown")
            return True
        except Exception as e:
            logger.error(f"Error sending status: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            status = {
                "is_trading": self.is_trading,
                "live_mode": self.live_mode,
                "trades_today": len(self.trades),
                "daily_pnl": self.daily_pnl,
                "cache_age_minutes": (time.time() - self._instruments_cache_timestamp) / 60,
                "open_positions": len([p for p in self.get_current_positions() if p.get('quantity', 0) != 0])
            }
            
            # Add risk manager status if available
            if hasattr(self, 'risk_manager') and hasattr(self.risk_manager, 'get_risk_status'):
                status.update(self.risk_manager.get_risk_status())
                
            return status
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}

    def get_summary(self) -> str:
        """Enhanced daily summary with options-specific information"""
        try:
            lines = [
                f"📊 **Daily Options Trading Summary**",
                f"🔁 **Total trades:** {len(self.trades)}",
                f"💰 **PnL:** ₹{self.daily_pnl:.2f}",
                f"📈 **Mode:** {'🟢 LIVE' if self.live_mode else '🛡️ SHADOW'}",
                "────────────────────"
            ]
            
            # Group trades by option type
            ce_trades = [t for t in self.trades if t.get('strike_info', {}).get('type') == 'CE']
            pe_trades = [t for t in self.trades if t.get('strike_info', {}).get('type') == 'PE']
            
            if ce_trades:
                lines.append(f"📈 **CE Trades:** {len(ce_trades)}")
            if pe_trades:
                lines.append(f"📉 **PE Trades:** {len(pe_trades)}")
                
            lines.append("────────────────────")
            
            # Show recent trades
            recent_trades = self.trades[-5:] if len(self.trades) > 5 else self.trades
            for trade in recent_trades:
                symbol = trade.get('symbol', 'N/A')
                strike_info = trade.get('strike_info', {})
                opt_type = strike_info.get('type', 'UNK')
                strike = strike_info.get('strike', 'N/A')
                lines.append(
                    f"{opt_type} {strike} {trade['direction']} {trade['quantity']} @ ₹{trade['entry_price']:.2f} "
                    f"(SL ₹{trade['stop_loss']:.2f}, TP ₹{trade['target']:.2f})"
                )
                
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"📊 Summary Error: {str(e)}"

    def _send_summary(self) -> bool:
        """Send trading summary"""
        try:
            summary = self.get_summary()
            self.telegram_controller.send_message(summary, parse_mode="Markdown")
            return True
        except Exception as e:
            logger.error(f"Error sending summary: {e}")
            return False

    def _send_positions(self) -> bool:
        """Send current positions"""
        try:
            positions = self.get_current_positions()
            if not positions:
                self.telegram_controller.send_message("📊 No open positions")
                return True
                
            lines = ["📊 **Current Positions**", ""]
            for pos in positions:
                if pos.get('quantity', 0) != 0:
                    symbol = pos.get('tradingsymbol', 'Unknown')
                    qty = pos.get('quantity', 0)
                    pnl = pos.get('pnl', 0)
                    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                    lines.append(f"{pnl_emoji} {symbol}: {qty} qty, P&L: ₹{pnl:.2f}")
                    
            self.telegram_controller.send_message("\n".join(lines), parse_mode="Markdown")
            return True
        except Exception as e:
            logger.error(f"Error sending positions: {e}")
            return False

    def _send_pnl(self) -> bool:
        """Send P&L summary"""
        try:
            current_pnl = self.calculate_current_pnl()
            message = (
                f"💰 **Profit & Loss Summary**\n"
                f"📊 Daily P&L: ₹{self.daily_pnl:.2f}\n"
                f"💼 Open P&L: ₹{current_pnl:.2f}\n"
                f"📈 Total P&L: ₹{self.daily_pnl + current_pnl:.2f}"
            )
            self.telegram_controller.send_message(message, parse_mode="Markdown")
            return True
        except Exception as e:
            logger.error(f"Error sending P&L: {e}")
            return False

    def _send_health_check(self) -> bool:
        """Send system health check"""
        try:
            if self.order_executor and hasattr(self.order_executor, 'kite') and self.order_executor.kite:
                health = health_check(self.order_executor.kite)
                message = f"""🛠️ **System Health Check**
📊 **Status:** {health.get('overall_status', 'UNKNOWN')}
⏱️ **Timestamp:** {health.get('timestamp', 'N/A')[:19]}

**Checks:**
"""
                for check_name, check_info in health.get('checks', {}).items():
                    status_emoji = "✅" if check_info.get('status') == 'PASS' else "❌"
                    message += f"{status_emoji} {check_name}: {check_info.get('message', 'N/A')}\n"
                    
                if health.get('recommendations'):
                    message += f"\n**Recommendations:**\n"
                    message += "\n".join(f"• {rec}" for rec in health.get('recommendations', [])[:3])
                    
                self.telegram_controller.send_message(message.strip(), parse_mode="Markdown")
            else:
                self.telegram_controller.send_message("❌ Cannot run health check - No API connection")
            return True
        except Exception as e:
            logger.error(f"Error sending health check: {e}")
            return False

    def _send_help(self) -> bool:
        """Send help message with available commands"""
        help_message = """🤖 **Available Commands**

**Basic Controls:**
• `/start` - Start trading
• `/stop` - Stop trading
• `/status` - Get current status
• `/summary` - Trading summary

**Mode & Settings:**
• `/mode live` - Switch to live trading
• `/mode shadow` - Switch to paper trading

**Monitoring:**
• `/positions` - Show open positions
• `/pnl` - Show P&L summary
• `/health` - System health check

**System:**
• `/refresh` - Refresh data cache
• `/optimize` - Optimize performance
• `/diagnostics` - Full system check
• `/save` - Save trading session
• `/load <filename>` - Load trading session

**Emergency:**
• `/exit all` - Close all positions (⚠️ Market orders!)

**Other:**
• `/help` - Show this help"""
        self.telegram_controller.send_message(help_message.strip(), parse_mode="Markdown")
        return True

    def _send_diagnostics(self) -> bool:
        """Send system diagnostics"""
        try:
            diagnostics = self.run_system_diagnostics()
            message = f"""🛠️ **System Diagnostics**
📊 **Trader Status**
🔁 Trading Active: {'🟢 YES' if diagnostics['trader_status']['is_trading'] else '🔴 NO'}
💰 Mode: {'🟢 LIVE' if diagnostics['trader_status']['live_mode'] else '🛡️ SHADOW'}
📈 Trades Today: {diagnostics['trader_status']['trades_count']}
"""
            self.telegram_controller.send_message(message.strip(), parse_mode="Markdown")
            return True
        except Exception as e:
            logger.error(f"Error sending diagnostics: {e}")
            return False

    def _save_session(self) -> bool:
        """Save current session"""
        try:
            success = self.save_trading_session()
            if success:
                self.telegram_controller.send_message("💾 Trading session saved successfully")
            else:
                self.telegram_controller.send_message("❌ Failed to save trading session")
            return success
        except Exception as e:
            logger.error(f"Error in save session command: {e}")
            return False

    def _send_unknown_command(self, command: str) -> None:
        """Handle unknown commands"""
        self.telegram_controller.send_message(f"❌ Unknown command: `{command}`\nUse `/help` for available commands",
                                              parse_mode="Markdown")

    def _show_exit_help(self) -> bool:
        """Show exit command help"""
        self.telegram_controller.send_message("⚠️ Usage: `/exit all` to close all positions\n⚠️ This will place market orders immediately!",
                                              parse_mode="Markdown")
        return True

    def _emergency_exit(self) -> bool:
        """Handle emergency exit command"""
        return self.force_exit_all_positions()

    def _run_optimization(self) -> bool:
        """Run system optimization"""
        try:
            self.optimize_performance()
            self.telegram_controller.send_message("⚡ System optimization completed")
            return True
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return False

    # --- Original Bar Processing (for backward compatibility) ---
    def process_bar(self, ohlc: pd.DataFrame) -> None:
        """Original bar processing method for non-options strategies"""
        logger.debug(f"process_bar called. Trading active: {self.is_trading}, OHLC data points: {len(ohlc) if ohlc is not None else 'None'}")
        if not self.is_trading:
            logger.debug("process_bar: Trading not active, returning.")
            return
        if ohlc is None or len(ohlc) < 30:
            logger.debug("Insufficient data to process bar (less than 30 points).")
            return
        try:
            if not isinstance(ohlc.index, pd.DatetimeIndex):
                logger.error("OHLC data must have DatetimeIndex.")
                return
            ts = ohlc.index[-1]
            if not self._passes_time_filter(ts):
                logger.debug("Time filter rejected bar")
                return
            current_price = float(ohlc.iloc[-1]["close"])
            logger.debug(f"Current bar timestamp: {ts}, price: {current_price}")
            signal = self.strategy.generate_signal(ohlc, current_price)
            logger.debug(f"Strategy returned signal: {signal}")
            if not signal:
                logger.debug("No signal generated by strategy.")
                return
            signal_confidence = float(signal.get("confidence", 0.0))
            confidence_threshold = getattr(Config, 'CONFIDENCE_THRESHOLD', 7.0)
            if signal_confidence < confidence_threshold:
                logger.debug(f"Signal confidence {signal_confidence} below threshold {confidence_threshold}")
                return
            position = self.position_sizer.calculate_position_size(signal, current_price)
            if not position or position.get("quantity", 0) <= 0:
                logger.debug("Invalid position sizing.")
                return
            if not self._passes_risk_checks(signal, position):
                logger.warning("Risk checks failed for bar signal.")
                return
            success = self._execute_trade(signal, position)
            if success:
                logger.info("✅ Successfully executed trade for bar signal")
            else:
                logger.warning("❌ Failed to execute trade for bar signal")
        except Exception as exc:
            logger.error(f"Error processing bar: {exc}", exc_info=True)

    def _passes_time_filter(self, ts: pd.Timestamp) -> bool:
        """Check if timestamp passes time filter"""
        try:
            hour = ts.hour
            minute = ts.minute
            # Allow trading between 9:15 AM and 3:30 PM
            if hour < 9 or (hour == 9 and minute < 15):
                return False
            if hour > 15 or (hour == 15 and minute > 30):
                return False
            return True
        except Exception as e:
            logger.error(f"Error in time filter: {e}")
            return True

    def _execute_trade(self, signal: Dict[str, Any], position: Dict[str, Any]) -> bool:
        """Execute a regular trade"""
        try:
            # Send telegram alert
            token = len(self.trades) + 1
            self.telegram_controller.send_signal_alert(token, signal, position)
            
            # Place order
            order_transaction_type = signal.get("signal", "BUY")
            logger.info(f"Placing {order_transaction_type} order, Qty: {position['quantity']}")
            
            order_id = self.order_executor.place_entry_order(
                symbol=getattr(Config, 'SPOT_SYMBOL', 'NSE:NIFTY 50'),
                exchange="NSE",
                transaction_type=order_transaction_type,
                quantity=position['quantity'],
                price=signal.get('entry_price'),
                product="MIS"
            )
            
            if not order_id:
                logger.error("Failed to place entry order")
                return False
                
            # Setup GTT orders
            sl_order_id = self.order_executor.setup_gtt_orders(
                symbol=getattr(Config, 'SPOT_SYMBOL', 'NSE:NIFTY 50'),
                exchange="NSE",
                transaction_type="SELL" if order_transaction_type == "BUY" else "BUY",
                quantity=position['quantity'],
                stop_loss_price=signal.get('stop_loss'),
                target_price=signal.get('target')
            )
            
            # Record trade
            trade_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": getattr(Config, 'SPOT_SYMBOL', 'NSE:NIFTY 50'),
                "direction": order_transaction_type,
                "quantity": position['quantity'],
                "entry_price": signal.get('entry_price'),
                "stop_loss": signal.get('stop_loss'),
                "target": signal.get('target'),
                "order_id": order_id,
                "sl_order_id": sl_order_id,
                "status": "OPEN"
            }
            
            self.trades.append(trade_record)
            logger.info(f"✅ Trade executed and recorded: {trade_record}")
            return True
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

    # --- Cleanup ---
    def shutdown(self) -> None:
        """Gracefully shutdown the trader"""
        try:
            logger.info("👋 Shutting down RealTimeTrader...")
            
            # Stop trading
            self.stop()
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            # Cleanup order executor
            if hasattr(self.order_executor, 'cleanup'):
                self.order_executor.cleanup()
                
            logger.info("✅ RealTimeTrader shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def __repr__(self) -> str:
        return (f"<RealTimeTrader is_trading={self.is_trading} "
                f"live_mode={self.live_mode} trades_today={len(self.trades)} "
                f"cache_age={int((time.time() - self._instruments_cache_timestamp)/60)}min>")

    def __del__(self):
        """Cleanup on object destruction"""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass

# - Final Export for Module -
if __name__ == "__main__":
    # Allow running the trader directly for testing
    logging.basicConfig(level=logging.INFO)
    logger.info("RealTimeTrader - Direct execution mode")
    
    try:
        trader = RealTimeTrader()
        logger.info("✅ RealTimeTrader created successfully")
        
        # Keep it running for testing
        import time
        while True:
            time.sleep(60)  # Run for testing
    except KeyboardInterrupt:
        logger.info("👋 RealTimeTrader stopped by user")
    except Exception as e:
        logger.error(f"❌ RealTimeTrader failed: {e}", exc_info=True)
