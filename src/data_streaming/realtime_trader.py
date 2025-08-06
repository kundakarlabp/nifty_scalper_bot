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

from src.config import Config
# Assuming you'll have an OptionsStrategy or modify the existing one
# from src.strategies.options_strategy import OptionsStrategy
from src.strategies.scalping_strategy import EnhancedScalpingStrategy # Placeholder, might need new strategy
from src.risk.position_sizing import PositionSizing
from src.execution.order_executor import OrderExecutor
from src.notifications.telegram_controller import TelegramController
from src.utils.strike_selector import get_next_expiry_date, get_instrument_tokens

logger = logging.getLogger(__name__)

class RealTimeTrader:
    def __init__(self) -> None:
        self.is_trading: bool = False
        self.daily_pnl: float = 0.0
        self.trades: List[Dict[str, Any]] = []
        self.live_mode: bool = Config.ENABLE_LIVE_TRADING

        # Assuming you'll have an OptionsStrategy or modify the existing one
        self.strategy = EnhancedScalpingStrategy(
            base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
            base_target_points=Config.BASE_TARGET_POINTS,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
            min_score_threshold=int(Config.MIN_SIGNAL_SCORE),
        )
        self.risk_manager = PositionSizing()
        self.order_executor = self._init_order_executor()
        self.telegram_controller = TelegramController(
            status_callback=self.get_status,
            control_callback=self._handle_control,
            summary_callback=self.get_summary,
        )
        self._polling_thread: Optional[threading.Thread] = None

        # Start Telegram polling in a daemon thread
        self._start_polling()

        # Schedule the data fetching and processing task
        schedule.every(1).minutes.do(self.fetch_and_process_data)
        logger.info("Scheduled fetch_and_process_data to run every 1 minute.")

        atexit.register(self.shutdown)
        logger.info("RealTimeTrader initialized and ready to receive commands.")

    def _init_order_executor(self) -> OrderExecutor:
        if not self.live_mode:
            logger.info("Live trading disabled. Using simulated order executor.")
            return OrderExecutor()
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
            kite.set_access_token(Config.KITE_ACCESS_TOKEN)
            logger.info("✅ Live order executor initialized with Kite Connect.")
            return OrderExecutor(kite=kite)
        except Exception as exc:
            logger.error("Failed to initialize live trading. Falling back to simulation: %s", exc, exc_info=True)
            self.live_mode = False
            return OrderExecutor()

    def start(self) -> bool:
        if self.is_trading:
            logger.info("Trader already running.")
            self.telegram_controller.send_message("🛑 Trader already running.")
            return True
        self.is_trading = True
        try:
            self.telegram_controller.send_realtime_session_alert("START")
            logger.info("✅ Trading started.")
        except Exception as exc:
            logger.warning("Failed to send START alert: %s", exc)
        return True

    def stop(self) -> bool:
        if not self.is_trading:
            logger.info("Trader is not running.")
            self.telegram_controller.send_message("🛑 Trader is already stopped.")
            return True
        self.is_trading = False
        try:
            self.telegram_controller.send_realtime_session_alert("STOP")
            logger.info("🛑 Trading stopped. Telegram polling remains active.")
        except Exception as exc:
            logger.warning("Failed to send STOP alert: %s", exc)
        return True

    def _handle_control(self, command: str, arg: str = "") -> bool:
        command = command.strip().lower()
        arg = arg.strip().lower() if arg else ""
        logger.info(f"Received command: /{command} {arg}")
        if command == "start":
            return self.start()
        elif command == "stop":
            return self.stop()
        elif command == "mode":
            if arg not in ["live", "shadow"]:
                logger.warning("Invalid mode argument: %s", arg)
                self.telegram_controller.send_message("⚠️ Usage: `/mode live` or `/mode shadow`", parse_mode="Markdown")
                return False
            return self._set_live_mode(arg)
        else:
            logger.warning("Unknown control command: %s", command)
            self.telegram_controller.send_message(f"❌ Unknown command: `{command}`", parse_mode="Markdown")
            return False

    def _set_live_mode(self, mode: str) -> bool:
        desired_live = (mode == "live")
        if desired_live == self.live_mode:
            current_mode = "LIVE" if self.live_mode else "SHADOW"
            logger.info(f"Already in {current_mode} mode.")
            self.telegram_controller.send_message(f"🟢 Already in *{current_mode}* mode.", parse_mode="Markdown")
            return True
        if self.is_trading:
            logger.warning("Cannot change mode while trading is active. Stop trading first.")
            self.telegram_controller.send_message("🛑 Cannot change mode while trading. Use `/stop` first.", parse_mode="Markdown")
            return False
        if desired_live:
            try:
                from kiteconnect import KiteConnect
                kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
                kite.set_access_token(Config.KITE_ACCESS_TOKEN)
                self.order_executor = OrderExecutor(kite=kite)
                self.live_mode = True
                logger.info("🟢 Switched to LIVE mode.")
                self.telegram_controller.send_message("🚀 Switched to *LIVE* trading mode.", parse_mode="Markdown")
                return True
            except Exception as exc:
                logger.error("Failed to switch to LIVE mode: %s", exc, exc_info=True)
                self.telegram_controller.send_message(
                    f"❌ Failed to switch to LIVE mode: `{str(exc)[:100]}...` Reverted to SHADOW mode.", parse_mode="Markdown"
                )
                self.live_mode = False
                self.order_executor = OrderExecutor()
                return False
        else:
            self.order_executor = OrderExecutor()
            self.live_mode = False
            logger.info("🛡️ Switched to SHADOW (simulation) mode.")
            self.telegram_controller.send_message("🛡️ Switched to *SHADOW* (simulation) mode.", parse_mode="Markdown")
            return True

    def _start_polling(self) -> None:
        if self._polling_thread and self._polling_thread.is_alive():
            logger.debug("Polling thread already running.")
            return
        try:
            self.telegram_controller.send_startup_alert()
        except Exception as e:
            logger.warning("Failed to send startup alert: %s", e)
        self._polling_thread = threading.Thread(
            target=self.telegram_controller.start_polling,
            daemon=True
        )
        self._polling_thread.start()
        logger.info("✅ Telegram polling started (daemon).")

    def _stop_polling(self) -> None:
        logger.info("🛑 Stopping Telegram polling (app shutdown)...")
        self.telegram_controller.stop_polling()
        if self._polling_thread and self._polling_thread.is_alive():
            if threading.current_thread() != self._polling_thread:
                self._polling_thread.join(timeout=3)
        self._polling_thread = None

    def shutdown(self) -> None:
        if not self.is_trading and (not self._polling_thread or not self._polling_thread.is_alive()):
            return
        logger.info("👋 Shutting down RealTimeTrader...")
        self.stop()
        self._stop_polling()
        logger.info("✅ RealTimeTrader shutdown complete.")

    def _analyze_options_data(self, spot_price: float, options_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Analyze fetched options data to select optimal strikes based on OI, Delta, etc.
        This is a simplified example. You can add more complex logic here.
        """
        selected_strikes = []
        atm_strike = round(spot_price / 50) * 50

        for opt_type in ['CE', 'PE']:
            # Example logic: Select ATM and one ITM/OTM based on OI change or Delta
            relevant_strikes = []
            for symbol, df in options_data.items():
                if opt_type not in symbol or df.empty:
                    continue
                # Extract strike from symbol (assuming format like NIFTY23SEP24000CE)
                try:
                    # Find the position of 'CE' or 'PE' and extract the strike before it
                    opt_index = symbol.find(opt_type)
                    if opt_index != -1:
                        # Assuming strike is 5 digits before CE/PE
                        strike_str = symbol[opt_index-5:opt_index]
                        strike = int(strike_str)
                    else:
                        logger.warning(f"Could not extract strike from symbol {symbol}")
                        continue
                except ValueError:
                    logger.warning(f"Could not extract strike from symbol {symbol}")
                    continue

                if opt_type == 'CE':
                    otm_condition = strike > atm_strike
                    itm_condition = strike < atm_strike
                else: # PE
                    otm_condition = strike < atm_strike
                    itm_condition = strike > atm_strike

                # Get last few rows for analysis (e.g., last 2 bars for delta)
                if len(df) < 2:
                    continue

                last_row = df.iloc[-1]
                prev_row = df.iloc[-2]

                # Basic OI Change calculation (placeholder, real OI data needed)
                # Kite historical data might not have 'oi' by default.
                # You might need to fetch LTP and calculate changes or use quotes.
                # This example assumes 'oi' column exists.
                oi_change = last_row.get('oi', 0) - prev_row.get('oi', 0)

                # Basic Delta approximation (requires more sophisticated calculation)
                # This is a very rough proxy using price change vs underlying change
                # A real implementation would use Greeks from the API or calculate from the book
                delta_approx = 0.5 # Placeholder, needs real calculation

                relevant_strikes.append({
                    'symbol': symbol,
                    'strike': strike,
                    'type': opt_type,
                    'ltp': last_row.get('last_price', 0),
                    'oi': last_row.get('oi', 0),
                    'oi_change': oi_change,
                    'delta': delta_approx,
                    'is_atm': strike == atm_strike,
                    'is_otm': otm_condition,
                    'is_itm': itm_condition
                })

            # Sort by OI Change (descending) to find strongest buildup
            # Note: If 'oi' is not available, this sorting won't work as intended.
            relevant_strikes.sort(key=lambda x: x['oi_change'], reverse=True)

            # Select ATM
            atm_option = next((s for s in relevant_strikes if s['is_atm']), None)
            if atm_option:
                selected_strikes.append(atm_option)
                logger.info(f"Selected ATM {opt_type}: {atm_option['symbol']}")

            # Select one with highest OI change (could be ITM or OTM)
            # Add logic to prefer ITM or OTM based on strategy
            if Config.STRIKE_SELECTION_TYPE == "ITM":
                 best_other = next((s for s in relevant_strikes if s['is_itm']), None)
            elif Config.STRIKE_SELECTION_TYPE == "OTM":
                 best_other = next((s for s in relevant_strikes if s['is_otm']), None)
            else: # Default to highest OI change non-ATM or ATM logic
                 # If OI data is not reliable, fallback to ATM/ATM+offset logic
                 # Or implement a different selection criterion here.
                 # For now, let's just pick the first non-ATM if available.
                 best_other = next((s for s in relevant_strikes if not s['is_atm']), None)

            if best_other:
                selected_strikes.append(best_other)
                logger.info(f"Selected {Config.STRIKE_SELECTION_TYPE} {opt_type}: {best_other['symbol']} (OI Change: {best_other['oi_change']})")
            elif atm_option: # If no other found, ensure ATM is added if not already
                 # ATM already added above
                 pass
            else:
                 # If nothing found, log a warning
                 logger.warning(f"No suitable {opt_type} strike found based on criteria.")

        return selected_strikes

    def fetch_and_process_data(self):
        """
        Fetches the latest spot price, options data, analyzes it, and triggers processing.
        """
        logger.debug("fetch_and_process_data triggered by schedule.")
        if not self.is_trading:
             logger.debug("fetch_and_process_data: Trading not active, skipping.")
             return

        try:
            if not hasattr(self.order_executor, 'kite') or not self.order_executor.kite:
                 logger.error("KiteConnect instance not found in order_executor. Cannot fetch data. Is live mode enabled?")
                 return

            # 1. Fetch current spot price (e.g., NIFTY 50 Index)
            spot_symbol = f"NSE:{Config.SPOT_SYMBOL}" # e.g., NSE:NIFTY 50
            ltp_data = self.order_executor.kite.ltp([spot_symbol])
            spot_price = ltp_data.get(spot_symbol, {}).get('last_price')
            if spot_price is None:
                logger.warning("Failed to fetch spot price.")
                # Fallback attempt
                ltp_data_fb = self.order_executor.kite.ltp([Config.SPOT_SYMBOL])
                spot_price = ltp_data_fb.get(Config.SPOT_SYMBOL, {}).get('last_price')
                if spot_price is None:
                    logger.error("Failed to fetch spot price even with fallback.")
                    return
            logger.info(f"Current {Config.SPOT_SYMBOL} spot price: {spot_price}")

            # 2. Determine expiry and get instrument tokens for ATM
            # This part might need adjustment based on your exact symbol format
            instruments_data = get_instrument_tokens(symbol=Config.SPOT_SYMBOL, kite_instance=self.order_executor.kite)
            if not instruments_data:
                logger.error("Failed to get instrument tokens for ATM strike.")
                return

            atm_strike = instruments_data['atm_strike']
            expiry = instruments_data['expiry']
            spot_token = instruments_data.get('spot_token')

            logger.info(f"ATM Strike: {atm_strike}, Expiry: {expiry}")

            # 3. Define timeframe for historical data
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=Config.DATA_LOOKBACK_MINUTES) # e.g., 30 mins

            # 4. Fetch historical data for spot (for trend context)
            spot_df = pd.DataFrame()
            if spot_token:
                try:
                    spot_historical = self.order_executor.kite.historical_data(
                        instrument_token=spot_token,
                        from_date=start_time,
                        to_date=end_time,
                        interval="minute"
                    )
                    spot_df = pd.DataFrame(spot_historical) if spot_historical else pd.DataFrame()
                    if not spot_df.empty and 'date' in spot_df.columns:
                        spot_df['date'] = pd.to_datetime(spot_df['date'])
                        spot_df.set_index('date', inplace=True)
                except Exception as e:
                    logger.warning(f"Could not fetch spot historical data: {e}")

            # 5. Fetch historical data for a range of CE and PE options around ATM
            options_data = {}
            strike_range = Config.STRIKE_RANGE # e.g., 4 strikes on either side
            for offset in range(-strike_range, strike_range + 1):
                strike_to_check = atm_strike + (offset * 50) # Assuming 50 point strikes

                # Re-fetch tokens for these strikes
                temp_instruments = get_instrument_tokens(symbol=Config.SPOT_SYMBOL, offset=offset, kite_instance=self.order_executor.kite)
                if not temp_instruments:
                    continue # Skip if couldn't get tokens

                temp_ce_token = temp_instruments.get('ce_token')
                temp_pe_token = temp_instruments.get('pe_token')
                temp_ce_symbol = temp_instruments.get('ce_symbol')
                temp_pe_symbol = temp_instruments.get('pe_symbol')

                for token, symbol in [(temp_ce_token, temp_ce_symbol), (temp_pe_token, temp_pe_symbol)]:
                    if token and symbol:
                        try:
                            hist_data = self.order_executor.kite.historical_data(
                                instrument_token=token,
                                from_date=start_time,
                                to_date=end_time,
                                interval="minute"
                            )
                            if hist_data:
                                df = pd.DataFrame(hist_data)
                                if 'date' in df.columns:
                                    df['date'] = pd.to_datetime(df['date'])
                                    df.set_index('date', inplace=True)
                                options_data[symbol] = df
                                logger.debug(f"Fetched data for {symbol}")
                            else:
                                logger.debug(f"No data returned for {symbol}")
                        except Exception as e:
                            logger.error(f"Error fetching data for {symbol} (Token: {token}): {e}")

            # 6. Analyze options data to select strikes
            # Note: _analyze_options_data relies on 'oi' which might not be in historical data.
            # This is a limitation of the Kite historical data API for options.
            # You might need to fetch LTP for a list of options and analyze those,
            # or use a different data source for OI/Delta.
            # For now, we'll proceed with a simpler selection.
            selected_strikes_info = self._analyze_options_data(spot_price, options_data)

            # Fallback if analysis doesn't yield results or if OI data is unreliable
            if not selected_strikes_info:
                logger.warning("Analysis yielded no strikes or OI data unreliable. Using fallback selection (ATM and ATM+/-1).")
                selected_strikes_info = []
                # Simple fallback: ATM, ATM+50, ATM-50 for both CE and PE
                for offset in [0, 1, -1]:
                    fb_strike = atm_strike + (offset * 50)
                    fb_instruments = get_instrument_tokens(symbol=Config.SPOT_SYMBOL, offset=offset, kite_instance=self.order_executor.kite)
                    if fb_instruments:
                        if fb_instruments.get('ce_symbol') and fb_instruments.get('ce_token'):
                            selected_strikes_info.append({
                                'symbol': fb_instruments['ce_symbol'],
                                'strike': fb_strike,
                                'type': 'CE',
                                'ltp': 0, # Will be fetched or estimated
                                'oi': 0,
                                'oi_change': 0,
                                'delta': 0.5 if offset <= 0 else 0.7 if offset > 0 else 0.3, # Rough estimate
                                'is_atm': offset == 0,
                                'is_otm': offset > 0,
                                'is_itm': offset < 0
                            })
                        if fb_instruments.get('pe_symbol') and fb_instruments.get('pe_token'):
                             selected_strikes_info.append({
                                'symbol': fb_instruments['pe_symbol'],
                                'strike': fb_strike,
                                'type': 'PE',
                                'ltp': 0,
                                'oi': 0,
                                'oi_change': 0,
                                'delta': -0.5 if offset >= 0 else -0.7 if offset < 0 else -0.3, # Rough estimate
                                'is_atm': offset == 0,
                                'is_otm': offset < 0,
                                'is_itm': offset > 0
                            })

            if not selected_strikes_info:
                logger.warning("No strikes selected based on analysis or fallback.")
                return

            # 7. Process data for each selected strike
            for strike_info in selected_strikes_info:
                symbol = strike_info['symbol']
                df = options_data.get(symbol)
                if df is not None and not df.empty:
                    # Pass spot data and strike info for context in strategy
                    self.process_options_bar(symbol, df, spot_df, strike_info)
                else:
                    # If no historical data, we can still try to get LTP and process
                    # This is useful if we are using a different selection method
                    # that doesn't rely on historical OHLC but on current state (like LTP, Quotes)
                    logger.debug(f"No historical data for {symbol}, attempting LTP-based processing.")
                    # Example: Fetch LTP for the symbol
                    try:
                        ltp_option_data = self.order_executor.kite.ltp([f"NFO:{symbol}"])
                        ltp_price = ltp_option_data.get(f"NFO:{symbol}", {}).get('last_price', 0)
                        if ltp_price > 0:
                            # Create a minimal dataframe or pass LTP directly
                            # For simplicity, let's create a single-row df
                            dummy_df = pd.DataFrame([{
                                'date': pd.Timestamp.now(),
                                'last_price': ltp_price,
                                # Add other necessary fields with defaults or Nones
                                'oi': strike_info.get('oi', 0),
                                'volume': 0
                            }]).set_index('date')
                            self.process_options_bar(symbol, dummy_df, spot_df, strike_info)
                        else:
                            logger.warning(f"Could not fetch LTP for {symbol}")
                    except Exception as e:
                        logger.error(f"Error fetching LTP for {symbol}: {e}")


        except Exception as e:
            logger.error(f"Error in fetch_and_process_data (Options): {e}", exc_info=True)


    def process_options_bar(self, symbol: str, ohlc: pd.DataFrame, spot_ohlc: pd.DataFrame, strike_info: Dict[str, Any]) -> None:
        """
        Process a bar of options data.
        This function is called for each selected option contract.
        """
        logger.debug(f"process_options_bar called for {symbol}. Trading active: {self.is_trading}")
        if not self.is_trading:
            logger.debug("process_options_bar: Trading not active, returning.")
            return
        # Might need fewer bars for options or LTP-based logic
        # if ohlc is None or len(ohlc) < 10:
        #     logger.debug("Insufficient data to process options bar.")
        #     return

        try:
            # --- Time Filter Check (if applicable) ---
            # Use the last available timestamp
            if not ohlc.empty:
                ts = ohlc.index[-1]
            else:
                ts = pd.Timestamp.now() # Fallback if df is empty (e.g., LTP only)
            current_time_str = ts.strftime("%H:%M")
            if Config.TIME_FILTER_START and Config.TIME_FILTER_END:
                if current_time_str < Config.TIME_FILTER_START or current_time_str > Config.TIME_FILTER_END:
                    logger.debug(f"Time filter active. Skipping bar for {symbol}.")
                    return

            # Options use 'last_price' from Kite
            if not ohlc.empty:
                current_price = float(ohlc.iloc[-1].get("last_price", 0))
            else:
                # If df is empty (e.g., from LTP fetch), use strike_info or a default
                current_price = strike_info.get('ltp', 0)
                if current_price == 0:
                    logger.warning(f"Could not determine current price for {symbol}")
                    return

            logger.debug(f"Processing {symbol} bar at {ts}, price: {current_price}")

            # --- Call Strategy ---
            # Pass the options OHLC, spot OHLC, and strike information to the strategy
            # NOTE: You will need to implement or adapt `generate_options_signal` in your strategy.
            # Placeholder: Using the existing strategy method, which might not be suitable.
            # A new method `generate_options_signal` should be created.
            # signal = self.strategy.generate_options_signal(ohlc, spot_ohlc, strike_info, current_price)
            # For now, using a dummy signal generation based on price action
            # --- Dummy Signal Generation (Replace with real strategy) ---
            signal = self._generate_dummy_options_signal(ohlc, spot_ohlc, strike_info, current_price)
            # --- End Dummy Signal ---

            logger.debug(f"Strategy returned signal for {symbol}: {signal}")
            if not signal:
                logger.debug(f"No signal generated by strategy for {symbol}.")
                return

            signal_confidence = float(signal.get("confidence", 0.0))
            logger.debug(f"Signal confidence for {symbol}: {signal_confidence}, Threshold: {Config.CONFIDENCE_THRESHOLD}")
            if signal_confidence < Config.CONFIDENCE_THRESHOLD:
                logger.debug(f"Signal confidence below threshold for {symbol}, discarding.")
                return

            # --- Position Sizing ---
            # Might need options-specific position sizing logic
            position = self.risk_manager.calculate_position_size(
                entry_price=signal.get("entry_price", current_price),
                stop_loss=signal.get("stop_loss", current_price),
                signal_confidence=signal.get("confidence", 0.0),
                market_volatility=signal.get("market_volatility", 0.0), # Extract from options data if available
            )
            logger.debug(f"Position sizing returned for {symbol}: {position}")
            if not position or position.get("quantity", 0) <= 0:
                logger.debug(f"Position sizing failed for {symbol} or quantity is zero/negative.")
                return

            # --- Telegram Alert ---
            token = len(self.trades) + 1
            self.telegram_controller.send_signal_alert(token, signal, position)

            # --- Order Placement ---
            # Options are typically bought
            order_transaction_type = "BUY"

            logger.debug(f"Attempting to place entry order for {symbol}. Type: {order_transaction_type}, Qty: {position['quantity']}")
            order_id = self.order_executor.place_entry_order(
                symbol=symbol, # Full trading symbol from Kite
                exchange="NFO",
                transaction_type=order_transaction_type,
                quantity=position["quantity"],
            )
            if not order_id:
                logger.warning(f"Failed to place entry order for {symbol}.")
                return

            # --- GTT Orders (Stop Loss / Target) ---
            # Note: GTTs for options might work differently, check Kite documentation
            logger.debug("Attempting to setup GTT orders...")
            self.order_executor.setup_gtt_orders(
                entry_order_id=order_id,
                entry_price=signal.get("entry_price", current_price),
                stop_loss_price=signal.get("stop_loss", current_price),
                target_price=signal.get("target", current_price),
                symbol=symbol,
                exchange="NFO",
                quantity=position["quantity"],
                transaction_type=order_transaction_type, # Should match entry
            )

            # --- Record Trade ---
            self.trades.append({
                "order_id": order_id,
                "symbol": symbol,
                "direction": order_transaction_type,
                "quantity": position["quantity"],
                "entry_price": signal.get("entry_price", current_price),
                "stop_loss": signal.get("stop_loss", current_price),
                "target": signal.get("target", current_price),
                "confidence": signal.get("confidence", 0.0),
                "strike_info": strike_info # Store for reference
            })
            logger.info(f"✅ Options Trade recorded: {order_transaction_type} {position['quantity']}x {symbol} @ {signal.get('entry_price', current_price)}")

        except Exception as exc:
            logger.error(f"Error processing options bar for {symbol}: {exc}", exc_info=True)

    def _generate_dummy_options_signal(self, ohlc: pd.DataFrame, spot_ohlc: pd.DataFrame, strike_info: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        """
        A dummy signal generator for testing. Replace with your actual strategy logic.
        This is a very basic example.
        """
        try:
            signal_dict = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "target": None,
                "confidence": 0.0,
                "market_volatility": 0.0
            }

            # 1. Check if selected based on OI/Delta (already done in trader, but can double-check)
            # if strike_info.get('oi_change') < some_threshold:
            #     return None

            # 2. Analyze Options Price Action (if OHLC data is available)
            if ohlc is not None and len(ohlc) >= 2:
                # Example: Bullish breakout in option price on high volume (conceptual)
                last_close = ohlc['last_price'].iloc[-2] # Use 'last_price' for options
                current_close = ohlc['last_price'].iloc[-1]
                # Kite historical might not have volume for options, check
                last_volume = ohlc['volume'].iloc[-2] if 'volume' in ohlc.columns else 1
                current_volume = ohlc['volume'].iloc[-1] if 'volume' in ohlc.columns else 1

                avg_volume = ohlc['volume'][-10:-1].mean() if 'volume' in ohlc.columns and len(ohlc) > 10 else 1 # Avg of last 10 bars (excluding current)

                # Simple breakout logic
                if (current_close > last_close * 1.005) and (current_volume > avg_volume * 1.5):
                     signal_dict["signal"] = "BUY"
                     signal_dict["stop_loss"] = current_close * 0.98 # 2% SL
                     signal_dict["target"] = current_close * 1.05 # 5% Target
                     signal_dict["confidence"] = 8.5 # Set based on strength of pattern
                     # Adjust confidence based on spot trend, delta, etc.
                     if spot_ohlc is not None and not spot_ohlc.empty:
                         # Example: Boost confidence if spot is also trending up for CE
                         if strike_info['type'] == 'CE':
                             if len(spot_ohlc) >= 5:
                                 spot_return = (spot_ohlc['close'].iloc[-1] / spot_ohlc['close'].iloc[-5]) - 1
                                 if spot_return > 0.002: # 0.2% spot up
                                     signal_dict["confidence"] += 1.0
                         elif strike_info['type'] == 'PE':
                             if len(spot_ohlc) >= 5:
                                 spot_return = (spot_ohlc['close'].iloc[-1] / spot_ohlc['close'].iloc[-5]) - 1
                                 if spot_return < -0.002: # 0.2% spot down
                                     signal_dict["confidence"] += 1.0

                     logger.debug(f"Generated BUY signal for {strike_info.get('symbol', 'Unknown')}")
                     return signal_dict

            # Add more logic for PE, different patterns, etc.
            # If no condition met, return None
            return None

        except Exception as e:
            logger.error(f"Error generating dummy options signal: {e}", exc_info=True)
            return None


    # Keep the original process_bar for potential future use or mixed strategies
    def process_bar(self, ohlc: pd.DataFrame) -> None:
         # ... (original logic for futures/single asset) ...
         # This is the original method from Pasted_Text_1754457091365.txt
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
             current_time_str = ts.strftime("%H:%M")
             if Config.TIME_FILTER_START and Config.TIME_FILTER_END:
                 if current_time_str < Config.TIME_FILTER_START or current_time_str > Config.TIME_FILTER_END:
                     logger.debug(f"Time filter active ({Config.TIME_FILTER_START} - {Config.TIME_FILTER_END}). Current time {current_time_str} is outside range, skipping bar.")
                     return
             current_price = float(ohlc.iloc[-1]["close"])
             logger.debug(f"Current bar timestamp: {ts}, price: {current_price}")
             signal = self.strategy.generate_signal(ohlc, current_price)
             logger.debug(f"Strategy returned signal: {signal}")
             if not signal:
                 logger.debug("No signal generated by strategy.")
                 return
             signal_confidence = float(signal.get("confidence", 0.0))
             logger.debug(f"Signal confidence: {signal_confidence}, Threshold: {Config.CONFIDENCE_THRESHOLD}")
             if signal_confidence < Config.CONFIDENCE_THRESHOLD:
                 logger.debug("Signal confidence below threshold, discarding.")
                 return
             position = self.risk_manager.calculate_position_size(
                 entry_price=signal.get("entry_price", current_price),
                 stop_loss=signal.get("stop_loss", current_price),
                 signal_confidence=signal.get("confidence", 0.0),
                 market_volatility=signal.get("market_volatility", 0.0),
             )
             logger.debug(f"Position sizing returned: {position}")
             if not position or position.get("quantity", 0) <= 0:
                 logger.debug("Position sizing failed or quantity is zero/negative.")
                 return
             token = len(self.trades) + 1
             self.telegram_controller.send_signal_alert(token, signal, position)
             transaction_type = signal.get("signal") or signal.get("direction")
             if not transaction_type:
                 logger.warning("Missing signal direction.")
                 return
             symbol = getattr(Config, "TRADE_SYMBOL", "NIFTY50")
             exchange = getattr(Config, "TRADE_EXCHANGE", "NFO")
             logger.debug(f"Attempting to place entry order. Symbol: {symbol}, Exchange: {exchange}, Type: {transaction_type}, Qty: {position['quantity']}")
             order_id = self.order_executor.place_entry_order(
                 symbol=symbol,
                 exchange=exchange,
                 transaction_type=transaction_type,
                 quantity=position["quantity"],
             )
             if not order_id:
                 logger.warning("Failed to place entry order.")
                 return
             logger.debug("Attempting to setup GTT orders...")
             self.order_executor.setup_gtt_orders(
                 entry_order_id=order_id,
                 entry_price=signal.get("entry_price", current_price),
                 stop_loss_price=signal.get("stop_loss", current_price),
                 target_price=signal.get("target", current_price),
                 symbol=symbol,
                 exchange=exchange,
                 quantity=position["quantity"],
                 transaction_type=transaction_type,
             )
             self.trades.append({
                 "order_id": order_id,
                 "direction": transaction_type,
                 "quantity": position["quantity"],
                 "entry_price": signal.get("entry_price", current_price),
                 "stop_loss": signal.get("stop_loss", current_price),
                 "target": signal.get("target", current_price),
                 "confidence": signal.get("confidence", 0.0),
             })
             logger.info(f"✅ Trade recorded: {transaction_type} {position['quantity']} @ {signal.get('entry_price', current_price)}")
         except Exception as exc:
             logger.error("Error processing bar: %s", exc, exc_info=True)
         # ... (rest of original logic) ...


    def get_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "is_trading": self.is_trading,
            "open_orders": len(self.order_executor.get_active_orders()),
            "trades_today": len(self.trades),
            "live_mode": self.live_mode,
        }
        status.update(self.risk_manager.get_risk_status())
        return status

    def get_summary(self) -> str:
        lines = [
            f"📊 <b>Daily Summary</b>",
            f"🔁 <b>Total trades:</b> {len(self.trades)}",
            f"💰 <b>PNL:</b> {self.daily_pnl:.2f}",
            "────────────────────"
        ]
        for trade in self.trades:
            # Adjust summary for options
            lines.append(
                f"{trade.get('symbol', 'N/A')} {trade['direction']} {trade['quantity']} @ {trade['entry_price']:.2f} "
                f"(SL {trade['stop_loss']:.2f}, TP {trade['target']:.2f})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<RealTimeTrader is_trading={self.is_trading} "
                f"live_mode={self.live_mode} trades_today={len(self.trades)}>")
