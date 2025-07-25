# signal_generator.py

import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from utils import TechnicalIndicators
from config import Config

logger = logging.getLogger(__name__)

class SignalGenerator:
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.history_size = max(Config.EMA_SLOW, Config.RSI_PERIOD) + 100
        self.historical_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.min_data_size = max(Config.EMA_SLOW, Config.RSI_PERIOD) + 5

    def _update_history(self, market_data: Dict[str, any]):
        """Appends new market data to the historical DataFrame."""
        new_row = {'timestamp': market_data['timestamp'], 'open': market_data['open'], 'high': market_data['high'], 'low': market_data['low'], 'close': market_data['ltp'], 'volume': market_data['volume']}
        self.historical_data = pd.concat([self.historical_data, pd.DataFrame([new_row])], ignore_index=True)
        if len(self.historical_data) > self.history_size:
            self.historical_data = self.historical_data.iloc[-self.history_size:]

    def _get_market_regime(self, indicators: dict) -> str:
        """Filter 1: Determines the current market regime."""
        ema_fast = indicators['ema_fast']
        ema_slow = indicators['ema_slow']
        atr_norm = indicators['atr_norm'] # Normalized ATR

        if abs(ema_fast - ema_slow) / ema_slow > 0.005 and atr_norm > 0.015:
            return "Strong Trend"
        elif abs(ema_fast - ema_slow) / ema_slow > 0.002 and atr_norm > 0.01:
            return "Weak Trend"
        else:
            return "Sideways"

    def generate_signal(self, market_data: Dict[str, any]) -> Optional[Dict[str, any]]:
        """
        Main signal generation function using the three-filter system.
        Returns a complete, actionable trade signal dictionary.
        """
        self._update_history(market_data)
        if len(self.historical_data) < self.min_data_size:
            logger.debug(f"Collecting data... {len(self.historical_data)}/{self.min_data_size} points.")
            return None

        # --- Calculate all indicators at once ---
        close_prices = self.historical_data['close']
        indicators = {
            'ema_fast': self.indicators.calculate_ema(close_prices, Config.EMA_FAST),
            'ema_slow': self.indicators.calculate_ema(close_prices, Config.EMA_SLOW),
            'rsi': self.indicators.calculate_rsi(close_prices, Config.RSI_PERIOD),
            'atr': self.indicators.calculate_atr(self.historical_data, Config.ATR_PERIOD),
        }
        if indicators['ema_slow'] == 0: return None # Avoid division by zero
        indicators['atr_norm'] = indicators['atr'] / indicators['ema_slow'] # ATR normalized by price

        # --- Filter 1: Determine Market Regime ---
        regime = self._get_market_regime(indicators)
        
        # --- Filter 2: Signal Confirmation ---
        direction = None
        if regime == "Strong Trend":
            # In a strong trend, look for pullback entries
            if indicators['ema_fast'] > indicators['ema_slow'] and 45 < indicators['rsi'] < 55:
                direction = "BUY" # Buy the dip in a strong uptrend
            elif indicators['ema_fast'] < indicators['ema_slow'] and 55 > indicators['rsi'] > 45:
                direction = "SELL" # Sell the rally in a strong downtrend
        elif regime == "Weak Trend":
            # In a weak trend, require stronger momentum confirmation
            if indicators['ema_fast'] > indicators['ema_slow'] and indicators['rsi'] > 60:
                direction = "BUY"
            elif indicators['ema_fast'] < indicators['ema_slow'] and indicators['rsi'] < 40:
                direction = "SELL"
        # Note: We will ignore "Sideways" regime for now to preserve capital, as it's riskier.

        if not direction:
            return None # No high-conviction signal found

        # --- Filter 3: Define Execution Plan (Strike, SL, TP) ---
        option_type = "CE" if direction == "BUY" else "PE"
        underlying_price = market_data['ltp']
        
        # Dynamic Stop-Loss and Target based on Volatility (ATR)
        atr = indicators['atr']
        if atr <= 0:
            logger.warning("ATR is zero, cannot generate dynamic SL/TP. Skipping signal.")
            return None
            
        sl_points = atr * Config.ATR_SL_MULT
        tp_points = atr * Config.ATR_TP_MULT
        
        # For options, the SL/TP is on the underlying, which we translate to the option premium later
        if direction == "BUY":
            underlying_stop_loss = underlying_price - sl_points
            underlying_target = underlying_price + tp_points
        else: # SELL
            underlying_stop_loss = underlying_price + sl_points
            underlying_target = underlying_price - tp_points

        signal = {
            'underlying_price': underlying_price,
            'direction': direction,
            'option_type': option_type,
            'underlying_stop_loss': round(underlying_stop_loss, 2),
            'underlying_target': round(underlying_target, 2),
            'atr': round(atr, 2),
            'timestamp': market_data['timestamp']
        }
        
        logger.critical(f"High-Conviction Signal Generated: {signal}")
        return signal```

#### 3. Modified `nifty_scalper_bot.py`

This file is now updated to use the new `BrokerManager` and the more advanced signal dictionary. It also includes the logic for trailing the stop-loss.

```python
# nifty_scalper_bot.py

import asyncio
import logging
from datetime import datetime
import pytz

from config import Config
from utils import is_market_open, format_currency, get_market_status
from signal_generator import SignalGenerator
from telegram_bot import TelegramBot
from broker_manager import BrokerManager # Import the new BrokerManager

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
IST = pytz.timezone('Asia/Kolkata')

# (RiskManager class remains the same as before, no changes needed)
class RiskManager:
    # ... (copy the RiskManager class from the previous response) ...
    def __init__(self, initial_balance, telegram_bot=None):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.todays_pnl = 0.0
        # ... rest of the class ...

class NiftyScalperBot:
    def __init__(self):
        self.signal_generator = SignalGenerator()
        self.broker = BrokerManager() # Use the new BrokerManager
        self.telegram_bot = TelegramBot(trading_bot_instance=self)
        self.risk_manager = RiskManager(Config.INITIAL_CAPITAL, telegram_bot=self.telegram_bot)
        self.current_position = None
        self.auto_trade = True
        self.last_reset_day = -1

    async def run(self):
        logger.info("Nifty Scalper Bot v3.0 (Advanced) is starting...")
        telegram_task = asyncio.create_task(self.telegram_bot.start_bot())

        while True:
            try:
                if is_market_open():
                    market_data = self.get_market_data()
                    if self.current_position:
                        self.manage_open_position(market_data)
                    elif self.auto_trade and self.risk_manager.can_trade():
                        signal = self.signal_generator.generate_signal(market_data)
                        if signal:
                            await self.execute_trade(signal)
                await asyncio.sleep(Config.TICK_INTERVAL_SECONDS)
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

    def get_market_data(self):
        # *** CRITICAL PLACEHOLDER *** - MUST BE REPLACED WITH REAL DATA FEED
        import random
        price = 25000 + random.uniform(-50, 50)
        return {'ltp': price, 'volume': random.randint(10000, 50000), 'open': price-10, 'high': price+10, 'low': price-10, 'timestamp': datetime.now(IST)}

    async def execute_trade(self, signal: dict):
        """Converts a signal into a live GTT OCO trade."""
        instrument = self.broker.get_instrument_for_option(signal['underlying_price'], signal['option_type'])
        if not instrument:
            logger.error("Could not determine option instrument. Skipping trade.")
            return

        option_ltp = self.broker.get_ltp(instrument)
        if option_ltp <= 0:
            logger.error(f"Invalid LTP ({option_ltp}) for {instrument}. Skipping trade.")
            return

        # Translate underlying SL/TP to option premium SL/TP (simplified 0.5 delta)
        delta_factor = 0.5
        underlying_sl_points = abs(signal['underlying_price'] - signal['underlying_stop_loss'])
        underlying_tp_points = abs(signal['underlying_price'] - signal['underlying_target'])
        
        option_sl_points = underlying_sl_points * delta_factor
        option_tp_points = underlying_tp_points * delta_factor

        option_stop_loss = option_ltp - option_sl_points
        option_target = option_ltp + option_tp_points

        quantity = self.risk_manager.calculate_position_size(underlying_sl_points)
        if quantity <= 0: return

        order_id = self.broker.place_gtt_oco_order(instrument, "BUY", quantity, option_ltp, option_target, option_stop_loss)
        if not order_id:
            logger.error("Failed to place GTT OCO order.")
            return

        self.current_position = {
            'order_id': order_id,
            'instrument': instrument,
            'quantity': quantity,
            'entry_price': option_ltp,
            'stop_loss': option_stop_loss,
            'target': option_target,
            'trailing_sl_trigger_price': option_ltp + (option_tp_points / 2), # Start trailing after 50% of target is hit
            'trailing_sl_new_price': option_ltp, # Initial trail price is entry price
            'underlying_entry': signal['underlying_price'],
        }
        logger.critical(f"NEW GTT OCO POSITION PLACED: {self.current_position}")
        # self.telegram_bot.notify_trade_entry(...) # You can enhance this notification

    def manage_open_position(self, market_data: dict):
        """Manages trailing stop-loss for an open GTT order."""
        pos = self.current_position
        option_ltp = self.broker.get_ltp(pos['instrument'])

        # Trailing Stop-Loss Logic
        if option_ltp >= pos['trailing_sl_trigger_price']:
            # New SL is current price minus the original SL distance, ensuring we lock in profit
            new_sl = option_ltp - (pos['entry_price'] - pos['stop_loss'])
            
            # We only move the SL up, never down
            if new_sl > pos['stop_loss']:
                logger.info(f"Trailing Stop-Loss Triggered for {pos['instrument']}!")
                self.broker.modify_order_to_trail_sl(pos['order_id'], new_sl)
                pos['stop_loss'] = new_sl # Update the position's SL
                # Set the next trigger higher to continue trailing
                pos['trailing_sl_trigger_price'] = option_ltp + (pos['target'] - pos['entry_price']) * 0.1 # Move trigger by 10% increments

    # ... (The rest of the NiftyScalperBot and the main execution block remain the same) ...
    # ... (You need to add the logic to clear self.current_position when a GTT executes) ...
    # ... (This usually requires a postback/webhook from the broker, or periodic status checks) ...
