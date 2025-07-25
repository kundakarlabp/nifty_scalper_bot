import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from utils import TechnicalIndicators
from config import Config

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Generates high-conviction trading signals by requiring alignment across
    multiple technical indicators. This version is stateful.
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.history_size = max(Config.EMA_SLOW, Config.RSI_PERIOD, Config.BB_PERIOD) + 50
        self.historical_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.min_data_size = max(Config.EMA_SLOW, Config.RSI_PERIOD) + 2

    def _update_history(self, market_data: Dict[str, any]):
        """Appends new market data to the historical DataFrame."""
        new_row = {
            'timestamp': market_data.get('timestamp', pd.Timestamp.now()),
            'open': market_data.get('open', market_data['ltp']),
            'high': market_data.get('high', market_data['ltp']),
            'low': market_data.get('low', market_data['ltp']),
            'close': market_data['ltp'],
            'volume': market_data.get('volume', 0)
        }
        self.historical_data = pd.concat([self.historical_data, pd.DataFrame([new_row])], ignore_index=True)
        if len(self.historical_data) > self.history_size:
            self.historical_data = self.historical_data.iloc[-self.history_size:]

    def generate_signal(self, market_data: Dict[str, any]) -> Optional[Dict[str, any]]:
        """Main entry point. Updates history, calculates indicators, and generates a trade signal."""
        try:
            if 'ltp' not in market_data: return None
            
            self._update_history(market_data)
            if len(self.historical_data) < self.min_data_size:
                logger.debug(f"Collecting data... {len(self.historical_data)}/{self.min_data_size} points.")
                return None

            current_price = market_data['ltp']
            
            # --- Calculate all necessary indicators ---
            close_prices = self.historical_data['close']
            
            ema_fast = self.indicators.calculate_ema(close_prices, Config.EMA_FAST)
            ema_slow = self.indicators.calculate_ema(close_prices, Config.EMA_SLOW)
            rsi = self.indicators.calculate_rsi(close_prices, Config.RSI_PERIOD)
            macd_data = self.indicators.calculate_macd(close_prices, Config.MACD_FAST, Config.MACD_SLOW, Config.MACD_SIGNAL)
            atr = self.indicators.calculate_atr(self.historical_data, Config.ATR_PERIOD)

            # --- Define High-Conviction Trade Conditions ---
            
            # Bullish (BUY) Conditions: All must be true
            is_uptrend = ema_fast > ema_slow
            has_bullish_momentum = rsi > 55  # Stricter than 50
            has_macd_confirmation = macd_data['macd'] > macd_data['signal'] and macd_data['histogram'] > 0

            # Bearish (SELL) Conditions: All must be true
            is_downtrend = ema_fast < ema_slow
            has_bearish_momentum = rsi < 45  # Stricter than 50
            has_macd_confirmation_sell = macd_data['macd'] < macd_data['signal'] and macd_data['histogram'] < 0

            direction = None
            if is_uptrend and has_bullish_momentum and has_macd_confirmation:
                direction = "BUY"
            elif is_downtrend and has_bearish_momentum and has_macd_confirmation_sell:
                direction = "SELL"

            if not direction:
                return None # No clear signal where all conditions align

            # --- We have a valid signal, now prepare the trade details ---
            logger.info(f"High-conviction {direction} signal detected at price {current_price}")

            stop_loss, target = self.get_stop_loss_target(current_price, direction, atr)
            
            if stop_loss == target: # Avoid trades with no profit potential
                logger.warning("Stopping trade: Stop loss and target are the same.")
                return None

            signal = {
                'direction': direction,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'target': target,
                'timestamp': market_data.get('timestamp', pd.Timestamp.now()),
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in generate_signal: {e}", exc_info=True)
            return None

    def get_stop_loss_target(self, entry_price: float, direction: str, atr: float) -> Tuple[float, float]:
        """Calculate stop loss and target prices using ATR for risk management."""
        if atr <= 0:
            # Fallback to percentage if ATR is not available or zero
            logger.warning("ATR is zero or invalid. Using percentage-based SL/TP.")
            sl_offset = entry_price * (Config.SL_PERCENT / 100)
            tp_offset = entry_price * (Config.TP_PERCENT / 100)
        else:
            # Use ATR for dynamic risk management
            sl_offset = atr * Config.ATR_SL_MULT
            tp_offset = atr * Config.ATR_TP_MULT

        if direction == "BUY":
            stop_loss = entry_price - sl_offset
            target = entry_price + tp_offset
        else:  # SELL
            stop_loss = entry_price + sl_offset
            target = entry_price - tp_offset
        
        # Round to 2 decimal places, common for Indian markets
        return round(stop_loss, 2), round(target, 2)

