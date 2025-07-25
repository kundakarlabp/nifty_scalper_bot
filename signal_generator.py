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
        # ALIGNED: Uses new Config variables
        self.history_size = max(Config.EMA_SLOW, Config.RSI_PERIOD) + 100
        self.historical_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.min_data_size = max(Config.EMA_SLOW, Config.RSI_PERIOD) + 5

    def _update_history(self, market_data: Dict[str, any]):
        """Appends new market data to the historical DataFrame."""
        new_row = {
            'timestamp': market_data.get('timestamp'),
            'open': market_data.get('open'),
            'high': market_data.get('high'),
            'low': market_data.get('low'),
            'close': market_data.get('ltp'),
            'volume': market_data.get('volume')
        }
        self.historical_data = pd.concat([self.historical_data, pd.DataFrame([new_row])], ignore_index=True)
        if len(self.historical_data) > self.history_size:
            self.historical_data = self.historical_data.iloc[-self.history_size:]

    def _get_market_regime(self, indicators: dict) -> str:
        """Filter 1: Determines the current market regime."""
        ema_fast = indicators.get('ema_fast', 0)
        ema_slow = indicators.get('ema_slow', 0)
        atr_norm = indicators.get('atr_norm', 0)

        # Define thresholds for trend detection
        ema_spread_threshold_strong = 0.005 # 0.5% spread for strong trend
        atr_norm_threshold_strong = 0.015   # 1.5% normalized ATR for strong trend

        ema_spread_threshold_weak = 0.002   # 0.2% spread for weak trend
        atr_norm_threshold_weak = 0.01      # 1.0% normalized ATR for weak trend

        if abs(ema_fast - ema_slow) / ema_slow > ema_spread_threshold_strong and atr_norm > atr_norm_threshold_strong:
            return "Strong Trend"
        elif abs(ema_fast - ema_slow) / ema_slow > ema_spread_threshold_weak and atr_norm > atr_norm_threshold_weak:
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
        indicators['atr_norm'] = indicators['atr'] / indicators['ema_slow']

        # --- Filter 1: Determine Market Regime ---
        regime = self._get_market_regime(indicators)
        
        # --- Filter 2: Signal Confirmation ---
        direction = None
        if regime == "Strong Trend":
            if indicators['ema_fast'] > indicators['ema_slow'] and 45 < indicators['rsi'] < 55:
                direction = "BUY"
            elif indicators['ema_fast'] < indicators['ema_slow'] and 55 > indicators['rsi'] > 45:
                direction = "SELL"
        elif regime == "Weak Trend":
            if indicators['ema_fast'] > indicators['ema_slow'] and indicators['rsi'] > 60:
                direction = "BUY"
            elif indicators['ema_fast'] < indicators['ema_slow'] and indicators['rsi'] < 40:
                direction = "SELL"
        
        if not direction:
            return None

        # --- Filter 3: Define Execution Plan ---
        option_type = "CE" if direction == "BUY" else "PE"
        underlying_price = market_data['ltp']
        
        atr = indicators['atr']
        if atr <= 0:
            logger.warning("ATR is zero, cannot generate dynamic SL/TP. Skipping signal.")
            return None
            
        sl_points = atr * Config.ATR_SL_MULT
        tp_points = atr * Config.ATR_TP_MULT
        
        if direction == "BUY":
            underlying_stop_loss = underlying_price - sl_points
            underlying_target = underlying_price + tp_points
        else:
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
        return signal
