import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from collections import deque
from utils import TechnicalIndicators
from config import Config

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Generate trading signals based on technical indicators.
    This class is stateful and maintains a history of market data.
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.signal_history: List[Dict] = []
        self.adaptive_threshold = Config.SIGNAL_THRESHOLD
        
        # FIX: Maintain a stateful history of market data (DataFrame)
        self.history_size = max(Config.RSI_PERIOD, Config.EMA_SLOW, Config.BB_PERIOD, Config.VOL_SMA_PERIOD) + 50
        self.historical_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.min_data_size = max(Config.RSI_PERIOD, Config.EMA_SLOW, Config.BB_PERIOD) + 2 # Minimum data needed for calculations

    def _update_history(self, market_data: Dict[str, any]):
        """Appends new market data to the historical DataFrame and trims it."""
        new_row = {
            'timestamp': market_data.get('timestamp', pd.Timestamp.now()),
            'open': market_data.get('open', market_data['ltp']),
            'high': market_data.get('high', market_data['ltp']),
            'low': market_data.get('low', market_data['ltp']),
            'close': market_data['ltp'],
            'volume': market_data.get('volume', 0)
        }
        
        # Use concat for appending new row, which is more robust with modern pandas
        self.historical_data = pd.concat([self.historical_data, pd.DataFrame([new_row])], ignore_index=True)
        
        # Trim the DataFrame to maintain the desired history size
        if len(self.historical_data) > self.history_size:
            self.historical_data = self.historical_data.iloc[-self.history_size:]

    def calculate_all_indicators(self) -> Dict[str, float]:
        """Calculate all technical indicators using the stored historical data."""
        # Use the stateful DataFrame `self.historical_data`
        df = self.historical_data
        
        if df.empty or len(df) < self.min_data_size:
            logger.debug(f"Not enough data to calculate indicators. Have {len(df)}, need {self.min_data_size}.")
            return {}
        
        indicators = {}
        try:
            close_prices = df['close']
            
            # RSI, EMAs, MACD, Bollinger Bands, ATR
            indicators['rsi'] = self.indicators.calculate_rsi(close_prices, Config.RSI_PERIOD)
            indicators['ema_fast'] = self.indicators.calculate_ema(close_prices, Config.EMA_FAST)
            indicators['ema_slow'] = self.indicators.calculate_ema(close_prices, Config.EMA_SLOW)
            indicators.update(self.indicators.calculate_macd(close_prices, Config.MACD_FAST, Config.MACD_SLOW, Config.MACD_SIGNAL))
            indicators.update(self.indicators.calculate_bollinger_bands(close_prices, Config.BB_PERIOD, Config.BB_STDDEV))
            indicators['atr'] = self.indicators.calculate_atr(df, Config.ATR_PERIOD)
            
            # Volume indicators
            if 'volume' in df.columns and df['volume'].sum() > 0:
                vol_sma = df['volume'].rolling(window=Config.VOL_SMA_PERIOD).mean().iloc[-1]
                indicators['volume_sma'] = vol_sma
                indicators['volume_ratio'] = (df['volume'].iloc[-1] / vol_sma) if vol_sma > 0 else 1.0
                
                # VWAP
                vwap_window = df.tail(Config.VWAP_WINDOW) if Config.VWAP_WINDOW > 0 else df
                total_volume = vwap_window['volume'].sum()
                if total_volume > 0:
                    indicators['vwap'] = (vwap_window['close'] * vwap_window['volume']).sum() / total_volume
                else:
                    indicators['vwap'] = df['close'].iloc[-1]

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return {}
        
        return indicators

    def generate_signal(self, market_data: Dict[str, any]) -> Optional[Dict[str, any]]:
        """
        Main entry point. Updates history, calculates indicators, and generates a trade signal.
        """
        try:
            if 'ltp' not in market_data:
                return None
            
            # FIX: Update the historical data with the new tick
            self._update_history(market_data)
            
            # FIX: Check if we have enough historical data to proceed
            if len(self.historical_data) < self.min_data_size:
                return None

            current_price = market_data['ltp']
            
            # FIX: Calculate indicators on the full historical DataFrame
            indicators = self.calculate_all_indicators()
            if not indicators:
                return None # Not enough data or an error occurred

            # --- The rest of the logic remains largely the same, but is now using valid data ---

            trend_signal = self.generate_trend_signal(indicators)
            momentum_signal = self.generate_momentum_signal(indicators)
            volume_signal = self.generate_volume_signal(indicators, current_price)
            mean_reversion_signal = self.generate_mean_reversion_signal(indicators, current_price)
            
            total_signal = (
                trend_signal * 0.4 +
                momentum_signal * 0.3 +
                volume_signal * 0.2 +
                mean_reversion_signal * 0.1
            )
            
            signal_components = {
                'trend': trend_signal, 'momentum': momentum_signal,
                'volume': volume_signal, 'mean_reversion': mean_reversion_signal,
                'total': total_signal
            }
            
            should_trade, reason = self.should_trade(total_signal, signal_components)
            if not should_trade:
                logger.debug(f"No trade signal: {reason}")
                return None
            
            direction = "BUY" if total_signal > 0 else "SELL"
            
            # FIX: Ensure ATR is valid before using it for SL/TP
            atr = indicators.get('atr', 0)
            if atr <= 0:
                logger.warning("ATR is zero or invalid. Falling back to percentage-based SL/TP.")
                # Use a default ATR based on a small percentage of price as a fallback
                atr = current_price * (Config.SL_PERCENT / 100) / Config.ATR_SL_MULT

            stop_loss, target = self.get_stop_loss_target(current_price, direction, atr)
            
            signal = {
                'direction': direction, 'entry_price': current_price,
                'stop_loss': stop_loss, 'target': target,
                'strength': abs(total_signal),
                'timestamp': market_data.get('timestamp', pd.Timestamp.now()),
                'components': signal_components
            }
            
            self.signal_history.append(signal)
            logger.info(f"Generated {direction} signal at ₹{current_price:.2f} (Strength: {total_signal:.2f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error in generate_signal: {e}", exc_info=True)
            return None

    # --- Other helper methods (generate_trend_signal, should_trade, etc.) remain unchanged ---
    # --- They are correct, they just needed valid input data to work with. ---
    # --- You should copy them from your original file to complete this one. ---

    def generate_trend_signal(self, indicators: Dict[str, float]) -> float:
        """Generate trend-following signal"""
        signal = 0.0
        if 'ema_fast' in indicators and 'ema_slow' in indicators:
            if indicators['ema_fast'] > indicators['ema_slow']:
                signal += 2.0
            else:
                signal -= 2.0
        if 'macd' in indicators and 'signal' in indicators:
            if indicators['macd'] > indicators['signal']:
                signal += 1.5
            else:
                signal -= 1.5
        return signal
    
    def generate_momentum_signal(self, indicators: Dict[str, float]) -> float:
        """Generate momentum-based signal"""
        signal = 0.0
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi > 70: signal -= 1.0
            elif rsi < 30: signal += 1.0
            elif rsi > 60: signal += 0.5
            elif rsi < 40: signal -= 0.5
        if 'histogram' in indicators:
            if indicators['histogram'] > 0: signal += 0.5
            else: signal -= 0.5
        return signal
    
    def generate_volume_signal(self, indicators: Dict[str, float], current_price: float) -> float:
        """Generate volume-based confirmation signal"""
        signal = 0.0
        if 'volume_ratio' in indicators and indicators.get('vwap'):
            if indicators['volume_ratio'] > 1.5:
                if current_price > indicators['vwap']: signal += 0.5
                else: signal -= 0.5
        return signal
    
    def generate_mean_reversion_signal(self, indicators: Dict[str, float], current_price: float) -> float:
        """Generate mean reversion signal using Bollinger Bands"""
        signal = 0.0
        if all(k in indicators for k in ['upper', 'lower', 'middle']):
            if current_price <= indicators['lower']: signal += 1.0
            elif current_price >= indicators['upper']: signal -= 1.0
        return signal

    def should_trade(self, signal_strength: float, signal_components: Dict[str, float]) -> Tuple[bool, str]:
        """Determine if we should trade based on signal strength"""
        abs_signal = abs(signal_strength)
        if abs_signal < self.adaptive_threshold:
            return False, f"Signal too weak: {abs_signal:.2f} < {self.adaptive_threshold:.2f}"
        
        trend_signal = signal_components.get('trend', 0)
        momentum_signal = signal_components.get('momentum', 0)
        if (trend_signal > 0 and momentum_signal < -1) or (trend_signal < 0 and momentum_signal > 1):
            return False, "Trend and momentum divergence"
        
        direction = "BUY" if signal_strength > 0 else "SELL"
        return True, f"Trade signal: {direction} (Strength: {signal_strength:.2f})"

    def get_stop_loss_target(self, entry_price: float, direction: str, atr: float) -> Tuple[float, float]:
        """Calculate stop loss and target prices"""
        if Config.USE_ATR_SL and atr > 0:
            sl_offset = atr * Config.ATR_SL_MULT
            tp_offset = atr * Config.ATR_TP_MULT
        else:
            sl_offset = entry_price * (Config.SL_PERCENT / 100)
            tp_offset = entry_price * (Config.TP_PERCENT / 100)

        if direction == "BUY":
            stop_loss = entry_price - sl_offset
            target = entry_price + tp_offset
        else:  # SELL
            stop_loss = entry_price + sl_offset
            target = entry_price - tp_offset
        
        return round(stop_loss, 2), round(target, 2)

