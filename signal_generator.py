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
        
        # Define the size of the historical data window needed for indicators
        self.history_size = max(Config.RSI_PERIOD, Config.EMA_SLOW, Config.BB_PERIOD, Config.VOL_SMA_PERIOD) + 50
        self.historical_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Define the minimum number of data points required before generating signals
        self.min_data_size = max(Config.RSI_PERIOD, Config.EMA_SLOW, Config.BB_PERIOD) + 2

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
        self.historical_data = pd.concat([self.historical_data, pd.DataFrame([new_row])], ignore_index=True)
        
        # Trim the DataFrame to maintain the desired history size
        if len(self.historical_data) > self.history_size:
            self.historical_data = self.historical_data.iloc[-self.history_size:]

    def calculate_all_indicators(self) -> Dict[str, float]:
        """Calculate all technical indicators using the stored historical data."""
        df = self.historical_data
        if df.empty or len(df) < self.min_data_size:
            return {}
        
        indicators = {}
        try:
            close_prices = df['close']
            indicators['rsi'] = self.indicators.calculate_rsi(close_prices, Config.RSI_PERIOD)
            indicators['ema_fast'] = self.indicators.calculate_ema(close_prices, Config.EMA_FAST)
            indicators['ema_slow'] = self.indicators.calculate_ema(close_prices, Config.EMA_SLOW)
            indicators.update(self.indicators.calculate_macd(close_prices, Config.MACD_FAST, Config.MACD_SLOW, Config.MACD_SIGNAL))
            indicators.update(self.indicators.calculate_bollinger_bands(close_prices, Config.BB_PERIOD, Config.BB_STDDEV))
            indicators['atr'] = self.indicators.calculate_atr(df, Config.ATR_PERIOD)
            
            if 'volume' in df.columns and df['volume'].sum() > 0:
                vol_sma = df['volume'].rolling(window=Config.VOL_SMA_PERIOD).mean().iloc[-1]
                if vol_sma > 0:
                    indicators['volume_ratio'] = df['volume'].iloc[-1] / vol_sma
                    vwap_window = df.tail(Config.VWAP_WINDOW) if Config.VWAP_WINDOW > 0 else df
                    total_volume = vwap_window['volume'].sum()
                    if total_volume > 0:
                        indicators['vwap'] = (vwap_window['close'] * vwap_window['volume']).sum() / total_volume
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return {}
        return indicators

    def generate_signal(self, market_data: Dict[str, any]) -> Optional[Dict[str, any]]:
        """Main entry point to generate a trade signal."""
        try:
            if 'ltp' not in market_data: return None
            
            self._update_history(market_data)
            if len(self.historical_data) < self.min_data_size: return None

            current_price = market_data['ltp']
            indicators = self.calculate_all_indicators()
            if not indicators: return None

            trend_signal = self.generate_trend_signal(indicators)
            momentum_signal = self.generate_momentum_signal(indicators)
            volume_signal = self.generate_volume_signal(indicators, current_price)
            mean_reversion_signal = self.generate_mean_reversion_signal(indicators, current_price)
            
            total_signal = (trend_signal * 0.4 + momentum_signal * 0.3 + volume_signal * 0.2 + mean_reversion_signal * 0.1)
            
            signal_components = {'trend': trend_signal, 'momentum': momentum_signal, 'volume': volume_signal, 'mean_reversion': mean_reversion_signal, 'total': total_signal}
            
            should_trade, reason = self.should_trade(total_signal, signal_components)
            if not should_trade:
                logger.debug(f"No trade signal: {reason}")
                return None
            
            direction = "BUY" if total_signal > 0 else "SELL"
            atr = indicators.get('atr', 0)
            stop_loss, target = self.get_stop_loss_target(current_price, direction, atr)
            
            signal = {'direction': direction, 'entry_price': current_price, 'stop_loss': stop_loss, 'target': target, 'strength': abs(total_signal), 'timestamp': market_data.get('timestamp', pd.Timestamp.now()), 'components': signal_components}
            
            self.signal_history.append(signal)
            logger.info(f"Generated {direction} signal at ₹{current_price:.2f} (Strength: {total_signal:.2f})")
            return signal
            
        except Exception as e:
            logger.error(f"Error in generate_signal: {e}", exc_info=True)
            return None

    def adapt_threshold(self, recent_performance: list):
        """Adapt signal threshold based on recent performance."""
        if not Config.ADAPT_THRESHOLD or len(recent_performance) < Config.PERFORMANCE_WINDOW:
            return
        
        wins = sum(1 for trade in recent_performance if trade.get('pnl', 0) > 0)
        win_rate = wins / len(recent_performance)
        
        if win_rate < 0.3:
            self.adaptive_threshold = min(Config.MAX_THRESHOLD, self.adaptive_threshold + 0.5)
        elif win_rate > 0.7:
            self.adaptive_threshold = max(Config.MIN_THRESHOLD, self.adaptive_threshold - 0.3)
        else:
            if self.adaptive_threshold > Config.SIGNAL_THRESHOLD: self.adaptive_threshold -= 0.1
            elif self.adaptive_threshold < Config.SIGNAL_THRESHOLD: self.adaptive_threshold += 0.1
        
        logger.info(f"Adaptive threshold updated to: {self.adaptive_threshold:.2f} (Win rate: {win_rate:.2%})")

    def generate_trend_signal(self, indicators: Dict[str, float]) -> float:
        signal = 0.0
        if indicators.get('ema_fast') > indicators.get('ema_slow', indicators.get('ema_fast')): signal += 2.0
        else: signal -= 2.0
        if indicators.get('macd') > indicators.get('signal', indicators.get('macd')): signal += 1.5
        else: signal -= 1.5
        return signal
    
    def generate_momentum_signal(self, indicators: Dict[str, float]) -> float:
        signal = 0.0
        rsi = indicators.get('rsi', 50)
        if rsi > 70: signal -= 1.0
        elif rsi < 30: signal += 1.0
        if indicators.get('histogram', 0) > 0: signal += 0.5
        else: signal -= 0.5
        return signal
    
    def generate_volume_signal(self, indicators: Dict[str, float], current_price: float) -> float:
        signal = 0.0
        if indicators.get('volume_ratio', 0) > 1.5 and 'vwap' in indicators:
            if current_price > indicators['vwap']: signal += 0.5
            else: signal -= 0.5
        return signal
    
    def generate_mean_reversion_signal(self, indicators: Dict[str, float], current_price: float) -> float:
        signal = 0.0
        if all(k in indicators for k in ['upper', 'lower']):
            if current_price <= indicators['lower']: signal += 1.0
            elif current_price >= indicators['upper']: signal -= 1.0
        return signal

    def should_trade(self, signal_strength: float, signal_components: Dict[str, float]) -> Tuple[bool, str]:
        abs_signal = abs(signal_strength)
        if abs_signal < self.adaptive_threshold:
            return False, f"Signal too weak: {abs_signal:.2f} < {self.adaptive_threshold:.2f}"
        
        if (signal_components.get('trend', 0) > 0 and signal_components.get('momentum', 0) < -1) or \
           (signal_components.get('trend', 0) < 0 and signal_components.get('momentum', 0) > 1):
            return False, "Trend and momentum divergence"
        
        direction = "BUY" if signal_strength > 0 else "SELL"
        return True, f"Trade signal: {direction} (Strength: {signal_strength:.2f})"

    def get_stop_loss_target(self, entry_price: float, direction: str, atr: float) -> Tuple[float, float]:
        if Config.USE_ATR_SL and atr > 0:
            sl_offset = atr * Config.ATR_SL_MULT
            tp_offset = atr * Config.ATR_TP_MULT
        else:
            sl_offset = entry_price * (Config.SL_PERCENT / 100)
            tp_offset = entry_price * (Config.TP_PERCENT / 100)

        if direction == "BUY":
            stop_loss, target = entry_price - sl_offset, entry_price + tp_offset
        else:
            stop_loss, target = entry_price + sl_offset, entry_price - tp_offset
        return round(stop_loss, 2), round(target, 2)

