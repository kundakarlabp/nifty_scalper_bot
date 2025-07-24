import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from utils import TechnicalIndicators
from config import Config

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Generate trading signals based on technical indicators"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.signal_history = []
        self.adaptive_threshold = Config.SIGNAL_THRESHOLD
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators"""
        if df.empty or len(df) < max(Config.RSI_PERIOD, Config.EMA_SLOW, Config.ATR_PERIOD):
            return {}
        
        indicators = {}
        
        try:
            # Price-based indicators
            close_prices = df['close']
            
            # RSI
            indicators['rsi'] = self.indicators.calculate_rsi(close_prices, Config.RSI_PERIOD)
            
            # EMAs
            indicators['ema_fast'] = self.indicators.calculate_ema(close_prices, Config.EMA_FAST)
            indicators['ema_slow'] = self.indicators.calculate_ema(close_prices, Config.EMA_SLOW)
            
            # MACD
            macd_data = self.indicators.calculate_macd(
                close_prices, Config.MACD_FAST, Config.MACD_SLOW, Config.MACD_SIGNAL
            )
            indicators.update(macd_data)
            
            # Bollinger Bands
            bb_data = self.indicators.calculate_bollinger_bands(
                close_prices, Config.BB_PERIOD, Config.BB_STDDEV
            )
            indicators.update(bb_data)
            
            # ATR
            indicators['atr'] = self.indicators.calculate_atr(df, Config.ATR_PERIOD)
            
            # Volume indicators (if volume data available)
            if 'volume' in df.columns:
                vol_sma = df['volume'].rolling(window=Config.VOL_SMA_PERIOD).mean()
                indicators['volume_sma'] = vol_sma.iloc[-1] if not vol_sma.empty else 0
                indicators['volume_ratio'] = (df['volume'].iloc[-1] / indicators['volume_sma'] 
                                            if indicators['volume_sma'] > 0 else 1.0)
            
            # VWAP (if volume data available)
            if 'volume' in df.columns and len(df) > 0:
                if Config.VWAP_WINDOW > 0 and len(df) >= Config.VWAP_WINDOW:
                    window_df = df.tail(Config.VWAP_WINDOW)
                else:
                    window_df = df
                
                total_volume = window_df['volume'].sum()
                if total_volume > 0:
                    indicators['vwap'] = (window_df['close'] * window_df['volume']).sum() / total_volume
                else:
                    indicators['vwap'] = df['close'].iloc[-1]
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
        
        return indicators
    
    def generate_trend_signal(self, indicators: Dict[str, float]) -> float:
        """Generate trend-following signal"""
        signal = 0.0
        
        # EMA crossover signal
        if 'ema_fast' in indicators and 'ema_slow' in indicators:
            if indicators['ema_fast'] > indicators['ema_slow']:
                signal += 2.0  # Bullish trend
            else:
                signal -= 2.0  # Bearish trend
        
        # MACD signal
        if 'macd' in indicators and 'signal' in indicators:
            if indicators['macd'] > indicators['signal']:
                signal += 1.5  # Bullish momentum
            else:
                signal -= 1.5  # Bearish momentum
        
        return signal
    
    def generate_momentum_signal(self, indicators: Dict[str, float]) -> float:
        """Generate momentum-based signal"""
        signal = 0.0
        
        # RSI signals
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi > 70:
                signal -= 1.0  # Overbought
            elif rsi < 30:
                signal += 1.0  # Oversold
            elif rsi > 60:
                signal += 0.5  # Bullish momentum
            elif rsi < 40:
                signal -= 0.5  # Bearish momentum
        
        # MACD histogram
        if 'histogram' in indicators:
            if indicators['histogram'] > 0:
                signal += 0.5
            else:
                signal -= 0.5
        
        return signal
    
    def generate_volume_signal(self, indicators: Dict[str, float], current_price: float) -> float:
        """Generate volume-based confirmation signal"""
        signal = 0.0
        
        # Volume confirmation
        if 'volume_ratio' in indicators:
            volume_ratio = indicators['volume_ratio']
            if volume_ratio > 1.5:  # High volume
                # Check price vs VWAP for direction
                if 'vwap' in indicators:
                    if current_price > indicators['vwap']:
                        signal += 0.5  # High volume with price above VWAP
                    else:
                        signal -= 0.5  # High volume with price below VWAP
                else:
                    signal += 0.3  # Just high volume
        
        return signal
    
    def generate_mean_reversion_signal(self, indicators: Dict[str, float], current_price: float) -> float:
        """Generate mean reversion signal using Bollinger Bands"""
        signal = 0.0
        
        if all(k in indicators for k in ['upper', 'lower', 'middle']):
            if current_price <= indicators['lower']:
                signal += 1.0  # Oversold - potential bounce
            elif current_price >= indicators['upper']:
                signal -= 1.0  # Overbought - potential decline
            elif current_price > indicators['middle']:
                signal += 0.3  # Above middle band
            else:
                signal -= 0.3  # Below middle band
        
        return signal
    
    def calculate_signal_strength(self, df: pd.DataFrame, current_price: float) -> Tuple[float, Dict[str, float]]:
        """Calculate overall signal strength"""
        indicators = self.calculate_all_indicators(df)
        
        if not indicators:
            return 0.0, {}
        
        # Generate different types of signals
        trend_signal = self.generate_trend_signal(indicators)
        momentum_signal = self.generate_momentum_signal(indicators)
        volume_signal = self.generate_volume_signal(indicators, current_price)
        mean_reversion_signal = self.generate_mean_reversion_signal(indicators, current_price)
        
        # Weighted combination of signals
        total_signal = (
            trend_signal * 0.4 +           # 40% weight to trend
            momentum_signal * 0.3 +        # 30% weight to momentum
            volume_signal * 0.2 +          # 20% weight to volume
            mean_reversion_signal * 0.1    # 10% weight to mean reversion
        )
        
        # Store signal components for analysis
        signal_components = {
            'trend': trend_signal,
            'momentum': momentum_signal,
            'volume': volume_signal,
            'mean_reversion': mean_reversion_signal,
            'total': total_signal
        }
        
        return total_signal, signal_components
    
    def adapt_threshold(self, recent_performance: list) -> None:
        """Adapt signal threshold based on recent performance"""
        if not Config.ADAPT_THRESHOLD or len(recent_performance) < Config.PERFORMANCE_WINDOW:
            return
        
        # Calculate win rate
        wins = sum(1 for trade in recent_performance if trade.get('pnl', 0) > 0)
        win_rate = wins / len(recent_performance)
        
        # Adjust threshold based on performance
        if win_rate < 0.3:  # Poor performance
            self.adaptive_threshold = min(Config.MAX_THRESHOLD, self.adaptive_threshold + 0.5)
        elif win_rate > 0.7:  # Good performance
            self.adaptive_threshold = max(Config.MIN_THRESHOLD, self.adaptive_threshold - 0.3)
        else:  # Average performance
            # Gradually move towards default threshold
            if self.adaptive_threshold > Config.SIGNAL_THRESHOLD:
                self.adaptive_threshold -= 0.1
            elif self.adaptive_threshold < Config.SIGNAL_THRESHOLD:
                self.adaptive_threshold += 0.1
        
        logger.info(f"Adaptive threshold updated to: {self.adaptive_threshold:.2f} (Win rate: {win_rate:.2%})")
    
    def should_trade(self, signal_strength: float, signal_components: Dict[str, float]) -> Tuple[bool, str]:
        """Determine if we should trade based on signal strength"""
        abs_signal = abs(signal_strength)
        
        # Check if signal exceeds threshold
        if abs_signal < self.adaptive_threshold:
            return False, f"Signal too weak: {abs_signal:.2f} < {self.adaptive_threshold:.2f}"
        
        # Additional filters
        # Ensure trend and momentum agree for strong signals
        if abs_signal > Config.MAX_THRESHOLD:
            trend_signal = signal_components.get('trend', 0)
            momentum_signal = signal_components.get('momentum', 0)
            
            # Check if trend and momentum are in same direction
            if (trend_signal > 0 and momentum_signal < -1) or (trend_signal < 0 and momentum_signal > 1):
                return False, "Trend and momentum divergence on strong signal"
        
        direction = "BUY" if signal_strength > 0 else "SELL"
        return True, f"Trade signal: {direction} (Strength: {signal_strength:.2f})"
    
    def get_stop_loss_target(self, entry_price: float, direction: str, atr: float) -> Tuple[float, float]:
        """Calculate stop loss and target prices"""
        if Config.USE_ATR_SL and Config.USE_ATR_TP and atr > 0:
            # ATR-based SL/TP
            if direction == "BUY":
                stop_loss = entry_price - (atr * Config.ATR_SL_MULT)
                target = entry_price + (atr * Config.ATR_TP_MULT)
            else:  # SELL
                stop_loss = entry_price + (atr * Config.ATR_SL_MULT)
                target = entry_price - (atr * Config.ATR_TP_MULT)
        else:
            # Percentage-based SL/TP
            if direction == "BUY":
                stop_loss = entry_price * (1 - Config.SL_PERCENT / 100)
                target = entry_price * (1 + Config.TP_PERCENT / 100)
            else:  # SELL
                stop_loss = entry_price * (1 + Config.SL_PERCENT / 100)
                target = entry_price * (1 - Config.TP_PERCENT / 100)
        
        return round(stop_loss, 2), round(target, 2)
