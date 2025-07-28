import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import pytz
from config import (BASE_STOP_LOSS_POINTS, BASE_TARGET_POINTS, 
                   CONFIDENCE_THRESHOLD, ACCOUNT_SIZE, 
                   RISK_PER_TRADE, MAX_DRAWDOWN, NIFTY_LOT_SIZE)

logger = logging.getLogger(__name__)

class DynamicScalpingStrategy:
    def __init__(self, base_stop_loss_points: int = None, base_target_points: int = None, 
                 confidence_threshold: float = None):
        self.base_stop_loss_points = base_stop_loss_points or BASE_STOP_LOSS_POINTS
        self.base_target_points = base_target_points or BASE_TARGET_POINTS
        self.confidence_threshold = confidence_threshold or CONFIDENCE_THRESHOLD
        self.account_size = ACCOUNT_SIZE
        self.risk_per_trade = RISK_PER_TRADE
        self.max_drawdown = MAX_DRAWDOWN
        self.lot_size = NIFTY_LOT_SIZE
        self.timezone = pytz.timezone('Asia/Kolkata')
        
        logger.info(f"✅ Dynamic Scalping Strategy initialized")
        logger.info(f"   Base SL Points: {self.base_stop_loss_points}")
        logger.info(f"   Base Target Points: {self.base_target_points}")
        logger.info(f"   Confidence Threshold: {self.confidence_threshold:.2%}")
    
    def generate_signal(self, ohlc_data: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """Generate trading signal based on OHLC data and current price"""
        try:
            # Validate inputs
            if ohlc_data is None or ohlc_data.empty:
                logger.warning("⚠️  No OHLC data provided for signal generation")
                return None
            
            if current_price is None or current_price <= 0:
                logger.warning("⚠️  Invalid current price for signal generation")
                return None
            
            if len(ohlc_data) < 50:
                logger.warning("⚠️  Insufficient OHLC data for signal generation")
                return None
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(ohlc_data)
            if not indicators:
                logger.warning("⚠️  Failed to calculate technical indicators")
                return None
            
            # Generate signal based on indicators
            signal = self._generate_signal_from_indicators(indicators, current_price)
            
            if signal and signal['confidence'] >= self.confidence_threshold:
                logger.info(f"🎯 Signal generated: {signal['signal']} with confidence {signal['confidence']:.2f}")
                return signal
            else:
                logger.debug(f"🔍 No strong signal generated. Confidence: {signal['confidence'] if signal else 0:.2f}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating signal: {e}")
            return None
    
    def _calculate_indicators(self, ohlc_data: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        try:
            if ohlc_data.empty or len(ohlc_data) < 50:
                return {}
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in ohlc_data.columns:
                    logger.warning(f"⚠️  Missing required column: {col}")
                    return {}
            
            indicators = {}
            
            close = ohlc_data['close']
            high = ohlc_data['high']
            low = ohlc_data['low']
            volume = ohlc_data['volume']
            open_prices = ohlc_data['open']
            
            # Moving Averages
            indicators['sma_9'] = self._calculate_sma(close, 9)
            indicators['sma_21'] = self._calculate_sma(close, 21)
            indicators['sma_50'] = self._calculate_sma(close, 50)
            indicators['ema_12'] = self._calculate_ema(close, 12)
            indicators['ema_26'] = self._calculate_ema(close, 26)
            
            # RSI
            indicators['rsi_14'] = self._calculate_rsi(close, 14)
            
            # MACD
            macd_data = self._calculate_macd(close)
            indicators.update(macd_data)
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(close)
            indicators.update(bb_data)
            
            # Stochastic
            stoch_data = self._calculate_stochastic(high, low, close)
            indicators.update(stoch_data)
            
            # ATR
            indicators['atr_14'] = self._calculate_atr(high, low, close, 14)
            
            # Volume Profile
            indicators['volume_profile'] = self._calculate_volume_profile(volume, 20)
            
            # OBV
            indicators['obv'] = self._calculate_obv(close, volume)
            
            logger.info(f"✅ Calculated {len(indicators)} technical indicators")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return {}
    
    def _generate_signal_from_indicators(self, indicators: Dict, current_price: float) -> Optional[Dict]:
        """Generate signal from calculated indicators"""
        try:
            if not indicators:
                return None
            
            # Calculate signal scores
            buy_score = 0
            sell_score = 0
            reasons = []
            
            # Trend analysis
            if 'sma_9' in indicators and 'sma_21' in indicators and 'sma_50' in indicators:
                sma_9 = indicators['sma_9']
                sma_21 = indicators['sma_21']
                sma_50 = indicators['sma_50']
                
                if sma_9 > sma_21 > sma_50:
                    buy_score += 2.5
                    reasons.append("Uptrend confirmed")
                elif sma_9 < sma_21 < sma_50:
                    sell_score += 2.5
                    reasons.append("Downtrend confirmed")
            
            # RSI analysis
            if 'rsi_14' in indicators:
                rsi = indicators['rsi_14']
                if 30 <= rsi <= 70:
                    # Neutral RSI is good for momentum trades
                    if rsi > 50:
                        buy_score += 1.0
                        reasons.append("RSI momentum positive")
                    else:
                        sell_score += 1.0
                        reasons.append("RSI momentum negative")
                elif rsi < 30:
                    buy_score += 1.5
                    reasons.append("RSI oversold")
                elif rsi > 70:
                    sell_score += 1.5
                    reasons.append("RSI overbought")
            
            # MACD analysis
            if 'macd' in indicators and 'signal' in indicators:
                macd = indicators['macd']
                signal_line = indicators['signal']
                
                if macd > signal_line:
                    buy_score += 1.5
                    reasons.append("MACD bullish")
                elif macd < signal_line:
                    sell_score += 1.5
                    reasons.append("MACD bearish")
            
            # Bollinger Bands analysis
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                bb_upper = indicators['bb_upper']
                bb_lower = indicators['bb_lower']
                
                if current_price < bb_lower:
                    buy_score += 1.0
                    reasons.append("Price below lower BB")
                elif current_price > bb_upper:
                    sell_score += 1.0
                    reasons.append("Price above upper BB")
            
            # Stochastic analysis
            if 'k' in indicators and 'd' in indicators:
                k = indicators['k']
                d = indicators['d']
                
                if k < 20 and d < 20:
                    buy_score += 1.0
                    reasons.append("Stochastic oversold")
                elif k > 80 and d > 80:
                    sell_score += 1.0
                    reasons.append("Stochastic overbought")
            
            # ATR analysis
            if 'atr_14' in indicators:
                atr = indicators['atr_14']
                market_volatility = atr / current_price if current_price > 0 else 0.01
                indicators['market_volatility'] = market_volatility
            else:
                market_volatility = 0.01
                indicators['market_volatility'] = market_volatility
            
            # Volume analysis
            if 'volume_profile' in indicators:
                volume_profile = indicators['volume_profile']
                if volume_profile > 1.5:
                    if buy_score > sell_score:
                        buy_score += 0.5
                        reasons.append("High volume confirming buy")
                    elif sell_score > buy_score:
                        sell_score += 0.5
                        reasons.append("High volume confirming sell")
            
            # Calculate final signal
            total_score = buy_score + sell_score
            if total_score == 0:
                return None
            
            # Normalize scores to 0-1 range
            normalized_buy_score = buy_score / max(1, total_score)
            normalized_sell_score = sell_score / max(1, total_score)
            
            # Determine signal direction and confidence
            if normalized_buy_score > normalized_sell_score and normalized_buy_score >= 0.6:
                signal_direction = "BUY"
                confidence = min(1.0, normalized_buy_score)
            elif normalized_sell_score > normalized_buy_score and normalized_sell_score >= 0.6:
                signal_direction = "SELL"
                confidence = min(1.0, normalized_sell_score)
            else:
                return None
            
            # Calculate stop loss and target based on volatility
            stop_loss_points = self.base_stop_loss_points * (1 + market_volatility)
            target_points = self.base_target_points * (1 + market_volatility)
            
            if signal_direction == "BUY":
                stop_loss = current_price - stop_loss_points
                target = current_price + target_points
            else:
                stop_loss = current_price + stop_loss_points
                target = current_price - target_points
            
            signal = {
                'signal': signal_direction,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'target': target,
                'confidence': confidence,
                'market_volatility': market_volatility,
                'reasons': reasons[:5],  # Limit to 5 reasons
                'timestamp': datetime.now(self.timezone).isoformat()
            }
            
            logger.info(f"✅ Signal generated: {signal_direction} with confidence {confidence:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating signal from indicators: {e}")
            return None
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> float:
        """Calculate Simple Moving Average"""
        try:
            if len(prices) < period:
                return prices.iloc[-1] if len(prices) > 0 else 0
            
            return prices.tail(period).mean()
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return 0
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return prices.iloc[-1] if len(prices) > 0 else 0
            
            return prices.tail(period).ewm(span=period).mean().iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return 0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if len(rsi) > 0 else 50.0
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast_period: int = 12, 
                        slow_period: int = 26, signal_period: int = 9) -> Dict[str, float]:
        """Calculate MACD"""
        try:
            if len(prices) < max(fast_period, slow_period, signal_period):
                return {'macd': 0, 'signal': 0, 'histogram': 0}
            
            exp1 = prices.ewm(span=fast_period).mean()
            exp2 = prices.ewm(span=slow_period).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=signal_period).mean()
            histogram = macd - signal
            
            return {
                'macd': macd.iloc[-1] if len(macd) > 0 else 0,
                'signal': signal.iloc[-1] if len(signal) > 0 else 0,
                'histogram': histogram.iloc[-1] if len(histogram) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                current_price = prices.iloc[-1] if len(prices) > 0 else 0
                return {
                    'upper': current_price + 50,
                    'middle': current_price,
                    'lower': current_price - 50,
                    'bandwidth': 0.1
                }
            
            rolling_mean = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            upper_band = rolling_mean + (rolling_std * std_dev)
            lower_band = rolling_mean - (rolling_std * std_dev)
            bandwidth = (upper_band - lower_band) / rolling_mean
            
            return {
                'upper': upper_band.iloc[-1] if len(upper_band) > 0 else prices.iloc[-1],
                'middle': rolling_mean.iloc[-1] if len(rolling_mean) > 0 else prices.iloc[-1],
                'lower': lower_band.iloc[-1] if len(lower_band) > 0 else prices.iloc[-1],
                'bandwidth': bandwidth.iloc[-1] if len(bandwidth) > 0 else 0.1
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            current_price = prices.iloc[-1] if len(prices) > 0 else 0
            return {
                'upper': current_price + 50,
                'middle': current_price,
                'lower': current_price - 50,
                'bandwidth': 0.1
            }
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                              k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """Calculate Stochastic Oscillator"""
        try:
            if len(high) < max(k_period, d_period):
                return {'k': 50.0, 'd': 50.0}
            
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d = k.rolling(window=d_period).mean()
            
            return {
                'k': k.iloc[-1] if len(k) > 0 else 50.0,
                'd': d.iloc[-1] if len(d) > 0 else 50.0
            }
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
            return {'k': 50.0, 'd': 50.0}
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(high) < period:
                return close.iloc[-1] * 0.01 if len(close) > 0 else 1.0
            
            tr0 = abs(high - low)
            tr1 = abs(high - close.shift())
            tr2 = abs(low - close.shift())
            tr = pd.DataFrame({'tr0': tr0, 'tr1': tr1, 'tr2': tr2}).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr.iloc[-1] if len(atr) > 0 else close.iloc[-1] * 0.01
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return close.iloc[-1] * 0.01 if len(close) > 0 else 1.0
    
    def _calculate_volume_profile(self, volume: pd.Series, period: int = 20) -> float:
        """Calculate Volume Profile"""
        try:
            if len(volume) < period:
                return 1.0
            
            avg_volume = volume.rolling(window=period).mean()
            return volume.iloc[-1] / avg_volume.iloc[-1] if len(avg_volume) > 0 and avg_volume.iloc[-1] > 0 else 1.0
        except Exception as e:
            logger.error(f"Error calculating Volume Profile: {e}")
            return 1.0
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> float:
        """Calculate On-Balance Volume"""
        try:
            if len(close) < 2 or len(volume) < 2:
                return volume.iloc[-1] if len(volume) > 0 else 0
            
            obv = pd.Series(index=close.index, dtype='float64')
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            return volume.iloc[-1] if len(volume) > 0 else 0

# Example usage
if __name__ == "__main__":
    print("Dynamic Scalping Strategy ready!")
    print("Import and use: from src.strategies.scalping_strategy import DynamicScalpingStrategy")
