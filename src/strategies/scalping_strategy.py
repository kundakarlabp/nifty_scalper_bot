# src/strategies/scalping_strategy.py
import pandas as pd
import numpy as np
import hashlib
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

class DynamicScalpingStrategy:
    """
    A dynamic scalping strategy using multiple technical indicators
    and market regime detection to generate trading signals.
    """

    def __init__(self, 
                 ema_fast_period: int = 9,
                 ema_slow_period: int = 21,
                 rsi_period: int = 14,
                 bb_window: int = 20,
                 bb_std: int = 2,
                 stoch_k_period: int = 14,
                 stoch_d_period: int = 3,
                 atr_period: int = 14,
                 scoring_threshold: int = 9):
        """
        Initialize the scalping strategy with configurable parameters.

        Args:
            ema_fast_period (int): Period for fast EMA.
            ema_slow_period (int): Period for slow EMA.
            rsi_period (int): Period for RSI calculation.
            bb_window (int): Window for Bollinger Bands.
            bb_std (int): Standard deviation multiplier for Bollinger Bands.
            stoch_k_period (int): Period for Stochastic %K.
            stoch_d_period (int): Period for Stochastic %D (signal line).
            atr_period (int): Period for ATR calculation.
            scoring_threshold (int): Minimum score to generate a signal.
        """
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.rsi_period = rsi_period
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.stoch_k_period = stoch_k_period
        self.stoch_d_period = stoch_d_period
        self.atr_period = atr_period
        self.scoring_threshold = scoring_threshold
        self.last_signal_hash = None

    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate a trading signal based on technical indicators.

        Args:
            df (pd.DataFrame): OHLCV data with required columns.

        Returns:
            Optional[Dict]: Signal dictionary or None if no signal.
        """
        try:
            # Validate input data
            if df is None or df.empty:
                logger.warning("Empty DataFrame provided to strategy")
                return None

            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns. Required: {required_columns}, Found: {list(df.columns)}")
                return None

            # Ensure sufficient data points
            min_required = max(self.ema_slow_period, self.bb_window, self.rsi_period, self.atr_period) + 5
            if len(df) < min_required:
                logger.warning(f"Insufficient data points. Required: {min_required}, Provided: {len(df)}")
                return None

            # Calculate indicators
            indicators = self._calculate_indicators(df)
            if not indicators:
                logger.warning("Failed to calculate indicators")
                return None

            # Detect market regime
            regime = self._detect_market_regime(indicators)

            # Score the potential signal
            score, reasons = self._score_signal(indicators, regime)

            # Determine signal direction
            signal_direction = None
            if score >= self.scoring_threshold:
                signal_direction = 'BUY'
            elif score <= -self.scoring_threshold:
                signal_direction = 'SELL'

            # If we have a valid signal
            if signal_direction:
                current_price = float(df['close'].iloc[-1])
                
                # Create a unique hash to prevent duplicate signals
                signal_string = f"{signal_direction}_{current_price}_{df.index[-1]}"
                signal_hash = hashlib.md5(signal_string.encode()).hexdigest()

                # Check for duplicate signals
                if signal_hash == self.last_signal_hash:
                    logger.debug("Duplicate signal detected, skipping")
                    return None

                # Update last signal hash
                self.last_signal_hash = signal_hash

                # Calculate risk management parameters
                atr = float(indicators.get('atr', 0))
                stop_loss = (
                    current_price - 2 * atr if signal_direction == 'BUY' 
                    else current_price + 2 * atr
                )
                target = (
                    current_price + 3 * atr if signal_direction == 'BUY' 
                    else current_price - 3 * atr
                )

                # Return the signal
                return {
                    'signal': signal_direction,
                    'confidence': abs(score),
                    'entry_price': round(current_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'target': round(target, 2),
                    'reasons': reasons,
                    'signal_hash': signal_hash,
                    'market_regime': regime,
                    'market_volatility': round(atr, 2)
                }

            return None

        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return None

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate all technical indicators needed for the strategy.

        Args:
            df (pd.DataFrame): OHLCV data.

        Returns:
            Dict: Dictionary of calculated indicators.
        """
        try:
            # Calculate moving averages
            ema_fast = df['close'].ewm(span=self.ema_fast_period, adjust=False).mean()
            ema_slow = df['close'].ewm(span=self.ema_slow_period, adjust=False).mean()
            sma = df['close'].rolling(window=self.bb_window).mean()

            # Calculate RSI
            rsi = self._calculate_rsi(df['close'], self.rsi_period)

            # Calculate MACD
            macd_data = self._calculate_macd(df['close'])
            
            # Calculate Bollinger Bands
            bb_std_dev = df['close'].rolling(window=self.bb_window).std()
            bb_upper = sma + (bb_std_dev * self.bb_std)
            bb_lower = sma - (bb_std_dev * self.bb_std)
            bb_bandwidth = (bb_upper - bb_lower) / sma

            # Calculate Stochastic Oscillator
            stoch_data = self._calculate_stochastic(df, self.stoch_k_period, self.stoch_d_period)

            # Calculate ATR
            atr = self._calculate_atr(df, self.atr_period)

            # Calculate OBV
            obv = self._calculate_obv(df)

            # Return the latest values of all indicators
            return {
                'sma': float(sma.iloc[-1]),
                'ema_fast': float(ema_fast.iloc[-1]),
                'ema_slow': float(ema_slow.iloc[-1]),
                'rsi': float(rsi.iloc[-1]),
                'macd_diff': float(macd_data['macd_diff'].iloc[-1]),
                'bb_upper': float(bb_upper.iloc[-1]),
                'bb_lower': float(bb_lower.iloc[-1]),
                'bb_bandwidth': float(bb_bandwidth.iloc[-1]),
                'stoch_k': float(stoch_data['%K'].iloc[-1]),
                'stoch_d': float(stoch_data['%D'].iloc[-1]),
                'atr': float(atr.iloc[-1]),
                'obv': float(obv.iloc[-1]),
                'volume': float(df['volume'].iloc[-1]),
                'close': float(df['close'].iloc[-1])
            }

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return {}

    def _score_signal(self, indicators: Dict, regime: str) -> Tuple[int, List[str]]:
        """
        Score a potential signal based on indicator values and market regime.

        Args:
            indicators (Dict): Dictionary of calculated indicators.
            regime (str): Current market regime.

        Returns:
            Tuple[int, List[str]]: Signal score and reasons.
        """
        score = 0
        reasons = []

        # Moving Average Crossover
        if indicators['ema_fast'] > indicators['ema_slow']:
            score += 2
            reasons.append("EMA Bullish Crossover")
        elif indicators['ema_fast'] < indicators['ema_slow']:
            score -= 2
            reasons.append("EMA Bearish Crossover")

        # RSI
        if indicators['rsi'] > 70:
            score -= 1
            reasons.append("RSI Overbought")
        elif indicators['rsi'] < 30:
            score += 1
            reasons.append("RSI Oversold")
        elif 50 < indicators['rsi'] < 60:
            score += 1
            reasons.append("RSI Bullish")
        elif 40 < indicators['rsi'] < 50:
            score -= 1
            reasons.append("RSI Bearish")

        # MACD
        if indicators['macd_diff'] > 0:
            score += 1
            reasons.append("MACD Bullish")
        elif indicators['macd_diff'] < 0:
            score -= 1
            reasons.append("MACD Bearish")

        # Bollinger Bands
        if indicators['close'] < indicators['bb_lower']:
            score += 2
            reasons.append("Price below BB Lower")
        elif indicators['close'] > indicators['bb_upper']:
            score -= 2
            reasons.append("Price above BB Upper")
        elif indicators['close'] < indicators['sma']:
            score += 1
            reasons.append("Price below SMA")
        else:
            score -= 1
            reasons.append("Price above SMA")

        # Stochastic
        if indicators['stoch_k'] > 80 and indicators['stoch_k'] < indicators['stoch_d']:
            score -= 2
            reasons.append("Stochastic Bearish Cross")
        elif indicators['stoch_k'] < 20 and indicators['stoch_k'] > indicators['stoch_d']:
            score += 2
            reasons.append("Stochastic Bullish Cross")
        elif indicators['stoch_k'] > 80:
            score -= 1
            reasons.append("Stochastic Overbought")
        elif indicators['stoch_k'] < 20:
            score += 1
            reasons.append("Stochastic Oversold")

        # Market Regime Adjustment
        if regime == "TRENDING":
            score *= 1.2  # Boost trending signals
            reasons.append("Trending Market")
        elif regime == "RANGING":
            score *= 0.8  # Reduce ranging signals
            reasons.append("Ranging Market")

        # Normalize score to integer
        score = int(round(score))
        return score, reasons

    def _detect_market_regime(self, indicators: Dict) -> str:
        """
        Detect the current market regime (TRENDING, RANGING, NEUTRAL).

        Args:
            indicators (Dict): Dictionary of calculated indicators.

        Returns:
            str: Market regime classification.
        """
        try:
            bandwidth = indicators.get('bb_bandwidth', 0)
            atr = indicators.get('atr', 0)
            close = indicators.get('close', 1)  # Avoid division by zero

            # Normalize ATR to price
            normalized_atr = atr / close if close > 0 else 0

            # Ranging market: Low bandwidth and low volatility
            if bandwidth < 0.01 and normalized_atr < 0.005:
                return "RANGING"
            
            # Trending market: High bandwidth
            elif bandwidth > 0.03:
                return "TRENDING"
            
            # Neutral market
            else:
                return "NEUTRAL"

        except Exception as e:
            logger.error(f"Error detecting market regime: {e}", exc_info=True)
            return "NEUTRAL"

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI).

        Args:
            series (pd.Series): Price series.
            period (int): RSI period.

        Returns:
            pd.Series: RSI values.
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate the initial average gain and loss
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_macd(self, series: pd.Series) -> pd.DataFrame:
        """
        Calculate MACD line, signal line, and histogram.

        Args:
            series (pd.Series): Price series.

        Returns:
            pd.DataFrame: MACD data with columns 'macd', 'signal', 'macd_diff'.
        """
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_diff = macd - signal
        return pd.DataFrame({'macd': macd, 'signal': signal, 'macd_diff': macd_diff})

    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Calculate the Stochastic Oscillator.

        Args:
            df (pd.DataFrame): OHLC data.
            k_period (int): Period for %K line.
            d_period (int): Period for %D (signal) line.

        Returns:
            pd.DataFrame: Stochastic data with columns '%K', '%D'.
        """
        low_min = df['low'].rolling(window=k_period, min_periods=1).min()
        high_max = df['high'].rolling(window=k_period, min_periods=1).max()
        
        # Avoid division by zero
        k = 100 * ((df['close'] - low_min) / (high_max - low_min).replace(0, 1e-10))
        d = k.rolling(window=d_period, min_periods=1).mean()
        
        return pd.DataFrame({'%K': k, '%D': d})

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the Average True Range (ATR).

        Args:
            df (pd.DataFrame): OHLC data.
            period (int): ATR period.

        Returns:
            pd.Series: ATR values.
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        # True range is the maximum of these three
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR using Wilder's smoothing method
        atr = pd.Series(index=tr.index, dtype='float64')
        atr.iloc[0] = tr.iloc[:period].mean()  # Initial value
        
        for i in range(1, len(atr)):
            atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + tr.iloc[i]) / period
            
        return atr

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the On-Balance Volume (OBV).

        Args:
            df (pd.DataFrame): OHLCV data.

        Returns:
            pd.Series: OBV values.
        """
        obv = [0]  # Initial OBV value
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
                
        return pd.Series(obv, index=df.index)