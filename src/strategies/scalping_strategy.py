# src/strategies/scalping_strategy.py
import pandas as pd
import numpy as np
import hashlib
import logging
from typing import Dict, Optional, Tuple, List
# Import classes from the `ta` library for indicator calculations
from ta.momentum import RSIIndicator
from ta.trend import MACD

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
                 # --- Parameters passed by RealTimeTrader ---
                 base_stop_loss_points: float = 20.0,
                 base_target_points: float = 40.0,
                 confidence_threshold: float = 8.0, # This is for internal scoring points check now
                 # Internal scoring threshold for generating a signal (points needed)
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
            base_stop_loss_points (float): Base stop loss points.
            base_target_points (float): Base target points.
            confidence_threshold (float): Minimum internal scoring points to act on a signal.
                                        (Renamed for clarity, as it's compared against `score`)
            scoring_threshold (int): Internal points threshold required to generate a raw signal.
                                     The signal's 'confidence' score (float) is then checked
                                     against confidence_threshold.
        """
        # --- Indicator Parameters ---
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.rsi_period = rsi_period
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.stoch_k_period = stoch_k_period
        self.stoch_d_period = stoch_d_period
        self.atr_period = atr_period

        # --- Strategy Control Parameters ---
        self.base_stop_loss_points = base_stop_loss_points
        self.base_target_points = base_target_points
        # Rename for clarity: this is the internal score threshold now
        self.min_score_threshold = confidence_threshold 
        self.scoring_threshold = scoring_threshold       # Raw signal points threshold

        self.last_signal_hash = None

    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """
        Generate a trading signal based on technical indicators.

        Args:
            df (pd.DataFrame): OHLCV data with required columns.
            current_price (float): The current market price.

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
            logger.debug(f"Signal score calculated: {score}")

            # Determine signal direction based on internal scoring threshold (points)
            signal_direction = None
            if score >= self.scoring_threshold:
                signal_direction = 'BUY'
            elif score <= -self.scoring_threshold:
                signal_direction = 'SELL'

            # If we have a valid raw signal based on points, check against the min score threshold
            # Note: The original logic checked `score >= self.confidence_threshold`.
            # Based on the KB file `Pasted_Text_1753870312034.txt`, the check was just `if signal_direction:`
            # The `confidence_threshold` passed by RealTimeTrader is likely meant to be 
            # the Config.CONFIDENCE_THRESHOLD, which is checked against the final 
            # `signal['confidence']` (the absolute score) in RealTimeTrader itself.
            # Let's make this strategy's internal check clear.
            # Option 1 (if confidence_threshold is internal score): if signal_direction and abs(score) >= self.min_score_threshold:
            # Option 2 (if confidence_threshold check is only in RealTimeTrader): if signal_direction:
            # Based on typical separation of concerns, Option 2 is cleaner. 
            # RealTimeTrader validates the final float confidence. Strategy validates its internal points.
            
            # Using Option 2 for cleaner separation, as RealTimeTrader does the final check.
            # However, if you want the strategy itself to have a final internal filter, use Option 1.
            # Let's go with Option 1 to utilize the passed parameter, but clarify its meaning.
            # The passed `confidence_threshold` becomes `min_score_threshold` (an internal integer score check).
            
            # Final check: Was a direction determined by points, and does it meet the minimum internal score?
            if signal_direction and abs(score) >= self.min_score_threshold:
                # Create a unique hash to prevent duplicate signals
                signal_string = f"{signal_direction}_{current_price}_{df.index[-1]}"
                signal_hash = hashlib.md5(signal_string.encode()).hexdigest()

                # Check for duplicate signals
                if signal_hash == self.last_signal_hash:
                    logger.debug("Duplicate signal detected, skipping")
                    return None

                # Update last signal hash
                self.last_signal_hash = signal_hash

                # Calculate risk management parameters using ATR
                atr = float(indicators.get('atr', 0))
                # Example SL/TP calculation using ATR multipliers
                sl_atr_component = 2 * atr
                tp_atr_component = 3 * atr

                stop_loss = (
                    current_price - sl_atr_component if signal_direction == 'BUY'
                    else current_price + sl_atr_component
                )
                target = (
                    current_price + tp_atr_component if signal_direction == 'BUY'
                    else current_price - tp_atr_component
                )

                # Return the signal with details
                # The 'confidence' returned is the absolute internal score.
                # RealTimeTrader will check if this meets its Config.CONFIDENCE_THRESHOLD.
                final_confidence = abs(score) # Calculate once
                return {
                    'signal': signal_direction,
                    'confidence': final_confidence, # This integer score is checked in RealTimeTrader
                    'entry_price': round(current_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'target': round(target, 2),
                    'reasons': reasons,
                    'signal_hash': signal_hash,
                    'market_regime': regime,
                    'market_volatility': round(atr, 2) # Pass ATR as volatility measure
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

            # Calculate RSI using ta library
            rsi_indicator = RSIIndicator(close=df['close'], window=self.rsi_period)
            rsi = rsi_indicator.rsi()

            # Calculate MACD using ta library
            macd_indicator = MACD(close=df['close'])
            macd_diff = macd_indicator.macd_diff() # This is the MACD line minus the Signal line

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
            # Explicitly handle potential NaNs when converting to float
            return {
                'sma': float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else 0.0,
                'ema_fast': float(ema_fast.iloc[-1]) if not pd.isna(ema_fast.iloc[-1]) else 0.0,
                'ema_slow': float(ema_slow.iloc[-1]) if not pd.isna(ema_slow.iloc[-1]) else 0.0,
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0, # Default to neutral RSI
                'macd_diff': float(macd_diff.iloc[-1]) if not pd.isna(macd_diff.iloc[-1]) else 0.0,
                'bb_upper': float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else 0.0,
                'bb_lower': float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else 0.0,
                'bb_bandwidth': float(bb_bandwidth.iloc[-1]) if not pd.isna(bb_bandwidth.iloc[-1]) else 0.0,
                'stoch_k': float(stoch_data['%K'].iloc[-1]) if not pd.isna(stoch_data['%K'].iloc[-1]) else 50.0,
                'stoch_d': float(stoch_data['%D'].iloc[-1]) if not pd.isna(stoch_data['%D'].iloc[-1]) else 50.0,
                'atr': float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 1.0, # Default small ATR
                'obv': float(obv.iloc[-1]) if not pd.isna(obv.iloc[-1]) else 0.0,
                'volume': float(df['volume'].iloc[-1]) if not pd.isna(df['volume'].iloc[-1]) else 0.0,
                'close': float(df['close'].iloc[-1]) if not pd.isna(df['close'].iloc[-1]) else 0.0
            }

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            # Return a default dict with neutral values to prevent downstream errors
            return {
                'sma': 0.0, 'ema_fast': 0.0, 'ema_slow': 0.0, 'rsi': 50.0,
                'macd_diff': 0.0, 'bb_upper': 0.0, 'bb_lower': 0.0, 'bb_bandwidth': 0.0,
                'stoch_k': 50.0, 'stoch_d': 50.0, 'atr': 1.0, 'obv': 0.0,
                'volume': 0.0, 'close': 0.0
            }

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
            score = int(score * 1.2) # Boost trending signals
            reasons.append("Trending Market")
        elif regime == "RANGING":
            score = int(score * 0.8) # Reduce ranging signals
            reasons.append("Ranging Market")

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
        try:
            low_min = df['low'].rolling(window=k_period, min_periods=1).min()
            high_max = df['high'].rolling(window=k_period, min_periods=1).max()

            # Avoid division by zero
            k_numerator = df['close'] - low_min
            k_denominator = (high_max - low_min).replace(0, 1e-10)
            k = 100 * (k_numerator / k_denominator)

            # Calculate %D (signal line)
            d = k.rolling(window=d_period, min_periods=1).mean()

            # Return as DataFrame
            return pd.DataFrame({'%K': k, '%D': d})

        except Exception as e:
            logger.error(f"Error calculating stochastic: {e}", exc_info=True)
            # Return a DataFrame with NaNs or neutral values to prevent downstream errors
            return pd.DataFrame({'%K': [50.0], '%D': [50.0]}, index=df.index[-1:])

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate the Average True Range (ATR).

        Args:
            df (pd.DataFrame): OHLC data.
            period (int): ATR period.

        Returns:
            pd.Series: ATR values.
        """
        try:
            # Calculate True Range components
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())

            # True range is the maximum of these three
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # Calculate ATR using Wilder's smoothing method
            atr = pd.Series(index=tr.index, dtype='float64')
            # Initial value is the simple average of the first `period` TR values
            atr.iloc[0] = tr.iloc[:period].mean() if len(tr) >= period else tr.mean()

            # Apply Wilder's recursive formula for subsequent values
            for i in range(1, len(atr)):
                atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + tr.iloc[i]) / period

            # Return the ATR Series
            return atr

        except Exception as e:
            logger.error(f"Error calculating ATR: {e}", exc_info=True)
            # Return a Series with NaN or a small default value
            return pd.Series([1.0], index=df.index[-1:], dtype='float64')

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the On-Balance Volume (OBV).

        Args:
            df (pd.DataFrame): OHLCV data.

        Returns:
            pd.Series: OBV values.
        """
        try:
            # Initialize OBV list
            obv = [0]  # Initial OBV value is 0

            # Iterate through the DataFrame to calculate OBV
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                    # If the closing price is higher, add volume
                    obv.append(obv[-1] + df['volume'].iloc[i])
                elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                    # If the closing price is lower, subtract volume
                    obv.append(obv[-1] - df['volume'].iloc[i])
                else:
                    # If the closing price is the same, OBV is unchanged
                    obv.append(obv[-1])

            # Convert list to Pandas Series with the same index as the input DataFrame
            return pd.Series(obv, index=df.index)

        except Exception as e:
            logger.error(f"Error calculating OBV: {e}", exc_info=True)
            # Return a Series with NaN or zero
            return pd.Series([0.0], index=df.index[-1:], dtype='float64')

# Example usage (if run directly)
# if __name__ == "__main__":
#     # Configure logging for testing
#     import os
#     os.makedirs("logs", exist_ok=True)
#     import logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.FileHandler("logs/strategy.log"),
#             logging.StreamHandler()
#         ]
#     )
#
#     # Example DataFrame (replace with actual data)
#     # data = pd.read_csv("your_data.csv", index_col='timestamp', parse_dates=True)
#     # signal = DynamicScalpingStrategy().generate_signal(data, data['close'].iloc[-1])
#     # print(signal)
#     print("DynamicScalpingStrategy class defined. Ready for use.")
