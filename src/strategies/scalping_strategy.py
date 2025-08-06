# src/strategies/scalping_strategy.py
"""
Advanced scalping strategy combining multiple technical indicators to
generate trading signals for both spot/futures and options.

For spot/futures, it uses a composite scoring system based on technical indicators.
For options, it provides a framework for generating signals based
on price action, volume, and underlying trend.

Signals for spot/futures are scored on an integer scale and converted into a confidence
score on a 1–10 range. The strategy also calculates adaptive stop-loss
and take-profit levels using ATR and supports market regime detection
to adapt scoring for trending versus ranging markets.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, Optional, Tuple, List, Any
import pandas as pd

# Corrected import assuming standard structure
from src.config import Config

# Assuming indicators.py is in src/utils/
from src.utils.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_atr,
    calculate_supertrend,
    calculate_vwap,
    calculate_adx,
    calculate_bb_width,
)

logger = logging.getLogger(__name__)


class EnhancedScalpingStrategy:
    """A dynamic scalping strategy for Nifty spot/futures/options."""

    def __init__(
        self,
        base_stop_loss_points: float = Config.BASE_STOP_LOSS_POINTS,
        base_target_points: float = Config.BASE_TARGET_POINTS,
        confidence_threshold: float = Config.CONFIDENCE_THRESHOLD,
        min_score_threshold: int = int(Config.MIN_SIGNAL_SCORE),
        ema_fast_period: int = 9,
        ema_slow_period: int = 21,
        rsi_period: int = 14,
        rsi_overbought: int = 60,
        rsi_oversold: int = 40,
        macd_fast_period: int = 12,
        macd_slow_period: int = 26,
        macd_signal_period: int = 9,
        atr_period: int = 14,
        supertrend_atr_multiplier: float = 2.0,
        bb_window: int = 20,
        bb_std_dev: float = 2.0,
        adx_period: int = 14,
        adx_trend_strength: int = 25,
        vwap_period: int = 20,
        # --- New parameters for options ---
        option_sl_percent: float = 0.05, # 5% Stop Loss
        option_tp_percent: float = 0.15, # 15% Target
        # --- End new parameters ---
    ) -> None:
        self.base_stop_loss_points = base_stop_loss_points
        self.base_target_points = base_target_points
        self.confidence_threshold = confidence_threshold
        self.min_score_threshold = min_score_threshold

        # Indicator parameters
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast_period = macd_fast_period
        self.macd_slow_period = macd_slow_period
        self.macd_signal_period = macd_signal_period
        self.atr_period = atr_period
        self.supertrend_atr_multiplier = supertrend_atr_multiplier
        self.bb_window = bb_window
        self.bb_std_dev = bb_std_dev
        self.adx_period = adx_period
        self.adx_trend_strength = adx_trend_strength
        self.vwap_period = vwap_period

        # --- New parameters for options ---
        # Load from Config or use defaults passed to __init__
        self.option_sl_percent = getattr(Config, 'OPTION_SL_PERCENT', option_sl_percent)
        self.option_tp_percent = getattr(Config, 'OPTION_TP_PERCENT', option_tp_percent)
        # --- End new parameters ---

        # Precompute max possible score for confidence normalisation (based on indicator logic)
        # EMA (1), RSI (1), MACD (2), Supertrend (1), BB (1), ADX (1), VWAP (1) = 8
        self.max_possible_score = 8
        self.last_signal_hash: Optional[str] = None

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculates all required technical indicators."""
        indicators = {}
        # Ensure enough data
        min_len = max(self.ema_slow_period, self.rsi_period, self.atr_period,
                      self.bb_window, self.adx_period, self.vwap_period) + 10
        if len(df) < min_len:
            logger.warning(f"Insufficient data for indicator calculation. Need {min_len}, got {len(df)}")
            return indicators # Return empty dict

        # EMA
        indicators['ema_fast'] = calculate_ema(df['close'], self.ema_fast_period)
        indicators['ema_slow'] = calculate_ema(df['close'], self.ema_slow_period)

        # RSI
        indicators['rsi'] = calculate_rsi(df['close'], self.rsi_period)

        # MACD
        macd_line, signal_line, hist = calculate_macd(df['close'], self.macd_fast_period,
                                                     self.macd_slow_period, self.macd_signal_period)
        indicators['macd_line'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = hist

        # ATR
        indicators['atr'] = calculate_atr(df['high'], df['low'], df['close'], self.atr_period)

        # Supertrend
        supertrend, supertrend_u, supertrend_l = calculate_supertrend(
            df['high'], df['low'], df['close'], self.atr_period, self.supertrend_atr_multiplier
        )
        indicators['supertrend'] = supertrend
        indicators['supertrend_upper'] = supertrend_u
        indicators['supertrend_lower'] = supertrend_l

        # Bollinger Bands
        bb_upper, bb_lower = calculate_bb_width(df['close'], self.bb_window, self.bb_std_dev)
        indicators['bb_upper'] = bb_upper
        indicators['bb_lower'] = bb_lower

        # ADX
        adx, di_plus, di_minus = calculate_adx(df['high'], df['low'], df['close'], self.adx_period)
        indicators['adx'] = adx
        indicators['di_plus'] = di_plus
        indicators['di_minus'] = di_minus

        # VWAP
        indicators['vwap'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'], self.vwap_period)

        return indicators

    def _detect_market_regime(self, df: pd.DataFrame, adx: pd.Series, di_plus: pd.Series, di_minus: pd.Series) -> str:
        """Detects if the market is trending or ranging."""
        if len(adx) < 2 or len(di_plus) < 2 or len(di_minus) < 2:
             return "unknown"
        # Simple ADX + DI logic
        current_adx = adx.iloc[-1]
        current_di_plus = di_plus.iloc[-1]
        current_di_minus = di_minus.iloc[-1]

        if current_adx > self.adx_trend_strength and abs(current_di_plus - current_di_minus) > 10:
            if current_di_plus > current_di_minus:
                return "trend_up"
            else:
                return "trend_down"
        else:
            return "range"

    def _score_signal(self, df: pd.DataFrame, indicators: Dict[str, pd.Series], current_price: float) -> Tuple[int, List[str]]:
        """Scores the potential signal based on indicator confluence."""
        if not indicators:
            return 0, ["No indicators calculated"]

        score = 0
        reasons = []
        # Get last values
        last_idx = df.index[-1]
        ema_fast = indicators['ema_fast'].loc[last_idx]
        ema_slow = indicators['ema_slow'].loc[last_idx]
        rsi = indicators['rsi'].loc[last_idx]
        macd_hist = indicators['macd_histogram'].loc[last_idx]
        supertrend = indicators['supertrend'].loc[last_idx]
        bb_upper = indicators['bb_upper'].loc[last_idx]
        bb_lower = indicators['bb_lower'].loc[last_idx]
        adx = indicators['adx']
        di_plus = indicators['di_plus']
        di_minus = indicators['di_minus']
        vwap = indicators['vwap'].loc[last_idx]

        # 1. EMA Crossover
        if ema_fast > ema_slow:
            score += 1
            reasons.append("EMA Fast > EMA Slow")
        elif ema_fast < ema_slow:
            score -= 1
            reasons.append("EMA Fast < EMA Slow")

        # 2. RSI
        if rsi < self.rsi_oversold:
            score += 1
            reasons.append("RSI Oversold")
        elif rsi > self.rsi_overbought:
            score -= 1
            reasons.append("RSI Overbought")

        # 3. MACD Histogram
        if macd_hist > 0:
            score += 1
            reasons.append("MACD Histogram > 0")
        elif macd_hist < 0:
            score -= 1
            reasons.append("MACD Histogram < 0")

        # 4. MACD Zero Cross (check previous bar)
        if len(indicators['macd_histogram']) >= 2:
            prev_macd_hist = indicators['macd_histogram'].iloc[-2]
            if prev_macd_hist <= 0 and macd_hist > 0:
                score += 1
                reasons.append("MACD Zero Cross Up")
            elif prev_macd_hist >= 0 and macd_hist < 0:
                score -= 1
                reasons.append("MACD Zero Cross Down")

        # 5. Supertrend
        if current_price > supertrend:
            score += 1
            reasons.append("Price > Supertrend")
        elif current_price < supertrend:
            score -= 1
            reasons.append("Price < Supertrend")

        # 6. Bollinger Bands
        if current_price < bb_lower:
            score += 1
            reasons.append("Price < BB Lower")
        elif current_price > bb_upper:
            score -= 1
            reasons.append("Price > BB Upper")

        # 7. ADX Trend Strength
        regime = self._detect_market_regime(df, adx, di_plus, di_minus)
        if regime == "trend_up" and score >= 0: # Encourage longs in uptrend
            score += 1
            reasons.append("Trending Up Regime")
        elif regime == "trend_down" and score <= 0: # Encourage shorts in downtrend
            score -= 1
            reasons.append("Trending Down Regime")
        elif regime == "range":
            # In range, reduce score magnitude
            score = int(score * 0.7)
            reasons.append("Ranging Regime")

        # 8. VWAP
        if current_price > vwap:
            score += 0.5 # Smaller weight
            reasons.append("Price > VWAP")
        elif current_price < vwap:
            score -= 0.5
            reasons.append("Price < VWAP")

        return score, reasons

    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[Dict[str, Any]]:
        """Generate a trading signal for spot/futures based on technical indicators."""
        # Sanity checks
        if df is None or df.empty:
            logger.warning("Strategy received empty DataFrame")
            return None
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            logger.error(f"DataFrame missing required columns: {required_cols - set(df.columns)}")
            return None

        # Ensure we have enough historical points
        min_required = max(self.ema_slow_period, self.rsi_period, self.atr_period,
                           self.bb_window, self.adx_period, self.vwap_period) + 5
        if len(df) < min_required:
            logger.debug(f"Insufficient data for signal generation. Need {min_required}, got {len(df)}")
            return None

        try:
            indicators = self._calculate_indicators(df)
            if not indicators:
                return None

            score, reasons = self._score_signal(df, indicators, current_price)

            direction: Optional[str] = None
            if score >= self.min_score_threshold: # Use min_score_threshold for BUY/SELL decision
                direction = "BUY"
            elif score <= -self.min_score_threshold:
                direction = "SELL"

            # Final filter: ensure minimum score threshold is met for direction
            if not direction:
                logger.debug(f"Signal score {score} below min threshold {self.min_score_threshold} for direction.")
                return None

            # De-duplicate signals
            signal_key = f"{direction}_{current_price}_{df.index[-1]}"
            new_hash = hashlib.md5(signal_key.encode()).hexdigest()
            if new_hash == self.last_signal_hash:
                logger.debug("Duplicate signal skipped")
                return None
            self.last_signal_hash = new_hash

            # Compute confidence on a 0–10 scale based on the absolute score.
            # Normalize score by max possible score and scale to 0-10
            normalized_score = min(abs(score) / self.max_possible_score, 1.0)
            confidence = max(1.0, min(10.0, normalized_score * 10))

            # Calculate adaptive SL and TP using ATR
            atr_value = indicators.get('atr', pd.Series([0])).iloc[-1] if not indicators['atr'].empty else 0
            if atr_value <= 0:
                # Fallback to fixed points if ATR is not available/calculation failed
                sl_points = self.base_stop_loss_points
                tp_points = self.base_target_points
            else:
                # Example adaptive logic: SL = 1.5 * ATR, TP = 3 * ATR
                sl_points = atr_value * Config.ATR_SL_MULTIPLIER
                tp_points = atr_value * Config.ATR_TP_MULTIPLIER

            # Apply confidence-based adjustments
            sl_points *= (1 + (10 - confidence) * Config.SL_CONFIDENCE_ADJ) # Wider SL if low confidence
            tp_points *= (1 + (confidence - 5) * Config.TP_CONFIDENCE_ADJ)  # Higher TP if high confidence

            entry_price = current_price
            stop_loss = entry_price - sl_points if direction == "BUY" else entry_price + sl_points
            take_profit = entry_price + tp_points if direction == "BUY" else entry_price - tp_points

            # Ensure SL/TP are positive and reasonable
            stop_loss = max(0, stop_loss)
            take_profit = max(0, take_profit)

            signal_result = {
                "signal": direction,
                "score": score,
                "confidence": round(confidence, 2),
                "entry_price": round(entry_price, 2),
                "stop_loss": round(stop_loss, 2),
                "target": round(take_profit, 2),
                "reasons": reasons,
                "market_volatility": round(atr_value, 2) if atr_value > 0 else 0.0
            }

            logger.debug(f"✅ Signal generated: {signal_result}")
            return signal_result

        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return None

    def generate_options_signal(
        self,
        options_ohlc: pd.DataFrame, # OHLC data for the specific option contract
        spot_ohlc: pd.DataFrame,   # OHLC data for the underlying spot (for context)
        strike_info: Dict[str, Any], # Information about the strike (from strike_selector)
        current_option_price: float # Last traded price of the option
    ) -> Optional[Dict[str, Any]]:
        """
        Generates a trading signal for an options contract.

        This is a basic example framework. A real options strategy would involve:
        - Analyzing the option's price action (options_ohlc).
        - Analyzing the underlying's trend and volatility (spot_ohlc).
        - Using Greeks (Delta, Gamma, Theta, Vega) - might need to fetch or calculate.
        - Considering Open Interest (OI) changes (if data available).
        - Applying specific options strategies (Long Call, Long Put, Straddle, Butterfly, etc.).

        Args:
            options_ohlc: DataFrame with OHLCV data for the option.
            spot_ohlc: DataFrame with OHLCV data for the underlying.
            strike_info: Dictionary with details like strike price, type (CE/PE), delta, etc.
            current_option_price: The current market price of the option.

        Returns:
            A dictionary containing the signal details, or None if no signal.
            Example:
            {
                "signal": "BUY",
                "entry_price": current_option_price,
                "stop_loss": sl_price,
                "target": tp_price,
                "confidence": 8.5, # 1-10 scale
                "market_volatility": 0.6, # Example metric
                "strategy_notes": "Bullish breakout on high volume with rising spot"
            }
        """
        try:
            signal_dict = {
                "signal": None,
                "entry_price": current_option_price,
                "stop_loss": None,
                "target": None,
                "confidence": 0.0,
                "market_volatility": 0.0,
                "strategy_notes": ""
            }

            # Ensure we have data
            if options_ohlc is None or options_ohlc.empty:
                logger.warning("No OHLC data provided for options signal generation.")
                return None

            if len(options_ohlc) < 5:
                logger.debug("Insufficient data points for options signal generation.")
                return None

            # --- Example Signal Logic (Replace with your strategy) ---

            # 1. Basic Price Action Breakout (Conceptual)
            # Check for a breakout in the option price with increasing volume (if available)
            last_close = options_ohlc['close'].iloc[-2]
            current_close = options_ohlc['close'].iloc[-1]
            # Kite historical for options might not have volume, check
            if 'volume' in options_ohlc.columns and len(options_ohlc) > 10:
                last_volume = options_ohlc['volume'].iloc[-2]
                current_volume = options_ohlc['volume'].iloc[-1]
                avg_volume = options_ohlc['volume'][-10:-1].mean()
                volume_condition = current_volume > avg_volume * 1.5
            else:
                volume_condition = True # If volume data is unreliable, ignore this check

            # Simple breakout condition using configurable percentage
            breakout_threshold = 1.0 + (getattr(Config, 'OPTION_BREAKOUT_PCT', 0.01)) # Default 1%
            breakout_condition = current_close > last_close * breakout_threshold

            # 2. Underlying Trend Confirmation (Conceptual)
            spot_trend_bullish = False
            spot_trend_bearish = False
            spot_return = 0.0
            if spot_ohlc is not None and not spot_ohlc.empty and len(spot_ohlc) >= 5:
                spot_return = (spot_ohlc['close'].iloc[-1] / spot_ohlc['close'].iloc[-5]) - 1
                spot_trend_threshold = getattr(Config, 'OPTION_SPOT_TREND_PCT', 0.005) # Default 0.5%
                if spot_return > spot_trend_threshold:
                    spot_trend_bullish = True
                elif spot_return < -spot_trend_threshold:
                    spot_trend_bearish = True

            # 3. Strike Selection Context (Conceptual)
            # Use information from strike_info (e.g., if it's ATM, OTM, Delta)
            is_atm = strike_info.get('is_atm', False)
            option_type = strike_info.get('type', '')
            # delta = strike_info.get('delta', 0.5) # If calculated/fetched

            # --- Signal Generation ---
            # Example: Buy Call if option breaks out and spot is bullish
            if option_type == 'CE' and breakout_condition and volume_condition and spot_trend_bullish:
                signal_dict["signal"] = "BUY"
                # Use configurable SL/TP percentages
                signal_dict["stop_loss"] = round(current_option_price * (1 - self.option_sl_percent), 2)
                signal_dict["target"] = round(current_option_price * (1 + self.option_tp_percent), 2)

                # --- Dynamic Confidence Scaling (Example) ---
                # Base confidence
                base_confidence = 7.0
                # Increase based on spot strength and volume
                vol_ratio = (current_volume / avg_volume) if 'volume' in options_ohlc.columns and avg_volume > 0 else 1
                vol_bonus = max(0, min(2.0, (vol_ratio - 1.5))) # Cap bonus at 2.0
                spot_bonus = max(0, min(2.0, abs(spot_return) * 200)) # Scale spot return (0.5% -> 1.0 bonus point)
                dynamic_confidence = min(10.0, base_confidence + vol_bonus + spot_bonus)
                # --- End Dynamic Confidence ---

                signal_dict["confidence"] = dynamic_confidence
                signal_dict["market_volatility"] = 0.5 # Placeholder, could use ATR or IV
                signal_dict["strategy_notes"] = "CE breakout with bullish spot trend"
                logger.info(f"Generated BUY signal for CE option {strike_info.get('symbol', 'N/A')}")

            # Example: Buy Put if option breaks out and spot is bearish
            elif option_type == 'PE' and breakout_condition and volume_condition and spot_trend_bearish:
                signal_dict["signal"] = "BUY"
                signal_dict["stop_loss"] = round(current_option_price * (1 - self.option_sl_percent), 2)
                signal_dict["target"] = round(current_option_price * (1 + self.option_tp_percent), 2)

                # --- Dynamic Confidence Scaling (Example) ---
                base_confidence = 7.0
                vol_ratio = (current_volume / avg_volume) if 'volume' in options_ohlc.columns and avg_volume > 0 else 1
                vol_bonus = max(0, min(2.0, (vol_ratio - 1.5)))
                spot_bonus = max(0, min(2.0, abs(spot_return) * 200))
                dynamic_confidence = min(10.0, base_confidence + vol_bonus + spot_bonus)
                # --- End Dynamic Confidence ---

                signal_dict["confidence"] = dynamic_confidence
                signal_dict["market_volatility"] = 0.5 # Placeholder
                signal_dict["strategy_notes"] = "PE breakout with bearish spot trend"
                logger.info(f"Generated BUY signal for PE option {strike_info.get('symbol', 'N/A')}")

            # --- End Example Logic ---

            # If a signal was generated, return it
            if signal_dict["signal"]:
                return signal_dict
            else:
                logger.debug(f"No options signal generated for {strike_info.get('symbol', 'N/A')}")
                return None

        except Exception as e:
            logger.error(f"Error in generate_options_signal: {e}", exc_info=True)
            return None
