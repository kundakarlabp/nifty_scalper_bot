# src/strategies/scalping_strategy.py
"""
Advanced scalping strategy combining multiple technical indicators to
generate trading signals for both spot/futures and options.

Signals are scored on an integer scale and converted into a 1–10 confidence
score. SL/TP are ATR-adaptive; regime detection nudges scoring.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.config import settings
from src.utils.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_supertrend,
    calculate_vwap,
    calculate_adx,
)
from src.utils.atr_helper import compute_atr_df, latest_atr_value

logger = logging.getLogger(__name__)


def _bollinger_bands_from_close(close: pd.Series, window: int, std: float) -> Tuple[pd.Series, pd.Series]:
    """Compute Bollinger Bands (upper, lower) from the close series using population std (ddof=0)."""
    ma = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = ma + std * sd
    lower = ma - std * sd
    return upper, lower


class EnhancedScalpingStrategy:
    """A dynamic scalping strategy for Nifty spot/futures/options."""

    def __init__(
        self,
        base_stop_loss_points: float = getattr(settings, "BASE_STOP_LOSS_POINTS", 20.0),
        base_target_points: float = getattr(settings, "BASE_TARGET_POINTS", 40.0),
        confidence_threshold: float = getattr(settings, "CONFIDENCE_THRESHOLD", 6.0),
        min_score_threshold: int = int(getattr(settings, "MIN_SIGNAL_SCORE", 5)),
        # Indicator params
        ema_fast_period: int = 9,
        ema_slow_period: int = 21,
        rsi_period: int = 14,
        rsi_overbought: int = 60,
        rsi_oversold: int = 40,
        macd_fast_period: int = 8,
        macd_slow_period: int = 17,
        macd_signal_period: int = 9,
        atr_period: int = getattr(settings, "ATR_PERIOD", 14),
        supertrend_atr_multiplier: float = 2.0,
        bb_window: int = 20,
        bb_std_dev: float = 2.0,
        adx_period: int = 14,
        adx_trend_strength: int = 25,
        vwap_period: int = 20,
        # Options params
        option_sl_percent: float = getattr(settings, "OPTION_SL_PERCENT", 0.05),
        option_tp_percent: float = getattr(settings, "OPTION_TP_PERCENT", 0.15),
    ) -> None:
        self.base_stop_loss_points = base_stop_loss_points
        self.base_target_points = base_target_points
        self.confidence_threshold = confidence_threshold
        self.min_score_threshold = min_score_threshold

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

        self.option_sl_percent = option_sl_percent
        self.option_tp_percent = option_tp_percent

        # We score 6 key axes: EMA, RSI, MACD hist sign, MACD zero-cross, Supertrend dir, VWAP.
        self.max_possible_score = 6
        self.last_signal_hash: Optional[str] = None

    # ------------------------------- internals ------------------------------- #

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate indicators needed by the scoring function."""
        indicators: Dict[str, pd.Series] = {}

        min_len = max(
            self.ema_slow_period,
            self.rsi_period,
            self.atr_period,
            self.bb_window,
            self.adx_period,
            self.vwap_period,
        ) + 10

        if len(df) < min_len:
            logger.debug(f"Insufficient data for indicators. Need {min_len}, got {len(df)}")
            return indicators

        # EMA
        indicators["ema_fast"] = calculate_ema(df, self.ema_fast_period)
        indicators["ema_slow"] = calculate_ema(df, self.ema_slow_period)

        # RSI
        indicators["rsi"] = calculate_rsi(df, self.rsi_period)

        # MACD
        macd_line, macd_signal, macd_hist = calculate_macd(
            df, self.macd_fast_period, self.macd_slow_period, self.macd_signal_period
        )
        indicators["macd_line"] = macd_line
        indicators["macd_signal"] = macd_signal
        indicators["macd_histogram"] = macd_hist

        # ATR (robust)
        indicators["atr"] = compute_atr_df(df, period=self.atr_period, method="rma")

        # Supertrend
        st_dir, st_u, st_l = calculate_supertrend(
            df, period=self.atr_period, multiplier=self.supertrend_atr_multiplier
        )
        indicators["supertrend"] = st_dir
        indicators["supertrend_upper"] = st_u
        indicators["supertrend_lower"] = st_l

        # Bollinger (upper/lower)
        bb_u, bb_l = _bollinger_bands_from_close(df["close"], self.bb_window, self.bb_std_dev)
        indicators["bb_upper"] = bb_u
        indicators["bb_lower"] = bb_l

        # ADX + DI
        adx, di_pos, di_neg = calculate_adx(df, period=self.adx_period)
        indicators["adx"] = adx
        indicators["di_plus"] = di_pos
        indicators["di_minus"] = di_neg

        # VWAP (rolling)
        indicators["vwap"] = calculate_vwap(df, period=self.vwap_period)

        return indicators

    def _detect_market_regime(
        self, df: pd.DataFrame, adx: pd.Series, di_plus: pd.Series, di_minus: pd.Series
    ) -> str:
        """Rough regime classification for nudging the score."""
        if len(adx) < 2 or len(di_plus) < 2 or len(di_minus) < 2:
            return "unknown"

        current_adx = adx.iloc[-1]
        current_di_plus = di_plus.iloc[-1]
        current_di_minus = di_minus.iloc[-1]

        if current_adx > self.adx_trend_strength and abs(current_di_plus - current_di_minus) > 10:
            return "trend_up" if current_di_plus > current_di_minus else "trend_down"
        return "range"

    def _score_signal(
        self, df: pd.DataFrame, indicators: Dict[str, pd.Series], current_price: float
    ) -> Tuple[int, List[str]]:
        """Scoring based on indicator confluence."""
        if not indicators:
            return 0, ["No indicators calculated"]

        score = 0
        reasons: List[str] = []
        last_idx = df.index[-1]

        ema_fast = indicators["ema_fast"].loc[last_idx]
        ema_slow = indicators["ema_slow"].loc[last_idx]
        rsi = indicators["rsi"].loc[last_idx]
        macd_hist = indicators["macd_histogram"].loc[last_idx]
        supertrend = indicators["supertrend"].loc[last_idx]
        adx = indicators["adx"]
        di_plus = indicators["di_plus"]
        di_minus = indicators["di_minus"]
        vwap = indicators["vwap"].loc[last_idx]
        bb_upper = indicators["bb_upper"].loc[last_idx]
        bb_lower = indicators["bb_lower"].loc[last_idx]

        # 1) EMA
        if pd.notna(ema_fast) and pd.notna(ema_slow):
            if ema_fast > ema_slow:
                score += 1
                reasons.append("EMA Fast > EMA Slow")
            elif ema_fast < ema_slow:
                score -= 1
                reasons.append("EMA Fast < EMA Slow")

        # 2) RSI
        if pd.notna(rsi):
            if rsi < self.rsi_oversold:
                score += 1
                reasons.append("RSI Oversold")
            elif rsi > self.rsi_overbought:
                score -= 1
                reasons.append("RSI Overbought")

        # 3) MACD histogram sign
        if pd.notna(macd_hist):
            if macd_hist > 0:
                score += 1
                reasons.append("MACD Histogram > 0")
            elif macd_hist < 0:
                score -= 1
                reasons.append("MACD Histogram < 0")

        # 4) MACD zero-cross (lookback 1)
        if len(indicators["macd_histogram"]) >= 2:
            prev_macd_hist = indicators["macd_histogram"].iloc[-2]
            if pd.notna(prev_macd_hist) and pd.notna(macd_hist):
                if prev_macd_hist <= 0 < macd_hist:
                    score += 1
                    reasons.append("MACD Zero Cross Up")
                elif prev_macd_hist >= 0 > macd_hist:
                    score -= 1
                    reasons.append("MACD Zero Cross Down")

        # 5) Supertrend dir (light touch with band alignment)
        if pd.notna(supertrend):
            st_lower = indicators["supertrend_lower"].loc[last_idx]
            st_upper = indicators["supertrend_upper"].loc[last_idx]
            if supertrend == 1 and pd.notna(st_lower) and current_price > st_lower:
                score += 1
                reasons.append("Price aligned with Supertrend Up")
            elif supertrend == -1 and pd.notna(st_upper) and current_price < st_upper:
                score -= 1
                reasons.append("Price aligned with Supertrend Down")

        # 6) VWAP
        if pd.notna(vwap):
            if current_price > vwap:
                score += 1
                reasons.append("Price > VWAP")
            elif current_price < vwap:
                score -= 1
                reasons.append("Price < VWAP")

        # BB as contextual filter (don’t count to max_possible_score)
        if pd.notna(bb_upper) and pd.notna(bb_lower):
            if current_price < bb_lower:
                reasons.append("Price < BB Lower")
            elif current_price > bb_upper:
                reasons.append("Price > BB Upper")

        # Regime nudge (softer than multiplicative scaling)
        regime = self._detect_market_regime(df, adx, di_plus, di_minus)
        if regime == "trend_up" and score >= 0:
            score += 1
            reasons.append("Trending Up Regime")
        elif regime == "trend_down" and score <= 0:
            score -= 1
            reasons.append("Trending Down Regime")
        elif regime == "range":
            if score > 0:
                score -= 1  # soft penalty
            reasons.append("Ranging Regime")

        return score, reasons

    # --------------------------- public API: spot/fut --------------------------- #

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_price: float,
        spot_df: Optional[pd.DataFrame] = None,  # accepted for runner compatibility
    ) -> Optional[Dict[str, Any]]:
        """Generate a signal for spot/futures (also used on options DF by the runner)."""
        if df is None or df.empty:
            logger.debug("Strategy received empty DataFrame")
            return None

        required_cols = {"open", "high", "low", "close"}  # volume optional
        if not required_cols.issubset(df.columns):
            logger.error(f"DataFrame missing required columns: {required_cols - set(df.columns)}")
            return None

        min_required = max(
            self.ema_slow_period,
            self.rsi_period,
            self.atr_period,
            self.bb_window,
            self.adx_period,
            self.vwap_period,
        ) + 5
        if len(df) < min_required:
            logger.debug(f"Insufficient data for signal generation. Need {min_required}, got {len(df)}")
            return None

        try:
            indicators = self._calculate_indicators(df)
            if not indicators:
                return None

            score, reasons = self._score_signal(df, indicators, current_price)

            direction: Optional[str] = None
            if score >= self.min_score_threshold:
                direction = "BUY"
            elif score <= -self.min_score_threshold:
                direction = "SELL"

            if not direction:
                logger.debug(f"Score {score} below threshold {self.min_score_threshold}. No trade.")
                return None

            # De-duplicate signals
            signal_key = f"{direction}_{round(current_price, 2)}_{df.index[-1]}"
            new_hash = hashlib.md5(signal_key.encode()).hexdigest()
            if new_hash == self.last_signal_hash:
                logger.debug("Duplicate signal skipped")
                return None
            self.last_signal_hash = new_hash

            # Confidence on 0–10 based on score magnitude
            normalized = min(abs(score) / max(1, self.max_possible_score), 1.0)
            confidence = max(1.0, min(10.0, normalized * 10.0))

            # ATR-based SL/TP (fallback to fixed points if ATR missing)
            atr_series = indicators.get("atr", pd.Series(0.0, index=df.index))
            atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
            if atr_value <= 0:
                v = latest_atr_value(df, period=self.atr_period, method="rma")
                atr_value = float(v) if v is not None else 0.0

            if atr_value <= 0:
                sl_points = self.base_stop_loss_points
                tp_points = self.base_target_points
            else:
                sl_mult = float(getattr(settings, "ATR_SL_MULTIPLIER", 1.5))
                tp_mult = float(getattr(settings, "ATR_TP_MULTIPLIER", 3.0))
                sl_points = atr_value * sl_mult
                tp_points = atr_value * tp_mult

            # Confidence adjustments
            sl_adj = float(getattr(settings, "SL_CONFIDENCE_ADJ", 0.2))
            tp_adj = float(getattr(settings, "TP_CONFIDENCE_ADJ", 0.3))
            sl_points *= (1 + (10 - confidence) * sl_adj / 10.0)
            tp_points *= (1 + (confidence - 5) * tp_adj / 10.0)

            entry_price = float(current_price)
            stop_loss = entry_price - sl_points if direction == "BUY" else entry_price + sl_points
            target = entry_price + tp_points if direction == "BUY" else entry_price - tp_points

            stop_loss = max(0.0, float(stop_loss))
            target = max(0.0, float(target))

            result = {
                "signal": direction,
                "score": int(score),
                "confidence": round(confidence, 2),
                "entry_price": round(entry_price, 2),
                "stop_loss": round(stop_loss, 2),
                "target": round(target, 2),
                "reasons": reasons,
                "market_volatility": round(float(atr_value), 2) if atr_value > 0 else 0.0,
            }
            logger.debug(f"Signal: {result}")
            return result

        except Exception as e:
            logger.error(f"Error generating signal: {e}", exc_info=True)
            return None

    # --------------------------- public API: options --------------------------- #

    def generate_options_signal(
        self,
        options_ohlc: pd.DataFrame,
        spot_ohlc: pd.DataFrame,
        strike_info: Dict[str, Any],
        current_option_price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Lightweight options signal framework:
        - price breakout on the option
        - volume confirmation (if present)
        - underlying trend confirmation
        """
        try:
            if options_ohlc is None or options_ohlc.empty:
                logger.debug("Options OHLC is empty")
                return None
            if len(options_ohlc) < 5:
                return None

            last_close = float(options_ohlc["close"].iloc[-2])
            curr_close = float(options_ohlc["close"].iloc[-1])

            # Volume check (graceful if missing or zeros)
            if "volume" in options_ohlc.columns and len(options_ohlc) > 10:
                recent = options_ohlc["volume"].iloc[-10:-1]
                avg_vol = float(recent.mean()) if recent.notna().any() else 0.0
                curr_vol = float(options_ohlc["volume"].iloc[-1] or 0.0)
                volume_condition = (avg_vol > 0) and (curr_vol > 1.5 * avg_vol)
            else:
                volume_condition = True

            breakout_pct = float(getattr(settings, "OPTION_BREAKOUT_PCT", 0.01))
            breakout_condition = curr_close > last_close * (1.0 + breakout_pct)

            spot_trend_bullish = False
            spot_trend_bearish = False
            spot_return = 0.0
            if spot_ohlc is not None and not spot_ohlc.empty and len(spot_ohlc) >= 5:
                spot_return = (spot_ohlc["close"].iloc[-1] / spot_ohlc["close"].iloc[-5]) - 1.0
                th = float(getattr(settings, "OPTION_SPOT_TREND_PCT", 0.005))
                if spot_return > th:
                    spot_trend_bullish = True
                elif spot_return < -th:
                    spot_trend_bearish = True

            option_type = str(strike_info.get("type", "")).upper()

            signal: Optional[Dict[str, Any]] = None
            if option_type == "CE" and breakout_condition and volume_condition and spot_trend_bullish:
                signal = {"signal": "BUY"}
            elif option_type == "PE" and breakout_condition and volume_condition and spot_trend_bearish:
                signal = {"signal": "BUY"}

            if not signal:
                return None

            sl_pct = float(getattr(settings, "OPTION_SL_PERCENT", self.option_sl_percent))
            tp_pct = float(getattr(settings, "OPTION_TP_PERCENT", self.option_tp_percent))

            entry = float(current_option_price)
            sl = round(entry * (1 - sl_pct), 2)
            tp = round(entry * (1 + tp_pct), 2)

            # Confidence: base 7 + bonuses from vol & spot strength (capped at 10)
            base_conf = 7.0
            if "volume" in options_ohlc.columns and len(options_ohlc) > 10:
                recent = options_ohlc["volume"].iloc[-10:-1]
                avg_vol = float(recent.mean()) if recent.notna().any() else 0.0
                curr_vol = float(options_ohlc["volume"].iloc[-1] or 0.0)
                vol_ratio = (curr_vol / avg_vol) if avg_vol > 0 else 1.0
            else:
                vol_ratio = 1.0

            vol_bonus = max(0.0, min(2.0, vol_ratio - 1.5))
            spot_bonus = max(0.0, min(2.0, abs(spot_return) * 200.0))
            confidence = min(10.0, base_conf + vol_bonus + spot_bonus)

            # Option ATR as volatility proxy (if available)
            opt_atr = latest_atr_value(options_ohlc, period=getattr(settings, "ATR_PERIOD", 14), method="rma")
            mv = float(opt_atr) if opt_atr is not None else 0.0

            signal.update(
                {
                    "entry_price": round(entry, 2),
                    "stop_loss": sl,
                    "target": tp,
                    "confidence": round(confidence, 2),
                    "market_volatility": round(mv, 4),
                    "strategy_notes": f"{option_type} breakout with underlying confirmation",
                }
            )
            return signal

        except Exception as e:
            logger.error(f"Error in generate_options_signal: {e}", exc_info=True)
            return None
