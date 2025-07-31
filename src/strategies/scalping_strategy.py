"""
Advanced scalping strategy combining multiple technical indicators to
generate option trading signals.  Signals are scored on an integer scale
and converted into a confidence score on a 1–10 range.  The strategy
also calculates adaptive stop‑loss and take‑profit levels using ATR and
supports market regime detection to adapt scoring for trending versus
ranging markets.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, Optional, Tuple, List

import pandas as pd

# Import Config via the package hierarchy.  Relative imports are not used
# here so that ``Config`` resolves correctly when the module is executed
# directly or via ``python -m``.  The ``src`` package is created via
# empty ``__init__.py`` files.
# Use a relative import so that this module can be loaded as part of the
# ``src`` package hierarchy.  See ``nifty_scalper_bot/src/__init__.py``.
from src.config import Config
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
    """
    A dynamic scalping strategy for Nifty options using a composite of
    technical indicators.  Each indicator contributes positively or
    negatively to an integer ``score``.  A valid trade signal is
    generated when the absolute score meets or exceeds
    ``scoring_threshold`` and the direction‑specific score meets
    ``min_score_threshold``.  The confidence value returned is on a
    0–10 scale.
    """

    def __init__(
        self,
        ema_fast_period: int = 9,
        ema_slow_period: int = 21,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        adx_period: int = 14,
        bb_window: int = 20,
        bb_std: int = 2,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0,
        base_stop_loss_points: float = Config.BASE_STOP_LOSS_POINTS,
        base_target_points: float = Config.BASE_TARGET_POINTS,
        confidence_threshold: float = Config.CONFIDENCE_THRESHOLD,
        scoring_threshold: int = 4,
    ) -> None:
        # Indicator parameters
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier

        # Strategy thresholds
        self.base_stop_loss_points = base_stop_loss_points
        self.base_target_points = base_target_points
        # Minimum integer score required for a trade (internal threshold)
        self.min_score_threshold = int(confidence_threshold)
        # Raw score threshold to determine direction; lower than max possible
        self.scoring_threshold = scoring_threshold

        # Precompute max possible score for confidence normalisation
        # EMA (±2) + RSI (±1) + MACD (±1) + SuperTrend (±2) + VWAP (±1) + regime (±1)
        self.max_possible_score = 8

        self.last_signal_hash: Optional[str] = None

    # ------------------------------------------------------------------
    # Indicator calculation
    # ------------------------------------------------------------------
    def _calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calculate indicator values based on the latest window of data."""
        try:
            ema_fast = calculate_ema(df, self.ema_fast_period)
            ema_slow = calculate_ema(df, self.ema_slow_period)
            rsi = calculate_rsi(df, self.rsi_period)
            macd_vals = calculate_macd(df)
            atr = calculate_atr(df, self.atr_period)
            supertrend_dir = calculate_supertrend(df, self.supertrend_period, self.supertrend_multiplier)
            vwap = calculate_vwap(df)
            adx = calculate_adx(df, self.adx_period)
            bb_width = calculate_bb_width(df, self.bb_window, self.bb_std)

            # Use the last available values
            idx = df.index[-1]
            indicators = {
                "ema_fast": float(ema_fast.loc[idx]),
                "ema_slow": float(ema_slow.loc[idx]),
                "rsi": float(rsi.loc[idx]),
                "macd": float(macd_vals["macd"].loc[idx]),
                "macd_signal": float(macd_vals["macd_signal"].loc[idx]),
                "macd_hist": float(macd_vals["macd_hist"].loc[idx]),
                "atr": float(atr.loc[idx]),
                "supertrend": int(supertrend_dir.loc[idx]),
                "vwap": float(vwap.loc[idx]),
                "adx": float(adx.loc[idx]),
                "bb_width": float(bb_width.loc[idx]),
            }
            return indicators
        except Exception as exc:
            logger.error(f"Indicator calculation failed: {exc}", exc_info=True)
            return None

    def _detect_market_regime(self, indicators: Dict[str, float]) -> str:
        """
        Detect whether the market is trending or ranging.  Returns
        ``'trend'`` if the ADX is above 25 and Bollinger width is above a
        small threshold, otherwise ``'range'``.
        """
        adx = indicators.get("adx", 0.0)
        bb_width = indicators.get("bb_width", 0.0)
        # Heuristic thresholds
        trending = adx >= 25 and bb_width >= 0.05
        return "trend" if trending else "range"

    def _score_signal(self, indicators: Dict[str, float], regime: str) -> Tuple[int, List[str]]:
        """
        Assign integer points based on indicator conditions.  Returns a tuple
        of (score, reasons) where ``reasons`` lists the contributing
        signals for logging/debugging.
        """
        score = 0
        reasons: List[str] = []

        # EMA crossover
        if indicators["ema_fast"] > indicators["ema_slow"]:
            score += 2
            reasons.append("EMA fast > slow (+2)")
        elif indicators["ema_fast"] < indicators["ema_slow"]:
            score -= 2
            reasons.append("EMA fast < slow (-2)")

        # RSI extremes
        rsi_val = indicators["rsi"]
        if rsi_val < 30:
            score += 1
            reasons.append("RSI oversold (+1)")
        elif rsi_val > 70:
            score -= 1
            reasons.append("RSI overbought (-1)")

        # MACD
        if indicators["macd"] > indicators["macd_signal"]:
            score += 1
            reasons.append("MACD above signal (+1)")
        elif indicators["macd"] < indicators["macd_signal"]:
            score -= 1
            reasons.append("MACD below signal (-1)")

        # SuperTrend direction
        if indicators["supertrend"] == 1:
            score += 2
            reasons.append("SuperTrend up (+2)")
        elif indicators["supertrend"] == -1:
            score -= 2
            reasons.append("SuperTrend down (-2)")

        # VWAP position
        # Use the current close for comparison
        if indicators["vwap"] < indicators["ema_fast"]:
            score += 1
            reasons.append("Price above VWAP (+1)")
        else:
            score -= 1
            reasons.append("Price below VWAP (-1)")

        # Regime based adjustment: trending markets favour momentum signals,
        # ranging markets favour mean reversion.  We award +1 if the regime
        # aligns with the dominant direction implied by other scores.
        if regime == "trend" and abs(score) >= self.scoring_threshold:
            # trending regime: encourage strong signals
            score += 1 if score > 0 else -1
            reasons.append("Trending regime adjustment")
        elif regime == "range" and abs(score) < self.scoring_threshold:
            # ranging regime: dampen weak signals
            score = int(score * 0.5)
            reasons.append("Ranging regime dampening")

        return score, reasons

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[Dict[str, object]]:
        """
        Generate a trading signal based on the provided OHLCV ``DataFrame``
        and current price.  Returns a dictionary containing the signal
        direction, score, confidence, stop‑loss and take‑profit levels or
        ``None`` if no trade should be taken.
        """
        # Sanity checks
        if df is None or df.empty:
            logger.warning("Strategy received empty DataFrame")
            return None
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            logger.error(f"DataFrame missing required columns: {required_cols - set(df.columns)}")
            return None
        # Ensure we have enough historical points
        min_required = max(
            self.ema_slow_period,
            self.rsi_period,
            self.atr_period,
            self.bb_window,
            self.adx_period,
            self.supertrend_period,
        ) + 5
        if len(df) < min_required:
            logger.warning(f"Insufficient data for indicators: have {len(df)}, need {min_required}")
            return None

        indicators = self._calculate_indicators(df)
        if not indicators:
            return None
        regime = self._detect_market_regime(indicators)
        score, reasons = self._score_signal(indicators, regime)
        logger.debug(f"Computed score {score} with reasons: {reasons}")

        # Determine direction if score meets scoring threshold
        direction: Optional[str] = None
        if score >= self.scoring_threshold:
            direction = "BUY"
        elif score <= -self.scoring_threshold:
            direction = "SELL"

        # Final filter: ensure minimum score threshold is met
        if not direction or abs(score) < self.min_score_threshold:
            return None

        # De‑duplicate signals
        signal_key = f"{direction}_{current_price}_{df.index[-1]}"
        new_hash = hashlib.md5(signal_key.encode()).hexdigest()
        if new_hash == self.last_signal_hash:
            logger.debug("Duplicate signal skipped")
            return None
        self.last_signal_hash = new_hash

        # Adaptive stop and target using ATR
        atr = indicators.get("atr", 0.0)
        sl_distance = max(self.base_stop_loss_points, atr * 2)
        tp_distance = max(self.base_target_points, atr * 3)
        if direction == "BUY":
            stop_loss = current_price - sl_distance
            target = current_price + tp_distance
        else:
            stop_loss = current_price + sl_distance
            target = current_price - tp_distance

        # Confidence on a 0–10 scale
        confidence_raw = (abs(score) / self.max_possible_score) * 10
        confidence = round(min(confidence_raw, 10.0), 1)

        return {
            "direction": direction,
            "score": score,
            "confidence": confidence,
            "stop_loss": float(stop_loss),
            "target": float(target),
        }
