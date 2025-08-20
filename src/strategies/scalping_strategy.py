# src/strategies/scalping_strategy.py
"""
Advanced scalping strategy combining multiple technical indicators to
generate trading signals for both spot/futures and options.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, List

import pandas as pd

from src.config import StrategyConfig
from src.signals.signal import Signal
from src.signals.regime_detector import detect_market_regime
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
    """Compute Bollinger Bands (upper, lower) from the close series (ddof=0)."""
    ma = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = ma + std * sd
    lower = ma - std * sd
    return upper, lower


class EnhancedScalpingStrategy:
    """A dynamic scalping strategy for Nifty spot/futures/options."""

    def __init__(self, config: StrategyConfig):
        if not isinstance(config, StrategyConfig):
            raise TypeError("A valid StrategyConfig instance is required.")
        self.config = config

        # Indicator parameters
        self.ema_fast_period = 9
        self.ema_slow_period = 21
        self.rsi_period = 14
        self.rsi_overbought = 60
        self.rsi_oversold = 40
        self.macd_fast_period = 8
        self.macd_slow_period = 17
        self.macd_signal_period = 9
        self.supertrend_atr_multiplier = 2.0
        self.bb_window = 20
        self.bb_std_dev = 2.0
        self.adx_period = 14
        self.adx_trend_strength = 25   # regime threshold
        self.vwap_period = 20
        self.max_possible_score = 6
        self.last_signal_hash: Optional[str] = None

    # --------------------------- indicators --------------------------- #

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate all technical indicators required for the strategy."""
        indicators: Dict[str, pd.Series] = {}
        if df is None or df.empty:
            return indicators

        # Need enough history before computing
        min_required = max(
            self.ema_slow_period,
            self.rsi_period,
            self.config.atr_period,
            self.bb_window,
            self.adx_period,
            self.vwap_period,
        )
        if len(df) < min_required:
            return indicators

        try:
            indicators["ema_fast"] = calculate_ema(df, self.ema_fast_period)
            indicators["ema_slow"] = calculate_ema(df, self.ema_slow_period)

            indicators["rsi"] = calculate_rsi(df, self.rsi_period)

            macd_line, macd_signal, macd_hist = calculate_macd(
                df, self.macd_fast_period, self.macd_slow_period, self.macd_signal_period
            )
            indicators["macd_line"] = macd_line
            indicators["macd_signal"] = macd_signal
            indicators["macd_histogram"] = macd_hist

            indicators["atr"] = compute_atr_df(df, period=self.config.atr_period, method="rma")

            st_dir, st_u, st_l = calculate_supertrend(
                df, period=self.config.atr_period, multiplier=self.supertrend_atr_multiplier
            )
            indicators["supertrend"] = st_dir
            indicators["supertrend_upper"] = st_u
            indicators["supertrend_lower"] = st_l

            indicators["bb_upper"], indicators["bb_lower"] = _bollinger_bands_from_close(
                df["close"], self.bb_window, self.bb_std_dev
            )

            adx, di_pos, di_neg = calculate_adx(df, period=self.adx_period)
            indicators["adx"] = adx
            indicators["di_plus"] = di_pos
            indicators["di_minus"] = di_neg

            indicators["vwap"] = calculate_vwap(df, period=self.vwap_period)
        except Exception as e:
            logger.error("Indicator calc failed: %s", e, exc_info=True)
            return {}

        return indicators

    # ----------------------------- scoring ---------------------------- #

    def _score_signal(
        self, df: pd.DataFrame, indicators: Dict[str, pd.Series], current_price: float
    ) -> Tuple[int, List[str]]:
        """Scoring based on indicator confluence."""
        if not indicators:
            return 0, ["No indicators calculated"]

        # Guard against NaN on last bar; skip if any core indicator is NaN
        core_keys = ("ema_fast", "ema_slow", "rsi", "macd_histogram", "supertrend", "bb_upper", "bb_lower", "vwap")
        for k in core_keys:
            s = indicators.get(k)
            if s is not None and (len(s) == 0 or pd.isna(s.iloc[-1])):
                return 0, [f"Indicator {k} NaN on last bar"]

        last_idx = -1
        score = 0
        reasons: List[str] = []

        ema_fast = indicators["ema_fast"].iloc[last_idx]
        ema_slow = indicators["ema_slow"].iloc[last_idx]
        rsi = indicators["rsi"].iloc[last_idx]
        macd_hist = indicators["macd_histogram"].iloc[last_idx]
        supertrend = indicators["supertrend"].iloc[last_idx]
        bb_upper = indicators["bb_upper"].iloc[last_idx]
        bb_lower = indicators["bb_lower"].iloc[last_idx]
        adx = indicators["adx"]
        di_plus = indicators["di_plus"]
        di_minus = indicators["di_minus"]
        vwap = indicators["vwap"].iloc[last_idx]

        # 1) EMA
        if pd.notna(ema_fast) and pd.notna(ema_slow):
            if ema_fast > ema_slow:
                score += 1; reasons.append("EMA Fast > EMA Slow")
            elif ema_fast < ema_slow:
                score -= 1; reasons.append("EMA Fast < EMA Slow")

        # 2) RSI
        if pd.notna(rsi):
            if rsi < self.rsi_oversold:
                score += 1; reasons.append("RSI Oversold")
            elif rsi > self.rsi_overbought:
                score -= 1; reasons.append("RSI Overbought")

        # 3) MACD histogram sign
        if pd.notna(macd_hist):
            if macd_hist > 0:
                score += 1; reasons.append("MACD Histogram > 0")
            elif macd_hist < 0:
                score -= 1; reasons.append("MACD Histogram < 0")

        # 4) MACD zero-cross (lookback 1)
        if len(indicators["macd_histogram"]) >= 2:
            prev_macd_hist = indicators["macd_histogram"].iloc[-2]
            if pd.notna(prev_macd_hist) and pd.notna(macd_hist):
                if prev_macd_hist <= 0 < macd_hist:
                    score += 1; reasons.append("MACD Zero Cross Up")
                elif prev_macd_hist >= 0 > macd_hist:
                    score -= 1; reasons.append("MACD Zero Cross Down")

        # 5) Supertrend dir
        if pd.notna(supertrend):
            if supertrend == 1:
                score += 1; reasons.append("Supertrend Up")
            elif supertrend == -1:
                score -= 1; reasons.append("Supertrend Down")

        # 6) VWAP
        if pd.notna(vwap):
            if current_price > vwap:
                score += 1; reasons.append("Price > VWAP")
            elif current_price < vwap:
                score -= 1; reasons.append("Price < VWAP")

        # 7) Regime nudge (shared detector with smoothing)
        try:
            regime = detect_market_regime(
                df,
                adx=adx,
                di_plus=di_plus,
                di_minus=di_minus,
                adx_trend_strength=self.adx_trend_strength,
            )
        except Exception:
            regime = "unknown"

        if regime == "trend_up" and score > 0:
            score += 1; reasons.append("Trending Up Regime")
        elif regime == "trend_down" and score < 0:
            score -= 1; reasons.append("Trending Down Regime")

        return score, reasons

    # --------------------------- signal API --------------------------- #

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_price: float,
        spot_df: pd.DataFrame | None = None,  # reserved for future cross-checks
    ) -> Optional[Signal]:
        """
        Build a Signal using indicator confluence + ATR based SL/TP.
        Returns None if quality/thresholds are not met.
        """
        if df is None or df.empty:
            return None
        try:
            indicators = self._calculate_indicators(df)
            if not indicators:
                return None

            score, reasons = self._score_signal(df, indicators, float(current_price))

            # Direction by score vs min_signal_score
            direction: Optional[str] = None
            if score >= int(self.config.min_signal_score):
                direction = "BUY"
            elif score <= -int(self.config.min_signal_score):
                direction = "SELL"
            if not direction:
                return None

            # Confidence scaled to 0..10, thresholded
            confidence = min(abs(score) / float(self.max_possible_score), 1.0) * 10.0
            if confidence < float(self.config.confidence_threshold):
                return None

            # ATR-based stops/targets
            atr_value = latest_atr_value(df, period=int(self.config.atr_period), method="rma")
            if not atr_value or float(atr_value) <= 0:
                return None

            sl_points = float(atr_value) * float(self.config.atr_sl_multiplier)
            tp_points = float(atr_value) * float(self.config.atr_tp_multiplier)

            entry_price = float(current_price)
            stop_loss = entry_price - sl_points if direction == "BUY" else entry_price + sl_points
            target = entry_price + tp_points if direction == "BUY" else entry_price - tp_points

            sig = Signal(
                signal=direction,
                score=int(score),
                confidence=float(round(confidence, 2)),
                entry_price=float(round(entry_price, 2)),
                stop_loss=float(round(stop_loss, 2)),
                target=float(round(target, 2)),
                reasons=reasons,
                market_volatility=float(round(float(atr_value), 2)),
            )
            sig.compute_hash()
            self.last_signal_hash = sig.hash
            return sig

        except Exception as e:
            logger.error("Error generating signal: %s", e, exc_info=True)
            return None

    # -------------------------- legacy method -------------------------- #

    def generate_options_signal(
        self,
        options_ohlc: pd.DataFrame,
        spot_ohlc: pd.DataFrame,
        strike_info: Dict[str, any],
        current_option_price: float,
    ) -> Optional[Dict[str, any]]:
        """
        DEPRECATED: Old breakout strategy retained for reference during refactor.
        """
        breakout_pct = 0.01
        sl_pct = 0.05
        tp_pct = 0.15

        if options_ohlc is None or len(options_ohlc) < 5:
            return None

        if current_option_price <= options_ohlc["close"].iloc[-2] * (1 + breakout_pct):
            return None

        entry = float(current_option_price)
        sl = round(entry * (1 - sl_pct), 2)
        tp = round(entry * (1 + tp_pct), 2)
        return {
            "signal": "BUY",
            "entry_price": round(entry, 2),
            "stop_loss": sl,
            "target": tp,
            "confidence": 5.0,
            "market_volatility": 0.0,
            "strategy_notes": "DEPRECATED breakout signal",
        }
