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
from typing import Dict, Optional, Tuple, List, Any

import pandas as pd

from src.config import Config
from src.utils.indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    # calculate_atr,  # replaced by atr_helper for robustness
    calculate_supertrend,
    calculate_vwap,
    calculate_adx,
)
from src.utils.atr_helper import compute_atr_df, latest_atr_value  # robust ATR

logger = logging.getLogger(__name__)


def _bollinger_bands_from_close(close: pd.Series, window: int, std: float) -> Tuple[pd.Series, pd.Series]:
    """
    Compute Bollinger Bands (upper, lower) from the close series.
    Uses population std (ddof=0) to be stable on short windows.
    """
    ma = close.rolling(window=window, min_periods=window).mean()
    sd = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = ma + std * sd
    lower = ma - std * sd
    return upper, lower


class EnhancedScalpingStrategy:
    """A dynamic scalping strategy for Nifty spot/futures/options."""

    def __init__(
        self,
        base_stop_loss_points: float = getattr(Config, "BASE_STOP_LOSS_POINTS", 20.0),
        base_target_points: float = getattr(Config, "BASE_TARGET_POINTS", 40.0),
        # Align defaults with .env to avoid runner/strategy mismatch:
        confidence_threshold: float = getattr(Config, "CONFIDENCE_THRESHOLD", 6.0),
        min_score_threshold: int = int(getattr(Config, "MIN_SIGNAL_SCORE", 5)),
        # Faster MACD for 1-min scalping:
        ema_fast_period: int = 9,
        ema_slow_period: int = 21,
        rsi_period: int = 14,
        rsi_overbought: int = 60,
        rsi_oversold: int = 40,
        macd_fast_period: int = 8,      # was 12
        macd_slow_period: int = 17,     # was 26
        macd_signal_period: int = 9,    # keep 9
        atr_period: int = getattr(Config, "ATR_PERIOD", 14),
        supertrend_atr_multiplier: float = 2.0,
        bb_window: int = 20,
        bb_std_dev: float = 2.0,
        adx_period: int = 14,
        adx_trend_strength: int = 25,
        vwap_period: int = 20,
        # --- options params (used in generate_options_signal) ---
        option_sl_percent: float = getattr(Config, "OPTION_SL_PERCENT", 0.05),
        option_tp_percent: float = getattr(Config, "OPTION_TP_PERCENT", 0.15),
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

        # We score 6 key “axes”: EMA, RSI, MACD hist sign, MACD zerocross, Supertrend dir, VWAP.
        # (BB and regime nudge modify score but aren’t counted toward max_possible_score here)
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

        # Bollinger (upper/lower) – compute locally to avoid API mismatch
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
        self, df: pd.DataFrame, indicators: Dict[s]()
