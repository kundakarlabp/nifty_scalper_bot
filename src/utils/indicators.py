# src/utils/indicators.py
"""
Helper functions to calculate technical indicators used by the scalping strategy.

Design goals
- Vectorised, NaN-safe, minimal lookbacks respected (warmup-friendly)
- Uses python-ta if available; otherwise fast numpy/pandas fallbacks
- Shapes & names match scalping_strategy.py expectations exactly

Exports
- calculate_ema(df_or_series, period) -> Series
- calculate_rsi(df_or_series, period=14) -> Series
- calculate_macd(df_or_series, fast=12, slow=26, signal=9) -> (macd, signal, hist)
- calculate_atr(df_or_h, low=None, close=None, period=14) -> Series  (SMA ATR)
- calculate_supertrend(df_or_h, low=None, close=None, period=10, multiplier=3.0)
    -> (trend_dir_series(+1/-1), final_upper, final_lower)
- calculate_bb_width(df_or_close, window=20, std=2.0) -> (upper_band, lower_band)
- calculate_adx(df_or_h, low=None, close=None, period=14) -> (adx, di_plus, di_minus)
- calculate_vwap(df_or_h, low=None, close=None, volume=None, period=None) -> Series
"""

from __future__ import annotations

from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np

# Try python-ta first
try:
    from ta.momentum import RSIIndicator  # type: ignore
    from ta.trend import EMAIndicator, MACD, ADXIndicator  # type: ignore
    from ta.volatility import AverageTrueRange, BollingerBands  # type: ignore
    TA_AVAILABLE = True
except Exception:
    RSIIndicator = EMAIndicator = MACD = ADXIndicator = AverageTrueRange = BollingerBands = None  # type: ignore
    TA_AVAILABLE = False

SeriesOrDF = Union[pd.Series, pd.DataFrame]


# ----------------------------- internals ----------------------------- #
def _series(s_or_df: SeriesOrDF, col: str) -> pd.Series:
    """Return a Series from either a Series or a DataFrame column."""
    if isinstance(s_or_df, pd.Series):
        return s_or_df
    return s_or_df[col]


def _sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window, min_periods=window).mean()


def _rma(s: pd.Series, window: int) -> pd.Series:
    """
    Wilder's RMA (a.k.a. SMMA). Uses EMA(alpha=1/window) with adjust=False.
    NaNs are preserved then backfilled minimally to avoid propagating NaN tails.
    """
    if window <= 1:
        return s.copy()
    r = s.ewm(alpha=1.0 / float(window), adjust=False).mean()
    return r


# ----------------------------- Moving Average ----------------------------- #
def calculate_ema(close: SeriesOrDF, period: int) -> pd.Series:
    """EMA on close series (works with DataFrame or Series)."""
    s = _series(close, "close").astype(float)
    if TA_AVAILABLE and EMAIndicator:
        out = EMAIndicator(close=s, window=period, fillna=False).ema_indicator()
        return out
    return s.ewm(span=period, adjust=False).mean()


# ---------------------------------- RSI ----------------------------------- #
def calculate_rsi(close: SeriesOrDF, period: int = 14) -> pd.Series:
    """RSI (Wilder)."""
    s = _series(close, "close").astype(float)
    if TA_AVAILABLE and RSIIndicator:
        return RSIIndicator(close=s, window=period, fillna=False).rsi()

    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder smoothing via RMA
    roll_up = _rma(up, period)
    roll_down = _rma(down, period)
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.astype(float)


# --------------------------------- MACD ----------------------------------- #
def calculate_macd(
    close: SeriesOrDF,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD tuple (macd_line, signal_line, hist)."""
    s = _series(close, "close").astype(float)
    if TA_AVAILABLE and MACD:
        macd_calc = MACD(close=s, window_slow=slow, window_fast=fast, window_sign=signal, fillna=False)
        macd_line = macd_calc.macd()
        signal_line = macd_calc.macd_signal()
        hist = macd_calc.macd_diff()
        return macd_line, signal_line, hist

    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ---------------------------------- ATR ----------------------------------- #
def calculate_atr(
    high_or_df: SeriesOrDF,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    period: int = 14,
) -> pd.Series:
    """
    ATR; accepts either:
      - DataFrame with high/low/close (pass as first arg), or
      - high, low, close Series as first three args.
    Default = SMA ATR (vectorised). For Wilder-style ATR use _rma(tr, period).
    """
    if isinstance(high_or_df, pd.DataFrame):
        h = high_or_df["high"].astype(float)
        l = high_or_df["low"].astype(float)
        c = high_or_df["close"].astype(float)
    else:
        assert low is not None and close is not None, "Provide low & close series when passing high as Series."
        h, l, c = high_or_df.astype(float), low.astype(float), close.astype(float)

    if TA_AVAILABLE and AverageTrueRange:
        atr_calc = AverageTrueRange(high=h, low=l, close=c, window=period, fillna=False)
        return atr_calc.average_true_range()

    prev_close = c.shift(1)
    tr1 = (h - l)
    tr2 = (h - prev_close).abs()
    tr3 = (l - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = _sma(tr, period)
    return atr.astype(float)


# ------------------------------- SuperTrend ------------------------------- #
def calculate_supertrend(
    high_or_df: SeriesOrDF,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    period: int = 10,
    multiplier: float = 3.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    SuperTrend returning (trend_dir, upper, lower).
    trend_dir is +1 for uptrend and -1 for downtrend.
    Accepts DataFrame or (high, low, close) Series.
    """
    if isinstance(high_or_df, pd.DataFrame):
        h = high_or_df["high"].astype(float)
        l = high_or_df["low"].astype(float)
        c = high_or_df["close"].astype(float)
    else:
        assert low is not None and close is not None, "Provide low & close series when passing high as Series."
        h, l, c = high_or_df.astype(float), low.astype(float), close.astype(float)

    atr = calculate_atr(h, l, c, period).astype(float)
    hl2 = (h + l) / 2.0
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    # initialise
    direction = np.ones(len(c), dtype=int)
    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    # vector-friendly but looped for clarity & ST rules
    for i in range(1, len(c)):
        prev_close = c.iat[i - 1]

        # carry/flip logic for bands
        if (upperband.iat[i] < final_upper.iat[i - 1]) or (prev_close > final_upper.iat[i - 1]):
            final_upper.iat[i] = upperband.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i - 1]

        if (lowerband.iat[i] > final_lower.iat[i - 1]) or (prev_close < final_lower.iat[i - 1]):
            final_lower.iat[i] = lowerband.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i - 1]

        # direction flips
        if (direction[i - 1] == 1) and (c.iat[i] < final_upper.iat[i]):
            direction[i] = -1
        elif (direction[i - 1] == -1) and (c.iat[i] > final_lower.iat[i]):
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]

        # ensure only the active band is used on each side
        if direction[i] == 1:
            final_lower.iat[i] = lowerband.iat[i]
        else:
            final_upper.iat[i] = upperband.iat[i]

    trend = pd.Series(direction, index=c.index).astype(int)
    return trend, final_upper.astype(float), final_lower.astype(float)


# ---------------------------- Bollinger Bands ----------------------------- #
def calculate_bb_width(
    close: SeriesOrDF,
    window: int = 20,
    std: float = 2.0,
) -> Tuple[pd.Series, pd.Series]:
    """Return Bollinger upper & lower bands (strategy only needs these two)."""
    s = _series(close, "close").astype(float)
    if TA_AVAILABLE and BollingerBands:
        bb = BollingerBands(close=s, window=window, window_dev=std, fillna=False)
        return bb.bollinger_hband(), bb.bollinger_lband()

    sma = s.rolling(window=window, min_periods=window).mean()
    rolling_std = s.rolling(window=window, min_periods=window).std(ddof=0)
    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    return upper.astype(float), lower.astype(float)


# ---------------------------------- ADX ----------------------------------- #
def calculate_adx(
    high_or_df: SeriesOrDF,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return (adx, di_plus, di_minus).
    Accepts DataFrame or (high, low, close) Series.
    """
    if isinstance(high_or_df, pd.DataFrame):
        h = high_or_df["high"].astype(float)
        l = high_or_df["low"].astype(float)
        c = high_or_df["close"].astype(float)
    else:
        assert low is not None and close is not None, "Provide low & close series when passing high as Series."
        h, l, c = high_or_df.astype(float), low.astype(float), close.astype(float)

    if TA_AVAILABLE and ADXIndicator:
        adx_calc = ADXIndicator(high=h, low=l, close=c, window=period, fillna=False)
        return adx_calc.adx(), adx_calc.adx_pos(), adx_calc.adx_neg()

    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_c = c.shift(1)

    up_move = h - prev_h
    down_move = prev_l - l
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=c.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=c.index)

    tr1 = (h - l)
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.Series(np.maximum.reduce([tr1, tr2, tr3]), index=c.index)

    atr = _sma(tr, period).replace(0.0, np.nan)
    plus_di = 100.0 * _sma(plus_dm, period) / atr
    minus_di = 100.0 * _sma(minus_dm, period) / atr

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = _sma(dx, period).bfill()
    return adx.astype(float), plus_di.astype(float), minus_di.astype(float)


# --------------------------------- VWAP ----------------------------------- #
def calculate_vwap(
    high_or_df: SeriesOrDF,
    low: Optional[pd.Series] = None,
    close: Optional[pd.Series] = None,
    volume: Optional[pd.Series] = None,
    period: Optional[int] = None,
) -> pd.Series:
    """
    VWAP. Accepts DataFrame or individual Series (high, low, close, volume).
    - If period is None -> cumulative session VWAP.
    - If period is int -> rolling VWAP over that window.
    """
    if isinstance(high_or_df, pd.DataFrame):
        h = high_or_df["high"].astype(float)
        l = high_or_df["low"].astype(float)
        c = high_or_df["close"].astype(float)
        v = high_or_df["volume"].astype(float)
    else:
        assert low is not None and close is not None and volume is not None, \
            "Provide low, close, volume series when passing high as Series."
        h, l, c, v = high_or_df.astype(float), low.astype(float), close.astype(float), volume.astype(float)

    typical_price = (h + l + c) / 3.0
    pv = typical_price * v

    if period is None or period <= 1:
        out = (pv.cumsum() / v.cumsum()).replace([np.inf, -np.inf], np.nan)
    else:
        pv_roll = pv.rolling(window=period, min_periods=period).sum()
        v_roll = v.rolling(window=period, min_periods=period).sum()
        out = (pv_roll / v_roll.replace(0.0, np.nan))
    return out.astype(float)