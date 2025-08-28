# src/utils/indicators.py
"""
Helper functions to calculate technical indicators used by the scalping strategy.

All functions accept either a Pandas DataFrame (with columns: open, high, low, close, volume)
or individual Series where appropriate. When the `ta` library is available we use it; otherwise
we fall back to numpy/pandas implementations.

Return shapes are aligned with scalping_strategy.py:
- calculate_macd -> (macd_line, macd_signal, macd_hist)
- calculate_atr -> Series
- calculate_supertrend -> (trend_dir_series, upper_band, lower_band)
- calculate_bb_width -> (upper_band, lower_band)
- calculate_adx -> (adx, di_plus, di_minus)
- calculate_vwap -> Series
"""

from __future__ import annotations

from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np

from src.utils.atr_helper import compute_atr_df  # unify ATR (RMA/Wilder) across codebase

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


def _series(s_or_df: SeriesOrDF, col: str) -> pd.Series:
    """Return a Series from either a Series or a DataFrame column, coerced to float."""
    if isinstance(s_or_df, pd.Series):
        return pd.to_numeric(s_or_df, errors="coerce").astype(float)
    return pd.to_numeric(s_or_df[col], errors="coerce").astype(float)


# ----------------------------- Moving Average ----------------------------- #
def calculate_ema(close: SeriesOrDF, period: int) -> pd.Series:
    """EMA on close series (works with DataFrame or Series)."""
    s = _series(close, "close")
    if TA_AVAILABLE and EMAIndicator:
        return EMAIndicator(close=s, window=period, fillna=False).ema_indicator()
    return s.ewm(span=period, adjust=False).mean()


# ---------------------------------- RSI ----------------------------------- #
def calculate_rsi(close: SeriesOrDF, period: int = 14) -> pd.Series:
    """RSI of close (DataFrame or Series)."""
    s = _series(close, "close")
    if TA_AVAILABLE and RSIIndicator:
        return RSIIndicator(close=s, window=period, fillna=False).rsi()

    # Wilder-style RSI fallback (RMA smoothing)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    roll_up = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    roll_down = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# --------------------------------- MACD ----------------------------------- #
def calculate_macd(
    close: SeriesOrDF,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD tuple (macd_line, signal_line, hist)."""
    s = _series(close, "close")
    if TA_AVAILABLE and MACD:
        macd_calc = MACD(close=s, window_slow=slow, window_fast=fast, window_sign=signal, fillna=False)
        macd_line = macd_calc.macd()
        signal_line = macd_calc.macd_signal()
        hist = macd_calc.macd_diff()
        return macd_line, signal_line, hist

    short_ema = s.ewm(span=fast, adjust=False).mean()
    long_ema = s.ewm(span=slow, adjust=False).mean()
    macd_line = short_ema - long_ema
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

    Uses atr_helper.compute_atr_df (Wilder/RMA by default) for consistency and
    returns the resulting `atr` column as a standalone Series.
    """
    if isinstance(high_or_df, pd.DataFrame):
        df = high_or_df
    else:
        if low is None or close is None:
            raise ValueError("Provide low & close series when passing high as Series.")
        df = pd.DataFrame({
            "high": pd.to_numeric(high_or_df, errors="coerce").astype(float),
            "low": pd.to_numeric(low, errors="coerce").astype(float),
            "close": pd.to_numeric(close, errors="coerce").astype(float),
        })
    # ``compute_atr_df`` returns the original DataFrame with an ``atr`` column
    # appended.  ``calculate_atr`` is expected to return only the ATR values
    # as a ``Series`` to keep function outputs simple and consistent with
    # callers such as ``calculate_supertrend``.  Extract the ``atr`` column
    # explicitly.
    return compute_atr_df(df, period=period, method="rma")["atr"]


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
        h = _series(high_or_df, "high")
        low_s = _series(high_or_df, "low")
        c = _series(high_or_df, "close")
    else:
        if low is None or close is None:
            raise ValueError("Provide low & close series when passing high as Series.")
        h, low_s, c = (
            pd.to_numeric(high_or_df, errors="coerce").astype(float),
            pd.to_numeric(low, errors="coerce").astype(float),
            pd.to_numeric(close, errors="coerce").astype(float),
        )

    atr = calculate_atr(pd.DataFrame({"high": h, "low": low_s, "close": c}), period=period)
    hl2 = (h + low_s) / 2.0
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    direction = np.ones(len(c), dtype=int)
    final_upper = upperband.copy()
    final_lower = lowerband.copy()

    for i in range(1, len(c)):
        prev_close = c.iat[i - 1]

        final_upper.iat[i] = (
            upperband.iat[i]
            if (upperband.iat[i] < final_upper.iat[i - 1]) or (prev_close > final_upper.iat[i - 1])
            else final_upper.iat[i - 1]
        )
        final_lower.iat[i] = (
            lowerband.iat[i]
            if (lowerband.iat[i] > final_lower.iat[i - 1]) or (prev_close < final_lower.iat[i - 1])
            else final_lower.iat[i - 1]
        )

        if (direction[i - 1] == 1) and (c.iat[i] < final_upper.iat[i]):
            direction[i] = -1
        elif (direction[i - 1] == -1) and (c.iat[i] > final_lower.iat[i]):
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]

        if direction[i] == 1:
            final_lower.iat[i] = lowerband.iat[i]
        else:
            final_upper.iat[i] = upperband.iat[i]

    trend = pd.Series(direction, index=c.index)
    return trend, final_upper, final_lower


# ---------------------------- Bollinger Bands ----------------------------- #
def calculate_bb_width(
    close: SeriesOrDF,
    window: int = 20,
    std: float = 2.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Return Bollinger upper & lower bands (your strategy only needs these two).
    """
    s = _series(close, "close")
    if TA_AVAILABLE and BollingerBands:
        bb = BollingerBands(close=s, window=window, window_dev=std, fillna=False)
        return bb.bollinger_hband(), bb.bollinger_lband()

    sma = s.rolling(window=window, min_periods=window).mean()
    rolling_std = s.rolling(window=window, min_periods=window).std(ddof=0)
    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    return upper, lower


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
        h = _series(high_or_df, "high")
        low_s = _series(high_or_df, "low")
        c = _series(high_or_df, "close")
    else:
        assert low is not None and close is not None, "Provide low & close series when passing high as Series."
        h, low_s, c = (
            pd.to_numeric(high_or_df, errors="coerce").astype(float),
            pd.to_numeric(low, errors="coerce").astype(float),
            pd.to_numeric(close, errors="coerce").astype(float),
        )

    if TA_AVAILABLE and ADXIndicator:
        adx_calc = ADXIndicator(high=h, low=low_s, close=c, window=period, fillna=False)
        return adx_calc.adx(), adx_calc.adx_pos(), adx_calc.adx_neg()

    # Fallback (SMA-smoothing): adequate when TA unavailable
    prev_h = h.shift(1)
    prev_l = low_s.shift(1)
    prev_c = c.shift(1)

    up_move = h - prev_h
    down_move = prev_l - low_s
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = (h - low_s)
    tr2 = (h - prev_c).abs()
    tr3 = (low_s - prev_c).abs()
    tr = np.maximum.reduce([tr1, tr2, tr3])

    plus_dm_s = pd.Series(plus_dm, index=c.index)
    minus_dm_s = pd.Series(minus_dm, index=c.index)
    tr_s = pd.Series(tr, index=c.index)

    atr = tr_s.rolling(window=period, min_periods=period).mean()
    plus_di = 100.0 * plus_dm_s.rolling(window=period, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_s.rolling(window=period, min_periods=period).mean() / atr.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx, plus_di, minus_di


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

    If volume is missing, falls back to (unweighted) rolling mean of typical price.
    """
    if isinstance(high_or_df, pd.DataFrame):
        if {"high", "low", "close"}.issubset(high_or_df.columns):
            h = _series(high_or_df, "high")
            low_series = _series(high_or_df, "low")
            c = _series(high_or_df, "close")
        else:
            raise KeyError("DataFrame must contain 'high','low','close' columns.")
        v = pd.to_numeric(high_or_df.get("volume", pd.Series(index=high_or_df.index, dtype=float)), errors="coerce").astype(float)
    else:
        assert low is not None and close is not None, "Provide low & close series when passing high as Series."
        h = pd.to_numeric(high_or_df, errors="coerce").astype(float)
        low_series = pd.to_numeric(low, errors="coerce").astype(float)
        c = pd.to_numeric(close, errors="coerce").astype(float)
        v = pd.to_numeric(volume if volume is not None else pd.Series(index=c.index, dtype=float), errors="coerce").astype(float)

    typical_price = (h + low_series + c) / 3.0
    if v.isna().all() or (v.fillna(0) == 0).all():
        # Graceful degradation when volume missing
        if period is None or period <= 1:
            return typical_price.expanding(min_periods=1).mean()
        return typical_price.rolling(window=period, min_periods=period).mean()

    pv = typical_price * v

    if period is None or period <= 1:
        vwap = (pv.cumsum() / v.cumsum()).replace([np.inf, -np.inf], np.nan)
        return vwap
    else:
        pv_roll = pv.rolling(window=period, min_periods=period).sum()
        v_roll = v.rolling(window=period, min_periods=period).sum()
        vwap = pv_roll / v_roll.replace(0, np.nan)
        return vwap
