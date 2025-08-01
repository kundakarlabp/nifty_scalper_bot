"""
Helper functions to calculate technical indicators used by the scalping strategy.
All functions operate on a Pandas ``DataFrame`` containing at least the
columns ``open``, ``high``, ``low``, ``close`` and ``volume``.  Where
possible the `ta` library is used to perform calculations, otherwise
numpy/pandas formulas are implemented directly.
"""
from typing import Dict
import pandas as pd
import numpy as np
"""
This module attempts to import indicator functions from the ``ta`` library.  If
that library is not available in the environment, fallback implementations
using pandas and numpy are provided.  The fallback versions are approximate
but sufficient for demonstration purposes.
"""

try:
    from ta.momentum import RSIIndicator  # type: ignore
    from ta.trend import EMAIndicator, MACD, ADXIndicator  # type: ignore
    from ta.volatility import AverageTrueRange  # type: ignore
except ImportError:
    # Define dummy classes to satisfy type checkers when ``ta`` is absent.
    RSIIndicator = None  # type: ignore
    EMAIndicator = None  # type: ignore
    MACD = None  # type: ignore
    ADXIndicator = None  # type: ignore
    AverageTrueRange = None  # type: ignore



def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Return the Exponential Moving Average of the closing price.

    Uses ``ta.trend.EMAIndicator`` when available.  Falls back to pandas'
    exponential weighted mean otherwise.
    """
    if EMAIndicator:
        return EMAIndicator(close=df["close"], window=period, fillna=False).ema_indicator()
    # Fallback: use pandas' ewm
    return df["close"].ewm(span=period, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return the Relative Strength Index of the closing price.

    When the ``ta`` library is unavailable a manual RSI calculation is
    performed using simple moving averages of gains and losses.
    """
    if RSIIndicator:
        return RSIIndicator(close=df["close"], window=period, fillna=False).rsi()
    # Fallback RSI implementation
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    roll_up = gain.rolling(window=period, min_periods=period).mean()
    roll_down = loss.rolling(window=period, min_periods=period).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(method="bfill")  # backfill initial values
    return rsi


def calculate_macd(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return the MACD line, signal line and histogram as a dictionary.

    Falls back to manual calculation when the ``ta`` library is absent.
    """
    if MACD:
        macd_calc = MACD(close=df["close"], fillna=False)
        return {
            "macd": macd_calc.macd(),
            "macd_signal": macd_calc.macd_signal(),
            "macd_hist": macd_calc.macd_diff(),
        }
    # Fallback MACD calculation
    short_ema = df["close"].ewm(span=12, adjust=False).mean()
    long_ema = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return {
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_hist,
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return the Average True Range.

    Falls back to a manual ATR computation when the ``ta`` library is
    unavailable.
    """
    if AverageTrueRange:
        atr_calc = AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=period, fillna=False
        )
        return atr_calc.average_true_range()
    # Fallback ATR calculation
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the Volume Weighted Average Price (VWAP).
    VWAP is computed cumulatively: sum(price*volume) / sum(volume).
    """
    price_volume = (df["close"] * df["volume"]).cumsum()
    cumulative_volume = df["volume"].cumsum()
    vwap = price_volume / cumulative_volume
    return vwap


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """
    Compute the SuperTrend indicator.
    This implementation follows a common formulation where an ATR based band
    is trailed by price to determine trend direction.  Returns a series
    containing +1 for uptrend and -1 for downtrend.
    """
    atr = calculate_atr(df, period)
    # Basic upper and lower bands
    hl2 = (df["high"] + df["low"]) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    # Initialise trend direction
    direction = np.ones(len(df))
    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()
    for i in range(1, len(df)):
        curr_close = df.loc[df.index[i], "close"]
        prev_close = df.loc[df.index[i - 1], "close"]
        # Adjust final upper band
        if (upperband[i] < final_upperband[i - 1]) or (prev_close > final_upperband[i - 1]):
            final_upperband[i] = upperband[i]
        else:
            final_upperband[i] = final_upperband[i - 1]
        # Adjust final lower band
        if (lowerband[i] > final_lowerband[i - 1]) or (prev_close < final_lowerband[i - 1]):
            final_lowerband[i] = lowerband[i]
        else:
            final_lowerband[i] = final_lowerband[i - 1]
        # Determine direction
        if (direction[i - 1] == 1) and (curr_close < final_upperband[i]):
            direction[i] = -1
        elif (direction[i - 1] == -1) and (curr_close > final_lowerband[i]):
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]
        # When direction flips, reset band to opposite band
        if direction[i] == 1:
            final_lowerband[i] = lowerband[i]
        else:
            final_upperband[i] = upperband[i]
    return pd.Series(direction, index=df.index)


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return the Average Directional Index (ADX) which measures trend strength.

    Falls back to a manual ADX calculation when the ``ta`` library is
    unavailable.
    """
    if ADXIndicator:
        adx_calc = ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=period, fillna=False
        )
        return adx_calc.adx()
    # Fallback ADX calculation
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    # Directional movement
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # True range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum.reduce([tr1, tr2, tr3])
    # Convert numpy arrays to Series with the original index
    plus_dm_s = pd.Series(plus_dm, index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)
    tr_s = pd.Series(tr, index=df.index)
    # Smoothed moving averages of DM and TR
    atr = tr_s.rolling(window=period, min_periods=period).mean()
    plus_di = 100.0 * plus_dm_s.rolling(window=period, min_periods=period).mean() / atr
    minus_di = 100.0 * minus_dm_s.rolling(window=period, min_periods=period).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx.fillna(method="bfill")


def calculate_bb_width(df: pd.DataFrame, window: int = 20, std: int = 2) -> pd.Series:
    """
    Compute Bollinger Band width as (upper - lower) / middle.
    A larger width implies higher volatility and a trending market.
    """
    sma = df["close"].rolling(window=window, min_periods=window).mean()
    rolling_std = df["close"].rolling(window=window, min_periods=window).std(ddof=0)
    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    width = (upper - lower) / sma
    return width