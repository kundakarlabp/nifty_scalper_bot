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
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Return the Exponential Moving Average of the closing price."""
    return EMAIndicator(close=df["close"], window=period, fillna=False).ema_indicator()


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return the Relative Strength Index of the closing price."""
    return RSIIndicator(close=df["close"], window=period, fillna=False).rsi()


def calculate_macd(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return the MACD line, signal line and histogram as a dictionary."""
    macd_calc = MACD(close=df["close"], fillna=False)
    return {
        "macd": macd_calc.macd(),
        "macd_signal": macd_calc.macd_signal(),
        "macd_hist": macd_calc.macd_diff(),
    }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return the Average True Range."""
    atr_calc = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=period, fillna=False)
    return atr_calc.average_true_range()


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
    """Return the Average Directional Index (ADX) which measures trend strength."""
    adx_calc = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=period, fillna=False)
    return adx_calc.adx()


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