# src/utils/indicators.py
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd


# ---------- helpers ----------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ---------- public indicators ----------

def ema(series: pd.Series, period: int) -> pd.Series:
    return _ema(series.astype(float), int(period))


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = series.astype(float)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    s = series.astype(float)
    macd = _ema(s, fast) - _ema(s, slow)
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def macd_cross(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd, sig, _ = macd_hist(series, fast, slow, signal)
    diff = macd - sig
    cross_up = (diff > 0) & (diff.shift(1) <= 0)
    cross_dn = (diff < 0) & (diff.shift(1) >= 0)
    # encode +1 for bull cross, -1 for bear cross, 0 otherwise
    out = pd.Series(0, index=series.index, dtype=int)
    out[cross_up] = 1
    out[cross_dn] = -1
    return out


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing (EMA with alpha=1/period works as approximation)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (trend_direction, upper_band, lower_band)
    trend_direction: +1 up, -1 down
    """
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    hl2 = (h + l) / 2.0
    atr_val = atr(df, period)
    upper_basic = hl2 + multiplier * atr_val
    lower_basic = hl2 - multiplier * atr_val

    upper = upper_basic.copy()
    lower = lower_basic.copy()

    for i in range(1, len(df)):
        upper.iloc[i] = min(upper_basic.iloc[i], upper.iloc[i - 1]) if c.iloc[i - 1] <= upper.iloc[i - 1] else upper_basic.iloc[i]
        lower.iloc[i] = max(lower_basic.iloc[i], lower.iloc[i - 1]) if c.iloc[i - 1] >= lower.iloc[i - 1] else lower_basic.iloc[i]

    # trend direction
    dir_series = pd.Series(1, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if c.iloc[i] > upper.iloc[i - 1]:
            dir_series.iloc[i] = 1
        elif c.iloc[i] < lower.iloc[i - 1]:
            dir_series.iloc[i] = -1
        else:
            dir_series.iloc[i] = dir_series.iloc[i - 1]

        # band selection
        if dir_series.iloc[i] == 1:
            upper.iloc[i] = np.nan
        else:
            lower.iloc[i] = np.nan

    return dir_series.fillna(1), upper, lower


def bollinger_bandwidth(series: pd.Series, period: int = 20, std_mul: float = 2.0) -> pd.Series:
    s = series.astype(float)
    ma = s.rolling(period).mean()
    sd = s.rolling(period).std(ddof=0)
    upper = ma + std_mul * sd
    lower = ma - std_mul * sd
    width = (upper - lower) / (ma.replace(0, np.nan).abs())
    return width.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Classic VWAP using typical price.
    Requires 'volume' column; if missing, returns rolling mean of price.
    """
    c = df["close"].astype(float)
    if "volume" not in df.columns:
        return c.rolling(20).mean().fillna(method="bfill")

    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float).clip(lower=0.0)
    tp = (h + l + c) / 3.0
    cum_v = v.cumsum().replace(0, np.nan)
    cum_vp = (tp * v).cumsum()
    out = (cum_vp / cum_v).fillna(method="bfill")
    return out


def di_plus_minus(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_series = atr(df, period) * period  # approximate true range sum
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (tr_series + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (tr_series + 1e-12)

    return plus_di.fillna(0.0), minus_di.fillna(0.0)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    plus_di, minus_di = di_plus_minus(df, period)
    dx = (100 * (plus_di - minus_di).abs() / ((plus_di + minus_di).replace(0, np.nan))).fillna(0.0)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)
