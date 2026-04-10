"""Centralised, cache-aware indicator utilities for NIFTY scalper strategies.

All functions operate on OHLC bars supplied by MarketDataManager.
Results are cached per (symbol, function) to avoid recalculating on every tick.

Public API
----------
compute_rsi(bars, period) -> float | None
compute_ema(bars, period) -> float | None
compute_macd(bars) -> tuple[float, float, float] | None  (macd, signal, hist)
compute_atr(bars, period) -> float | None
compute_vwap(bars) -> float | None
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Internal cache — avoids recomputing full series each tick.
# Key: (symbol, indicator, params_hash)  Value: (result, computed_at_mono)
# ---------------------------------------------------------------------------
_CACHE: dict[str, tuple[Any, float]] = {}
_CACHE_TTL_S: float = 1.0  # re-compute at most once per second per key


def _cache_key(*parts: Any) -> str:
    return "|".join(str(p) for p in parts)


def _cached(key: str, value: Any) -> Any:
    _CACHE[key] = (value, time.monotonic())
    return value


def _from_cache(key: str) -> tuple[bool, Any]:
    entry = _CACHE.get(key)
    if entry is None:
        return False, None
    val, ts = entry
    if time.monotonic() - ts < _CACHE_TTL_S:
        return True, val
    return False, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _closes(bars: Sequence[dict[str, Any]]) -> list[float]:
    """Extract close prices from bar list, skipping invalid entries."""
    result: list[float] = []
    for b in bars:
        try:
            c = float(b.get("close") or b.get("c") or 0.0)
            if c > 0:
                result.append(c)
        except (TypeError, ValueError):
            continue
    return result


def _highs(bars: Sequence[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for b in bars:
        try:
            v = float(b.get("high") or b.get("h") or 0.0)
            if v > 0:
                out.append(v)
        except (TypeError, ValueError):
            continue
    return out


def _lows(bars: Sequence[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for b in bars:
        try:
            v = float(b.get("low") or b.get("l") or 0.0)
            if v > 0:
                out.append(v)
        except (TypeError, ValueError):
            continue
    return out


def _volumes(bars: Sequence[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for b in bars:
        try:
            v = float(b.get("volume") or b.get("v") or 0.0)
            out.append(max(v, 0.0))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _ema_series(values: list[float], period: int) -> list[float]:
    """Compute EMA series using Wilder smoothing (k = 2/(n+1))."""
    if not values or period <= 0:
        return []
    k = 2.0 / (period + 1)
    result: list[float] = []
    for v in values:
        if not result:
            result.append(v)
        else:
            result.append(v * k + result[-1] * (1 - k))
    return result


# ---------------------------------------------------------------------------
# Phase 6 Public Compute Functions
# ---------------------------------------------------------------------------

def compute_rsi(
    bars: Sequence[dict[str, Any]],
    period: int = 14,
    symbol: str = "",
) -> float | None:
    """Compute RSI(period) from the supplied OHLC bars.

    Args:
        bars: Sequence of OHLC bar dicts with 'close'/'c' key.
        period: RSI lookback period (default 14).
        symbol: Optional symbol tag used for cache keying.

    Returns:
        RSI value 0–100, or None if insufficient data.
    """
    key = _cache_key("rsi", symbol, period, len(bars))
    hit, cached_val = _from_cache(key)
    if hit:
        return cached_val

    closes = _closes(bars)
    if len(closes) < period + 1:
        return None

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    # Initial averages (simple mean for first period)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Wilder smoothing for remaining bars
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return _cached(key, 100.0)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return _cached(key, round(rsi, 4))


def compute_ema(
    bars: Sequence[dict[str, Any]],
    period: int = 20,
    symbol: str = "",
) -> float | None:
    """Compute the latest EMA(period) value.

    Args:
        bars: Sequence of OHLC bar dicts.
        period: EMA period (default 20).
        symbol: Optional cache key tag.

    Returns:
        Latest EMA value, or None if insufficient bars.
    """
    key = _cache_key("ema", symbol, period, len(bars))
    hit, cached_val = _from_cache(key)
    if hit:
        return cached_val

    closes = _closes(bars)
    if len(closes) < period:
        return None

    series = _ema_series(closes, period)
    if not series:
        return None
    return _cached(key, round(series[-1], 4))


def compute_macd(
    bars: Sequence[dict[str, Any]],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    symbol: str = "",
) -> tuple[float, float, float] | None:
    """Compute MACD line, signal line, and histogram.

    Args:
        bars: Sequence of OHLC bar dicts.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal_period: Signal EMA period (default 9).
        symbol: Optional cache key tag.

    Returns:
        (macd, signal, histogram) tuple, or None if insufficient data.
    """
    key = _cache_key("macd", symbol, fast, slow, signal_period, len(bars))
    hit, cached_val = _from_cache(key)
    if hit:
        return cached_val

    closes = _closes(bars)
    if len(closes) < slow + signal_period:
        return None

    fast_ema = _ema_series(closes, fast)
    slow_ema = _ema_series(closes, slow)

    if len(fast_ema) != len(slow_ema):
        return None

    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = _ema_series(macd_line, signal_period)

    if not signal_line:
        return None

    m = macd_line[-1]
    s = signal_line[-1]
    h = m - s
    result = (round(m, 4), round(s, 4), round(h, 4))
    return _cached(key, result)


def compute_atr(
    bars: Sequence[dict[str, Any]],
    period: int = 14,
    symbol: str = "",
) -> float | None:
    """Compute ATR(period) using Wilder smoothing.

    Args:
        bars: Sequence of OHLC bar dicts with high/low/close keys.
        period: ATR period (default 14).
        symbol: Optional cache key tag.

    Returns:
        ATR value > 0, or None if insufficient data.
    """
    key = _cache_key("atr", symbol, period, len(bars))
    hit, cached_val = _from_cache(key)
    if hit:
        return cached_val

    if len(bars) < period + 1:
        return None

    highs = _highs(bars)
    lows = _lows(bars)
    closes = _closes(bars)

    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return None

    true_ranges: list[float] = []
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        true_ranges.append(max(hl, hpc, lpc))

    if len(true_ranges) < period:
        return None

    # Initial ATR = simple average of first `period` TRs
    atr = sum(true_ranges[:period]) / period
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period

    return _cached(key, round(atr, 4))


def compute_vwap(
    bars: Sequence[dict[str, Any]],
    symbol: str = "",
) -> float | None:
    """Compute session VWAP from OHLC+volume bars.

    Uses typical price = (H + L + C) / 3 weighted by volume.

    Args:
        bars: Sequence of OHLC bar dicts with volume key.
        symbol: Optional cache key tag.

    Returns:
        VWAP value, or None if no volume data available.
    """
    key = _cache_key("vwap", symbol, len(bars))
    hit, cached_val = _from_cache(key)
    if hit:
        return cached_val

    if not bars:
        return None

    highs = _highs(bars)
    lows = _lows(bars)
    closes = _closes(bars)
    volumes = _volumes(bars)

    n = min(len(highs), len(lows), len(closes), len(volumes))
    if n == 0:
        return None

    cum_tp_vol = 0.0
    cum_vol = 0.0
    for i in range(n):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        v = volumes[i]
        cum_tp_vol += tp * v
        cum_vol += v

    if cum_vol <= 0:
        return None

    vwap = cum_tp_vol / cum_vol
    return _cached(key, round(vwap, 4))


# ---------------------------------------------------------------------------
# Legacy weighted score (unchanged — backward compat)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class WeightedSignalInputs:
    """Indicator inputs for weighted scoring."""

    ema: float
    rsi: float
    macd: float
    vwap: float
    adx: float


def weighted_score(inputs: WeightedSignalInputs) -> float:
    """Compute configured weighted score."""
    return (
        1.2 * inputs.ema
        + 1.0 * inputs.rsi
        + 0.8 * inputs.macd
        + 1.5 * inputs.vwap
        + 1.3 * inputs.adx
    )


__all__ = [
    "WeightedSignalInputs",
    "weighted_score",
    "compute_rsi",
    "compute_ema",
    "compute_macd",
    "compute_atr",
    "compute_vwap",
]
