"""Vectorized technical indicator functions. Pure numpy — no pandas on hot path."""

from __future__ import annotations

import numpy as np


def _validate(arr: np.ndarray, min_len: int) -> bool:
    return len(arr) >= min_len and not np.all(np.isnan(arr[-min_len:]))


def calc_ema(closes: np.ndarray, period: int) -> float | None:
    if not _validate(closes, period):
        return None
    alpha = 2.0 / (period + 1)
    ema = float(closes[0])
    for price in closes[1:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema


def calc_sma(closes: np.ndarray, period: int) -> float | None:
    if not _validate(closes, period):
        return None
    return float(np.mean(closes[-period:]))


def calc_rsi(closes: np.ndarray, period: int = 14) -> float | None:
    if not _validate(closes, period + 1):
        return None
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float | None:
    min_len = period + 1
    if len(highs) < min_len or len(lows) < min_len or len(closes) < min_len:
        return None
    h = highs[-min_len:]
    l = lows[-min_len:]
    c = closes[-min_len:]
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:]))


def calc_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float | None:
    if len(closes) == 0 or np.sum(volumes) == 0:
        return None
    typical = (highs + lows + closes) / 3.0
    return float(np.sum(typical * volumes) / np.sum(volumes))


def calc_bollinger(closes: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple[float, float, float] | None:
    if not _validate(closes, period):
        return None
    window = closes[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window, ddof=1))
    return mid + std_dev * std, mid, mid - std_dev * std


def calc_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> tuple[float, float, float] | None:
    """Returns (adx, +di, -di)."""
    min_len = period * 2 + 1
    if len(highs) < min_len:
        return None
    h, l, c = highs[-min_len:], lows[-min_len:], closes[-min_len:]
    n = len(h)
    tr_arr, pdm_arr, ndm_arr = np.zeros(n - 1), np.zeros(n - 1), np.zeros(n - 1)
    for i in range(1, n):
        tr_arr[i-1] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up = h[i] - h[i-1]
        dn = l[i-1] - l[i]
        pdm_arr[i-1] = up if up > dn and up > 0 else 0.0
        ndm_arr[i-1] = dn if dn > up and dn > 0 else 0.0
    # Wilder smoothing
    def wilder(arr: np.ndarray, p: int) -> float:
        s = float(np.sum(arr[:p]))
        for v in arr[p:]:
            s = s - s / p + float(v)
        return s
    atr14 = wilder(tr_arr, period)
    pdi = 100 * wilder(pdm_arr, period) / atr14 if atr14 else 0.0
    ndi = 100 * wilder(ndm_arr, period) / atr14 if atr14 else 0.0
    dx = 100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) > 0 else 0.0
    return dx, pdi, ndi


def calc_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float] | None:
    """Returns (macd_line, signal_line, histogram)."""
    min_len = slow + signal
    if not _validate(closes, min_len):
        return None
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    if ema_fast is None or ema_slow is None:
        return None
    macd_line = ema_fast - ema_slow
    # Approximate signal line from last 'signal' MACD values
    # Build MACD series for signal smoothing
    macd_vals = []
    for i in range(signal, 0, -1):
        ef = calc_ema(closes[:-i], fast)
        es = calc_ema(closes[:-i], slow)
        if ef is not None and es is not None:
            macd_vals.append(ef - es)
    macd_vals.append(macd_line)
    sig_line = calc_ema(np.array(macd_vals), signal) or macd_line
    return macd_line, sig_line, macd_line - sig_line


def calc_cpr(prev_high: float, prev_low: float, prev_close: float) -> tuple[float, float, float]:
    """Central Pivot Range: (top, pivot, bottom)."""
    pivot = (prev_high + prev_low + prev_close) / 3.0
    bc = (prev_high + prev_low) / 2.0
    tc = (pivot - bc) + pivot
    return max(tc, bc), pivot, min(tc, bc)
