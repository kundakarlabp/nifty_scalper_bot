from __future__ import annotations

import pandas as pd


def _compute_adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Minimal ADX/DI implementation (Wilder).
    Assumes df has 'high','low','close'.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = (high.combine(low, max) - low.combine(low, min)).abs()
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, 1e-9)

    plus_di = (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr) * 100.0
    minus_di = (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr) * 100.0
    dx = (plus_di.subtract(minus_di).abs() / (plus_di.add(minus_di).abs() + 1e-9)) * 100.0
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx, plus_di, minus_di


def detect_market_regime(
    *,
    df: pd.DataFrame,
    adx: pd.Series | None = None,
    di_plus: pd.Series | None = None,
    di_minus: pd.Series | None = None,
    adx_trend_strength: int = 20,
) -> str:
    """
    Returns: "trend_up" | "trend_down" | "range"
    """
    if df is None or df.empty:
        return "range"

    if adx is None or di_plus is None or di_minus is None:
        adx, di_plus, di_minus = _compute_adx(df, period=14)

    try:
        adx_val = float(adx.iloc[-1])
        dplus = float(di_plus.iloc[-1])
        dminus = float(di_minus.iloc[-1])
    except Exception:
        return "range"

    if adx_val < float(adx_trend_strength):
        return "range"
    # trend: use DI dominance
    if dplus > dminus:
        return "trend_up"
    return "trend_down"