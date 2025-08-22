# src/signals/regime_detector.py
from __future__ import annotations

"""
Market regime detection utilities.

- Works whether ADX/DI are precomputed on `spot_df` (e.g., columns adx_14, di_plus_14, di_minus_14),
  or provided as Series, or missing (falls back to an internal calculation).
- Output is one of: "trend_up" | "trend_down" | "range".
- Includes a verbose variant for diagnostics (/diag, /health).

Expected usage (matches scalping_strategy.py):
  regime = detect_market_regime(
      df=spot_df,
      adx=spot_df.get("adx") or spot_df.get("adx_14"),
      di_plus=spot_df.get("di_plus") or spot_df.get("di_plus_14"),
      di_minus=spot_df.get("di_minus") or spot_df.get("di_minus_14"),
      adx_trend_strength=settings.strategy.adx_trend_strength,
      di_diff_threshold=settings.strategy.di_diff_threshold,
  )
"""

from typing import Optional, Literal, Dict, Any, Tuple
import pandas as pd

try:
    from ta.trend import ADXIndicator  # type: ignore
    _TA_OK = True
except Exception:  # pragma: no cover
    ADXIndicator = None  # type: ignore
    _TA_OK = False

Regime = Literal["trend_up", "trend_down", "range"]


# ---------- internal helpers ----------

def _compute_adx_di_manual(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Manual Wilder-style ADX / +DI / -DI (no external TA dependency)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = (high - low).abs()
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0.0, 1e-9)

    di_plus = (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr) * 100.0
    di_minus = (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr) * 100.0
    dx = (di_plus.subtract(di_minus).abs() / (di_plus.add(di_minus).abs() + 1e-9)) * 100.0
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    adx.name, di_plus.name, di_minus.name = "adx", "di_plus", "di_minus"
    return adx, di_plus, di_minus


def _ensure_adx_di(
    df: pd.DataFrame,
    adx: Optional[pd.Series],
    di_plus: Optional[pd.Series],
    di_minus: Optional[pd.Series],
    period: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return valid (adx, di+, di-) series; compute them if not supplied."""
    if adx is not None and di_plus is not None and di_minus is not None:
        return adx.astype(float), di_plus.astype(float), di_minus.astype(float)

    # Try TA library
    if _TA_OK:
        try:
            ind = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=period)
            a = ind.adx().astype(float)
            dpos = ind.adx_pos().astype(float)
            dneg = ind.adx_neg().astype(float)
            a.name, dpos.name, dneg.name = "adx", "di_plus", "di_minus"
            return a, dpos, dneg
        except Exception:
            pass

    # Manual fallback
    return _compute_adx_di_manual(df, period=period)


# ---------- public API ----------

def detect_market_regime(
    *,
    df: pd.DataFrame,
    adx: Optional[pd.Series] = None,
    di_plus: Optional[pd.Series] = None,
    di_minus: Optional[pd.Series] = None,
    adx_trend_strength: float = 20.0,
    di_diff_threshold: float = 8.0,
    period: int = 14,
) -> Regime:
    """
    Decide whether the market is "trend_up", "trend_down", or "range".

    Heuristics:
      1) If ADX < adx_trend_strength => "range"
      2) Else, if DI+ - DI- >= di_diff_threshold => "trend_up"
         If DI- - DI+ >= di_diff_threshold => "trend_down"
      3) Else tie-break using fast vs very-fast EMA slope on close.
    """
    if df is None or df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return "range"

    a, dpos, dneg = _ensure_adx_di(df, adx, di_plus, di_minus, period=period)
    if len(a) == 0:
        return "range"

    a_last = float(a.iloc[-1])
    if a_last < float(adx_trend_strength):
        return "range"

    dpos_last = float(dpos.iloc[-1]) if len(dpos) else 0.0
    dneg_last = float(dneg.iloc[-1]) if len(dneg) else 0.0
    diff = dpos_last - dneg_last

    if diff >= di_diff_threshold:
        return "trend_up"
    if diff <= -di_diff_threshold:
        return "trend_down"

    # Tie-break with EMA slope (on close)
    close = df["close"].astype(float)
    ema_fast = close.ewm(span=max(4, period // 2), adjust=False).mean()
    ema_ultra = close.ewm(span=max(2, period // 3), adjust=False).mean()
    slope = float(ema_ultra.iloc[-1] - ema_fast.iloc[-1])
    if slope > 0:
        return "trend_up"
    if slope < 0:
        return "trend_down"
    return "range"


def detect_market_regime_verbose(
    *,
    df: pd.DataFrame,
    adx: Optional[pd.Series] = None,
    di_plus: Optional[pd.Series] = None,
    di_minus: Optional[pd.Series] = None,
    adx_trend_strength: float = 20.0,
    di_diff_threshold: float = 8.0,
    period: int = 14,
) -> Dict[str, Any]:
    """Diagnostic variant with numeric context for /diag or debugging."""
    if df is None or df.empty:
        return {"regime": "range", "reason": "empty_df"}

    a, dpos, dneg = _ensure_adx_di(df, adx, di_plus, di_minus, period=period)
    regime = detect_market_regime(
        df=df,
        adx=a,
        di_plus=dpos,
        di_minus=dneg,
        adx_trend_strength=adx_trend_strength,
        di_diff_threshold=di_diff_threshold,
        period=period,
    )
    return {
        "regime": regime,
        "adx_last": float(a.iloc[-1]) if len(a) else None,
        "di_plus_last": float(dpos.iloc[-1]) if len(dpos) else None,
        "di_minus_last": float(dneg.iloc[-1]) if len(dneg) else None,
        "adx_trend_strength": float(adx_trend_strength),
        "di_diff_threshold": float(di_diff_threshold),
    }