from __future__ import annotations
import pandas as pd


def detect_regime(spot_df: pd.DataFrame) -> str:
    """
    Very light regime detector:
    - Trend if EMA(9) and EMA(21) spread > 0.25% of price and same direction
    - Range otherwise
    """
    if len(spot_df) < 30:
        return "range"
    close = spot_df["close"].astype(float)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    spread = float(abs(ema9.iloc[-1] - ema21.iloc[-1]) / close.iloc[-1])
    if spread > 0.0025:
        return "trend"
    return "range"