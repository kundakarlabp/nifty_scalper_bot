# src/strategies/trend_strategy.py
from __future__ import annotations
from typing import Dict, Optional
import pandas as pd

class TrendScalpingStrategy:
    """
    Simple trend-following entry:
      - Direction from fast/slow EMA alignment
      - Pullback entry with tight SL behind recent swing / EMA
      - Target as RR multiple
    Output schema matches your existing EnhancedScalpingStrategy.
    """

    def __init__(
        self,
        fast: int = 9,
        slow: int = 21,
        rr: float = 1.6,
        base_sl_points: float = 6.0,
        min_slope: float = 0.0008,  # 0.08% over 5 bars
        confidence_floor: float = 0.55,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.rr = rr
        self.base_sl_points = base_sl_points
        self.min_slope = min_slope
        self.confidence_floor = confidence_floor

    def generate_signal(self, df: pd.DataFrame, last_price: float) -> Optional[Dict[str, float]]:
        if df is None or df.empty or len(df) < max(self.slow, 30):
            return None

        close = df["close"].astype(float)
        fast = close.ewm(span=self.fast, adjust=False).mean()
        slow = close.ewm(span=self.slow, adjust=False).mean()

        # trend direction & slope
        dir_up = fast.iloc[-1] > slow.iloc[-1]
        slope = (slow.iloc[-1] - slow.iloc[-5]) / max(1e-9, slow.iloc[-5])
        if abs(float(slope)) < self.min_slope:
            return None

        # entry at market price (you already have spread guards elsewhere)
        entry = float(last_price)

        # SL just beyond slow EMA (trend protected) or base points
        if dir_up:
            sl = min(entry - self.base_sl_points, float(slow.iloc[-1]))
            target = entry + self.rr * abs(entry - sl)
            direction = "BUY"
        else:
            sl = max(entry + self.base_sl_points, float(slow.iloc[-1]))
            target = entry - self.rr * abs(entry - sl)
            direction = "SELL"

        confidence = max(self.confidence_floor, min(0.99, abs(float(slope)) / (self.min_slope * 2.0)))
        return {
            "signal": direction,
            "entry_price": float(entry),
            "stop_loss": float(sl),
            "target": float(target),
            "confidence": float(confidence),
            "market_volatility": 0.0,  # optional hook
        }
