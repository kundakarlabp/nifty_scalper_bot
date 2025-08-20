"""
Enhanced scalping strategy.

Signature constraint:
  generate_signal(df, current_price)  # NO spot_df here

Output format (dict) example:
  {
    "side": "BUY" | "SELL",
    "confidence": float(0..10),
    "sl_points": float,
    "tp_points": float,
    "score": int,
  }
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.config import settings
from src.utils.atr_helper import compute_atr, atr_sl_tp_points


class EnhancedScalpingStrategy:
    """
    Minimal but effective rules (works with 1-min bars):
      - Fast/slow EMA cross bias
      - RSI momentum bias (computed ad-hoc if present)
      - ATR-based SL/TP with confidence nudges
    """

    def __init__(
        self,
        *,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period

    @staticmethod
    def _ema(s: pd.Series, period: int) -> pd.Series:
        return s.ewm(span=max(1, int(period)), adjust=False).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        up = (delta.clip(lower=0)).ewm(alpha=1 / period, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = (up / (down.replace(0, 1e-12)))
        return 100 - (100 / (1 + rs))

    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Build a single actionable signal from recent bars.
        df must include: 'high','low','close'
        """
        if df is None or df.empty:
            return None
        need_cols = {"high", "low", "close"}
        if not need_cols.issubset(df.columns):
            return None

        close = df["close"].astype(float)
        ema_f = self._ema(close, self.ema_fast)
        ema_s = self._ema(close, self.ema_slow)
        ema_bias_up = ema_f.iloc[-1] > ema_s.iloc[-1]

        # Optional RSI-based momentum tilt
        rsi = self._rsi(close, self.rsi_period)
        rsi_val = float(rsi.iloc[-1])
        momentum_up = rsi_val >= 50.0

        # Score (integer) -> map to confidence
        score = 0
        score += 1 if ema_bias_up else -1
        score += 1 if momentum_up else -1

        # Confidence scale 0..10 centered
        confidence = max(0.0, min(10.0, 5.0 + score * 2.0))

        # ATR and dynamic SL/TP (prefer settings.strategy; fallback to flat)
        atr = compute_atr(df, period=getattr(settings.strategy, "atr_period", 14))
        atr_val = float(atr.iloc[-1]) if len(atr) else 0.0

        base_sl = float(getattr(settings.strategy, "base_stop_loss_points", getattr(settings, "BASE_STOP_LOSS_POINTS", 20.0)))
        base_tp = float(getattr(settings.strategy, "base_target_points", getattr(settings, "BASE_TARGET_POINTS", 40.0)))
        sl_mult = float(getattr(settings.strategy, "atr_sl_multiplier", getattr(settings, "ATR_SL_MULTIPLIER", 1.5)))
        tp_mult = float(getattr(settings.strategy, "atr_tp_multiplier", getattr(settings, "ATR_TP_MULTIPLIER", 3.0)))
        sl_adj = float(getattr(settings.strategy, "sl_confidence_adj", getattr(settings, "SL_CONFIDENCE_ADJ", 0.2)))
        tp_adj = float(getattr(settings.strategy, "tp_confidence_adj", getattr(settings, "TP_CONFIDENCE_ADJ", 0.3)))

        sl_points, tp_points = atr_sl_tp_points(
            base_sl_points=base_sl,
            base_tp_points=base_tp,
            atr_value=atr_val,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            confidence=confidence,
            sl_conf_adj=sl_adj,
            tp_conf_adj=tp_adj,
        )

        side = "BUY" if ema_bias_up else "SELL"
        min_conf = float(getattr(settings.strategy, "confidence_threshold", getattr(settings, "CONFIDENCE_THRESHOLD", 6.0)))
        min_score = int(getattr(settings.strategy, "min_signal_score", getattr(settings, "MIN_SIGNAL_SCORE", 5)))

        # Convert confidence back to rough integer score for gating (inverse of mapping above)
        gated_score = int(round((confidence - 5.0) / 2.0))

        if confidence < min_conf or abs(gated_score) < min_score // 5:
            return None

        return {
            "side": side,
            "confidence": float(confidence),
            "sl_points": float(sl_points),
            "tp_points": float(tp_points),
            "score": int(gated_score),
        }