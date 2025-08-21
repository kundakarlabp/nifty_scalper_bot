from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Literal

import pandas as pd

from src.config import settings
from src.utils.atr_helper import compute_atr, atr_sl_tp_points
from src.utils.indicators import calculate_vwap
from src.signals.regime_detector import detect_market_regime

logger = logging.getLogger(__name__)

Side = Literal["BUY", "SELL"]
SignalOutput = Optional[Dict[str, Any]]


class EnhancedScalpingStrategy:
    """
    Trend-vs-Range aware scalping strategy that combines EMA, RSI, ADX/DI, and VWAP.

    generate_signal(df, current_price, spot_df)
      - df:   option OHLCV
      - current_price: current option price (float)
      - spot_df: spot OHLCV (with ADX/DI columns if available)
    """

    def __init__(
        self,
        *,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        adx_period: int = 14,
        adx_trend_strength: int = 20,
    ) -> None:
        strat = getattr(settings, "strategy", object())
        self.ema_fast = int(getattr(strat, "ema_fast", ema_fast))
        self.ema_slow = int(getattr(strat, "ema_slow", ema_slow))
        self.rsi_period = int(getattr(strat, "rsi_period", rsi_period))
        self.adx_period = int(getattr(strat, "adx_period", adx_period))
        self.adx_trend_strength = int(getattr(strat, "adx_trend_strength", adx_trend_strength))
        self.atr_period = int(getattr(strat, "atr_period", 14))

    @staticmethod
    def _ema(s: pd.Series, period: int) -> pd.Series:
        return s.ewm(span=max(1, int(period)), adjust=False).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
        rs = (avg_gain / (avg_loss.replace(0.0, 1e-9))).fillna(0.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _extract_adx_columns(spot_df: pd.DataFrame) -> tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        if spot_df is None or spot_df.empty:
            return None, None, None
        adx_cols = sorted([c for c in spot_df.columns if c.startswith("adx_")])
        dip_cols = sorted([c for c in spot_df.columns if c.startswith("di_plus_")])
        dim_cols = sorted([c for c in spot_df.columns if c.startswith("di_minus_")])
        adx = spot_df[adx_cols[-1]] if adx_cols else spot_df.get("adx")
        di_plus = spot_df[dip_cols[-1]] if dip_cols else spot_df.get("di_plus")
        di_minus = spot_df[dim_cols[-1]] if dim_cols else spot_df.get("di_minus")
        return adx, di_plus, di_minus

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_price: float,
        spot_df: pd.DataFrame,
    ) -> SignalOutput:
        if df is None or df.empty or len(df) < self.ema_slow:
            return None

        reasons: list[str] = []
        score = 0

        # 1) EMA bias on OPTION
        ema_fast = self._ema(df["close"], self.ema_fast)
        ema_slow = self._ema(df["close"], self.ema_slow)
        ema_bias_up = bool(ema_fast.iloc[-1] > ema_slow.iloc[-1])
        ema_cross_up = bool((ema_fast.iloc[-2] <= ema_slow.iloc[-2]) and ema_bias_up)
        ema_cross_dn = bool((ema_fast.iloc[-2] >= ema_slow.iloc[-2]) and not ema_bias_up)

        if ema_cross_up:
            score += 2
            reasons.append(f"EMA fast({self.ema_fast}) crossed above slow({self.ema_slow}).")
        elif ema_cross_dn:
            score += 2
            reasons.append(f"EMA fast({self.ema_fast}) crossed below slow({self.ema_slow}).")

        # 2) RSI on OPTION
        rsi_val = float(self._rsi(df["close"], self.rsi_period).iloc[-1])
        if ema_bias_up and rsi_val > 50:
            score += 1
            reasons.append(f"RSI({self.rsi_period})>50 (up momentum).")
        elif not ema_bias_up and rsi_val < 50:
            score += 1
            reasons.append(f"RSI({self.rsi_period})<50 (down momentum).")

        # 3) Market regime from SPOT
        regime = "unknown"
        if spot_df is not None and len(spot_df) >= max(10, self.adx_period):
            adx_series, di_plus_series, di_minus_series = self._extract_adx_columns(spot_df)
            regime = detect_market_regime(
                df=spot_df,
                adx=adx_series,
                di_plus=di_plus_series,
                di_minus=di_minus_series,
                adx_trend_strength=self.adx_trend_strength,
            )
            if regime == "trend_up" and ema_bias_up:
                score += 2
                reasons.append("Spot ADX trend ↑.")
            elif regime == "trend_down" and not ema_bias_up:
                score += 2
                reasons.append("Spot ADX trend ↓.")
            elif regime == "range":
                reasons.append("Spot indicates range.")

        # 4) VWAP on SPOT
        vwap_ok = True
        if spot_df is not None and len(spot_df) > 0:
            vwap_series = calculate_vwap(spot_df)
            if vwap_series is not None and len(vwap_series) > 0:
                vwap_val = float(vwap_series.iloc[-1])
                current_spot = float(spot_df["close"].iloc[-1])

                # VWAP filter adapts to regime:
                if regime.startswith("trend"):
                    if ema_bias_up and current_spot > vwap_val:
                        score += 1
                        reasons.append("Spot above VWAP in trend.")
                    elif (not ema_bias_up) and current_spot < vwap_val:
                        score += 1
                        reasons.append("Spot below VWAP in trend.")
                    else:
                        vwap_ok = False
                elif regime == "range":
                    # near VWAP (±0.2%) is fine in range
                    if abs(current_spot - vwap_val) / max(1e-9, vwap_val) <= 0.002:
                        score += 1
                        reasons.append("Spot near VWAP (range).")
                    else:
                        vwap_ok = False

        # Gate by score and confidence
        min_score = int(getattr(getattr(settings, "strategy", object()), "min_signal_score", 5))
        if score < min_score or not vwap_ok:
            return None

        # Confidence mapping (slightly regime-aware)
        # 0-2->0, 3-4->2.5, 5-6->5, 7-8->7.5, 9+->10 ; boost trend by +0.5
        base_map = {0: 0.0, 1: 0.0, 2: 0.0, 3: 2.5, 4: 2.5, 5: 5.0, 6: 5.0, 7: 7.5, 8: 7.5, 9: 10.0}
        confidence = float(base_map.get(score, 10.0 if score >= 9 else 0.0))
        if regime.startswith("trend"):
            confidence += 0.5
        confidence = min(confidence, 10.0)

        min_conf = float(getattr(getattr(settings, "strategy", object()), "confidence_threshold", 6.0))
        if confidence < min_conf:
            return None

        # ATR-based SL/TP on OPTION
        atr_series = compute_atr(df, period=self.atr_period)
        if atr_series is None or len(atr_series) == 0:
            return None
        atr_val = float(atr_series.iloc[-1] or 0.0)
        if atr_val <= 0:
            return None

        strat = getattr(settings, "strategy", object())
        sl_mult = float(getattr(strat, "atr_sl_multiplier", getattr(settings, "ATR_SL_MULTIPLIER", 1.5)))
        tp_mult = float(getattr(strat, "atr_tp_multiplier", getattr(settings, "ATR_TP_MULTIPLIER", 3.0)))
        sl_adj = float(getattr(strat, "sl_confidence_adj", getattr(settings, "SL_CONFIDENCE_ADJ", 0.2)))
        tp_adj = float(getattr(strat, "tp_confidence_adj", getattr(settings, "TP_CONFIDENCE_ADJ", 0.3)))

        # Regime adaptation: trend = wider TP, range = tighter TP, tighter SL
        if regime.startswith("trend"):
            tp_mult *= 1.1
        elif regime == "range":
            tp_mult *= 0.8
            sl_mult *= 0.9

        sl_points, tp_points = atr_sl_tp_points(
            base_sl_points=atr_val * sl_mult,
            base_tp_points=atr_val * tp_mult,
            atr_value=atr_val,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            confidence=confidence,
            sl_confidence_adj=sl_adj,
            tp_confidence_adj=tp_adj,
        )

        side: Side = "BUY" if ema_bias_up else "SELL"
        entry_price = float(current_price)

        if side == "BUY":
            stop_loss = entry_price - sl_points
            target = entry_price + tp_points
        else:
            stop_loss = entry_price + sl_points
            target = entry_price - tp_points

        signal: Dict[str, Any] = {
            "side": side,
            "confidence": float(confidence),
            "sl_points": float(sl_points),
            "tp_points": float(tp_points),
            "score": int(score),
            "entry_price": entry_price,
            "stop_loss": float(stop_loss),
            "target": float(target),
            "reasons": reasons + [f"regime={regime}"],
        }
        logger.info("Generated signal: %s", signal)
        return signal