from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Literal, Tuple

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
    EMA/RSI + ADX/DI (on spot) + VWAP confirmation.
    Adapts SL/TP to regime (trend vs range) for more fluid exits.
    """

    def __init__(self) -> None:
        strat = settings.strategy
        self.ema_fast = int(strat.ema_fast)
        self.ema_slow = int(strat.ema_slow)
        self.rsi_period = int(strat.rsi_period)
        self.adx_period = int(strat.adx_period)
        self.adx_trend_strength = float(strat.adx_trend_strength)
        self.di_diff_threshold = float(strat.di_diff_threshold)
        self.atr_period = int(strat.atr_period)

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
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _extract_adx_cols(spot_df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
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
        opt_df: pd.DataFrame,
        spot_df: pd.DataFrame,
        current_price: float,
    ) -> SignalOutput:
        # Sanity
        if opt_df is None or opt_df.empty or len(opt_df) < max(self.ema_slow, settings.strategy.min_bars_for_signal):
            return None

        reasons: list[str] = []
        score = 0

        # 1) EMA crossover on option
        ema_fast = self._ema(opt_df["close"], self.ema_fast)
        ema_slow = self._ema(opt_df["close"], self.ema_slow)
        bias_up = bool(ema_fast.iloc[-1] > ema_slow.iloc[-1])
        cross_up = bool((ema_fast.iloc[-2] <= ema_slow.iloc[-2]) and bias_up)
        cross_down = bool((ema_fast.iloc[-2] >= ema_slow.iloc[-2]) and not bias_up)
        if cross_up:
            score += 2; reasons.append(f"EMA {self.ema_fast}>{self.ema_slow} cross ↑")
        elif cross_down:
            score += 2; reasons.append(f"EMA {self.ema_fast}<{self.ema_slow} cross ↓")

        # 2) RSI bias
        rsi = float(self._rsi(opt_df["close"], self.rsi_period).iloc[-1])
        if bias_up and rsi > 50:
            score += 1; reasons.append(f"RSI {self.rsi_period}>50 (up)")
        elif (not bias_up) and rsi < 50:
            score += 1; reasons.append(f"RSI {self.rsi_period}<50 (down)")

        # 3) Regime via ADX/DI on spot
        regime, regime_strength = ("range", 0.0)
        if spot_df is not None and len(spot_df) >= self.adx_period:
            adx, di_plus, di_minus = self._extract_adx_cols(spot_df)
            regime, regime_strength = detect_market_regime(
                spot_df,
                adx=adx, di_plus=di_plus, di_minus=di_minus,
                adx_trend_strength=self.adx_trend_strength,
                di_diff_threshold=self.di_diff_threshold,
            )
            if regime == "trend_up" and bias_up:
                score += 2; reasons.append("Spot ADX/DI trend ↑")
            elif regime == "trend_down" and not bias_up:
                score += 2; reasons.append("Spot ADX/DI trend ↓")
            else:
                reasons.append("Spot regime range")

        # 4) VWAP on spot
        if spot_df is not None and not spot_df.empty:
            vwap = calculate_vwap(spot_df)
            if vwap is not None and len(vwap):
                spot_last = float(spot_df["close"].iloc[-1])
                v = float(vwap.iloc[-1])
                if bias_up and spot_last > v:
                    score += 1; reasons.append("Spot above VWAP")
                elif (not bias_up) and spot_last < v:
                    score += 1; reasons.append("Spot below VWAP")

        # scoring gates
        if score < int(settings.strategy.min_signal_score):
            return None

        # confidence map
        conf_map = {0:0.0,1:0.0,2:0.0,3:2.5,4:2.5,5:5.0,6:5.0,7:7.5,8:7.5}
        confidence = float(conf_map.get(score, 10.0 if score >= 9 else 0.0))
        if confidence < float(settings.strategy.confidence_threshold):
            return None

        # ATR on option to size SL/TP
        atr = compute_atr(opt_df, period=self.atr_period)
        if atr is None or not len(atr):
            return None
        atr_v = float(atr.iloc[-1] or 0.0)
        if atr_v <= 0:
            return None

        # Base SL/TP from settings
        strat = settings.strategy
        sl_mult = float(strat.atr_sl_multiplier)
        tp_mult = float(strat.atr_tp_multiplier)

        # Regime nudges (trend: bigger TP, slightly looser SL; range: tighter TP/SL)
        if regime == "trend_up" or regime == "trend_down":
            tp_mult += float(strat.trend_tp_boost)
            sl_mult += float(strat.trend_sl_relax)
        else:
            tp_mult += float(strat.range_tp_tighten)
            sl_mult += float(strat.range_sl_tighten)

        # Blend with confidence nudges
        sl_points, tp_points = atr_sl_tp_points(
            base_sl_points=atr_v * sl_mult,
            base_tp_points=atr_v * tp_mult,
            atr_value=atr_v,
            sl_mult=sl_mult,
            tp_mult=tp_mult,
            confidence=confidence,
            sl_confidence_adj=float(strat.sl_confidence_adj),
            tp_confidence_adj=float(strat.tp_confidence_adj),
        )

        side: Side = "BUY" if bias_up else "SELL"
        entry = float(current_price)

        if side == "BUY":
            stop = entry - sl_points
            tgt = entry + tp_points
        else:
            stop = entry + sl_points
            tgt = entry - tp_points

        return {
            "side": side,
            "confidence": confidence,
            "score": int(score),
            "regime": regime,
            "regime_strength": float(regime_strength),
            "sl_points": float(sl_points),
            "tp_points": float(tp_points),
            "entry_price": entry,
            "stop_loss": float(stop),
            "target": float(tgt),
            "reasons": reasons,
        }