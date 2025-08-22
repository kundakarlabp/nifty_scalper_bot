from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Literal, Tuple

import pandas as pd

from src.config import settings
from src.utils.atr_helper import compute_atr, atr_sl_tp_points, latest_atr_value
from src.utils.indicators import calculate_vwap
from src.signals.regime_detector import detect_market_regime

logger = logging.getLogger(__name__)

Side = Literal["BUY", "SELL"]
SignalOutput = Optional[Dict[str, Any]]


class EnhancedScalpingStrategy:
    """
    Regime-aware scalping strategy:
      - Price EMAs & RSI on OPTION
      - ADX/DI & VWAP on SPOT
      - ATR-based SL/TP with regime & confidence shaping
      - Signature: generate_signal(df, current_price, spot_df)

    Output example:
      {
        "side": "BUY" | "SELL",
        "confidence": float,
        "sl_points": float,
        "tp_points": float,
        "score": int,
        "entry_price": float,
        "stop_loss": float,
        "target": float,
        "reasons": list[str],
        "regime": "trend_up" | "trend_down" | "range" | "unknown",
        "emas": {"fast": float, "slow": float},
        "rsi": float,
        "atr": float,
      }
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
        # keep adx_period knob (spot indicators)
        self.adx_period = int(getattr(strat, "adx_period", adx_period))
        self.adx_trend_strength = int(getattr(strat, "adx_trend_strength", adx_trend_strength))
        self.atr_period = int(getattr(strat, "atr_period", 14))

        # regime shaping (all floats; may be +/-)
        self.trend_tp_boost = float(getattr(strat, "trend_tp_boost", 0.6))
        self.trend_sl_relax = float(getattr(strat, "trend_sl_relax", 0.2))
        self.range_tp_tighten = float(getattr(strat, "range_tp_tighten", -0.4))
        self.range_sl_tighten = float(getattr(strat, "range_sl_tighten", -0.2))

        # bars threshold for validity
        self.min_bars_for_signal = int(getattr(strat, "min_bars_for_signal", max(self.ema_slow, 10)))

    # ------------- tech utils -------------
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
    def _extract_adx_columns(spot_df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """
        Return (adx, di_plus, di_minus) from spot_df; tolerant to suffix (_{n}) naming.
        """
        if spot_df is None or spot_df.empty:
            return None, None, None
        adx_cols = sorted([c for c in spot_df.columns if c.startswith("adx_")])
        dip_cols = sorted([c for c in spot_df.columns if c.startswith("di_plus_")])
        dim_cols = sorted([c for c in spot_df.columns if c.startswith("di_minus_")])
        adx = spot_df[adx_cols[-1]] if adx_cols else spot_df.get("adx")
        di_plus = spot_df[dip_cols[-1]] if dip_cols else spot_df.get("di_plus")
        di_minus = spot_df[dim_cols[-1]] if dim_cols else spot_df.get("di_minus")
        return adx, di_plus, di_minus

    # ------------- main -------------
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_price: float,
        spot_df: pd.DataFrame,
    ) -> SignalOutput:
        """
        Generate signal with regime & confidence aware SL/TP shaping.
        """
        if df is None or df.empty or len(df) < self.min_bars_for_signal:
            logger.debug("DataFrame too short to generate signal.")
            return None

        reasons: list[str] = []
        score = 0

        # 1) EMA crossover bias (OPTION)
        ema_fast = self._ema(df["close"], self.ema_fast)
        ema_slow = self._ema(df["close"], self.ema_slow)
        ema_bias_up = bool(ema_fast.iloc[-1] > ema_slow.iloc[-1])
        ema_cross_up = bool((ema_fast.iloc[-2] <= ema_slow.iloc[-2]) and ema_bias_up)
        ema_cross_down = bool((ema_fast.iloc[-2] >= ema_slow.iloc[-2]) and not ema_bias_up)

        if ema_cross_up:
            score += 2
            reasons.append(f"EMA fast ({self.ema_fast}) crossed above slow ({self.ema_slow}).")
        elif ema_cross_down:
            score += 2
            reasons.append(f"EMA fast ({self.ema_fast}) crossed below slow ({self.ema_slow}).")

        # 2) RSI (OPTION)
        rsi_val = float(self._rsi(df["close"], self.rsi_period).iloc[-1])
        if ema_bias_up and rsi_val > 50:
            score += 1
            reasons.append(f"RSI ({self.rsi_period}) > 50 (up momentum).")
        elif not ema_bias_up and rsi_val < 50:
            score += 1
            reasons.append(f"RSI ({self.rsi_period}) < 50 (down momentum).")

        # 3) Regime detection (SPOT: ADX/DI)
        regime = None
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
                reasons.append("Spot ADX: trend_up (aligned).")
            elif regime == "trend_down" and not ema_bias_up:
                score += 2
                reasons.append("Spot ADX: trend_down (aligned).")
            elif regime == "range":
                reasons.append("Spot ADX: range.")

        # 4) VWAP (SPOT)
        if spot_df is not None and len(spot_df) > 0:
            vwap_series = calculate_vwap(spot_df)
            if vwap_series is not None and len(vwap_series) > 0:
                vwap_val = float(vwap_series.iloc[-1])
                current_spot_price = float(spot_df["close"].iloc[-1])
                if ema_bias_up and current_spot_price > vwap_val:
                    score += 1
                    reasons.append("Spot > VWAP (risk-on).")
                elif not ema_bias_up and current_spot_price < vwap_val:
                    score += 1
                    reasons.append("Spot < VWAP (risk-off).")

        # 5) scoring gate
        min_score = int(getattr(getattr(settings, "strategy", object()), "min_signal_score", 5))
        if score < min_score:
            logger.debug("Score %s < min_score %s — no signal.", score, min_score)
            return None

        # score → confidence (coarse)
        confidence_map = {0: 0.0, 1: 0.0, 2: 0.0, 3: 2.5, 4: 2.5, 5: 5.0, 6: 5.0, 7: 7.5, 8: 7.5, 9: 10.0}
        confidence = float(confidence_map.get(score, 10.0 if score >= 9 else 0.0))

        min_conf = float(getattr(getattr(settings, "strategy", object()), "confidence_threshold", 6.0))
        if confidence < min_conf:
            logger.debug("Confidence %.2f < threshold %.2f — no signal.", confidence, min_conf)
            return None

        # 6) ATR SL/TP base (OPTION)
        atr_series = compute_atr(df, period=self.atr_period)
        atr_val = latest_atr_value(atr_series, default=0.0)
        if atr_val <= 0:
            logger.debug("ATR invalid/missing — no SL/TP.")
            return None

        strat = getattr(settings, "strategy", object())
        base_sl_mult = float(getattr(strat, "atr_sl_multiplier", 1.5))
        base_tp_mult = float(getattr(strat, "atr_tp_multiplier", 3.0))
        sl_adj = float(getattr(strat, "sl_confidence_adj", 0.2))
        tp_adj = float(getattr(strat, "tp_confidence_adj", 0.3))

        # regime shaping (additive to multipliers; can be negative)
        sl_mult = base_sl_mult
        tp_mult = base_tp_mult
        if regime == "trend_up" or regime == "trend_down":
            tp_mult += self.trend_tp_boost
            sl_mult += self.trend_sl_relax
            reasons.append(f"Regime boost: trend (tp+{self.trend_tp_boost}, sl+{self.trend_sl_relax}).")
        elif regime == "range":
            tp_mult += self.range_tp_tighten
            sl_mult += self.range_sl_tighten
            reasons.append(f"Regime tighten: range (tp{self.range_tp_tighten:+}, sl{self.range_sl_tighten:+}).")

        # ensure bounds
        sl_mult = max(0.2, sl_mult)
        tp_mult = max(0.4, tp_mult)

        base_sl = atr_val * sl_mult
        base_tp = atr_val * tp_mult

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

        side: Side = "BUY" if ema_bias_up else "SELL"
        entry_price = float(current_price)

        # Final SL/TP prices (with safety clamp on SL)
        eps = 1e-4  # ultra-small floor to avoid zero/negative SL
        if side == "BUY":
            stop_loss = max(eps, entry_price - sl_points)
            target = entry_price + tp_points
        else:
            stop_loss = max(eps, entry_price + sl_points)
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
            "reasons": reasons,
            "regime": regime or "unknown",
            "emas": {"fast": float(ema_fast.iloc[-1]), "slow": float(ema_slow.iloc[-1])},
            "rsi": float(rsi_val),
            "atr": float(atr_val),
        }

        logger.info("Generated signal: %s", signal)
        return signal