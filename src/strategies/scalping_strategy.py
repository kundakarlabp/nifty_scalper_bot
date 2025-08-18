# src/strategies/scalping_strategy.py
"""
EnhancedScalpingStrategy:
- Legacy confluence scoring preserved.
- Regime-aware playbooks (RANGE vs TREND) adjust SL/TP/trailing and apply light filters.
- Returns a dict signal with: signal ("BUY"/"SELL"), entry_price, stop_loss, target, score, confidence, reasons[].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import StrategyConfig
from src.signals.regime_detector import detect_market_regime
from src.utils.indicators import (
    ema, rsi, macd_hist, macd_cross, atr, supertrend, bollinger_bandwidth, vwap,
    adx, di_plus_minus
)


@dataclass
class RegimePlaybook:
    name: str
    sl_mult: float
    tp_mult: float
    trail_mult: float
    min_score_bump: int
    notes: str


class EnhancedScalpingStrategy:
    """
    LEGACY behavior + RANGE/TREND playbooks.
    """
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

        # --- Regime playbooks (can be tweaked live via /config set) ---
        self.playbooks = {
            "range": RegimePlaybook(
                name="range",
                sl_mult=max(0.8, cfg.atr_sl_multiplier * 0.8),
                tp_mult=max(1.8, cfg.atr_tp_multiplier * 0.8),
                trail_mult=max(0.8, 1.0),
                min_score_bump=+1,
                notes="Tighter SL/TP; demand a bit more confluence."
            ),
            "trend_up": RegimePlaybook(
                name="trend_up",
                sl_mult=max(1.0, cfg.atr_sl_multiplier),
                tp_mult=max(3.0, cfg.atr_tp_multiplier * 1.2),
                trail_mult=max(1.0, 1.3),
                min_score_bump=0,
                notes="Let winners run; use slightly wider TP and trail."
            ),
            "trend_down": RegimePlaybook(
                name="trend_down",
                sl_mult=max(1.0, cfg.atr_sl_multiplier),
                tp_mult=max(3.0, cfg.atr_tp_multiplier * 1.2),
                trail_mult=max(1.0, 1.3),
                min_score_bump=0,
                notes="Mirror of trend_up for shorts."
            ),
            "unknown": RegimePlaybook(
                name="unknown",
                sl_mult=cfg.atr_sl_multiplier,
                tp_mult=cfg.atr_tp_multiplier,
                trail_mult=1.0,
                min_score_bump=0,
                notes="No regime context."
            ),
        }

        # Live-tweakable switches
        self.require_vwap_alignment: bool = True
        self.bb_squeeze_threshold: float = 0.08  # percent of price
        self.adx_trend_strength: int = 18        # regime detector threshold

    # ---------- helpers ----------

    def _compute_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        feats: Dict[str, pd.Series] = {}
        feats["ema12"] = ema(df["close"], 12)
        feats["ema26"] = ema(df["close"], 26)
        feats["rsi"] = rsi(df["close"], 14)
        macd_val, macd_signal, macd_hist_v = macd_hist(df["close"])
        feats["macd_hist"] = macd_hist_v
        feats["macd_cross"] = macd_cross(df["close"])
        feats["atr"] = atr(df, period=self.cfg.atr_period)
        st_dir, st_upper, st_lower = supertrend(df, period=10, multiplier=3)
        feats["st_dir"] = st_dir
        feats["bb_width"] = bollinger_bandwidth(df["close"], 20, 2.0)
        feats["vwap"] = vwap(df)
        feats["adx"] = adx(df, 14)
        di_p, di_m = di_plus_minus(df, 14)
        feats["di_p"] = di_p
        feats["di_m"] = di_m
        return feats

    def _detect_regime(self, df: pd.DataFrame, feats: Dict[str, pd.Series]) -> str:
        return detect_market_regime(
            df=df,
            adx=feats["adx"],
            di_plus=feats["di_p"],
            di_minus=feats["di_m"],
            adx_trend_strength=self.adx_trend_strength,
        )

    def _score_signal(self, df: pd.DataFrame, feats: Dict[str, pd.Series]) -> Tuple[int, float, List[str]]:
        score = 0
        reasons: List[str] = []

        # EMA alignment
        if feats["ema12"].iloc[-1] > feats["ema26"].iloc[-1]:
            score += 1; reasons.append("EMA12>EMA26")
        else:
            score -= 1; reasons.append("EMA12<=EMA26")

        # RSI momentum
        if feats["rsi"].iloc[-1] > 55:
            score += 1; reasons.append("RSI>55")
        elif feats["rsi"].iloc[-1] < 45:
            score -= 1; reasons.append("RSI<45")

        # MACD confluence
        if feats["macd_hist"].iloc[-1] > 0:
            score += 1; reasons.append("MACD hist +")
        else:
            score -= 1; reasons.append("MACD hist -")

        # Supertrend direction
        if feats["st_dir"].iloc[-1] > 0:
            score += 1; reasons.append("Supertrend up")
        else:
            score -= 1; reasons.append("Supertrend down")

        # VWAP proximity/side
        last = df["close"].iloc[-1]
        if (last >= feats["vwap"].iloc[-1]) == (feats["ema12"].iloc[-1] > feats["ema26"].iloc[-1]):
            score += 1; reasons.append("VWAP aligned")
        else:
            reasons.append("VWAP not aligned")

        # BB squeeze opportunity (mean reversion potential)
        if feats["bb_width"].iloc[-1] <= self.bb_squeeze_threshold * last / 100.0:
            score += 1; reasons.append("BB squeeze")

        # Confidence as normalized score proxy
        # (keep legacy semantics: threshold in config drives final allow/deny)
        confidence = max(0.0, min(10.0, 2.0 + score * 1.0))
        return score, confidence, reasons

    # ---------- public ----------

    def generate_signal(self, df: pd.DataFrame, current_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if len(df) < self.cfg.min_bars_for_signal:
            return None

        feats = self._compute_features(df)
        regime = self._detect_regime(df, feats)
        playbook = self.playbooks.get(regime, self.playbooks["unknown"])

        score, confidence, reasons = self._score_signal(df, feats)

        # Regime-aware bump and light filters
        score += playbook.min_score_bump
        if self.require_vwap_alignment:
            last = df["close"].iloc[-1]
            if not ((last >= feats["vwap"].iloc[-1]) == (feats["ema12"].iloc[-1] > feats["ema26"].iloc[-1])):
                reasons.append("Rejected: VWAP misalignment (playbook)")
                return None

        # Accept only if score passes min threshold
        if score < self.cfg.min_signal_score or confidence < self.cfg.confidence_threshold:
            return None

        # Direction by legacy bias (ema12 vs ema26)
        direction = "BUY" if feats["ema12"].iloc[-1] > feats["ema26"].iloc[-1] else "SELL"
        px = current_price if current_price is not None else float(df["close"].iloc[-1])
        cur_atr = float(feats["atr"].iloc[-1])

        # Base SL/TP from ATR with regime multipliers
        sl_pts = playbook.sl_mult * cur_atr
        tp_pts = playbook.tp_mult * cur_atr

        stop_loss = px - sl_pts if direction == "BUY" else px + sl_pts
        target = px + tp_pts if direction == "BUY" else px - tp_pts

        reasons.append(f"regime={regime} playbook={playbook.notes}")

        return {
            "signal": direction,
            "entry_price": px,
            "stop_loss": stop_loss,
            "target": target,
            "score": int(score),
            "confidence": float(confidence),
            "atr": float(cur_atr),
            "regime": regime,
            "reasons": reasons,
            # trailing uses playbook.trail_mult downstream (executor/runner)
            "trail_mult": float(playbook.trail_mult),
        }

    # --------- live tweak via Telegram /config ---------
    def set_param(self, key: str, value: str) -> str:
        """
        Safe runtime overrides. Returns a message string.
        """
        key = key.strip().lower()
        if key == "require_vwap_alignment":
            self.require_vwap_alignment = value.lower() in ("1","true","yes","on")
            return f"require_vwap_alignment={self.require_vwap_alignment}"
        if key == "bb_squeeze_threshold":
            try:
                self.bb_squeeze_threshold = float(value)
                return f"bb_squeeze_threshold={self.bb_squeeze_threshold}"
            except ValueError:
                return "Invalid float"
        if key == "adx_trend_strength":
            try:
                self.adx_trend_strength = int(value)
                return f"adx_trend_strength={self.adx_trend_strength}"
            except ValueError:
                return "Invalid int"
        return "Key not permitted"

    def get_params(self) -> Dict[str, Any]:
        return {
            "require_vwap_alignment": self.require_vwap_alignment,
            "bb_squeeze_threshold": self.bb_squeeze_threshold,
            "adx_trend_strength": self.adx_trend_strength,
        }
