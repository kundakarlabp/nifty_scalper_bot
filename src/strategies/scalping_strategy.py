# src/strategies/scalping_strategy.py
from __future__ import annotations
"""
EnhancedScalpingStrategy
------------------------

Signal engine used by RealTimeTrader.

Design:
- Multi-indicator blend (EMA slope/cross, MACD, Supertrend, VWAP, ADX, BB width)
- Regime gating: TREND vs RANGE decided by ADX + BB width
- Score -> confidence on 0–10 scale (PositionSizing expects 0..10)
- ATR-aware SL/TP with sensible fallbacks to Config base points
- Options-friendly: also returns 'sl_points'/'tp_points' for convenience

Output (when a signal is present):
{
  "signal": "BUY"|"SELL",
  "direction": "BUY"|"SELL",       # alias
  "entry_price": float,            # use last close
  "stop_loss": float,              # absolute price
  "target": float,                 # absolute price
  "sl_points": float,              # distance in points
  "tp_points": float,              # distance in points
  "atr": float,
  "score": int,                    # 0..100
  "confidence": float,             # 0..10
  "regime": "TREND"|"RANGE",
}
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Config
from src.utils.indicators import (
    calculate_ema,
    calculate_macd,
    calculate_adx,
    calculate_bb_width,
    calculate_supertrend,
    calculate_vwap,
    calculate_atr,
)


def _cfg(name: str, default):
    return getattr(Config, name, default)


@dataclass
class EnhancedScalpingStrategy:
    # Base point stops/targets used if ATR info is weak/unavailable
    base_stop_loss_points: float = float(_cfg("BASE_STOP_LOSS_POINTS", 8.0))
    base_target_points: float = float(_cfg("BASE_TARGET_POINTS", 14.0))

    # How strong a setup must be on 0..10 to emit a signal
    confidence_threshold: float = float(_cfg("CONFIDENCE_THRESHOLD", 6.0))
    # If caller passes MIN_SIGNAL_SCORE as int (e.g., 6), we compare to 0..10 confidence
    min_score_threshold: int = int(_cfg("MIN_SIGNAL_SCORE", 6))

    # Indicator params (can be overridden via Config)
    ema_fast: int = int(_cfg("EMA_FAST", 12))
    ema_slow: int = int(_cfg("EMA_SLOW", 26))
    macd_fast: int = int(_cfg("MACD_FAST", 12))
    macd_slow: int = int(_cfg("MACD_SLOW", 26))
    macd_signal: int = int(_cfg("MACD_SIGNAL", 9))
    adx_period: int = int(_cfg("ADX_PERIOD", 14))
    bb_window: int = int(_cfg("BB_WINDOW", 20))
    bb_std: float = float(_cfg("BB_STD", 2.0))
    st_period: int = int(_cfg("SUPERTREND_PERIOD", 10))
    st_mult: float = float(_cfg("SUPERTREND_MULT", 3.0))
    atr_period: int = int(_cfg("ATR_PERIOD", 14))

    # Regime gates
    trend_adx_min: float = float(_cfg("TREND_ADX_MIN", 18.0))  # > this → trending
    range_adx_max: float = float(_cfg("RANGE_ADX_MAX", 15.0))  # < this → ranging

    # ATR scaling for SL (e.g., 1.1 * ATR)
    atr_sl_mult: float = float(_cfg("ATR_SL_MULTIPLIER", 1.5))
    atr_tp_mult: float = float(_cfg("ATR_TP_MULTIPLIER", 2.2))

    # Misc
    use_close_for_entry: bool = True

    # ------------------------------- public -------------------------------- #

    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Emit a directional signal using the latest completed bar.

        Expects df with columns: open, high, low, close, (volume optional).
        """
        if not isinstance(df, pd.DataFrame) or len(df) < max(50, self.bb_window + 5, self.ema_slow + 5):
            return None

        df = df.copy()
        last_idx = df.index[-1]

        # --- indicators ---
        ema_f = calculate_ema(df, self.ema_fast)
        ema_s = calculate_ema(df, self.ema_slow)
        macd_line, macd_sig, macd_hist = calculate_macd(df, self.macd_fast, self.macd_slow, self.macd_signal)
        adx, di_pos, di_neg = calculate_adx(df, period=self.adx_period)
        bb_u, bb_l = calculate_bb_width(df, window=self.bb_window, std=self.bb_std)
        st_dir, st_up, st_lo = calculate_supertrend(df, period=self.st_period, multiplier=self.st_mult)
        vwap = calculate_vwap(df)

        # Attach
        df["ema_f"] = ema_f
        df["ema_s"] = ema_s
        df["macd"] = macd_line
        df["macd_sig"] = macd_sig
        df["macd_hist"] = macd_hist
        df["adx"] = adx
        df["di_pos"] = di_pos
        df["di_neg"] = di_neg
        df["bb_u"] = bb_u
        df["bb_l"] = bb_l
        df["st_dir"] = st_dir
        df["st_up"] = st_up
        df["st_lo"] = st_lo
        df["vwap"] = vwap
        df["atr"] = calculate_atr(df, period=self.atr_period)

        row = df.iloc[-1]
        px = float(row["close"])
        atr = float(row.get("atr") or 0.0)

        # --- regime ---
        regime = self._regime(df)

        # --- directional scoring ---
        long_score, short_score = self._scores(df)

        # Turn 0..100 → 0..10 confidence
        long_conf = self._to_conf(long_score)
        short_conf = self._to_conf(short_score)

        # Gate by regime and threshold
        chosen = None
        if long_conf >= max(self.confidence_threshold, float(self.min_score_threshold)):
            if regime == "TREND" or self._near_band_breakout(row, direction="BUY"):
                chosen = ("BUY", long_conf)
        if short_conf >= max(self.confidence_threshold, float(self.min_score_threshold)):
            if chosen is None and (regime == "TREND" or self._near_band_breakout(row, direction="SELL")):
                chosen = ("SELL", short_conf)

        if not chosen:
            return None

        side, conf = chosen
        sl, tp, sl_pts, tp_pts = self._sl_tp(px, atr, side)

        out = {
            "signal": side,
            "direction": side,
            "entry_price": float(px),
            "stop_loss": float(sl),
            "target": float(tp),
            "sl_points": float(sl_pts),
            "tp_points": float(tp_pts),
            "atr": float(atr or 0.0),
            "score": int(max(long_score, short_score)),
            "confidence": float(conf),
            "regime": regime,
        }
        return out

    def generate_options_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Same signal as generate_signal(), just returned with keys that are
        convenient for options legs. (RealTimeTrader handles strikes.)
        """
        sig = self.generate_signal(df)
        if not sig:
            return None
        # Keep payload identical — options flow upstream just needs entry/SL/TP
        return sig

    # ------------------------------- internals ------------------------------ #

    def _regime(self, df: pd.DataFrame) -> str:
        """TREND if ADX strong; otherwise RANGE. BB width can be added if desired."""
        adx = df["adx"].iloc[-1]
        if float(adx) >= self.trend_adx_min:
            return "TREND"
        if float(adx) <= self.range_adx_max:
            return "RANGE"
        # Fuzzy middle: look at BB compression/expansion around last n bars
        try:
            last = df.iloc[-1]
            width = float((last["bb_u"] - last["bb_l"]) / last["close"])
            if width <= 0.008:  # ~0.8% band width → range-ish
                return "RANGE"
            return "TREND"
        except Exception:
            return "TREND"

    def _ema_slope(self, s: pd.Series, lookback: int = 3) -> float:
        if len(s) < lookback + 1:
            return 0.0
        # simple slope normalized by price
        y2 = float(s.iloc[-1])
        y1 = float(s.iloc[-1 - lookback])
        return (y2 - y1) / max(1e-6, y1)

    def _scores(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Compute long/short scores on 0..100.
        Heuristic blend (tunable; keep simple & fast for real-time).
        """
        row = df.iloc[-1]
        prev = df.iloc[-2]

        px = float(row["close"])
        ema_f = float(row["ema_f"])
        ema_s = float(row["ema_s"])
        macd = float(row["macd"])
        macd_sig = float(row["macd_sig"])
        macd_hist = float(row["macd_hist"])
        adx = float(row["adx"])
        di_pos = float(row["di_pos"])
        di_neg = float(row["di_neg"])
        vwap = float(row.get("vwap") or px)
        st_dir = int(row["st_dir"])

        # Slopes
        slope_f = self._ema_slope(df["ema_f"], 3)
        slope_s = self._ema_slope(df["ema_s"], 5)

        long = 0.0
        short = 0.0

        # Trend alignment
        if ema_f > ema_s:
            long += 15
        if ema_f < ema_s:
            short += 15

        # Slope
        if slope_f > 0:
            long += min(15, 100 * slope_f)
        else:
            short += min(15, 100 * abs(slope_f))
        if slope_s > 0:
            long += min(10, 100 * slope_s)
        else:
            short += min(10, 100 * abs(slope_s))

        # MACD cross & histogram
        if macd > macd_sig:
            long += 15
        else:
            short += 15
        if macd_hist > 0:
            long += 10
        else:
            short += 10

        # DI dominance
        if di_pos > di_neg:
            long += 10
        if di_neg > di_pos:
            short += 10

        # Above/below VWAP
        if px > vwap:
            long += 10
        else:
            short += 10

        # Supertrend filter
        if st_dir > 0:
            long += 10
        else:
            short += 10

        # Recent momentum (close vs previous close)
        if px > float(prev["close"]):
            long += 5
        else:
            short += 5

        return int(round(long)), int(round(short))

    def _to_conf(self, score_100: int) -> float:
        """
        Map 0..100 → 0..10 (cap).
        """
        return float(max(0.0, min(10.0, score_100 / 10.0)))

    def _near_band_breakout(self, row: pd.Series, direction: str) -> bool:
        """
        If in RANGE regime but price is near BB edge, allow breakouts.
        """
        try:
            u = float(row["bb_u"])
            l = float(row["bb_l"])
            c = float(row["close"])
            if direction == "BUY":
                return c >= (u - 0.1 * (u - l))
            else:
                return c <= (l + 0.1 * (u - l))
        except Exception:
            return False

    def _sl_tp(self, entry: float, atr: float, side: str) -> Tuple[float, float, float, float]:
        """
        Compute absolute SL/TP and their point distances.
        Prefer ATR-based distances; fallback to base points.
        """
        # Robust guards
        atr = float(atr or 0.0)
        use_atr = np.isfinite(atr) and atr > 0

        sl_pts = (self.atr_sl_mult * atr) if use_atr else float(self.base_stop_loss_points)
        tp_pts = (self.atr_tp_mult * atr) if use_atr else float(self.base_target_points)

        sl_pts = max(0.5, float(sl_pts))
        tp_pts = max(sl_pts * 1.2, float(tp_pts))  # keep TP > SL

        if side == "BUY":
            sl = entry - sl_pts
            tp = entry + tp_pts
        else:
            sl = entry + sl_pts
            tp = entry - tp_pts

        return float(sl), float(tp), float(sl_pts), float(tp_pts)