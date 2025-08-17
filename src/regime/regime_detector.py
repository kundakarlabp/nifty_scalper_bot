# src/regime/regime_detector.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Literal, Tuple

import pandas as pd

Regime = Literal["TREND", "RANGE"]

@dataclass
class RegimeDetector:
    adx_period: int = 14
    ema_period: int = 20
    bb_period: int = 20
    bb_dev: float = 2.0
    adx_trend_threshold: float = 18.0      # > this ⇒ trending
    bb_range_threshold: float = 0.025      # < this ⇒ ranging (2.5%)
    min_bars: int = 60

    def detect(self, df: pd.DataFrame) -> Tuple[Regime, float]:
        """
        Returns (regime, confidence 0..1).
        - TREND: ADX high and/or EMA slope strong, Bollinger bandwidth moderate+.
        - RANGE: ADX low and Bollinger bandwidth tight.
        """
        if df is None or df.empty or len(df) < max(self.min_bars, self.adx_period + 5, self.bb_period + 5):
            return "RANGE", 0.0  # neutral/low info → be conservative

        close = df["close"].astype(float)

        # --- ADX (quick, dependency-free approximation)
        # True range components
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        prev_close = close.shift(1)
        tr = (high - low).abs()
        tr = tr.combine((high - prev_close).abs(), max)
        tr = tr.combine((low - prev_close).abs(), max)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = (up_move.where((up_move > down_move) & (up_move > 0), 0.0)).rolling(self.adx_period).sum()
        minus_dm = (down_move.where((down_move > up_move) & (down_move > 0), 0.0)).rolling(self.adx_period).sum()
        atr = tr.rolling(self.adx_period).sum().replace(0, 1e-9)
        plus_di = 100 * (plus_dm / atr)
        minus_di = 100 * (minus_dm / atr)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9))
        adx = dx.rolling(self.adx_period).mean()

        # --- EMA slope as trend proxy
        ema = close.ewm(span=self.ema_period, adjust=False).mean()
        ema_slope = (ema.iloc[-1] - ema.iloc[-5]) / max(1e-9, ema.iloc[-5])
        ema_slope_abs = abs(float(ema_slope))

        # --- Bollinger bandwidth as range proxy
        mid = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std(ddof=0)
        upper = mid + self.bb_dev * std
        lower = mid - self.bb_dev * std
        bandwidth = (upper - lower) / mid.replace(0, 1e-9)
        bb_bw = float(bandwidth.iloc[-1])

        # --- decision
        adx_now = float(adx.iloc[-1])
        trend_score = 0.0
        range_score = 0.0

        # trendy if ADX high and/or slope not tiny
        if adx_now >= self.adx_trend_threshold:
            trend_score += min(1.0, (adx_now - self.adx_trend_threshold) / 20.0)
        trend_score += min(1.0, ema_slope_abs / 0.005) * 0.5  # ~0.5% move over 5 bars → decent

        # rangy if tight bands and low ADX
        if bb_bw <= self.bb_range_threshold:
            range_score += min(1.0, (self.bb_range_threshold - bb_bw) / self.bb_range_threshold)
        if adx_now < self.adx_trend_threshold:
            range_score += 0.3

        if trend_score >= range_score:
            conf = max(0.0, min(1.0, trend_score))
            return "TREND", conf
        conf = max(0.0, min(1.0, range_score))
        return "RANGE", conf
