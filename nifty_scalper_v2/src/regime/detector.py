"""Market regime classifier."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..indicators.engine import IndicatorSnapshot


class RegimeType(str, Enum):
    TREND    = "TREND"
    RANGE    = "RANGE"
    VOLATILE = "VOLATILE"
    CALM     = "CALM"


@dataclass(slots=True)
class MarketRegime:
    regime: RegimeType
    confidence: float
    adx: float
    vix: float
    atr_ratio: float
    direction: str | None  # "BULLISH" or "BEARISH" — only in TREND
    detected_at: datetime


class RegimeDetector:
    """Classifies market regime from indicator snapshot. Cached for 5 minutes."""

    CACHE_TTL = timedelta(minutes=5)

    def __init__(self) -> None:
        self._cache: MarketRegime | None = None
        self._atr_history: list[float] = []

    def detect(self, snapshot: "IndicatorSnapshot") -> MarketRegime:
        now = datetime.now(timezone.utc)
        if self._cache and (now - self._cache.detected_at) < self.CACHE_TTL:
            return self._cache

        adx = snapshot.adx or 0.0
        vix = snapshot.vix or 0.0
        atr = snapshot.atr_14 or 0.0

        # Track ATR for ratio
        if atr > 0:
            self._atr_history.append(atr)
            if len(self._atr_history) > 20:
                self._atr_history.pop(0)

        avg_atr = sum(self._atr_history) / len(self._atr_history) if self._atr_history else atr
        atr_ratio = (atr / avg_atr) if avg_atr > 0 else 1.0

        direction = None
        confidence = 0.7

        if vix >= 30.0:
            regime = RegimeType.VOLATILE
            confidence = min(0.95, 0.7 + (vix - 30) * 0.02)
        elif atr_ratio > 1.5:
            regime = RegimeType.VOLATILE
            confidence = min(0.9, 0.65 + atr_ratio * 0.1)
        elif adx >= 25:
            regime = RegimeType.TREND
            confidence = min(0.95, 0.65 + (adx - 25) * 0.01)
            ema9 = snapshot.ema_9
            ema21 = snapshot.ema_21
            if ema9 and ema21:
                direction = "BULLISH" if ema9 > ema21 else "BEARISH"
        elif adx < 20 and vix < 15 and atr_ratio < 0.8:
            regime = RegimeType.CALM
            confidence = 0.75
        else:
            regime = RegimeType.RANGE
            confidence = 0.70

        self._cache = MarketRegime(
            regime=regime,
            confidence=confidence,
            adx=adx,
            vix=vix,
            atr_ratio=atr_ratio,
            direction=direction,
            detected_at=now,
        )
        return self._cache
