"""Market regime classification helpers for strategy gating."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class MarketRegime(str, Enum):
    """Supported market regime labels."""

    TREND = 'TREND'
    RANGE = 'RANGE'
    VOLATILE = 'VOLATILE'
    LOW_ACTIVITY = 'LOW_ACTIVITY'


@dataclass(slots=True)
class RegimeSnapshot:
    """Container for calculated regime inputs and output."""

    regime: MarketRegime
    adx: float
    atr: float
    atr_average: float
    vwap_slope: float
    volume_expansion: float


class MarketRegimeEngine:
    """Classify market regime from indicator inputs."""

    def classify(self, indicators: Mapping[str, float | int | None]) -> RegimeSnapshot:
        """Classify regime. Args: indicators; Returns: RegimeSnapshot; Raises: none."""

        adx = self._as_float(indicators.get('adx'))
        atr = self._as_float(indicators.get('atr'))
        atr_average = max(self._as_float(indicators.get('atr_average')), 1e-6)
        vwap_slope = self._as_float(indicators.get('vwap_slope'))
        volume_expansion = self._as_float(indicators.get('volume_expansion'))

        regime = MarketRegime.LOW_ACTIVITY
        if adx > 25.0 and abs(vwap_slope) > 0.01:
            regime = MarketRegime.TREND
        elif atr > 1.8 * atr_average:
            regime = MarketRegime.VOLATILE
        elif adx < 15.0:
            regime = MarketRegime.RANGE
        elif volume_expansion < 0.7:
            regime = MarketRegime.LOW_ACTIVITY

        return RegimeSnapshot(
            regime=regime,
            adx=adx,
            atr=atr,
            atr_average=atr_average,
            vwap_slope=vwap_slope,
            volume_expansion=volume_expansion,
        )

    @staticmethod
    def _as_float(value: float | int | None) -> float:
        """Convert numeric values to float. Args: value; Returns: float; Raises: none."""

        if isinstance(value, (int, float)):
            return float(value)
        return 0.0
