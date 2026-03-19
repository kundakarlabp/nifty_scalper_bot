"""Indicator score utilities used by deterministic strategies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeightedSignalInputs:
    """Indicator inputs for weighted scoring. Args: indicator values. Returns: inputs. Raises: None."""

    ema: float
    rsi: float
    macd: float
    vwap: float
    adx: float


def weighted_score(inputs: WeightedSignalInputs) -> float:
    """Compute configured weighted score. Args: inputs. Returns: score. Raises: None."""
    return (
        1.2 * inputs.ema
        + 1.0 * inputs.rsi
        + 0.8 * inputs.macd
        + 1.5 * inputs.vwap
        + 1.3 * inputs.adx
    )
