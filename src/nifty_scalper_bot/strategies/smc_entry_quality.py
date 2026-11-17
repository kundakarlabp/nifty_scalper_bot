"""Purpose: Audit SMC sweep age and option-premium extension without a veto.

Responsibilities: Provide numerical VWAP- and ATR-relative stretch flags with
explicit data completeness for downstream analysis of late liquidity sweeps.
Safety: Thresholds are research-only, not calibrated live admission gates.
"""

from __future__ import annotations

import math
from typing import Any


def _positive(value: Any) -> float | None:
    """Return a finite positive number, otherwise missing."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def assess_smc_entry_extension(
    *,
    premium: Any,
    premium_vwap: Any,
    option_atr: Any,
    sweep_age_minutes: Any,
    vwap_source: str,
    reference_fraction: float = 0.18,
    reference_atr: float = 3.0,
    reference_age_minutes: float = 3.0,
) -> dict[str, Any]:
    """Return diagnostic evidence; no result here is an order veto.

    The reference thresholds mirror existing VWAP diagnostics but are *not*
    calibrated SMC entry thresholds. A missing input remains unknown.
    """
    price = _positive(premium)
    vwap = _positive(premium_vwap)
    atr = _positive(option_atr)
    try:
        age = float(sweep_age_minutes)
    except (TypeError, ValueError, OverflowError):
        age = math.nan
    age = age if math.isfinite(age) and age >= 0 else None

    distance = (
        max(0.0, price - vwap) if price is not None and vwap is not None else None
    )
    fraction = (distance / vwap) if distance is not None and vwap else None
    atrs = (distance / atr) if distance is not None and atr else None
    extended_fraction = fraction > reference_fraction if fraction is not None else None
    extended_atr = atrs > reference_atr if atrs is not None else None
    delayed = age >= reference_age_minutes if age is not None else None
    enough_data = all(value is not None for value in (fraction, atrs, age))
    return {
        "policy": "shadow_observation_only",
        "reference_policy": "unvalidated_vwap_analogue",
        "premium_vwap_source": str(vwap_source or "unavailable"),
        "premium_above_vwap_fraction": (
            round(fraction, 5) if fraction is not None else None
        ),
        "premium_distance_atr": round(atrs, 4) if atrs is not None else None,
        "sweep_age_minutes": round(age, 4) if age is not None else None,
        "reference_fraction": reference_fraction,
        "reference_atr": reference_atr,
        "reference_age_minutes": reference_age_minutes,
        "extended_fraction": extended_fraction,
        "extended_atr": extended_atr,
        "aged_sweep": delayed,
        "combined_extension_flag": (
            bool(extended_fraction and extended_atr and delayed)
            if enough_data
            else None
        ),
        "data_complete": enough_data,
    }
