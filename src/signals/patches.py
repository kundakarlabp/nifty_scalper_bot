"""Signal-level helper utilities and runtime patches."""

from __future__ import annotations

import math

_BAND_EPSILON = 1e-4


def _within_band_floor(value: float, bound: float) -> bool:
    """Return ``True`` when ``value`` respects the lower ``bound`` within tolerance."""

    return value >= bound or math.isclose(value, bound, abs_tol=_BAND_EPSILON)


def _within_band_ceiling(value: float, bound: float) -> bool:
    """Return ``True`` when ``value`` respects the upper ``bound`` within tolerance."""

    return value <= bound or math.isclose(value, bound, abs_tol=_BAND_EPSILON)


def check_atr_band(atr_pct: float, min_val: float, max_val: float) -> tuple[bool, str | None]:
    """Return whether ``atr_pct`` falls within the inclusive ``[min_val, max_val]`` band.

    Parameters
    ----------
    atr_pct:
        Observed ATR percentage of the underlying instrument.
    min_val:
        Lower bound for the acceptable ATR percentage.
    max_val:
        Upper bound for the acceptable ATR percentage.

    Returns
    -------
    Tuple[bool, str | None]
        ``(True, None)`` when within the configured band, otherwise ``(False, reason)``
        where ``reason`` contains a diagnostic message including the offending value
        and configured limits.
    """

    min_bound = float(min_val)
    max_bound = float(max_val)
    if max_bound <= 0:
        max_bound = float("inf")
    elif max_bound < min_bound:
        max_bound = min_bound

    if not _within_band_floor(atr_pct, min_bound):
        return False, f"atr_out_of_band: atr={atr_pct:.4f} < min={min_bound}"
    if max_bound != float("inf") and not _within_band_ceiling(atr_pct, max_bound):
        return False, f"atr_out_of_band: atr={atr_pct:.4f} > max={max_bound}"
    return True, None


__all__ = ["check_atr_band"]
