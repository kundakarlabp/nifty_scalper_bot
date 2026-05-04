from __future__ import annotations

import math
from collections.abc import Sequence


def nearest_available_strike(spot: float, strikes: Sequence[float]) -> int:
    """Find nearest listed strike. Args: spot,strikes. Returns: strike. Raises: ValueError."""
    valid = sorted({int(float(s)) for s in strikes if float(s) > 0})
    if not valid:
        raise ValueError('No valid strikes available')
    return min(valid, key=lambda strike: (abs(strike - float(spot)), strike))


def round_to_nearest_strike(spot: float, step: int = 50) -> int:
    """Round spot to strike step. Args: spot,step. Returns: strike. Raises: ValueError."""
    if spot <= 0 or step <= 0:
        raise ValueError('spot and step must be positive')
    return int(math.floor((float(spot) / float(step)) + 0.5) * step)
