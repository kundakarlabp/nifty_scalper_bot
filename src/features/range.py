from __future__ import annotations

"""Helpers for scoring range-bound market conditions."""

from typing import Any


def range_score(features: Any) -> float:
    """Return a heuristic range score in ``[0, 1]``.

    Parameters
    ----------
    features:
        Object providing ``mom_norm`` (normalized momentum, ``-1..1``) and
        ``atr_pct`` (ATR percentage). Both attributes are optional.

    The score rewards quiet markets with low momentum and ATR within the
    preferred band of 2%–20%.
    """

    mom = float(getattr(features, "mom_norm", 0.0) or 0.0)
    atrp = float(getattr(features, "atr_pct", 0.0) or 0.0)
    band_ok = 0.02 <= atrp <= 0.20
    score = (1.0 - abs(mom)) * (1.0 if band_ok else 0.5)
    return max(0.0, min(1.0, score))
