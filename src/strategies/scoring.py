"""Scoring helpers for diagnostic breakdowns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class ScoreInfo:
    """Holds the total score and individual feature contributions."""

    total: float
    items: Dict[str, float]


def compute_score(
    weights: Mapping[str, float] | None,
    features: Mapping[str, float] | None,
) -> ScoreInfo:
    """Multiply ``weights`` by ``features`` and return totals.

    Parameters
    ----------
    weights:
        Mapping of feature name to weight.
    features:
        Mapping of feature name to value.

    Returns
    -------
    ScoreInfo
        Total score and per-feature contributions, both rounded to 6 decimals.
    """

    items: Dict[str, float] = {}
    w_map = weights or {}
    f_map = features or {}
    for key, weight in w_map.items():
        value = float(f_map.get(key, 0.0))
        items[key] = round(float(weight) * value, 6)
    total = round(sum(items.values()), 6)
    return ScoreInfo(total=total, items=items)
