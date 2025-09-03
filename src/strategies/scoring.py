from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ScoreInfo:
    """Total score and per-feature contributions."""

    total: float
    items: Dict[str, float]


def compute_score(weights: Dict[str, float] | None, features: Dict[str, float] | None) -> ScoreInfo:
    """Multiply ``weights`` by ``features`` and return totals.

    Both inputs are optional; missing entries are treated as ``0``.
    ``total`` and each item are rounded to 6 decimal places to avoid
    floating-point noise.
    """

    items: Dict[str, float] = {}
    for k, w in (weights or {}).items():
        v = float((features or {}).get(k, 0.0))
        items[k] = round(w * v, 6)
    total = round(sum(items.values()), 6)
    return ScoreInfo(total=total, items=items)
