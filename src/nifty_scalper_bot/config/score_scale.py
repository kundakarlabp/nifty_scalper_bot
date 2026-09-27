"""Canonical 0..10 score configuration boundary.

New score settings use an explicit 0..10 scale. Historical confidence settings
remain supported only through the named legacy converter so ambiguous 0..1 /
0..10 / 0..100 interpretation cannot spread into new configuration.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

SCORE_MIN = 0.0
SCORE_MAX = 10.0


def canonical_score(value: Any, *, field: str) -> float:
    """Parse a strictly canonical 0..10 score or raise ValueError."""

    try:
        score = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a numeric 0..10 score") from exc
    if not math.isfinite(score) or score < SCORE_MIN or score > SCORE_MAX:
        raise ValueError(f"{field} must be within 0..10")
    return round(score, 3)


def legacy_confidence_score(value: Any, *, field: str) -> float:
    """Convert one historical confidence value to the canonical 0..10 scale."""

    token = str(value or "").strip()
    if not token:
        raise ValueError(f"{field} must be non-empty")
    percent = token.endswith("%")
    if percent:
        token = token[:-1].strip()
    try:
        raw = float(token)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(raw) or raw < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    if percent:
        raw /= 10.0
    elif raw <= 1.0:
        raw *= 10.0
    elif raw > 10.0:
        if raw > 100.0:
            raise ValueError(f"{field} must not exceed 100%")
        raw /= 10.0
    return canonical_score(raw, field=field)


def resolve_score_setting(
    environment: Mapping[str, str],
    *,
    canonical_key: str,
    legacy_keys: tuple[str, ...] = (),
    default: float,
) -> float:
    """Resolve one score with canonical-key precedence and legacy compatibility."""

    if canonical_key in environment:
        return canonical_score(environment[canonical_key], field=canonical_key)
    for key in legacy_keys:
        if key in environment:
            return legacy_confidence_score(environment[key], field=key)
    return canonical_score(default, field=f"default:{canonical_key}")


__all__ = [
    "SCORE_MAX",
    "SCORE_MIN",
    "canonical_score",
    "legacy_confidence_score",
    "resolve_score_setting",
]
