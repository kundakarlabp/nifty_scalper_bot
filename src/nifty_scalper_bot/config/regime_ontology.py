"""File purpose:
    Own the single market-regime vocabulary used by detectors, gates and scoring.

Key responsibilities:
    - Define the canonical MarketRegime states.
    - Normalise every producer, config and persisted label onto that set.
    - Keep regime orthogonal to direction.

Operational constraints:
    - Unrecognised labels resolve to UNKNOWN, which is not a tradable state.
    - Direction stays with the underlying-direction observation; TREND_UP and
      TREND_DOWN collapse to TREND.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["MarketRegime", "normalize_regime", "regime_label"]


class MarketRegime(str, Enum):
    """The complete set of market states any component may act on."""

    TREND = "TREND"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"
    EVENT = "EVENT"
    LOW_ACTIVITY = "LOW_ACTIVITY"
    UNKNOWN = "UNKNOWN"


# Historical spellings accepted from configuration, persisted state and
# upstream producers. Directional trend labels collapse to TREND because
# direction is carried separately; CHOPPY collapses to RANGE because no
# producer ever emitted a distinct chop state.
_ALIASES: dict[str, MarketRegime] = {
    "TREND": MarketRegime.TREND,
    "TRENDING": MarketRegime.TREND,
    "TREND_UP": MarketRegime.TREND,
    "TREND_DOWN": MarketRegime.TREND,
    "UPTREND": MarketRegime.TREND,
    "DOWNTREND": MarketRegime.TREND,
    "RANGE": MarketRegime.RANGE,
    "RANGING": MarketRegime.RANGE,
    "RANGE_BOUND": MarketRegime.RANGE,
    "RANGEBOUND": MarketRegime.RANGE,
    "CHOPPY": MarketRegime.RANGE,
    "CHOP": MarketRegime.RANGE,
    "NORMAL": MarketRegime.RANGE,
    "VOLATILE": MarketRegime.VOLATILE,
    "HIGH_VOLATILITY": MarketRegime.VOLATILE,
    "HIGH_VOL": MarketRegime.VOLATILE,
    "HIGHVOL": MarketRegime.VOLATILE,
    "EVENT": MarketRegime.EVENT,
    "NEWS": MarketRegime.EVENT,
    "LOW_ACTIVITY": MarketRegime.LOW_ACTIVITY,
    "LOW_VOLATILITY": MarketRegime.LOW_ACTIVITY,
    "LOW_VOL": MarketRegime.LOW_ACTIVITY,
    "LOWVOL": MarketRegime.LOW_ACTIVITY,
    "QUIET": MarketRegime.LOW_ACTIVITY,
    "UNKNOWN": MarketRegime.UNKNOWN,
}


def normalize_regime(value: object) -> MarketRegime:
    """Return the canonical regime for any producer, config or persisted label.

    Args:
        value: Enum member, string, or object exposing ``value``/``regime``.

    Returns:
        The matching :class:`MarketRegime`, or ``UNKNOWN`` when the label is
        missing or unrecognised. ``UNKNOWN`` is deliberately not a tradable
        state, so an unrecognised label fails closed at the gates rather than
        silently scoring as neutral.

    Raises:
        None.
    """
    if isinstance(value, MarketRegime):
        return value
    candidate: object = value
    if candidate is not None and not isinstance(candidate, str):
        for attribute in ("value", "regime"):
            inner = getattr(candidate, attribute, None)
            if inner is not None and inner is not candidate:
                candidate = inner
                break
    if isinstance(candidate, MarketRegime):
        return candidate
    if candidate is None:
        return MarketRegime.UNKNOWN
    token = str(candidate).strip().upper().replace("-", "_").replace(" ", "_")
    if not token:
        return MarketRegime.UNKNOWN
    return _ALIASES.get(token, MarketRegime.UNKNOWN)


def regime_label(value: object) -> str:
    """Return the canonical regime name as a plain string for logs and metadata."""
    return normalize_regime(value).value
