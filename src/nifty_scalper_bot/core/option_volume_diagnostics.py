"""Read-only option candle provenance for diagnosing zero-volume VWAP rejections.

Responsibilities:
- Compare strategy indicator volume with the latest actual history bar.
- Record quote cumulative versus interval volume without conflating them.
- Classify missing, provisional, zero or contradictory candle evidence.

Safety boundary: This module never generates signals, synthesizes volume or
weakens the option-candle-no-trades admission gate. No broker calls.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _first_present_number(
    data: Mapping[str, Any], fields: tuple[str, ...]
) -> float | None:
    for name in fields:
        if name in data:
            return _finite_number(data[name])
    return None


def describe_option_volume_evidence(
    indicators: Mapping[str, Any],
    *,
    latest_bar: Mapping[str, Any] | None = None,
    quote: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a bounded, non-admission diagnostic of one option evaluation."""
    bar = latest_bar or {}
    quote_fields = quote or {}
    indicator_volume = _finite_number(indicators.get("volume"))
    history_volume = _finite_number(bar.get("volume")) if "volume" in bar else None
    bar_complete = bar.get("is_complete")
    bar_provisional = bar.get("is_provisional")
    ts = bar.get("timestamp")
    if isinstance(ts, datetime):
        ts = ts.isoformat()
    transition = quote_fields.get("volume_transition")
    state = transition.get("state") if isinstance(transition, Mapping) else None
    quote_delta = _first_present_number(
        quote_fields, ("volume_delta", "effective_volume_delta")
    )
    quote_cumulative = _first_present_number(
        quote_fields,
        ("volume_cumulative", "volume_traded_today", "volume_traded"),
    )

    if not bar:
        evidence_state = "history_tail_unavailable"
    elif bar_provisional is True or bar_complete is False:
        evidence_state = "unfinalized_history_tail"
    elif history_volume is None:
        evidence_state = "history_volume_missing_or_invalid"
    elif indicator_volume is not None and indicator_volume <= 0 < history_volume:
        evidence_state = "indicator_history_volume_mismatch"
    elif history_volume == 0:
        # Zero in the completed internal bar is not independent proof that
        # the exchange really reported no trades for that minute.
        evidence_state = "zero_recorded_bar_volume_unverified"
    elif history_volume > 0:
        evidence_state = "positive_recorded_bar_volume"
    else:
        evidence_state = "negative_history_volume_invalid"

    return {
        "state": evidence_state,
        "indicator_volume": indicator_volume,
        "history_bar_volume": history_volume,
        "history_bar_ts": ts,
        "history_bar_complete": bar_complete,
        "history_bar_provisional": bar_provisional,
        "history_bar_has_volume_field": "volume" in bar,
        "quote_volume_delta": quote_delta,
        "quote_volume_cumulative": quote_cumulative,
        "quote_volume_transition_state": state,
        "quote_volume_delta_untrusted": quote_fields.get("volume_delta_untrusted"),
        "quote_source": quote_fields.get("source"),
        "quote_update_version": quote_fields.get("quote_update_version"),
        "history_source": indicators.get("history_source"),
    }
