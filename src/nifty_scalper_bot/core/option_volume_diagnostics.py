"""Read-only option candle provenance for diagnosing zero-volume VWAP rejections.

Responsibilities:
- Compare strategy indicator volume with the latest actual history bar.
- Record quote cumulative versus interval volume without conflating them.
- Classify missing, provisional, zero or contradictory candle evidence.
- Normalize aware candle timestamps to UTC minutes with _minute_key.
- Compare describe_option_volume_evidence's last completed indicator bar
  against the same-minute canonical CandleEngine bar without zero-filling,
  using _bar_summary. A mismatched minute is not a volume mismatch.

Safety boundary: This module never generates signals, synthesizes volume or
weakens the option-candle-no-trades admission gate. No broker calls.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
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



def _minute_key(value: Any) -> str | None:
    """Align aware candle timestamps to their UTC minute without guessing TZ."""
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str):
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        return None
    return stamp.astimezone(timezone.utc).replace(
        second=0, microsecond=0
    ).isoformat()


def _bar_summary(bar: Mapping[str, Any] | None) -> tuple[str | None, float | None]:
    """Read a finalized bar from one named source, without zero-filling."""
    row = bar or {}
    stamp = row.get("timestamp") or row.get("start")
    return _minute_key(stamp), _finite_number(row.get("volume"))


def describe_option_volume_evidence(
    indicators: Mapping[str, Any],
    *,
    latest_bar: Mapping[str, Any] | None = None,
    quote: Mapping[str, Any] | None = None,
    indicator_completed_bar: Mapping[str, Any] | None = None,
    canonical_completed_bar: Mapping[str, Any] | None = None,
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

    indicator_ts, indicator_completed_volume = _bar_summary(
        indicator_completed_bar
    )
    canonical_ts, canonical_completed_volume = _bar_summary(
        canonical_completed_bar
    )
    if indicator_ts is None or canonical_ts is None:
        parity = "missing_or_invalid_completed_timestamp"
    elif indicator_ts != canonical_ts:
        parity = "different_completed_minutes"
    elif indicator_completed_volume is None or canonical_completed_volume is None:
        parity = "missing_or_invalid_completed_volume"
    elif abs(indicator_completed_volume - canonical_completed_volume) > 0.000001:
        parity = "same_minute_volume_mismatch"
    else:
        parity = "same_minute_volume_match"

    return {
        "state": evidence_state,
        "completed_bar_parity": parity,
        "indicator_completed_ts": indicator_ts,
        "indicator_completed_volume": indicator_completed_volume,
        "canonical_completed_ts": canonical_ts,
        "canonical_completed_volume": canonical_completed_volume,
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
