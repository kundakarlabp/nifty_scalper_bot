"""Polling fallback decision helpers.

The poller is a recovery path, not a parallel quote authority.  A stale flag from
feed-health diagnostics must therefore be cross-checked against the actual age of
the selected option quotes before fallback is activated while WebSocket is still
healthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class PollingFallbackDecision:
    """Structured fallback decision for runtime logging and tests."""

    activate: bool
    reason: str | None
    ws_ok: bool
    lagging: bool
    futures_fresh: bool
    options_fresh: bool
    max_age_ms: float | None
    threshold_ms: float
    futures_age_ms: float | None = None
    selected_ce_age_ms: float | None = None
    selected_pe_age_ms: float | None = None

    def as_log_extra(self) -> dict[str, Any]:
        """Return stable structured fields for operator logs."""

        return {
            "event": "POLLING_FALLBACK_DECISION",
            "activate": self.activate,
            "reason": self.reason,
            "ws_ok": self.ws_ok,
            "lagging": self.lagging,
            "futures_fresh": self.futures_fresh,
            "options_fresh": self.options_fresh,
            "max_age_ms": self.max_age_ms,
            "threshold_ms": self.threshold_ms,
            "futures_age_ms": self.futures_age_ms,
            "selected_ce_age_ms": self.selected_ce_age_ms,
            "selected_pe_age_ms": self.selected_pe_age_ms,
        }


def _coerce_age_ms(value: Any) -> float | None:
    """Return a finite non-negative age in milliseconds, or None."""

    try:
        age = float(value)
    except (TypeError, ValueError):
        return None
    if age != age or age < 0:  # NaN or negative
        return None
    return age


def _first_age_ms(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        age = _coerce_age_ms(payload.get(key))
        if age is not None:
            return age
    return None


def decide_polling_fallback(
    *,
    ws_ok: bool,
    lagging: bool,
    futures_fresh: bool,
    options_fresh: bool,
    quote_stale_ms: float,
    feed_health: Mapping[str, Any] | None = None,
    data_age_ms: float | None = None,
) -> PollingFallbackDecision:
    """Decide whether REST polling fallback should activate.

    Args:
        ws_ok: Whether the WebSocket transport is connected/healthy.
        lagging: Whether the event loop or tick path is lagging.
        futures_fresh: Freshness flag for futures context.
        options_fresh: Freshness flag for selected CE/PE option quotes.
        quote_stale_ms: Age threshold in milliseconds.
        feed_health: Optional detailed age payload from MarketDataManager.
        data_age_ms: Generic fallback age when role-specific ages are absent.

    Returns:
        PollingFallbackDecision with an explicit activation reason.

    Safety rule:
        If WebSocket is healthy and not lagging, a false ``options_fresh`` flag
        alone is not enough. At least one selected option age must be known and
        cross the threshold. This prevents activations like
        ``age_ms=70 threshold_ms=120000``.
    """

    threshold = max(0.0, float(quote_stale_ms or 0.0))
    health: Mapping[str, Any] = feed_health or {}
    generic_age = _coerce_age_ms(data_age_ms)
    selected_ce_age = _first_age_ms(
        health,
        "selected_ce_age_ms",
        "ce_age_ms",
        "atm_ce_age_ms",
    )
    selected_pe_age = _first_age_ms(
        health,
        "selected_pe_age_ms",
        "pe_age_ms",
        "atm_pe_age_ms",
    )
    option_age = _first_age_ms(
        health,
        "selected_option_age_ms",
        "options_age_ms",
        "option_age_ms",
    )
    futures_age = _first_age_ms(
        health,
        "selected_futures_age_ms",
        "futures_age_ms",
        "future_age_ms",
    )

    selected_ages = [age for age in (selected_ce_age, selected_pe_age, option_age) if age is not None]
    max_option_age = max(selected_ages) if selected_ages else None
    max_age = max_option_age if max_option_age is not None else generic_age

    reason: str | None = None
    activate = False
    if not ws_ok:
        activate = True
        reason = "websocket_unhealthy"
    elif lagging:
        activate = True
        reason = "event_loop_lagging"
    elif not futures_fresh and futures_age is not None and futures_age >= threshold:
        activate = True
        reason = "futures_stale"
    elif not futures_fresh and futures_age is None and generic_age is not None and generic_age >= threshold:
        activate = True
        reason = "futures_stale"
    elif not options_fresh and max_option_age is not None and max_option_age >= threshold:
        activate = True
        reason = "options_stale"
    elif not options_fresh and max_option_age is None and generic_age is not None and generic_age >= threshold:
        activate = True
        reason = "options_stale"

    return PollingFallbackDecision(
        activate=bool(activate),
        reason=reason,
        ws_ok=bool(ws_ok),
        lagging=bool(lagging),
        futures_fresh=bool(futures_fresh),
        options_fresh=bool(options_fresh),
        max_age_ms=max_age,
        threshold_ms=threshold,
        futures_age_ms=futures_age,
        selected_ce_age_ms=selected_ce_age,
        selected_pe_age_ms=selected_pe_age,
    )


__all__ = ["PollingFallbackDecision", "decide_polling_fallback"]
