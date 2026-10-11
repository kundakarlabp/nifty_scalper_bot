"""Pricing utilities to normalize price source labels."""

from __future__ import annotations

import math
import os

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__).getChild("pricing")

CANONICAL_PRICE_SOURCES: set[str] = {"live", "mdm", "historical", "est"}


def canonical_price_source(source: object) -> str:
    """Return canonical price source label for downstream consumers.

    Args:
        source: Raw price source descriptor received from adapters.

    Returns:
        Canonical label among ``live``, ``mdm``, ``historical``, or ``est``.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered canonical_price_source",
        extra={"event": "utils_canonical_price_source_enter", "source": source},
    )
    try:
        raw = str(source or "").strip().lower()
        if not raw:
            LOGGER.debug(
                "Condition met: canonical_price_source_default",
                extra={"event": "utils_canonical_price_source_default"},
            )
            return "mdm"
        if raw in CANONICAL_PRICE_SOURCES:
            LOGGER.debug(
                "Condition met: canonical_price_source_exact",
                extra={"event": "utils_canonical_price_source_exact", "source": raw},
            )
            return raw
        if "est" in raw or "approx" in raw or "guess" in raw:
            LOGGER.debug(
                "Condition met: canonical_price_source_est",
                extra={"event": "utils_canonical_price_source_est", "source": raw},
            )
            return "est"
        if any(
            keyword in raw for keyword in ("hist", "cache", "persist", "snapshot", "yday", "prev")
        ):
            LOGGER.debug(
                "Condition met: canonical_price_source_historical",
                extra={
                    "event": "utils_canonical_price_source_historical",
                    "source": raw,
                },
            )
            return "historical"
        if "mdm" in raw:
            LOGGER.debug(
                "Condition met: canonical_price_source_mdm",
                extra={"event": "utils_canonical_price_source_mdm", "source": raw},
            )
            return "mdm"
        if raw.startswith("live") or "broker" in raw or "kite" in raw:
            LOGGER.debug(
                "Condition met: canonical_price_source_live",
                extra={"event": "utils_canonical_price_source_live", "source": raw},
            )
            return "live"
        LOGGER.debug(
            "Condition met: canonical_price_source_fallback",
            extra={"event": "utils_canonical_price_source_fallback", "source": raw},
        )
        return "mdm"
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in canonical_price_source: %s",
            exc,
            extra={"event": "utils_canonical_price_source_error", "source": source},
            exc_info=exc,
        )
        return "mdm"




def configured_stop_limit_buffer_pct() -> float:
    """One configurable limit tolerance for protective and SL-M fallback orders."""
    raw = os.getenv(
        "STOP_LIMIT_BUFFER_PCT",
        os.getenv("EXCHANGE_PROTECTIVE_STOP_LIMIT_BUFFER_PCT", "5.0"),
    )
    try:
        pct = float(raw)
    except (TypeError, ValueError):
        pct = 5.0
    return max(0.1, min(pct, 99.0)) if math.isfinite(pct) else 5.0


def canonical_stop_limit_prices(
    trigger_price: float,
    side: str,
    *,
    buffer_pct: float | None = None,
) -> tuple[float, float]:
    """Return tick-aligned (trigger, limit), direction-correct for NSE SL orders.

    A ₹0.05 SELL trigger cannot have a lower positive tick-aligned limit;
    callers must reject an equal-priced pair instead of sending it to the broker.
    """
    trigger_raw = float(trigger_price)
    side_key = str(side).strip().upper()
    if not math.isfinite(trigger_raw) or trigger_raw <= 0:
        raise ValueError("stop trigger must be positive and finite")
    if side_key not in {"BUY", "SELL"}:
        raise ValueError("stop side must be BUY or SELL")
    pct = (
        configured_stop_limit_buffer_pct()
        if buffer_pct is None
        else float(buffer_pct)
    )
    if not math.isfinite(pct) or pct < 0.0 or pct >= 100.0:
        raise ValueError("invalid stop-limit buffer percentage")
    tick = 0.05
    trigger = round(round(trigger_raw / tick) * tick, 2)
    trigger = max(tick, trigger)
    multiplier = 1 + pct / 100.0 if side_key == "BUY" else 1 - pct / 100.0
    limit = round(round(trigger * multiplier / tick) * tick, 2)
    if side_key == "BUY":
        limit = max(round(trigger + tick, 2), limit)
    else:
        limit = max(tick, min(round(trigger - tick, 2), limit))
    return trigger, limit

__all__ = [
    "CANONICAL_PRICE_SOURCES",
    "canonical_price_source",
    "canonical_stop_limit_prices",
    "configured_stop_limit_buffer_pct",
]
