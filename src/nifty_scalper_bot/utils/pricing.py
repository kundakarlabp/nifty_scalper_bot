"""Pricing utilities to normalize price source labels."""

from __future__ import annotations

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


__all__ = ["CANONICAL_PRICE_SOURCES", "canonical_price_source"]
