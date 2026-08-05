"""Authoritative quote-version and evaluation-snapshot identity helpers."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
from typing import Any, Mapping

_QUOTE_VERSION_KEYS = (
    "quote_update_version",
    "update_version",
    "tick_version",
)


def coerce_quote_update_version(value: Any) -> int | None:
    """Return an exact positive integer quote version, otherwise ``None``."""
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        return None
    version = int(parsed)
    return version if version > 0 else None


def resolve_quote_update_identity(
    *sources: tuple[str, Mapping[str, Any] | None],
) -> tuple[int | None, str | None]:
    """Resolve the first authoritative quote version and its provenance."""
    for source_name, payload in sources:
        if not isinstance(payload, Mapping):
            continue
        for key in _QUOTE_VERSION_KEYS:
            version = coerce_quote_update_version(payload.get(key))
            if version is not None:
                return version, f"{source_name}:{key}"
    return None, None


def build_evaluation_snapshot_id(
    setup_signal_id: str, quote_update_version: int | None
) -> str | None:
    """Build exact-evaluation identity without changing setup idempotency."""
    version = coerce_quote_update_version(quote_update_version)
    setup_id = str(setup_signal_id or "").strip()
    if not setup_id or version is None:
        return None
    raw = f"{setup_id}:quote_update_version:{version}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


__all__ = [
    "build_evaluation_snapshot_id",
    "coerce_quote_update_version",
    "resolve_quote_update_identity",
]
