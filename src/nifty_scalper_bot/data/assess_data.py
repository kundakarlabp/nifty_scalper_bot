"""
World-class market data freshness assessment.

Designed for hybrid snapshot / stream trading systems.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Iterable, Tuple


PRIMARY_ROLES = {"index", "future"}
SECONDARY_ROLES = {"option"}


def _to_epoch_seconds(ts: object) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts) if ts < 1e12 else float(ts) / 1000.0
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except Exception:
        return None


def assess_datahub_fresh(
    hub: Any,
    symbols: Iterable[str],
    *,
    symbol_roles: Dict[str, str],
    freshness_ms: int,
    grace_ms: int = 2_000,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Assess market data freshness across symbols.

    Trading is allowed if:
    - ALL primary-role symbols are fresh
    - Secondary-role symbols may be stale within grace limits
    """

    now = getattr(hub, "_now", time.time)()
    results: Dict[str, Dict[str, Any]] = {}

    primary_failures = []
    secondary_failures = []

    for symbol in symbols:
        role = symbol_roles.get(symbol, "secondary")
        quote = hub.get_quote(symbol, allow_pull=False)

        if not quote:
            failure = {
                "symbol": symbol,
                "role": role,
                "reason": "no_quote",
            }
            results[symbol] = failure
            (primary_failures if role in PRIMARY_ROLES else secondary_failures).append(failure)
            continue

        ts = (
            quote.get("timestamp")
            or quote.get("ts")
            or quote.get("ts_ms")
        )

        server_ts = _to_epoch_seconds(ts)
        if server_ts is None:
            failure = {
                "symbol": symbol,
                "role": role,
                "reason": "bad_timestamp",
                "raw_ts": ts,
            }
            results[symbol] = failure
            (primary_failures if role in PRIMARY_ROLES else secondary_failures).append(failure)
            continue

        age_ms = max(0.0, (now - server_ts) * 1000.0)
        limit = freshness_ms + (grace_ms if role in SECONDARY_ROLES else 0)

        ok = age_ms <= limit

        results[symbol] = {
            "symbol": symbol,
            "role": role,
            "age_ms": age_ms,
            "limit_ms": limit,
            "ok": ok,
        }

        if not ok:
            (primary_failures if role in PRIMARY_ROLES else secondary_failures).append(
                results[symbol]
            )

    if primary_failures:
        return (
            False,
            "primary_data_stale",
            {
                "now": now,
                "primary_failures": primary_failures,
                "secondary_failures": secondary_failures,
                "mode": getattr(hub, "mode", "unknown"),
            },
        )

    if secondary_failures:
        return (
            True,
            "secondary_data_stale",
            {
                "now": now,
                "secondary_failures": secondary_failures,
                "mode": getattr(hub, "mode", "unknown"),
            },
        )

    return (
        True,
        "ok",
        {
            "now": now,
            "symbols": list(symbols),
            "mode": getattr(hub, "mode", "unknown"),
        },
    )


__all__ = ["assess_datahub_fresh"]
