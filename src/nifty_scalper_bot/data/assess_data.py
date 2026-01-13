from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Iterable, Tuple, Union

PRIMARY_ROLES = {"primary"}
SECONDARY_ROLES = {"secondary"}


def _to_epoch_seconds(ts: Any) -> Union[float, None]:
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
    symbols: Union[str, Iterable[str]],
    *,
    symbol_roles: Dict[str, str] | None = None,
    freshness_ms: int | None = None,
    grace_ms: int = 2_000,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Assess market data freshness.

    Backward compatible:
    - Old mode: assess_datahub_fresh(hub, symbol, freshness_ms)
    - New mode: role-aware multi-symbol assessment
    """

    # -------------------------------
    # Backward compatibility layer
    # -------------------------------
    if isinstance(symbols, str):
        if freshness_ms is None:
            raise TypeError("freshness_ms is required for single-symbol mode")

        quote = hub.get_quote(symbols, allow_pull=False)
        if not quote:
            return False, "no_quote", {"symbol": symbols}

        ts = quote.get("timestamp") or quote.get("ts") or quote.get("ts_ms")
        server_ts = _to_epoch_seconds(ts)
        if server_ts is None:
            return False, "bad_timestamp", {"symbol": symbols, "raw_ts": ts}

        now = getattr(hub, "_now", time.time)()
        age_ms = max(0.0, (now - server_ts) * 1000.0)

        return (
            age_ms <= freshness_ms,
            "ok" if age_ms <= freshness_ms else "stale",
            {
                "symbol": symbols,
                "age_ms": age_ms,
                "limit_ms": freshness_ms,
                "mode": getattr(hub, "mode", "unknown"),
            },
        )

    # -------------------------------
    # New multi-symbol role-aware path
    # -------------------------------
    if not symbols:
        raise ValueError("symbols cannot be empty")

    if not symbol_roles:
        raise ValueError("symbol_roles must be provided for multi-symbol mode")

    if freshness_ms is None:
        raise ValueError("freshness_ms must be provided")

    now = getattr(hub, "_now", time.time)()
    get_quote = hub.get_quote

    results: Dict[str, Dict[str, Any]] = {}
    primary_failures = []
    secondary_failures = []

    for symbol in symbols:
        role = symbol_roles.get(symbol, "secondary")
        quote = get_quote(symbol, allow_pull=False)

        if not quote:
            failure = {"symbol": symbol, "role": role, "reason": "no_quote"}
            results[symbol] = failure
            (primary_failures if role in PRIMARY_ROLES else secondary_failures).append(failure)
            continue

        ts = quote.get("timestamp") or quote.get("ts") or quote.get("ts_ms")
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

        entry = {
            "symbol": symbol,
            "role": role,
            "age_ms": age_ms,
            "limit_ms": limit,
            "ok": ok,
        }
        results[symbol] = entry

        if not ok:
            (primary_failures if role in PRIMARY_ROLES else secondary_failures).append(entry)

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
