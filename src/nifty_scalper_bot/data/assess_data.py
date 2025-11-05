"""Data layer assessment helpers."""

from __future__ import annotations

from datetime import datetime
import time
from typing import Any, Dict, Tuple


def assess_datahub_fresh(
    hub: Any, symbol: str, adaptive_ms: int
) -> Tuple[bool, str, Dict[str, Any]]:
    """Verify the latest quote for *symbol* is within the freshness window."""

    quote = hub.get_quote(symbol, allow_pull=False)
    if not quote:
        return False, "no_quote", {"symbol": symbol}
    now = getattr(hub, "_now", time.time)()
    ts = quote.get("timestamp") or quote.get("ts") or quote.get("ts_ms")
    if ts is None:
        return False, "no_timestamp", {"symbol": symbol}
    if isinstance(ts, (int, float)):
        server_seconds = float(ts) if ts < 1e12 else float(ts) / 1000.0
    else:
        try:
            server_seconds = datetime.fromisoformat(str(ts)).timestamp()
        except Exception:  # noqa: BLE001
            return False, "bad_timestamp", {"symbol": symbol, "ts": ts}
    age_ms = max(0.0, (now - server_seconds) * 1000.0)
    ok = age_ms <= float(adaptive_ms)
    detail = "ok" if ok else "stale"
    return ok, detail, {"symbol": symbol, "age_ms": age_ms, "limit_ms": adaptive_ms}


__all__ = ["assess_datahub_fresh"]
