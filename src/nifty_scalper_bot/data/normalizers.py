"""Shared market-data normalization helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Mapping


def normalize_history_row(symbol: str, row: Any, source: str = "historical") -> dict[str, Any] | None:
    """Normalize one OHLCV row. Args: symbol,row,source. Returns: canonical bar/None. Raises: none."""
    try:
        if isinstance(row, Mapping):
            ts = row.get("timestamp") or row.get("date") or row.get("time")
            open_ = row.get("open")
            high = row.get("high")
            low = row.get("low")
            close = row.get("close")
            volume = row.get("volume", 0)
        elif isinstance(row, (list, tuple)) and len(row) >= 5:
            ts, open_, high, low, close = row[:5]
            volume = row[5] if len(row) > 5 else 0
        else:
            return None
        if ts is None:
            return None
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif not isinstance(ts, datetime):
            ts = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        ts = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
        o, h, l, c = float(open_), float(high), float(low), float(close)
        if min(o, h, l, c) <= 0:
            return None
        symbol_upper = str(symbol or "").upper()
        is_option_symbol = symbol_upper.endswith("CE") or symbol_upper.endswith("PE")
        try:
            volume_value = int(float(volume or 0))
        except (TypeError, ValueError):
            volume_value = 0
        if is_option_symbol and volume_value > int(os.getenv("OPTION_MAX_REASONABLE_1M_VOLUME", "5000000")):
            volume_value = 0
        return {
            "symbol": str(symbol),
            "timestamp": ts,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": volume_value,
            "source": source,
        }
    except Exception:
        return None
