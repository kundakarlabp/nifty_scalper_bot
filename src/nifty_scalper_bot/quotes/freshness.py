"""Quote freshness helpers."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

from nifty_scalper_bot.utils.errors import DataStaleError


@dataclass(frozen=True)
class Quote:
    """Typed representation of a broker quote."""

    symbol: str
    ltp: float
    ts_ms: int


class FreshnessGuard:
    """Validate that quotes are within an acceptable freshness window."""

    def __init__(self, stale_threshold_ms: int) -> None:
        if stale_threshold_ms <= 0:
            raise ValueError("stale_threshold_ms must be positive")
        self._threshold = stale_threshold_ms

    def ensure_fresh(self, quote_data: Mapping[str, Any]) -> None:
        """Raise if the provided quote dictionary is stale."""

        symbol_raw = quote_data.get("symbol")
        ts_raw = quote_data.get("ts_ms")
        if symbol_raw is None or ts_raw is None:
            raise DataStaleError("Quote data missing required fields")
        try:
            ts_ms = self._to_int(ts_raw)
        except (TypeError, ValueError) as exc:
            raise DataStaleError("Quote timestamp is invalid") from exc
        now_ms = int(time.time() * 1000)
        if now_ms - ts_ms > self._threshold:
            symbol = str(symbol_raw)
            raise DataStaleError(f"Quote for {symbol} is stale")

    @staticmethod
    def _to_int(value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            return int(value)
        raise TypeError(f"Cannot convert {value!r} to int")


__all__ = ["FreshnessGuard", "Quote"]
