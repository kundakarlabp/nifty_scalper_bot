"""Strict market-data tick validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from nifty_scalper_bot.data.source import DataIntegrityError
from nifty_scalper_bot.data.time_contract import (
    coerce_market_timestamp,
    normalize_market_tick_timestamp,
    normalized_symbol,
)


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    timestamp: pd.Timestamp
    ltp: float
    volume: float = 0.0
    timestamp_source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "ltp": self.ltp,
            "volume": self.volume,
            "timestamp_source": self.timestamp_source,
        }


def _to_ist(value: Any) -> pd.Timestamp:
    try:
        return coerce_market_timestamp(value)
    except ValueError as exc:
        raise DataIntegrityError("Invalid timestamp") from exc


def validate_tick(raw_tick: Mapping[str, Any]) -> Tick:
    try:
        symbol = normalized_symbol(raw_tick.get("symbol") or raw_tick.get("trading_symbol"))
    except ValueError as exc:
        raise DataIntegrityError("Missing symbol") from exc

    try:
        normalized_ts = normalize_market_tick_timestamp(raw_tick)
    except ValueError as exc:
        raise DataIntegrityError("Missing or invalid timestamp") from exc
    timestamp = normalized_ts.timestamp

    ltp_raw = raw_tick.get("ltp") or raw_tick.get("last_price")
    if ltp_raw is None:
        raise DataIntegrityError("Missing ltp/last_price")
    ltp = float(ltp_raw)
    if ltp <= 0:
        raise DataIntegrityError("Invalid price")

    volume_raw: Any = 0.0
    if "volume_delta" in raw_tick:
        volume_raw = raw_tick.get("volume_delta")
    elif "volume" in raw_tick:
        volume_raw = raw_tick.get("volume")
    elif "volume_traded" in raw_tick:
        volume_raw = raw_tick.get("volume_traded")
    else:
        volume_raw = 0.0

    try:
        volume = float(volume_raw if volume_raw is not None else 0.0)
    except (TypeError, ValueError):
        volume = 0.0

    return Tick(
        symbol=symbol,
        timestamp=timestamp,
        ltp=ltp,
        volume=volume,
        timestamp_source=normalized_ts.source,
    )
