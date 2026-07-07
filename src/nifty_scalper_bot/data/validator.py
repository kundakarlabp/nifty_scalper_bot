"""Strict market-data tick validation."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from nifty_scalper_bot.data.source import DataIntegrityError

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    timestamp: pd.Timestamp
    ltp: float
    volume: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "ltp": self.ltp,
            "volume": self.volume,
        }


def validate_tick(raw_tick: Mapping[str, Any]) -> Tick:
    symbol_raw = raw_tick.get("symbol")
    if symbol_raw is None or str(symbol_raw).strip() == "":
        raise DataIntegrityError("Missing symbol")

    timestamp_raw = raw_tick.get("timestamp") or raw_tick.get("exchange_timestamp")
    if timestamp_raw is None:
        timestamp_raw = _dt.datetime.now(IST)
    timestamp = pd.to_datetime(timestamp_raw, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise DataIntegrityError("Invalid timestamp")
    timestamp = pd.Timestamp(timestamp).tz_convert(IST)

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
        symbol=str(symbol_raw),
        timestamp=timestamp,
        ltp=ltp,
        volume=volume,
    )
