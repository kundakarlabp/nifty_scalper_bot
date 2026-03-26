"""Strict market-data tick validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

import pandas as pd

from nifty_scalper_bot.data.source import DataIntegrityError


@dataclass(frozen=True, slots=True)
class Tick:
    """Validated tick payload."""

    symbol: str
    timestamp: pd.Timestamp
    ltp: float
    volume: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation for legacy call sites."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'ltp': self.ltp,
            'volume': self.volume,
        }


def validate_tick(raw_tick: Mapping[str, Any]) -> Tick:
    """Validate tick payload without fallbacks. Args: raw_tick. Returns: Tick. Raises: DataIntegrityError."""
    symbol_raw = raw_tick.get('symbol')
    if symbol_raw is None or str(symbol_raw).strip() == '':
        raise DataIntegrityError('Missing symbol')
    timestamp_raw = raw_tick.get('timestamp')
    if timestamp_raw is None:
        raise DataIntegrityError('Missing timestamp')
    timestamp = pd.to_datetime(timestamp_raw, utc=True, errors='coerce')
    if pd.isna(timestamp):
        raise DataIntegrityError('Invalid timestamp')
    if isinstance(timestamp_raw, datetime):
        pass
    ltp_raw = raw_tick.get('ltp')
    if ltp_raw is None:
        raise DataIntegrityError('Missing ltp')
    ltp = float(ltp_raw)
    if ltp <= 0:
        raise DataIntegrityError('Invalid price')
    volume = float(raw_tick.get('volume') or 0.0)
    return Tick(symbol=str(symbol_raw), timestamp=pd.Timestamp(timestamp), ltp=ltp, volume=volume)
