"""Strict market-data tick validation — Zerodha KiteTicker compatible."""

from __future__ import annotations

import datetime as _dt
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
    """Validate and normalise a raw broker tick.

    Accepts both internal format (symbol/ltp/timestamp) and the raw
    Zerodha KiteTicker format (instrument_token/last_price/exchange_timestamp).
    The caller must inject ``symbol`` from the token→symbol map before calling
    this function if it is absent.

    Args: raw_tick. Returns: Tick. Raises: DataIntegrityError.
    """
    # ── 1. SYMBOL ────────────────────────────────────────────────────────────
    symbol_raw = raw_tick.get('symbol')
    if symbol_raw is None or str(symbol_raw).strip() == '':
        raise DataIntegrityError('Missing symbol')

    # ── 2. TIMESTAMP — prefer exchange time, then internal timestamp ─────────
    # Zerodha sends 'exchange_timestamp'. Prefer it over wall-clock/internal
    # timestamps so candle ordering is based on exchange time consistently across
    # the legacy and deterministic pipelines.
    timestamp_raw = (
        raw_tick.get('exchange_timestamp')
        or raw_tick.get('timestamp')
    )
    if timestamp_raw is None:
        # Fallback is kept for non-live/tooling ticks only; production live
        # Zerodha ticks should carry exchange_timestamp.
        timestamp_raw = _dt.datetime.now(_dt.timezone.utc)
    timestamp = pd.to_datetime(timestamp_raw, utc=True, errors='coerce')
    if pd.isna(timestamp):
        raise DataIntegrityError('Invalid timestamp')

    # ── 3. LTP — accept last_price (Zerodha field name) as alias ─────────────
    ltp_raw = raw_tick.get('ltp') or raw_tick.get('last_price')
    if ltp_raw is None:
        raise DataIntegrityError('Missing ltp/last_price')
    ltp = float(ltp_raw)
    if ltp <= 0:
        raise DataIntegrityError('Invalid price')

    # ── 4. VOLUME ────────────────────────────────────────────────────────────
    volume_raw = 0.0
    if 'volume_delta' in raw_tick:
        volume_raw = raw_tick.get('volume_delta')
    elif 'volume' in raw_tick:
        volume_raw = raw_tick.get('volume')
    elif 'volume_traded' in raw_tick:
        # Legacy/raw fallback only. This path should not be used for MDM-normalized ticks.
        volume_raw = raw_tick.get('volume_traded')
    else:
        volume_raw = 0.0

    try:
        volume = float(volume_raw if volume_raw is not None else 0.0)
    except (TypeError, ValueError):
        volume = 0.0

    return Tick(
        symbol=str(symbol_raw),
        timestamp=pd.Timestamp(timestamp),
        ltp=ltp,
        volume=volume,
    )
