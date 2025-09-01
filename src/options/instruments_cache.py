from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, Optional, Tuple

try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore


Key = Tuple[str, str, int, str]


@dataclass
class InstrumentRecord:
    token: int
    lot_size: int


class InstrumentsCache:
    """Cache of option instruments indexed by symbol/expiry/strike/kind."""

    def __init__(self, kite: Optional[KiteConnect] = None, *, instruments: Optional[Iterable[dict]] = None) -> None:
        self._by_key: Dict[Key, InstrumentRecord] = {}
        self._lot_sizes: Dict[str, int] = {}
        data = instruments
        if data is None and kite is not None:
            try:
                data = kite.instruments("NFO")  # type: ignore[attr-defined]
            except Exception:
                data = []
        for ins in data or []:
            name = str(ins.get("name", "")).upper()
            exp_raw = ins.get("expiry")
            if isinstance(exp_raw, (datetime, date)):
                expiry = exp_raw.date().isoformat() if isinstance(exp_raw, datetime) else exp_raw.isoformat()
            else:
                expiry = str(exp_raw)
            strike = int(float(ins.get("strike", 0)))
            kind = str(ins.get("instrument_type", "")).upper()
            token = int(ins.get("instrument_token", 0))
            lot = int(ins.get("lot_size", 0))
            if name and expiry and strike and kind and token:
                key = (name, expiry, strike, kind)
                self._by_key[key] = InstrumentRecord(token=token, lot_size=lot)
                self._lot_sizes[name] = lot

    def token(self, symbol: str, expiry: str, strike: int, kind: str) -> Optional[int]:
        key = (symbol.upper(), expiry, int(strike), kind.upper())
        rec = self._by_key.get(key)
        return rec.token if rec else None

    def lot_size(self, symbol: str) -> Optional[int]:
        return self._lot_sizes.get(symbol.upper())


def nearest_weekly_expiry(now_ist: datetime) -> str:
    """Return the nearest weekly expiry date (Thursday) as YYYY-MM-DD."""
    d = now_ist.date()
    while d.weekday() != 3:  # Thursday
        d += timedelta(days=1)
    return d.isoformat()
