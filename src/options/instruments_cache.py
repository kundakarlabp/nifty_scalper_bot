from __future__ import annotations

from datetime import date, datetime
from typing import Dict, Iterable, Optional, Tuple

from src.utils.expiry import next_tuesday_expiry

try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore

Key = Tuple[str, str, int, str]


class InstrumentsCache:
    """Cache of option instruments indexed by symbol/expiry/strike/kind."""

    def __init__(
        self,
        kite: Optional[KiteConnect] = None,
        *,
        instruments: Optional[Iterable[dict]] = None,
    ) -> None:
        self._kite = kite
        self._by_key: Dict[Key, Dict[str, object]] = {}
        data = instruments
        if data is None and kite is not None:
            try:
                data = kite.instruments("NFO")  # type: ignore[attr-defined]
            except Exception:
                data = []
        self._populate(data)

    def _populate(self, data: Optional[Iterable[dict]]) -> None:
        for ins in data or []:
            name = str(ins.get("name", "")).upper()
            exp_raw = ins.get("expiry")
            if isinstance(exp_raw, (datetime, date)):
                expiry = (
                    exp_raw.date().isoformat()
                    if isinstance(exp_raw, datetime)
                    else exp_raw.isoformat()
                )
            else:
                expiry = str(exp_raw)
            strike = int(float(ins.get("strike", 0)))
            kind = str(ins.get("instrument_type", "")).upper()
            token = int(ins.get("instrument_token", 0))
            tsym = str(ins.get("tradingsymbol", ""))
            lot = int(ins.get("lot_size", 0))
            if name and expiry and strike and kind and token:
                key = (name, expiry, strike, kind)
                self._by_key[key] = {
                    "token": token,
                    "tradingsymbol": tsym,
                    "lot_size": lot,
                }

    def get(self, symbol: str, expiry: str, strike: int, kind: str) -> Optional[Dict[str, object]]:
        key = (symbol.upper(), expiry, int(strike), kind.upper())
        meta = self._by_key.get(key)
        if meta is not None or self._kite is None:
            return meta
        try:
            data = self._kite.instruments("NFO")  # type: ignore[attr-defined]
        except Exception:
            return None
        self._by_key.clear()
        self._populate(data)
        return self._by_key.get(key)


def nearest_weekly_expiry(now_ist: datetime) -> str:
    """Return the next Tuesday expiry as ``YYYY-MM-DD``.

    Rolls over to the following week if ``now_ist`` is past Tuesday 15:30 IST.
    """
    return next_tuesday_expiry(now_ist).date().isoformat()
