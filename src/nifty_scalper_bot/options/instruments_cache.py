"""NFO instrument dump cache for deterministic option resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from threading import RLock
import time
from typing import Any, Mapping

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class OptionInstrument:
    """Option instrument row. Args: broker row. Returns: option instrument. Raises: none."""

    instrument_token: int
    tradingsymbol: str
    name: str
    expiry: date
    strike: float
    option_type: str


class InstrumentsCache:
    """Daily-refresh in-memory cache for broker instruments."""

    def __init__(self, kite: Any) -> None:
        """Create cache for broker instruments. Args: kite. Returns: none. Raises: none."""

        self._kite = kite
        self._lock = RLock()
        self._last_loaded_day: date | None = None
        self._last_loaded_ts: float | None = None
        self._nifty_options: list[OptionInstrument] = []

    def get_nifty_options(self) -> list[OptionInstrument]:
        """Return cached NIFTY options, refreshing daily. Args: none. Returns: list. Raises: Exception."""

        today = date.today()
        with self._lock:
            if self._last_loaded_day == today and self._nifty_options:
                return list(self._nifty_options)
        rows = self._load_from_broker()
        with self._lock:
            self._nifty_options = rows
            self._last_loaded_day = today
            self._last_loaded_ts = time.time()
            return list(self._nifty_options)

    def _load_from_broker(self) -> list[OptionInstrument]:
        """Load NFO instruments from broker. Args: none. Returns: list. Raises: Exception."""

        try:
            raw_rows = self._kite.instruments("NFO")
        except Exception as e:
            LOGGER.exception("Failure in InstrumentsCache._load_from_broker: %s", e)
            raise
        results: list[OptionInstrument] = []
        for row in raw_rows:
            if not isinstance(row, Mapping):
                continue
            name = str(row.get("name") or "").strip().upper()
            if name != "NIFTY":
                continue
            instrument_type = str(row.get("instrument_type") or "").strip().upper()
            if instrument_type not in {"CE", "PE"}:
                continue
            expiry_value = row.get("expiry")
            if expiry_value is None:
                continue
            expiry = expiry_value if isinstance(expiry_value, date) else None
            if expiry is None:
                continue
            try:
                token = int(row.get("instrument_token"))
                strike = float(row.get("strike"))
            except (TypeError, ValueError):
                continue
            results.append(
                OptionInstrument(
                    instrument_token=token,
                    tradingsymbol=str(row.get("tradingsymbol") or "").strip().upper(),
                    name=name,
                    expiry=expiry,
                    strike=strike,
                    option_type=instrument_type,
                )
            )
        LOGGER.info("Loaded %d NIFTY option instruments", len(results))
        return results
