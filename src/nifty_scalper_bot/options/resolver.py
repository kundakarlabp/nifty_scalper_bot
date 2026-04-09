"""Deterministic token-based resolver for ATM NIFTY options."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from nifty_scalper_bot.options.instruments_cache import (
    InstrumentsCache,
    OptionInstrument,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AtmOptionPair:
    """ATM call/put pair. Args: CE/PE. Returns: pair. Raises: none."""

    ce: OptionInstrument
    pe: OptionInstrument
    expiry: date
    strike: float


class OptionResolver:
    """Resolve ATM CE/PE options from cached broker dump."""

    def __init__(self, kite: Any) -> None:
        """Create resolver from kite client. Args: kite. Returns: none. Raises: none."""

        self._cache = InstrumentsCache(kite)

    def resolve_atm_pair(
        self, spot: float, *, today: date | None = None
    ) -> AtmOptionPair:
        """Resolve ATM pair. Args: spot/today. Returns: pair. Raises: ValueError."""

        if spot <= 0:
            raise ValueError("spot must be positive")
        options = self._cache.get_nifty_options()
        if not options:
            raise ValueError("No NIFTY options available from broker dump")
        trade_day = today or date.today()
        expiries = sorted({row.expiry for row in options if row.expiry >= trade_day})
        if not expiries:
            raise ValueError("No valid upcoming expiries for NIFTY options")
        expiry = expiries[0]
        atm = float(round(spot / 50.0) * 50)
        filtered = [
            row for row in options if row.expiry == expiry and row.strike == atm
        ]
        ce = next((row for row in filtered if row.option_type == "CE"), None)
        pe = next((row for row in filtered if row.option_type == "PE"), None)
        if ce is None or pe is None:
            raise ValueError(
                f"Missing exact ATM CE/PE for expiry={expiry.isoformat()} strike={atm:.0f}"
            )
        return AtmOptionPair(ce=ce, pe=pe, expiry=expiry, strike=atm)
