"""Deterministic token-based resolver for ATM NIFTY options.

Runtime role:
- Deprecated ATM resolver; runtime must use InstrumentManager.
- Legacy InstrumentsCache path is tests/non-live compatibility only.
- Must not resolve live contracts without InstrumentManager."""

from __future__ import annotations

from dataclasses import dataclass
import os
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

        self._instrument_manager = kite if hasattr(kite, "get_active_nifty_contracts") else None
        self._cache = None if self._instrument_manager is not None else InstrumentsCache(kite)

    def resolve_atm_pair(
        self, spot: float, *, today: date | None = None
    ) -> AtmOptionPair:
        """Resolve ATM pair. Args: spot/today. Returns: pair. Raises: ValueError."""

        if spot <= 0:
            raise ValueError("spot must be positive")
        LOGGER.warning("DEPRECATED_OPTION_RESOLVER_WRAPPER_USED", extra={"event": "DEPRECATED_OPTION_RESOLVER_WRAPPER_USED"})
        if self._instrument_manager is not None:
            basket = self._instrument_manager.get_active_nifty_contracts(float(spot), strikes_around_atm=0)
            ce_row = self._instrument_manager.lookup(basket.selected_ce)
            pe_row = self._instrument_manager.lookup(basket.selected_pe)
            if ce_row and pe_row:
                def _to_inst(row):
                    return OptionInstrument(
                        instrument_token=int(row["instrument_token"]),
                        tradingsymbol=str(row["tradingsymbol"]),
                        expiry=row["expiry"],
                        strike=float(row["strike"]),
                        option_type=str(row["instrument_type"]),
                    )
                return AtmOptionPair(ce=_to_inst(ce_row), pe=_to_inst(pe_row), expiry=basket.option_expiry, strike=float(basket.atm_strike))
        if os.getenv("EXECUTION_MODE", os.getenv("MODE", "")).strip().upper() == "LIVE":
            raise RuntimeError("OptionResolver live path requires InstrumentManager")
        if self._cache is None:
            raise RuntimeError("OptionResolver legacy cache unavailable")
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
