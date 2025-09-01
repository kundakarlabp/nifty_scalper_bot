from __future__ import annotations

from datetime import datetime
from typing import Optional

from .instruments_cache import InstrumentsCache, nearest_weekly_expiry


class OptionResolver:
    """Resolve ATM option contracts."""

    def __init__(self, instruments: InstrumentsCache) -> None:
        self.inst = instruments

    @staticmethod
    def step_for(under_symbol: str) -> int:
        return 100 if "BANK" in under_symbol.upper() else 50

    def resolve_atm(self, under_symbol: str, under_ltp: float, kind: str, now_ist: datetime) -> dict:
        step = self.step_for(under_symbol)
        strike = int(round(under_ltp / step) * step)
        expiry = nearest_weekly_expiry(now_ist)
        tok = self.inst.token(under_symbol, expiry, strike, kind)
        return {"token": tok, "strike": strike, "expiry": expiry, "kind": kind}
