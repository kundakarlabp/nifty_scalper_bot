from __future__ import annotations

from datetime import datetime

from .instruments_cache import InstrumentsCache, nearest_weekly_expiry


class OptionResolver:
    """Resolve ATM option contracts."""

    def __init__(self, cache: InstrumentsCache) -> None:
        self.cache = cache

    @staticmethod
    def step_for(under_symbol: str) -> int:
        return 50 if "BANK" not in under_symbol.upper() else 100

    def resolve_atm(self, under_symbol: str, under_ltp: float, kind: str, now_ist: datetime) -> dict:
        step = self.step_for(under_symbol)
        strike = int(round(float(under_ltp) / step) * step)
        expiry = nearest_weekly_expiry(now_ist)
        meta = self.cache.get(under_symbol, expiry, strike, kind)
        return {"under": under_symbol, "strike": strike, "expiry": expiry, **(meta or {})}
