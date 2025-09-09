from __future__ import annotations

from datetime import datetime
from typing import Optional

from src.utils.strike_selector import resolve_weekly_atm

from .instruments_cache import InstrumentsCache, nearest_weekly_expiry

try:  # pragma: no cover - optional dependency
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore


class OptionResolver:
    """Resolve ATM option contracts.

    Parameters
    ----------
    cache:
        Preloaded instrument cache.
    kite:
        Optional `KiteConnect` instance used as a fallback when the
        instrument cache does not contain the requested option token.
    """

    def __init__(
        self, cache: InstrumentsCache, kite: Optional[KiteConnect] = None
    ) -> None:
        self.cache = cache
        self.kite = kite

    @staticmethod
    def step_for(under_symbol: str) -> int:
        return 50 if "BANK" not in under_symbol.upper() else 100

    def resolve_atm(
        self, under_symbol: str, under_ltp: float, kind: str, now_ist: datetime
    ) -> dict:
        step = self.step_for(under_symbol)
        strike = int(round(float(under_ltp) / step) * step)
        expiry = nearest_weekly_expiry(now_ist)
        meta = self.cache.get(under_symbol, expiry, strike, kind)

        if not meta and self.kite is not None:
            # Attempt a lightweight lookup via trading symbol -> token
            opt = resolve_weekly_atm(under_ltp)
            sym_info = opt.get(kind.lower()) if isinstance(opt, dict) else None
            if sym_info:
                tsym, lot = sym_info
                try:
                    q = self.kite.ltp(f"NFO:{tsym}").get(f"NFO:{tsym}", {})  # type: ignore[call-arg]
                    token = int(q.get("instrument_token", 0))
                except Exception:  # pragma: no cover - network/SDK issues
                    token = 0
                if token:
                    meta = {"token": token, "tradingsymbol": tsym, "lot_size": lot}
                    key = (under_symbol.upper(), expiry, strike, kind.upper())
                    self.cache._by_key[key] = meta  # type: ignore[attr-defined]

        return {
            "under": under_symbol,
            "strike": strike,
            "expiry": expiry,
            **(meta or {}),
        }
