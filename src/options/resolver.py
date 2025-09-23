from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from src.options.contract_registry import InstrumentRegistry
from src.utils.expiry import next_tuesday_expiry
from src.utils.strike_selector import (
    resolve_weekly_atm,
    _fetch_instruments_nfo,
    _infer_step,
    _nearest_strike,
)

from .instruments_cache import InstrumentsCache

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
        self,
        cache: InstrumentsCache,
        kite: Optional[KiteConnect] = None,
        registry: InstrumentRegistry | None = None,
    ) -> None:
        self.cache = cache
        self.kite = kite
        self.registry = registry

    def resolve_atm(
        self, under_symbol: str, under_ltp: float, kind: str, now_ist: datetime
    ) -> dict:
        step = _infer_step(under_symbol)
        strike = _nearest_strike(under_ltp, step)
        expiry_dt = None
        if self.registry is not None:
            expiry_dt = self.registry.next_weekly_expiry(under_symbol, now_ist)
        if expiry_dt is None:
            expiry_dt = next_tuesday_expiry(now_ist)
        expiry = expiry_dt.date().isoformat()

        meta: Optional[Dict[str, object]] = None
        if self.registry is not None:
            reg_meta = self.registry.lookup(under_symbol, expiry, strike, kind)
            if reg_meta:
                meta = {
                    "token": reg_meta.get("token"),
                    "tradingsymbol": reg_meta.get("tradingsymbol"),
                    "lot_size": reg_meta.get("lot_size"),
                }
                if meta.get("token") and meta.get("tradingsymbol"):
                    key = (under_symbol.upper(), expiry, int(strike), kind.upper())
                    self.cache._by_key[key] = dict(meta)
        if meta is None:
            meta = self.cache.get(under_symbol, expiry, strike, kind)

        if self.registry is not None:
            lot_lookup = self.registry.lot_size(under_symbol, expiry)
            if lot_lookup:
                if meta is None:
                    meta = {"lot_size": lot_lookup}
                else:
                    meta.setdefault("lot_size", lot_lookup)

        if not meta and self.kite is not None:
            # Attempt a lightweight lookup via trading symbol -> token
            inst_dump = _fetch_instruments_nfo(self.kite) or []
            opt = resolve_weekly_atm(under_ltp, inst_dump)
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
