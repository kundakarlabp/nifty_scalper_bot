"""Helpers for instrument-driven symbol universe selection.

Runtime role:
- Deprecated universe builder wrapper.
- Runtime delegates to InstrumentManager ActiveContractBasket.
- Legacy independent universe is non-live/env-gated only."""

from __future__ import annotations

from datetime import date, datetime
import os
from typing import Any


def _coerce_expiry(value: Any) -> date | None:
    """Coerce expiry field to date; Args: value. Returns: date|None. Raises: None."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def build_nifty_universe(kite: Any, spot_price: float) -> list[dict[str, int | str]]:
    """Build legacy NIFTY option universe; disabled in live runtime."""
    im_method = getattr(kite, "get_active_nifty_contracts", None)
    if callable(im_method):
        basket = im_method(float(spot_price))
        return [
            {"symbol": sym, "token": int(tok)}
            for sym, tok in zip(basket.option_symbols, basket.option_tokens)
        ]
    live_mode = os.getenv("EXECUTION_MODE", os.getenv("MODE", "")).strip().upper() == "LIVE"
    legacy_enabled = os.getenv("OPTION_UNIVERSE_LEGACY_FALLBACK_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    if live_mode or not legacy_enabled:
        raise RuntimeError("UniverseBuilder legacy broker instruments fallback disabled; use InstrumentManager ActiveContractBasket")
    instruments = kite.instruments('NFO')
    atm = round(float(spot_price) / 50.0) * 50
    expiries = [
        expiry
        for inst in instruments
        if str(inst.get('name', '')).upper() == 'NIFTY'
        if (expiry := _coerce_expiry(inst.get('expiry'))) is not None
    ]
    if not expiries:
        raise ValueError('NIFTY expiries not found in NFO instrument dump')
    nearest_expiry = min(expiries)
    selected: list[dict[str, int | str]] = []
    for inst in instruments:
        if str(inst.get('name', '')).upper() != 'NIFTY':
            continue
        expiry = _coerce_expiry(inst.get('expiry'))
        if expiry != nearest_expiry:
            continue
        option_type = str(inst.get('instrument_type', '')).upper()
        if option_type not in ('CE', 'PE'):
            continue
        strike = float(inst.get('strike', 0.0) or 0.0)
        if abs(strike - atm) > 100:
            continue
        token = int(inst.get('instrument_token'))
        tradingsymbol = str(inst.get('tradingsymbol', '')).strip()
        if not tradingsymbol:
            continue
        selected.append({'symbol': tradingsymbol, 'token': token})
    return selected
