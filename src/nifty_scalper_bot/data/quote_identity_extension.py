"""DataHub quote identity contract.

This module stamps quote payloads with canonical instrument identity and dynamic
freshness metadata at the data boundary. Execution may still fail closed, but
valid live quotes should now carry the identity fields required by the final
TradePlan guard.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
import time
from typing import Any, Mapping

from nifty_scalper_bot.data import data_hub as _data_hub

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}
_IDENTITY_KEYS = (
    "symbol",
    "tradingsymbol",
    "trading_symbol",
    "instrument_symbol",
    "exchange_symbol",
)


def _canonical(hub: Any, symbol: object) -> str:
    method = getattr(hub, "_canonical_quote_symbol", None)
    if callable(method):
        with suppress(Exception):
            return str(method(symbol) or "")
    normalizer = getattr(_data_hub, "normalize_symbol", None)
    if callable(normalizer):
        with suppress(Exception):
            return str(normalizer(symbol) or "")
    return str(symbol or "").strip().upper().replace(" ", "")


def _quote_symbol_hint(quote: Mapping[str, Any]) -> str:
    for key in _IDENTITY_KEYS:
        value = quote.get(key)
        if isinstance(value, str) and value.strip():
            return value
    instrument = quote.get("instrument")
    if isinstance(instrument, Mapping):
        for key in _IDENTITY_KEYS:
            value = instrument.get(key)
            if isinstance(value, str) and value.strip():
                return value
    meta = quote.get("meta") or quote.get("metadata")
    if isinstance(meta, Mapping):
        for key in _IDENTITY_KEYS:
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _coerce_timestamp_ms(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, datetime):
        return raw.timestamp() * 1000.0
    with suppress(Exception):
        value = float(raw)
        if value > 0:
            return value * 1000.0 if value < 1e11 else value
    with suppress(Exception):
        import pandas as pd

        ts = pd.to_datetime(raw, utc=True, errors="coerce")
        if not pd.isna(ts):
            return float(pd.Timestamp(ts).timestamp() * 1000.0)
    return None


def _token_for(hub: Any, symbol: str, quote: Mapping[str, Any]) -> int | None:
    for key in ("instrument_token", "token"):
        with suppress(Exception):
            value = quote.get(key)
            if value is not None:
                return int(value)
    mapping = getattr(hub, "_token_by_symbol", {})
    if isinstance(mapping, Mapping):
        with suppress(Exception):
            value = mapping.get(symbol)
            if value is not None:
                return int(value)
    resolver = getattr(hub, "_token_from_symbol", None)
    if callable(resolver):
        with suppress(Exception):
            value = resolver(symbol)
            if value is not None:
                return int(value)
    return None


def stamp_quote_identity(hub: Any, requested_symbol: object, quote: Any) -> Any:
    if not isinstance(quote, Mapping):
        return quote
    stamped = dict(quote)
    symbol = _canonical(hub, requested_symbol) or _canonical(hub, _quote_symbol_hint(stamped))
    if not symbol:
        return stamped
    stamped["symbol"] = symbol
    stamped["tradingsymbol"] = symbol
    stamped["trading_symbol"] = symbol
    stamped["instrument_symbol"] = symbol
    stamped["exchange_symbol"] = symbol
    token = _token_for(hub, symbol, stamped)
    if token is not None:
        stamped["instrument_token"] = token
        stamped["token"] = token
    version_getter = getattr(hub, "quote_update_version", None)
    if callable(version_getter):
        with suppress(Exception):
            version = version_getter(symbol)
            if version is not None:
                stamped["quote_update_version"] = int(version)
    ts_ms = _coerce_timestamp_ms(
        stamped.get("timestamp_ms")
        or stamped.get("last_tick_ts_ms")
        or stamped.get("exchange_timestamp")
        or stamped.get("timestamp")
    )
    if ts_ms is not None:
        stamped["last_tick_ts_ms"] = ts_ms
        now_fn = getattr(hub, "_now", None)
        with suppress(Exception):
            now_ms = float(now_fn() if callable(now_fn) else time.time()) * 1000.0
            stamped["tick_age_ms"] = max(0.0, now_ms - ts_ms)
    stamped["quote_identity_source"] = stamped.get("quote_identity_source") or "datahub_quote_contract"
    return stamped


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    cls = getattr(_data_hub, "DataHub", None)
    if cls is None or getattr(cls, "_quote_identity_contract_patch", False):
        _PATCH_APPLIED = True
        return
    _ORIGINALS["DataHub._canonicalize_tick_payload"] = cls._canonicalize_tick_payload
    _ORIGINALS["DataHub.get_quote"] = cls.get_quote
    _ORIGINALS["DataHub.get_tick_by_token"] = cls.get_tick_by_token

    def _canonicalize_tick_payload(self: Any, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        tick = _ORIGINALS["DataHub._canonicalize_tick_payload"](self, payload)
        if tick is None:
            return None
        return stamp_quote_identity(self, tick.get("symbol"), tick)

    def get_quote(self: Any, symbol: str, allow_pull: bool = True) -> Any:
        quote = _ORIGINALS["DataHub.get_quote"](self, symbol, allow_pull)
        return stamp_quote_identity(self, symbol, quote)

    def get_tick_by_token(self: Any, token: int) -> Any:
        tick = _ORIGINALS["DataHub.get_tick_by_token"](self, token)
        symbol = None
        with suppress(Exception):
            symbol = getattr(self, "_symbol_by_token", {}).get(int(token))
        return stamp_quote_identity(self, symbol or token, tick)

    cls._canonicalize_tick_payload = _canonicalize_tick_payload
    cls.get_quote = get_quote
    cls.get_tick_by_token = get_tick_by_token
    cls._quote_identity_contract_patch = True
    _PATCH_APPLIED = True


apply_patches()

__all__ = ["apply_patches", "stamp_quote_identity"]
