"""Trade-plan to quote identity guard.

This is a final live-entry choke-point guard: a TradePlan must be validated
against a quote snapshot that identifies the same canonical instrument before
broker placement can be attempted.
"""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.execution import order_manager_core as _core
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False
_ORIGINAL_VALIDATE = None
_IDENTITY_KEYS = (
    "symbol",
    "tradingsymbol",
    "trading_symbol",
    "instrument",
    "instrument_symbol",
    "exchange_symbol",
)


def _quote_identity_values(quote: Any) -> set[str]:
    if not isinstance(quote, Mapping):
        return set()
    values: set[str] = set()
    for key in _IDENTITY_KEYS:
        value = quote.get(key)
        if isinstance(value, str) and value.strip():
            values.add(normalize_symbol(value))
    instrument = quote.get("instrument")
    if isinstance(instrument, Mapping):
        for key in _IDENTITY_KEYS:
            value = instrument.get(key)
            if isinstance(value, str) and value.strip():
                values.add(normalize_symbol(value))
    meta = quote.get("meta") or quote.get("metadata")
    if isinstance(meta, Mapping):
        for key in _IDENTITY_KEYS:
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                values.add(normalize_symbol(value))
    return {value for value in values if value}


def _is_entry_plan(plan: Any) -> bool:
    intent = str(getattr(plan, "intent", "ENTRY") or "ENTRY").strip().upper()
    return intent in {"ENTRY", "SCALE_IN", "REVERSAL"}


def _details(symbol: str, quote_symbols: set[str]) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "quote_symbols": sorted(quote_symbols),
        "identity_chain": "quote.symbol->trade_plan.symbol",
    }


def apply_patches() -> None:
    global _PATCH_APPLIED, _ORIGINAL_VALIDATE
    if _PATCH_APPLIED:
        return
    cls = getattr(_core, "OrderManager", None)
    if cls is None or getattr(cls, "_trade_plan_identity_guard_patch", False):
        _PATCH_APPLIED = True
        return
    _ORIGINAL_VALIDATE = cls._validate_trade_plan

    def _validate_trade_plan(self: Any, plan: Any) -> Any:
        result = _ORIGINAL_VALIDATE(self, plan)
        if not getattr(result, "allowed", False):
            return result
        symbol = normalize_symbol(str(getattr(plan, "symbol", "") or ""))
        if not symbol or not _is_entry_plan(plan):
            return result
        live = False
        live_fn = getattr(self, "_order_live_execution_enabled", None)
        if callable(live_fn):
            try:
                live = bool(live_fn())
            except Exception:
                live = False
        if not live:
            return result
        quote = self._get_latest_quote_safe(symbol)
        quote_symbols = _quote_identity_values(quote)
        if quote_symbols and symbol not in quote_symbols:
            return _core.OrderPreflightResult(
                False,
                "quote_symbol_identity_mismatch",
                _details(symbol, quote_symbols),
            )
        if not quote_symbols:
            return _core.OrderPreflightResult(
                False,
                "quote_symbol_identity_missing",
                _details(symbol, quote_symbols),
            )
        return result

    cls._validate_trade_plan = _validate_trade_plan
    cls._trade_plan_identity_guard_patch = True
    _PATCH_APPLIED = True


apply_patches()

__all__ = ["apply_patches"]
