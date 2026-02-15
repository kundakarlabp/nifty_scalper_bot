"""Utility helpers for canonical trading symbol normalization."""

from __future__ import annotations

from collections.abc import Iterable


def canonical(symbol: str) -> str:
    """Args: symbol; Returns: canonical EXCHANGE:SYMBOL; Raises: none."""
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""
    compact = "".join(raw.replace("_", " ").split())
    compact = compact.replace("NIFTY50", "NIFTY").replace("NIFTYBANK", "BANKNIFTY")
    if ":" in compact:
        exchange, tradingsymbol = compact.split(":", 1)
        if not exchange:
            exchange = "NFO" if tradingsymbol.endswith(("CE", "PE", "FUT")) else "NSE"
        return f"{exchange}:{tradingsymbol}"
    if compact.isdigit():
        return compact
    exchange = "NFO" if compact.endswith(("CE", "PE", "FUT")) else "NSE"
    return f"{exchange}:{compact}"


def normalize_symbol(symbol: str) -> str:
    """Args: symbol; Returns: normalized exchange-qualified symbol; Raises: none."""
    return canonical(symbol)


def unique_normalized_symbols(symbols: Iterable[str]) -> list[str]:
    """Args: symbols; Returns: ordered normalized de-duplicated symbols; Raises: none."""
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in symbols:
        value = normalize_symbol(raw)
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def is_canonical_symbol(symbol: str) -> bool:
    """Args: symbol; Returns: whether symbol matches EXCHANGE:SYMBOL; Raises: none."""
    value = str(symbol or "").strip().upper()
    if not value or ":" not in value:
        return False
    exchange, tradingsymbol = value.split(":", 1)
    return bool(exchange and tradingsymbol and " " not in exchange)
