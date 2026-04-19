"""Utility helpers for canonical trading symbol normalization."""

from __future__ import annotations

from collections.abc import Iterable

_NIFTY_SPOT_ALIASES = {
    "NIFTY",
    "NIFTY50",
    "NIFTY50INDEX",
}


def _collapse_nifty_spot_alias(
    exchange: str,
    tradingsymbol: str,
) -> tuple[str, str]:
    exchange_value = str(exchange or "").strip().upper()
    trading_value = " ".join(str(tradingsymbol or "").replace("_", " ").split()).upper()
    compact = "".join(trading_value.split())
    if exchange_value in {"", "NSE"} and compact in _NIFTY_SPOT_ALIASES:
        return "NSE", "NIFTY"
    return exchange_value, trading_value


def canonical(symbol: str) -> str:
    """Args: symbol; Returns: canonical EXCHANGE:SYMBOL; Raises: none."""
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""
    normalized = raw.replace("_", " ")
    if ":" in normalized:
        exchange, tradingsymbol = normalized.split(":", 1)
        exchange, tradingsymbol = _collapse_nifty_spot_alias(exchange, tradingsymbol)
        if not exchange:
            compact = "".join(tradingsymbol.split())
            exchange = "NFO" if compact.endswith(("CE", "PE", "FUT")) else "NSE"
        return f"{exchange}:{tradingsymbol}"
    compact = "".join(normalized.split())
    if compact.isdigit():
        return compact
    if compact in _NIFTY_SPOT_ALIASES:
        return "NSE:NIFTY"
    exchange = "NFO" if compact.endswith(("CE", "PE", "FUT")) else "NSE"
    return f"{exchange}:{compact}"


def enforce_canonical(symbol: str) -> str:
    """Args: symbol; Returns: canonical symbol; Raises: ValueError for malformed values."""
    value = str(symbol or "").strip()
    if ":" not in value:
        raise ValueError(f"Non-canonical symbol: {symbol}")
    return value


def normalize_symbol(symbol: str) -> str:
    """Args: symbol; Returns: normalized exchange-qualified symbol; Raises: none."""
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""
    normalized = raw.replace("_", " ")
    if ":" in raw:
        exchange, tradingsymbol = normalized.split(":", 1)
        exchange, tradingsymbol = _collapse_nifty_spot_alias(exchange, tradingsymbol)
        if not exchange:
            compact = "".join(tradingsymbol.split())
            exchange = "NFO" if compact.endswith(("CE", "PE", "FUT")) else "NSE"
        return f"{exchange}:{tradingsymbol}"
    compact = "".join(normalized.split())
    if compact.isdigit():
        return compact
    if compact in _NIFTY_SPOT_ALIASES:
        return "NSE:NIFTY"
    # Infer exchange the same way canonical() does: derivatives get NFO,
    # everything else (indices, equities) gets NSE.
    if compact.endswith(("CE", "PE", "FUT")):
        return f"NFO:{normalized}"
    return f"NSE:{normalized}"


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


def is_strategy_instrument(symbol: str) -> bool:
    """Args: symbol; Returns: True for strategy instruments; Raises: none."""
    normalized = canonical(symbol)
    return normalized.startswith("NFO:NIFTY")
