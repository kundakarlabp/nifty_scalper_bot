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


# ---------------------------------------------------------------------------
# Zerodha REST quote alias helpers
#
# Zerodha's /quote endpoint keys the response by the exact instrument name it
# recognises (e.g. "NSE:NIFTY 50" for the NIFTY index).  Our internal canonical
# representation is "NSE:NIFTY".  These helpers translate canonical or shorthand
# index names into the Kite REST quote-compatible variants without changing the
# bot's internal symbol storage.  WebSocket token mapping is unaffected.
# ---------------------------------------------------------------------------

_ZERODHA_QUOTE_ALIAS_MAP: dict[str, str] = {
    "NSE:NIFTY": "NSE:NIFTY 50",
    "NIFTY": "NSE:NIFTY 50",
    "NIFTY50": "NSE:NIFTY 50",
    "NSE:NIFTY50": "NSE:NIFTY 50",
    "NSE:NIFTY 50": "NSE:NIFTY 50",
    "NSE:BANKNIFTY": "NSE:NIFTY BANK",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "NSE:BANKNIFTY 1": "NSE:NIFTY BANK",
    "NSE:NIFTY BANK": "NSE:NIFTY BANK",
    "NIFTYBANK": "NSE:NIFTY BANK",
    "NSE:FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "NSE:NIFTY FIN SERVICE": "NSE:NIFTY FIN SERVICE",
}


def to_zerodha_quote_symbol(symbol: str) -> str:
    """Translate a canonical/shorthand symbol to the Zerodha quote symbol.

    Args:
        symbol: Canonical or shorthand internal symbol (e.g. ``"NSE:NIFTY"``).

    Returns:
        Kite REST quote-compatible symbol (e.g. ``"NSE:NIFTY 50"``).  Symbols
        that are not known indices are returned unchanged so option strikes and
        equities continue to round-trip without modification.
    """

    raw = str(symbol or "").strip()
    if not raw:
        return ""
    # Normalize whitespace before mapping: "NSE:NIFTY  50" → "NSE:NIFTY 50".
    collapsed = " ".join(raw.split())
    upper = collapsed.upper()
    return _ZERODHA_QUOTE_ALIAS_MAP.get(upper, raw)


def zerodha_quote_aliases(symbol: str) -> list[str]:
    """Return all Kite quote-compatible aliases for ``symbol``.

    The canonical Kite name is preferred, but the original symbol and the
    bare tradingsymbol are kept as fallbacks so the response can be matched
    even when Kite returns the payload under a slightly different key.  Order
    is preserved and duplicates are removed.

    Args:
        symbol: Canonical or shorthand symbol to expand.

    Returns:
        List of alias strings, with the Kite-preferred form first.
    """

    raw = str(symbol or "").strip()
    if not raw:
        return []
    primary = to_zerodha_quote_symbol(raw)
    candidates = [
        primary,
        raw,
        primary.split(":", 1)[-1] if ":" in primary else primary,
        raw.split(":", 1)[-1] if ":" in raw else raw,
    ]
    out: list[str] = []
    for item in candidates:
        cleaned = str(item or "").strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out
