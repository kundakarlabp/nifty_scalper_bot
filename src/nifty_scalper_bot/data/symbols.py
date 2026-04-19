"""Shared symbol canonicalization helpers for market-data paths."""

from __future__ import annotations

from typing import Any

NIFTY_SPOT_TOKEN = 256265
NIFTY_SPOT_CANONICAL_SYMBOL = "NSE:NIFTY 50"
NIFTY_SPOT_CANONICAL_TRADINGSYMBOL = "NIFTY 50"

_NIFTY_SPOT_ALIASES = frozenset(
    {
        "NIFTY",
        "NIFTY 50",
        "NIFTY50",
        "NIFTY-50",
        "NSE:NIFTY",
        "NSE:NIFTY 50",
    }
)


def canonicalize_market_symbol(symbol: Any) -> str:
    """Return the canonical broker-facing symbol for market data operations."""

    if symbol is None:
        return ""
    normalized = str(symbol).strip().upper()
    if not normalized:
        return ""
    if normalized in _NIFTY_SPOT_ALIASES:
        return NIFTY_SPOT_CANONICAL_SYMBOL
    return normalized


def normalize_hub_symbol(symbol: Any) -> str:
    """Return the canonical cache key used by :class:`DataHub`."""

    normalized = canonicalize_market_symbol(symbol)
    if normalized == NIFTY_SPOT_CANONICAL_SYMBOL:
        return normalized
    if ":" in normalized:
        return normalized.split(":", 1)[-1].strip()
    return normalized


def is_nifty_spot_symbol(symbol: Any) -> bool:
    """Return ``True`` when *symbol* refers to the Nifty spot index."""

    return canonicalize_market_symbol(symbol) == NIFTY_SPOT_CANONICAL_SYMBOL


__all__ = [
    "NIFTY_SPOT_CANONICAL_SYMBOL",
    "NIFTY_SPOT_CANONICAL_TRADINGSYMBOL",
    "NIFTY_SPOT_TOKEN",
    "canonicalize_market_symbol",
    "is_nifty_spot_symbol",
    "normalize_hub_symbol",
]
