"""Utility helpers for canonical trading symbol normalization."""

from __future__ import annotations

from collections.abc import Iterable


def normalize_symbol(symbol: str) -> str:
    """Args: symbol; Returns: normalized exchange-qualified symbol; Raises: none."""
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""
    compact = " ".join(raw.replace("_", " ").split())
    if ":" in compact:
        return compact
    if compact.isdigit():
        return compact
    if compact.startswith("NIFTY") or compact.startswith("BANKNIFTY"):
        return f"NSE:{compact}"
    if compact.endswith("CE") or compact.endswith("PE"):
        return f"NFO:{compact}"
    return f"NSE:{compact}"


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
