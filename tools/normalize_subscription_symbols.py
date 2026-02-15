"""Normalize subscription symbols by trimming, uppercasing, and deduplicating."""

from __future__ import annotations

from collections.abc import Iterable


def normalize_subscription_symbols(symbols: Iterable[str]) -> list[str]:
    """Args: symbols; Returns: normalized symbols; Raises: none."""

    seen: set[str] = set()
    normalized: list[str] = []
    for raw_symbol in symbols:
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol:
            continue
        if ":" in symbol:
            exchange, name = symbol.split(":", 1)
            exchange = exchange.strip() or "NSE"
            name = name.strip()
        else:
            exchange = "NFO" if any(ch.isdigit() for ch in symbol) else "NSE"
            name = symbol
        if not name:
            continue
        canonical = f"{exchange}:{name}"
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


if __name__ == "__main__":
    sample = ["NFO:NIFTY26FEBFUT", "NIFTY26FEBFUT", "NSE:NIFTY 50", "NIFTY 50"]
    print(normalize_subscription_symbols(sample))
