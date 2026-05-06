"""Lot-size resolution helpers."""

from __future__ import annotations

from typing import Any, Callable

from nifty_scalper_bot.config import settings as app_settings
from nifty_scalper_bot.utils.symbols import normalize_symbol


def resolve_lot_size(symbol: str, lookup: Callable[[str], int] | None = None) -> tuple[int, str]:
    """Resolve lot size. Args: symbol/lookup. Returns: lot size and source. Raises: ValueError."""
    normalized_symbol = normalize_symbol(symbol) or str(symbol).strip().upper()
    candidates = [normalized_symbol]
    if ':' in normalized_symbol:
        candidates.append(normalized_symbol.split(':', 1)[-1])
    if lookup is not None:
        for candidate in candidates:
            resolved: Any = lookup(candidate)
            if resolved is None:
                continue
            lot = int(resolved)
            if lot > 0:
                return lot, 'instrument_dump'
    if 'NIFTY' in normalized_symbol and ('CE' in normalized_symbol or 'PE' in normalized_symbol):
        return 65, 'fallback'
    fallback_settings = app_settings.get_settings()
    fallback_lot = int(getattr(fallback_settings, 'contract_lot_size', 0) or 0)
    if fallback_lot > 0:
        return fallback_lot, 'settings'
    raise ValueError(f'No lot size available for symbol={normalized_symbol}')
