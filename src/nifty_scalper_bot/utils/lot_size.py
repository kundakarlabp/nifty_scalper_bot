"""Lot-size resolution helpers."""

from __future__ import annotations

import os
import re
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
    compact_symbol = candidates[-1]
    option_match = re.match(r"^(?P<underlying>NIFTY)\d{1,2}[A-Z]{3}\d+(?P<option_type>CE|PE)$", compact_symbol)
    if option_match:
        env_fallback = os.getenv("NIFTY_LOT_SIZE") or os.getenv("INSTRUMENTS__NIFTY_LOT_SIZE")
        if env_fallback:
            try:
                lot = int(env_fallback)
                if lot > 0:
                    return lot, "env_fallback"
            except ValueError:
                pass
        return 65, "fallback_default"
    fallback_settings = app_settings.get_settings()
    fallback_lot = int(getattr(fallback_settings, 'contract_lot_size', 0) or 0)
    if fallback_lot > 0:
        return fallback_lot, 'settings'
    raise ValueError(f'No lot size available for symbol={normalized_symbol}')
