"""Shared runtime-history role resolution.

Runtime role:
- Provides the single neutral SSOT for symbol-to-history-role classification.
- Consumes caller-provided selected/context/open-position symbols only.
- Must not import app, runner, broker, DataHub, or MarketDataManager.
"""

from __future__ import annotations

from collections.abc import Collection


def _normalize(value: str | None) -> str:
    return str(value or "").strip().upper()


def resolve_symbol_history_role(
    *,
    symbol: str | None,
    selected_ce: str | None = None,
    selected_pe: str | None = None,
    spot_symbol: str | None = None,
    futures_symbol: str | None = None,
    open_position_symbols: Collection[str] = (),
) -> str:
    """Resolve the canonical runtime-history role for a symbol."""
    normalized = _normalize(symbol)
    if not normalized:
        return "option_context"

    selected = {_normalize(selected_ce), _normalize(selected_pe)}
    if normalized in selected:
        return "selected_option"

    spot = _normalize(spot_symbol)
    if normalized == spot or (not spot and normalized == "NSE:NIFTY"):
        return "spot_context"

    future = _normalize(futures_symbol)
    if normalized == future or (not future and normalized.endswith("FUT")):
        return "futures_context"

    open_positions = {_normalize(str(item)) for item in open_position_symbols}
    if normalized in open_positions:
        return "recovery_or_open_position"

    return "option_context"
