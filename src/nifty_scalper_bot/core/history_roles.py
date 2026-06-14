"""Shared role resolution for canonical history hydration.

Runtime role:
- Owns symbol-to-history-role classification only.
- Consumes already-selected contract symbols from app/runner context.
- Must not fetch, hydrate, cache, or mutate history.
"""

from __future__ import annotations

from collections.abc import Collection


def _norm(value: object | None) -> str:
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
    """Return canonical history role for an already-selected runtime symbol."""
    normalized = _norm(symbol)
    if not normalized:
        return "option_context"
    selected = {_norm(selected_ce), _norm(selected_pe)} - {""}
    if normalized in selected:
        return "selected_option"
    if normalized in {_norm(spot_symbol), "NSE:NIFTY", "NIFTY", "NSE:NIFTY50", "NIFTY50"}:
        return "spot_context"
    if normalized == _norm(futures_symbol) or (normalized.startswith("NFO:NIFTY") and normalized.endswith("FUT")):
        return "futures_context"
    if normalized in {_norm(item) for item in open_position_symbols}:
        return "recovery_or_open_position"
    return "option_context"
