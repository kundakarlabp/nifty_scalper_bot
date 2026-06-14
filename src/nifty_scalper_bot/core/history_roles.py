"""Shared runtime-history role resolution.

Runtime role:
- Provides the single neutral SSOT for symbol-to-history-role classification.
- Consumes caller-provided selected/context/open-position symbols only.
- Must not import app, runner, broker, DataHub, or MarketDataManager.
"""

from __future__ import annotations

from collections.abc import Collection


HISTORY_ROLE_PRIORITY: dict[str, int] = {
    "option_context": 10,
    "spot_context": 20,
    "futures_context": 20,
    "selected_option": 30,
    "recovery_or_open_position": 40,
}


def _normalize(value: str | None) -> str:
    return str(value or "").strip().upper()


def _spot_alias_key(value: str | None) -> str:
    normalized = _normalize(value)
    if normalized.startswith("NSE:"):
        normalized = normalized.split(":", 1)[1]
    return normalized.replace(" ", "")


def is_same_spot_symbol(left: str | None, right: str | None) -> bool:
    left_key = _spot_alias_key(left)
    right_key = _spot_alias_key(right)
    if not left_key or not right_key:
        return False
    return left_key in {"NIFTY", "NIFTY50"} and right_key in {"NIFTY", "NIFTY50"}


def history_role_priority(role: str | None) -> int:
    return HISTORY_ROLE_PRIORITY.get(_normalize(role).lower(), HISTORY_ROLE_PRIORITY["option_context"])


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

    if is_same_spot_symbol(normalized, spot_symbol) or (not _normalize(spot_symbol) and is_same_spot_symbol(normalized, "NSE:NIFTY")):
        return "spot_context"

    future = _normalize(futures_symbol)
    if normalized == future or (not future and normalized.endswith("FUT")):
        return "futures_context"

    open_positions = {_normalize(str(item)) for item in open_position_symbols}
    if normalized in open_positions:
        return "recovery_or_open_position"

    return "option_context"
