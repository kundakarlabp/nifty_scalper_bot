"""Bracket ownership extension."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Mapping

from nifty_scalper_bot.execution import bracket_core as _bracket_core
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}
_TERMINAL_STATES = {"CLOSED", "EXIT_FILLED", "EXIT_RECONCILED_FLAT"}


def _canonical(symbol: object) -> str:
    return normalize_symbol(str(symbol or ""))


def _is_nonterminal(bracket: Any) -> bool:
    with suppress(Exception):
        if int(getattr(bracket, "remaining_quantity", 0) or 0) <= 0:
            return False
    if bool(getattr(bracket, "exit_executed", False)):
        return False
    state = str(getattr(bracket, "exit_state", "") or "").upper()
    return state not in _TERMINAL_STATES


def _existing_owner(manager: Any, symbol: str, order_id: str | None = None) -> Any | None:
    brackets = getattr(manager, "_brackets", {})
    if not isinstance(brackets, Mapping):
        return None
    canonical_symbol = _canonical(symbol)
    for bracket in brackets.values():
        if order_id and str(getattr(bracket, "entry_order_id", "")) == str(order_id):
            continue
        if _canonical(getattr(bracket, "symbol", "")) != canonical_symbol:
            continue
        if _is_nonterminal(bracket):
            return bracket
    return None


def _patch_bracket_manager() -> None:
    cls = getattr(_bracket_core, "BracketManager", None)
    if cls is None or getattr(cls, "_bracket_ownership_extension_patch", False):
        return
    if hasattr(cls, "register_virtual_bracket"):
        _ORIGINALS["BracketManager.register_virtual_bracket"] = cls.register_virtual_bracket

        def register_virtual_bracket(self: Any, order_id: str, symbol: str, *args: Any, **kwargs: Any) -> Any:
            canonical_symbol = _canonical(symbol)
            lock = getattr(self, "_lock", None)
            if lock is None:
                owner = _existing_owner(self, canonical_symbol, str(order_id))
            else:
                with lock:
                    owner = _existing_owner(self, canonical_symbol, str(order_id))
            if owner is not None:
                _bracket_core.LOGGER.warning(
                    "BRACKET_OWNERSHIP_CONFLICT_SKIPPED symbol=%s existing=%s attempted=%s",
                    canonical_symbol,
                    getattr(owner, "entry_order_id", None),
                    order_id,
                    extra={
                        "event": "BRACKET_OWNERSHIP_CONFLICT_SKIPPED",
                        "symbol": canonical_symbol,
                        "existing_entry_order_id": getattr(owner, "entry_order_id", None),
                        "attempted_order_id": str(order_id),
                    },
                )
                return None
            return _ORIGINALS["BracketManager.register_virtual_bracket"](
                self,
                order_id,
                canonical_symbol,
                *args,
                **kwargs,
            )

        cls.register_virtual_bracket = register_virtual_bracket
    cls._bracket_ownership_extension_patch = True


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    _patch_bracket_manager()
    _PATCH_APPLIED = True


apply_patches()

__all__ = ["apply_patches"]
