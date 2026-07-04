"""Reconciliation ownership runtime guards."""

from __future__ import annotations

from contextlib import suppress
import threading
from typing import Any, Mapping, Sequence

from nifty_scalper_bot.execution import position_manager as _position_manager
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}


def _canonical(symbol: object) -> str:
    return normalize_symbol(str(symbol or ""))


def _prepare_rows(manager: Any, rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], set[str]]:
    positions = getattr(manager, "_positions", {})
    prepared: list[dict[str, object]] = []
    unresolved: set[str] = set()
    for row in rows:
        cloned = dict(row)
        symbol = _canonical(cloned.get("tradingsymbol") or cloned.get("symbol"))
        if symbol:
            cloned["tradingsymbol"] = symbol
            cloned["symbol"] = symbol
        avg = 0.0
        for key in ("average_price", "avg_price", "buy_price", "price"):
            with suppress(Exception):
                avg = max(avg, float(cloned.get(key) or 0.0))
        if avg <= 0.0:
            existing = positions.get(symbol) if isinstance(positions, Mapping) else None
            existing_entry = float(getattr(existing, "entry_price", 0.0) or 0.0) if existing else 0.0
            if existing_entry > 0.0:
                cloned["average_price"] = existing_entry
            else:
                cloned["cost_basis_unresolved"] = True
                unresolved.add(symbol)
        prepared.append(cloned)
    return prepared, unresolved


def apply_patches() -> None:
    return None


apply_patches()

__all__ = ["apply_patches"]
