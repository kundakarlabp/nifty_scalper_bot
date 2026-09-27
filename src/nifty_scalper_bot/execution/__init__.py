# fmt: off
# ruff: noqa: E501,I001,F841
"""Canonical execution package exports.

Only production runtime owners are part of the package-level public API. Older
stage names are lazy compatibility aliases so existing imports do not create a
second live lifecycle authority.
"""

from __future__ import annotations

from typing import Any

from nifty_scalper_bot.execution.adaptive_trailing import (
    AdaptiveTrailingController,
    HardenedAdaptiveTrailingController,
)
from nifty_scalper_bot.execution.bracket_manager import BoundBracketManager, BracketManager
from nifty_scalper_bot.execution.order_manager import OrderManager, RuntimeOrderManager
from nifty_scalper_bot.execution.broker_pnl_authority_patch import apply_patches as _apply_broker_pnl_authority_patches
from nifty_scalper_bot.execution.pnl_session_rollover_patch import apply_patches as _apply_pnl_session_rollover_patches
import nifty_scalper_bot.execution.broker_order_ledger_patch as _broker_order_ledger_patch

_broker_order_ledger_patch.apply_patches()
_apply_broker_pnl_authority_patches()
_apply_pnl_session_rollover_patches()

CanonicalBracketManager = BracketManager
_COMPAT_BRACKET_ALIASES = {
    "FillIntegrityBracketManager",
    "HardenedBracketManager",
    "LedgerBracketManager",
    "RuntimeBracketManager",
}


def __getattr__(name: str) -> Any:
    if name in _COMPAT_BRACKET_ALIASES:
        return BracketManager
    raise AttributeError(name)


__all__ = [
    "AdaptiveTrailingController",
    "BoundBracketManager",
    "BracketManager",
    "CanonicalBracketManager",
    "HardenedAdaptiveTrailingController",
    "OrderManager",
    "RuntimeOrderManager",
]

# fmt: on
