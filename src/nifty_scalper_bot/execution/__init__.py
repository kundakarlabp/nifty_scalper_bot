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
from nifty_scalper_bot.execution.live_safety_identity import apply_patches as _apply_live_safety_identity_patches
from nifty_scalper_bot.execution.position_identity_extension import apply_patches as _apply_position_identity_extension_patches
import nifty_scalper_bot.data.quote_identity_extension as _quote_identity_extension
import nifty_scalper_bot.execution.bracket_ownership_extension as _bracket_ownership_extension
import nifty_scalper_bot.execution.broker_exposure_quarantine_extension as _broker_exposure_quarantine_extension
import nifty_scalper_bot.execution.trade_plan_identity_guard as _trade_plan_identity_guard

_apply_live_safety_identity_patches()
_apply_position_identity_extension_patches()
_quote_identity_extension.apply_patches()
_broker_exposure_quarantine_extension.apply_patches()
_bracket_ownership_extension.apply_patches()
_trade_plan_identity_guard.apply_patches()

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
