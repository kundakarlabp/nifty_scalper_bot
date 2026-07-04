"""Canonical execution package exports."""

from __future__ import annotations

from nifty_scalper_bot.execution.adaptive_trailing import (
    AdaptiveTrailingController,
    HardenedAdaptiveTrailingController,
)
from nifty_scalper_bot.execution.bracket_manager import (
    BoundBracketManager,
    BracketManager,
    RuntimeBracketManager,
)
from nifty_scalper_bot.execution.hardened_bracket_manager import HardenedBracketManager
from nifty_scalper_bot.execution.canonical_bracket_manager import (
    CanonicalBracketManager as FillIntegrityBracketManager,
)
from nifty_scalper_bot.execution.ledger_bracket_manager import LedgerBracketManager
from nifty_scalper_bot.execution.order_manager import (
    OrderManager,
    RuntimeOrderManager,
)
from nifty_scalper_bot.execution.live_safety_identity import apply_patches as _apply_live_safety_identity_patches
from nifty_scalper_bot.execution.position_identity_extension import apply_patches as _apply_position_identity_extension_patches
import nifty_scalper_bot.execution.bracket_ownership_extension as _bracket_ownership_extension

_apply_live_safety_identity_patches()
_apply_position_identity_extension_patches()
_bracket_ownership_extension.apply_patches()

CanonicalBracketManager = BracketManager

__all__ = [
    "AdaptiveTrailingController",
    "BoundBracketManager",
    "BracketManager",
    "CanonicalBracketManager",
    "FillIntegrityBracketManager",
    "HardenedAdaptiveTrailingController",
    "HardenedBracketManager",
    "LedgerBracketManager",
    "OrderManager",
    "RuntimeBracketManager",
    "RuntimeOrderManager",
]
