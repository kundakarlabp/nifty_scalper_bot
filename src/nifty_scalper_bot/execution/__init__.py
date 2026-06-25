"""Canonical execution package exports."""

from __future__ import annotations

from nifty_scalper_bot.execution.adaptive_trailing import (
    AdaptiveTrailingController,
    HardenedAdaptiveTrailingController,
    LegacyAdaptiveTrailingController,
)
from nifty_scalper_bot.execution.bracket_manager import (
    BoundBracketManager,
    BracketManager,
    LegacyBracketManager,
    RuntimeBracketManager,
)
from nifty_scalper_bot.execution.hardened_bracket_manager import HardenedBracketManager
from nifty_scalper_bot.execution.canonical_bracket_manager import (
    CanonicalBracketManager as FillIntegrityBracketManager,
)
from nifty_scalper_bot.execution.ledger_bracket_manager import LedgerBracketManager
from nifty_scalper_bot.execution.order_manager import (
    LegacyOrderManager,
    OrderManager,
    RuntimeOrderManager,
)

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
    "LegacyAdaptiveTrailingController",
    "LegacyBracketManager",
    "LegacyOrderManager",
    "OrderManager",
    "RuntimeBracketManager",
    "RuntimeOrderManager",
]
