"""Canonical execution package exports."""

from __future__ import annotations

from importlib import import_module


_adaptive_module = import_module(f"{__name__}.adaptive_trailing")
from nifty_scalper_bot.execution.hardened_adaptive_trailing import (
    HardenedAdaptiveTrailingController,
)

_adaptive_module.AdaptiveTrailingController = HardenedAdaptiveTrailingController

from nifty_scalper_bot.execution.bracket_manager import (
    BracketManager,
    LegacyBracketManager,
    RuntimeBracketManager,
)
from nifty_scalper_bot.execution.hardened_bracket_manager import HardenedBracketManager
from nifty_scalper_bot.execution.canonical_bracket_manager import (
    CanonicalBracketManager as FillIntegrityBracketManager,
)
from nifty_scalper_bot.execution.ledger_bracket_manager import LedgerBracketManager

CanonicalBracketManager = BracketManager

_order_manager_module = import_module(f"{__name__}.order_manager")
from nifty_scalper_bot.execution.entry_recovery import install_entry_recovery

install_entry_recovery(_order_manager_module.OrderManager)


__all__ = [
    "BracketManager",
    "CanonicalBracketManager",
    "FillIntegrityBracketManager",
    "HardenedAdaptiveTrailingController",
    "HardenedBracketManager",
    "LedgerBracketManager",
    "LegacyBracketManager",
    "RuntimeBracketManager",
]
