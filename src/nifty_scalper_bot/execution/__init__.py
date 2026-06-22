"""Canonical execution package exports and live-money safety hardening.

The public module paths remain unchanged. Import-time replacement keeps every
existing caller on the single production execution path while applying the
hardened implementations before any manager instance is constructed.
"""

from __future__ import annotations

from importlib import import_module


_adaptive_module = import_module(f"{__name__}.adaptive_trailing")
from nifty_scalper_bot.execution.hardened_adaptive_trailing import (  # noqa: E402
    HardenedAdaptiveTrailingController,
)

_adaptive_module.AdaptiveTrailingController = HardenedAdaptiveTrailingController

_bracket_module = import_module(f"{__name__}.bracket_manager")
from nifty_scalper_bot.execution.hardened_bracket_manager import (  # noqa: E402
    HardenedBracketManager,
)

# Preserve all established import paths, including
# ``from ...execution.bracket_manager import BracketManager``.
_bracket_module.AdaptiveTrailingController = HardenedAdaptiveTrailingController
_bracket_module.BracketManager = HardenedBracketManager


__all__ = [
    "HardenedAdaptiveTrailingController",
    "HardenedBracketManager",
]
