"""Canonical execution package exports and live-money safety hardening.

Public import paths remain stable while one bracket authority applies fill
integrity and durable accounting, and the established OrderManager receives a
bounded recovery layer without changing its class identity.
"""

from __future__ import annotations

from importlib import import_module


_adaptive_module = import_module(f"{__name__}.adaptive_trailing")
from nifty_scalper_bot.execution.hardened_adaptive_trailing import (  # noqa: E402
    HardenedAdaptiveTrailingController,
)

_adaptive_module.AdaptiveTrailingController = HardenedAdaptiveTrailingController

_bracket_module = import_module(f"{__name__}.bracket_manager")
_canonical_module = import_module(f"{__name__}.canonical_bracket_manager")
from nifty_scalper_bot.execution.hardened_bracket_manager import (  # noqa: E402
    HardenedBracketManager,
)
from nifty_scalper_bot.execution.canonical_bracket_manager import (  # noqa: E402
    CanonicalBracketManager as _FillIntegrityBracketManager,
)
from nifty_scalper_bot.execution.ledger_bracket_manager import (  # noqa: E402
    LedgerBracketManager,
)
from nifty_scalper_bot.execution.runtime_bracket_manager import (  # noqa: E402
    RuntimeBracketManager,
)

_bracket_module.AdaptiveTrailingController = HardenedAdaptiveTrailingController
_bracket_module.BracketManager = RuntimeBracketManager
_canonical_module.CanonicalBracketManager = RuntimeBracketManager
CanonicalBracketManager = RuntimeBracketManager

# Preserve the existing OrderManager class and all unbound method contracts.
_order_manager_module = import_module(f"{__name__}.order_manager")
from nifty_scalper_bot.execution.entry_recovery import (  # noqa: E402
    install_entry_recovery,
)

install_entry_recovery(_order_manager_module.OrderManager)


__all__ = [
    "CanonicalBracketManager",
    "HardenedAdaptiveTrailingController",
    "HardenedBracketManager",
    "LedgerBracketManager",
    "RuntimeBracketManager",
    "_FillIntegrityBracketManager",
]
