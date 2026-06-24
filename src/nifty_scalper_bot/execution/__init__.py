"""Canonical execution package exports and live-money safety hardening.

Public import paths remain stable while the runtime bracket authority adds
fill-integrity and durable confirmed-fill accounting.  Import replacement is
retained only as a compatibility bridge until composition-root construction is
migrated in the final canonicalisation stage.
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
from nifty_scalper_bot.execution.canonical_bracket_manager import (  # noqa: E402
    CanonicalBracketManager,
)
from nifty_scalper_bot.execution.ledger_bracket_manager import (  # noqa: E402
    LedgerBracketManager,
)

_bracket_module.AdaptiveTrailingController = HardenedAdaptiveTrailingController
_bracket_module.BracketManager = LedgerBracketManager


__all__ = [
    "CanonicalBracketManager",
    "HardenedAdaptiveTrailingController",
    "HardenedBracketManager",
    "LedgerBracketManager",
]
