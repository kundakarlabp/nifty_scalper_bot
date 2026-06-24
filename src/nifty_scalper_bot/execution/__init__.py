"""Canonical execution package exports and live-money safety hardening.

Public import paths remain stable while the runtime bracket authority adds
fill-integrity and durable confirmed-fill accounting. Import replacement is a
compatibility bridge only; explicit composition-root construction follows after
all lifecycle invariants are merged and proven.
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

_bracket_module.AdaptiveTrailingController = HardenedAdaptiveTrailingController
_bracket_module.BracketManager = LedgerBracketManager
# Existing tests and external callers importing CanonicalBracketManager continue
# to resolve to the single runtime authority during the compatibility stage.
_canonical_module.CanonicalBracketManager = LedgerBracketManager
CanonicalBracketManager = LedgerBracketManager


__all__ = [
    "CanonicalBracketManager",
    "HardenedAdaptiveTrailingController",
    "HardenedBracketManager",
    "LedgerBracketManager",
    "_FillIntegrityBracketManager",
]
