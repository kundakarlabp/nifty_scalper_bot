"""Canonical execution package exports and live-money safety hardening.

Public import paths remain stable while one runtime bracket authority applies
fill integrity, durable LIVE accounting and compatible PAPER/SHADOW behaviour.
Import replacement remains a temporary bridge until explicit composition-root
construction is migrated.
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


__all__ = [
    "CanonicalBracketManager",
    "HardenedAdaptiveTrailingController",
    "HardenedBracketManager",
    "LedgerBracketManager",
    "RuntimeBracketManager",
    "_FillIntegrityBracketManager",
]
