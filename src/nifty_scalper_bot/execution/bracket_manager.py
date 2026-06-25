"""Stable public bracket API with explicit runtime ownership.

The historical implementation remains in ``legacy_bracket_manager``. Public
state models and helpers are re-exported unchanged, while ``BracketManager`` is
the bound runtime authority that configures the native OrderManager entry gate.
"""

from __future__ import annotations

from nifty_scalper_bot.execution import legacy_bracket_manager as _legacy

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

LegacyBracketManager = _legacy.BracketManager

from nifty_scalper_bot.execution.runtime_bracket_manager import (  # noqa: E402
    RuntimeBracketManager,
)
from nifty_scalper_bot.execution.ownership import (  # noqa: E402
    BoundBracketManager,
)

BracketManager = BoundBracketManager

__all__ = sorted(
    {
        *[name for name in dir(_legacy) if not name.startswith("_")],
        "BoundBracketManager",
        "BracketManager",
        "LegacyBracketManager",
        "RuntimeBracketManager",
    }
)
