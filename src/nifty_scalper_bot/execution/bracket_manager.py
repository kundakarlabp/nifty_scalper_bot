"""Stable public bracket API with one canonical runtime manager.

The historical implementation is preserved in ``legacy_bracket_manager`` for
subclass compatibility and staged retirement. New and existing public imports
of ``BracketManager`` resolve directly to ``RuntimeBracketManager``; no package
initializer mutates this module after import.
"""

from __future__ import annotations

from nifty_scalper_bot.execution import legacy_bracket_manager as _legacy

# Re-export the complete historical public surface so callers importing state
# classes, helpers, constants or enums do not lose functionality.
for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

LegacyBracketManager = _legacy.BracketManager

# During this import the runtime inheritance chain may import this partially
# initialized facade. ``BracketManager`` already points to the legacy base at
# that point, allowing the subclasses to initialize without circular ownership.
from nifty_scalper_bot.execution.runtime_bracket_manager import (  # noqa: E402
    RuntimeBracketManager,
)

BracketManager = RuntimeBracketManager

__all__ = sorted(
    {
        *[name for name in dir(_legacy) if not name.startswith("_")],
        "BracketManager",
        "LegacyBracketManager",
        "RuntimeBracketManager",
    }
)
