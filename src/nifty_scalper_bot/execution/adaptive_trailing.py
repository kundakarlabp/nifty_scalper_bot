"""Stable adaptive trailing API with one hardened runtime controller."""

from __future__ import annotations

from nifty_scalper_bot.execution import adaptive_trailing_core as _core

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

# Export the core base before importing the hardened subclass so the subclass can
# safely import this partially initialized public facade.
AdaptiveTrailingController = _core.AdaptiveTrailingController

from nifty_scalper_bot.execution.hardened_adaptive_trailing import (  # noqa: E402
    HardenedAdaptiveTrailingController,
)

AdaptiveTrailingController = HardenedAdaptiveTrailingController

__all__ = sorted(
    {
        *[name for name in dir(_core) if not name.startswith("_")],
        "AdaptiveTrailingController",
        "HardenedAdaptiveTrailingController",
    }
)
