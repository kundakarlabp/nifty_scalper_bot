"""Stable adaptive trailing API with one hardened runtime controller.

The historical implementation is preserved in ``adaptive_trailing_legacy``.
Public data structures and helpers remain available from this module, while
``AdaptiveTrailingController`` resolves directly to the hardened controller.
"""

from __future__ import annotations

from nifty_scalper_bot.execution import adaptive_trailing_legacy as _legacy

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)

LegacyAdaptiveTrailingController = _legacy.AdaptiveTrailingController

# Export the legacy base before importing the hardened subclass. The hardened
# module can therefore import this partially initialized facade safely.
AdaptiveTrailingController = LegacyAdaptiveTrailingController

from nifty_scalper_bot.execution.hardened_adaptive_trailing import (  # noqa: E402
    HardenedAdaptiveTrailingController,
)

AdaptiveTrailingController = HardenedAdaptiveTrailingController

__all__ = sorted(
    {
        *[name for name in dir(_legacy) if not name.startswith("_")],
        "AdaptiveTrailingController",
        "HardenedAdaptiveTrailingController",
        "LegacyAdaptiveTrailingController",
    }
)
