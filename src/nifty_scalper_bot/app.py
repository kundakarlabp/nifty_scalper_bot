"""Backward compatible application launcher."""

from __future__ import annotations

from nifty_scalper_bot.core.app import (
    NiftyScalperApp,
    initialize_components,
    shutdown_sequence,
    startup_sequence,
)
from nifty_scalper_bot.main import main

__all__ = [
    "main",
    "NiftyScalperApp",
    "initialize_components",
    "startup_sequence",
    "shutdown_sequence",
]


if __name__ == "__main__":
    main()
