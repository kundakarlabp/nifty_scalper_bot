"""Compatibility shim for legacy ``nifty_scalper_bot.backtest.replay`` imports."""

from __future__ import annotations

from nifty_scalper_bot.backtesting.replay import *  # noqa: F401,F403
from nifty_scalper_bot.backtesting.replay import __all__ as _CANONICAL_ALL

__all__ = _CANONICAL_ALL
