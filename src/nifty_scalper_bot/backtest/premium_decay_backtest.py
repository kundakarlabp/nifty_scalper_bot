"""Compatibility shim for legacy premium-decay backtest imports."""

from __future__ import annotations

from nifty_scalper_bot.backtesting.premium_decay_backtest import *  # noqa: F401,F403
from nifty_scalper_bot.backtesting.premium_decay_backtest import __all__ as _CANONICAL_ALL

__all__ = _CANONICAL_ALL
