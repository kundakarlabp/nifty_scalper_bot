"""Compatibility shim for legacy premium-decay backtest imports."""

from __future__ import annotations

from nifty_scalper_bot.backtesting.premium_decay_backtest import *  # noqa: F401,F403

__all__ = ["BacktestBar", "run_premium_decay_backtest"]
