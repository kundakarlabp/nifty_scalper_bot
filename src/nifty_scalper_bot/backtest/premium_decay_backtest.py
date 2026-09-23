"""Compatibility shim for legacy premium-decay backtest imports."""

from __future__ import annotations

from nifty_scalper_bot.backtesting.premium_decay_backtest import (
    BacktestBar,
    run_premium_decay_backtest,
)

__all__ = ["BacktestBar", "run_premium_decay_backtest"]
