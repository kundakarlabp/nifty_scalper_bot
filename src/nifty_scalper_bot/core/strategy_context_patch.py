"""Route legacy strategy context entry points to the canonical builder."""

from __future__ import annotations

import importlib

from nifty_scalper_bot.strategies.context_builder import build_strategy_history_context


def apply_patches() -> None:
    """Install the canonical context builder at existing public call sites."""

    signal_generator = importlib.import_module("nifty_scalper_bot.strategies.signal_generator")
    signal_generator.build_strategy_history_context = build_strategy_history_context

    strategy_manager = importlib.import_module("nifty_scalper_bot.core.strategy_manager")
    strategy_manager.build_strategy_history_context = build_strategy_history_context


__all__ = ["apply_patches"]
