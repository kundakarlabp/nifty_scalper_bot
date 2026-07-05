"""Route legacy strategy context entry points to the canonical builder."""

from __future__ import annotations

from nifty_scalper_bot.strategies.context_builder import build_strategy_history_context


def apply_patches() -> None:
    """Install the canonical context builder at existing public call sites."""

    from nifty_scalper_bot.strategies import signal_generator

    signal_generator.build_strategy_history_context = build_strategy_history_context

    try:
        from nifty_scalper_bot.core import strategy_manager
    except Exception:
        return
    strategy_manager.build_strategy_history_context = build_strategy_history_context


__all__ = ["apply_patches"]
