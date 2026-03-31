"""Explicit elite strategy class registry."""

from __future__ import annotations

from nifty_scalper_bot.strategies.elite_tuesday_gamma_buyer import EliteTuesdayGammaBuyer

ELITE_STRATEGY_REGISTRY = {
    'tuesday_gamma_buyer': EliteTuesdayGammaBuyer,
}

__all__ = ['ELITE_STRATEGY_REGISTRY', 'EliteTuesdayGammaBuyer']
