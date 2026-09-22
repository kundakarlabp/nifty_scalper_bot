"""Compatibility layer importing the core market regime utilities."""

from __future__ import annotations

from nifty_scalper_bot.core.market_regime import (
    MarketRegimeDetector,
    RegimeParameters,
    RegimeSnapshot,
)

__all__ = ["MarketRegimeDetector", "RegimeParameters", "RegimeSnapshot"]
