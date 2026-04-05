"""Immutable hard limit definitions."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_day_loss_inr: float
    max_concurrent_positions: int
    max_daily_trades: int
    max_order_notional_inr: float
    consecutive_loss_cooldown: int
    cooldown_minutes: int
    min_option_oi: int
    max_spread_pct: float
    min_option_price: float
    max_option_price: float
    vix_high_threshold: float
    vix_extreme_threshold: float
