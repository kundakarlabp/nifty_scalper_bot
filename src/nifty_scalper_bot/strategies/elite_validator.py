"""Validation for the canonical elite-strategy configuration model."""

from __future__ import annotations

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategiesSettings,
    EliteStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class EliteConfigError(ValueError):
    """Raised when elite strategy configuration is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EliteConfigError(message)


def _validate_base(name: str, cfg: EliteStrategyConfig) -> None:
    _require(
        0.0 <= float(cfg.min_confidence) <= 100.0,
        f"{name}.min_confidence must be 0..100",
    )
    _require(
        float(cfg.cooldown_seconds) >= 0.0,
        f"{name}.cooldown_seconds must be >= 0",
    )
    _require(int(cfg.quantity) > 0, f"{name}.quantity must be > 0")


def validate_elite_config(elite: EliteStrategiesSettings) -> None:
    """Validate only fields owned by the current dataclass configuration model."""
    _require(
        int(elite.max_concurrent_strategies) >= 1,
        "max_concurrent_strategies must be >= 1",
    )
    _require(float(elite.position_size_pct) >= 0.0, "position_size_pct must be >= 0")

    configs = {
        "smc": elite.smc,
        "vwap": elite.vwap,
        "oi_max_pain": elite.oi_max_pain,
        "gamma_scalping": elite.gamma_scalping,
        "tuesday_gamma_buyer": elite.tuesday_gamma_buyer,
        "cpr": elite.cpr,
        "order_flow": elite.order_flow,
        "bb_squeeze": elite.bb_squeeze,
        "rsi_div": elite.rsi_div,
        "orb": elite.orb,
        "straddle": elite.straddle,
    }
    for name, cfg in configs.items():
        _validate_base(name, cfg)

    _require(
        elite.smc.sweep_distance_points > 0,
        "smc.sweep_distance_points must be > 0",
    )
    _require(elite.smc.volume_spike_mult > 0, "smc.volume_spike_mult must be > 0")
    _require(elite.vwap.ema_period >= 2, "vwap.ema_period must be >= 2")
    _require(elite.vwap.proximity_pct > 0, "vwap.proximity_pct must be > 0")
    _require(
        elite.oi_max_pain.min_deviation_pct >= 0,
        "oi_max_pain.min_deviation_pct must be >= 0",
    )
    _require(
        elite.gamma_scalping.min_gamma > 0,
        "gamma_scalping.min_gamma must be > 0",
    )
    _require(
        elite.tuesday_gamma_buyer.atr_multiplier > 0,
        "tuesday_gamma_buyer.atr_multiplier must be > 0",
    )
    _require(
        elite.tuesday_gamma_buyer.target_multiplier > 0,
        "tuesday_gamma_buyer.target_multiplier must be > 0",
    )
    _require(elite.cpr.narrow_cpr_threshold > 0, "cpr.narrow_cpr_threshold must be > 0")
    _require(
        elite.order_flow.imbalance_ratio_min >= 1.0,
        "order_flow.imbalance_ratio_min must be >= 1",
    )
    _require(
        0.0 < elite.order_flow.large_order_threshold_pct <= 100.0,
        "order_flow.large_order_threshold_pct must be in (0, 100]",
    )
    _require(
        elite.bb_squeeze.squeeze_threshold_pct > 0,
        "bb_squeeze.squeeze_threshold_pct must be > 0",
    )
    _require(elite.rsi_div.rsi_period >= 2, "rsi_div.rsi_period must be >= 2")
    _require(elite.orb.orb_minutes >= 1, "orb.orb_minutes must be >= 1")
    _require(
        0.0 <= elite.straddle.adx_threshold <= 100.0,
        "straddle.adx_threshold must be 0..100",
    )
    _require(elite.straddle.min_iv >= 0.0, "straddle.min_iv must be >= 0")

    LOGGER.info(
        "Elite config validated successfully",
        extra={
            "event": "elite_validated",
            "max_concurrent": elite.max_concurrent_strategies,
        },
    )
