"""Configuration dataclasses for elite strategy package."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EliteStrategyConfig:
    """Base configuration shared by elite strategies."""

    enabled: bool = True
    min_confidence: float = 70.0
    cooldown_seconds: float = 60.0
    
    # ✅ FIX: Add this field so strategies can read self.config.symbol
    symbol: str | None = None 

    def __post_init__(self) -> None:
        """Validate configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any configuration value is outside expected bounds.
        """
        if self.min_confidence < 0 or self.min_confidence > 100:
            raise ValueError("min_confidence must be between 0 and 100")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")


@dataclass(slots=True)
class SMCStrategyConfig(EliteStrategyConfig):
    """Configuration for liquidity sweep detection strategy."""

    equal_level_tolerance_pct: float = 2.0
    sweep_distance_points: float = 30.0
    volume_spike_mult: float = 2.0
    order_block_volume_mult: float = 1.5

    def __post_init__(self) -> None:
        """Validate configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any configuration value is outside expected bounds.
        """

        EliteStrategyConfig.__post_init__(self)
        if self.equal_level_tolerance_pct < 0:
            raise ValueError("equal_level_tolerance_pct must be non-negative")
        if self.sweep_distance_points < 0:
            raise ValueError("sweep_distance_points must be non-negative")
        if self.volume_spike_mult < 0:
            raise ValueError("volume_spike_mult must be non-negative")
        if self.order_block_volume_mult < 0:
            raise ValueError("order_block_volume_mult must be non-negative")


@dataclass(slots=True)
class VWAPProStrategyConfig(EliteStrategyConfig):
    """Configuration for VWAP mean reversion strategy."""

    stddev_mult: float = 1.5
    rsi_oversold: float = 35.0
    rsi_overbought: float = 65.0
    volume_spike: float = 1.8
    max_holding_minutes: float = 15.0

    def __post_init__(self) -> None:
        """Validate configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any configuration value is outside expected bounds.
        """

        EliteStrategyConfig.__post_init__(self)
        if self.stddev_mult <= 0:
            raise ValueError("stddev_mult must be positive")
        if not 0 <= self.rsi_oversold <= 100:
            raise ValueError("rsi_oversold must be between 0 and 100")
        if not 0 <= self.rsi_overbought <= 100:
            raise ValueError("rsi_overbought must be between 0 and 100")
        if self.volume_spike < 0:
            raise ValueError("volume_spike must be non-negative")
        if self.max_holding_minutes < 0:
            raise ValueError("max_holding_minutes must be non-negative")


@dataclass(slots=True)
class OIMaxPainStrategyConfig(EliteStrategyConfig):
    """Configuration for OI max pain strategy."""

    fetch_interval_min: float = 3.0
    min_distance_points: float = 50.0
    pcr_bullish: float = 1.2
    pcr_bearish: float = 0.8
    exit_time_minutes: float = 900.0

    def __post_init__(self) -> None:
        """Validate configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any configuration value is outside expected bounds.
        """

        EliteStrategyConfig.__post_init__(self)
        if self.fetch_interval_min < 0:
            raise ValueError("fetch_interval_min must be non-negative")
        if self.min_distance_points < 0:
            raise ValueError("min_distance_points must be non-negative")
        if self.exit_time_minutes < 0:
            raise ValueError("exit_time_minutes must be non-negative")


@dataclass(slots=True)
class GammaScalpingStrategyConfig(EliteStrategyConfig):
    """Configuration for gamma scalping strategy."""

    hedge_trigger_points: float = 20.0
    delta_threshold: float = 0.2
    target_profit: float = 2500.0
    max_theta_loss: float = 500.0
    exit_time_minutes: float = 840.0

    def __post_init__(self) -> None:
        """Validate configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any configuration value is outside expected bounds.
        """

        EliteStrategyConfig.__post_init__(self)
        if self.hedge_trigger_points < 0:
            raise ValueError("hedge_trigger_points must be non-negative")
        if self.delta_threshold < 0:
            raise ValueError("delta_threshold must be non-negative")
        if self.target_profit < 0:
            raise ValueError("target_profit must be non-negative")
        if self.max_theta_loss < 0:
            raise ValueError("max_theta_loss must be non-negative")
        if self.exit_time_minutes < 0:
            raise ValueError("exit_time_minutes must be non-negative")


@dataclass(slots=True)
class CPRBreakoutStrategyConfig(EliteStrategyConfig):
    """Configuration for CPR breakout strategy."""

    narrow_threshold_pct: float = 0.3
    volume_mult: float = 2.0
    gap_threshold_points: float = 30.0
    breakout_window_start: float = 9 * 60 + 30
    breakout_window_end: float = 10 * 60

    def __post_init__(self) -> None:
        """Validate configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any configuration value is outside expected bounds.
        """

        EliteStrategyConfig.__post_init__(self)
        if self.narrow_threshold_pct < 0:
            raise ValueError("narrow_threshold_pct must be non-negative")
        if self.volume_mult < 0:
            raise ValueError("volume_mult must be non-negative")
        if self.gap_threshold_points < 0:
            raise ValueError("gap_threshold_points must be non-negative")
        if self.breakout_window_end < self.breakout_window_start:
            raise ValueError("breakout window end must be after start")


@dataclass(slots=True)
class OrderFlowStrategyConfig(EliteStrategyConfig):
    """Configuration for order flow imbalance strategy."""

    imbalance_ratio_buy: float = 2.5
    imbalance_ratio_sell: float = 0.4
    large_order_pct: float = 15.0
    persistence_seconds: float = 5.0
    target_points: float = 25.0

    def __post_init__(self) -> None:
        """Validate configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any configuration value is outside expected bounds.
        """

        EliteStrategyConfig.__post_init__(self)
        if self.imbalance_ratio_buy < 0:
            raise ValueError("imbalance_ratio_buy must be non-negative")
        if self.imbalance_ratio_sell < 0:
            raise ValueError("imbalance_ratio_sell must be non-negative")
        if self.large_order_pct < 0:
            raise ValueError("large_order_pct must be non-negative")
        if self.persistence_seconds < 0:
            raise ValueError("persistence_seconds must be non-negative")
        if self.target_points < 0:
            raise ValueError("target_points must be non-negative")


@dataclass(slots=True)
class BBSqueezeStrategyConfig(EliteStrategyConfig):
    """Configuration for Bollinger Band squeeze strategy."""

    period: int = 20
    stddev: float = 2.0
    bandwidth_threshold: float = 0.4
    atr_percentile: float = 20.0
    volume_breakout: float = 2.5

    def __post_init__(self) -> None:
        """Validate configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any configuration value is outside expected bounds.
        """

        EliteStrategyConfig.__post_init__(self)
        if self.period <= 0:
            raise ValueError("period must be positive")
        if self.stddev <= 0:
            raise ValueError("stddev must be positive")
        if self.bandwidth_threshold < 0:
            raise ValueError("bandwidth_threshold must be non-negative")
        if self.atr_percentile < 0:
            raise ValueError("atr_percentile must be non-negative")
        if self.volume_breakout < 0:
            raise ValueError("volume_breakout must be non-negative")


@dataclass(slots=True)
class RSIDivergenceStrategyConfig(EliteStrategyConfig):
    """Configuration for RSI divergence strategy."""

    period: int = 14
    oversold: float = 35.0
    overbought: float = 65.0
    confirmation_volume: float = 1.3
    swing_lookback: int = 10

    def __post_init__(self) -> None:
        """Validate configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any configuration value is outside expected bounds.
        """

        EliteStrategyConfig.__post_init__(self)
        if self.period <= 0:
            raise ValueError("period must be positive")
        if not 0 <= self.oversold <= 100:
            raise ValueError("oversold must be between 0 and 100")
        if not 0 <= self.overbought <= 100:
            raise ValueError("overbought must be between 0 and 100")
        if self.confirmation_volume < 0:
            raise ValueError("confirmation_volume must be non-negative")
        if self.swing_lookback <= 0:
            raise ValueError("swing_lookback must be positive")


@dataclass(slots=True)
class ORBProStrategyConfig(EliteStrategyConfig):
    """Configuration for opening range breakout strategy."""

    duration_min: float = 15.0
    narrow_threshold_points: float = 50.0
    volume_mult: float = 2.0
    close_strength_pct: float = 70.0
    invalid_time_minutes: float = 690.0

    def __post_init__(self) -> None:
        """Validate configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any configuration value is outside expected bounds.
        """

        EliteStrategyConfig.__post_init__(self)
        if self.duration_min < 0:
            raise ValueError("duration_min must be non-negative")
        if self.narrow_threshold_points < 0:
            raise ValueError("narrow_threshold_points must be non-negative")
        if self.volume_mult < 0:
            raise ValueError("volume_mult must be non-negative")
        if not 0 <= self.close_strength_pct <= 100:
            raise ValueError("close_strength_pct must be between 0 and 100")
        if self.invalid_time_minutes < 0:
            raise ValueError("invalid_time_minutes must be non-negative")


@dataclass(slots=True)
class StraddleThetaStrategyConfig(EliteStrategyConfig):
    """Configuration for ATM straddle theta strategy."""

    vix_threshold: float = 13.0
    entry_time_minutes: float = 9 * 60 + 25
    profit_target_pct: float = 65.0
    max_spot_move_points: float = 150.0
    exit_time_minutes: float = 900.0

    def __post_init__(self) -> None:
        """Validate configuration values.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any configuration value is outside expected bounds.
        """

        EliteStrategyConfig.__post_init__(self)
        if self.vix_threshold < 0:
            raise ValueError("vix_threshold must be non-negative")
        if self.profit_target_pct < 0:
            raise ValueError("profit_target_pct must be non-negative")
        if self.max_spot_move_points < 0:
            raise ValueError("max_spot_move_points must be non-negative")
        if self.entry_time_minutes < 0:
            raise ValueError("entry_time_minutes must be non-negative")
        if self.exit_time_minutes < 0:
            raise ValueError("exit_time_minutes must be non-negative")


@dataclass(slots=True)
class EliteStrategiesSettings:
    """Aggregate configuration for the elite strategies bundle."""

    enabled: bool = True
    max_concurrent_strategies: int = 3
    position_size_pct: float = 2.0
    smc: SMCStrategyConfig = field(default_factory=SMCStrategyConfig)
    vwap: VWAPProStrategyConfig = field(default_factory=VWAPProStrategyConfig)
    oi_max_pain: OIMaxPainStrategyConfig = field(
        default_factory=OIMaxPainStrategyConfig
    )
    gamma: GammaScalpingStrategyConfig = field(
        default_factory=GammaScalpingStrategyConfig
    )
    cpr: CPRBreakoutStrategyConfig = field(default_factory=CPRBreakoutStrategyConfig)
    order_flow: OrderFlowStrategyConfig = field(default_factory=OrderFlowStrategyConfig)
    bb_squeeze: BBSqueezeStrategyConfig = field(default_factory=BBSqueezeStrategyConfig)
    rsi_div: RSIDivergenceStrategyConfig = field(
        default_factory=RSIDivergenceStrategyConfig
    )
    orb: ORBProStrategyConfig = field(default_factory=ORBProStrategyConfig)
    straddle: StraddleThetaStrategyConfig = field(
        default_factory=StraddleThetaStrategyConfig
    )

    def __post_init__(self) -> None:
        """Validate aggregate settings.

        Args:
            None.

        Returns:
            None.

        Raises:
            ValueError: If any configuration value is outside expected bounds.
        """
        if self.max_concurrent_strategies <= 0:
            raise ValueError("max_concurrent_strategies must be positive")
        if self.position_size_pct < 0:
            raise ValueError("position_size_pct must be non-negative")


__all__ = [
    "EliteStrategyConfig",
    "SMCStrategyConfig",
    "VWAPProStrategyConfig",
    "OIMaxPainStrategyConfig",
    "GammaScalpingStrategyConfig",
    "CPRBreakoutStrategyConfig",
    "OrderFlowStrategyConfig",
    "BBSqueezeStrategyConfig",
    "RSIDivergenceStrategyConfig",
    "ORBProStrategyConfig",
    "StraddleThetaStrategyConfig",
    "EliteStrategiesSettings",
]
