"""
Configuration dataclasses for elite strategy package.
Production-Grade: Full implementation with validation, slots, and self-healing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class EliteStrategyConfig:
    """Base configuration shared by all elite strategies."""

    enabled: bool = True
    min_confidence: float = 70.0
    cooldown_seconds: float = 60.0
    
    # Context Injection Fields: Populated at runtime by the Strategy Engine
    symbol: Optional[str] = None 
    quantity: int = 1

    def __post_init__(self) -> None:
        """Validate and self-heal configuration values."""
        # Confidence must be a percentage
        if self.min_confidence < 0:
            self.min_confidence = 0.0
        elif self.min_confidence > 100:
            self.min_confidence = 100.0
            
        if self.cooldown_seconds < 0:
            self.cooldown_seconds = 0.0
            
        if self.quantity <= 0:
            # Prevent zero-division or zero-order errors
            self.quantity = 1


# --- Strategy Specific Configurations ---

@dataclass(slots=True)
class SMCStrategyConfig(EliteStrategyConfig):
    """SMC Liquidity: Pivot and Sweep thresholds."""
    sweep_distance_points: float = 30.0
    volume_spike_mult: float = 2.0


@dataclass(slots=True)
class VWAPProStrategyConfig(EliteStrategyConfig):
    """VWAP Pro: Trend and Pullback thresholds."""
    ema_period: int = 50
    proximity_pct: float = 0.15


@dataclass(slots=True)
class OIMaxPainStrategyConfig(EliteStrategyConfig):
    """OI Max Pain: Mean reversion thresholds."""
    min_deviation_pct: float = 0.5


@dataclass(slots=True)
class GammaScalpingStrategyConfig(EliteStrategyConfig):
    """Gamma Scalper: Acceleration thresholds."""
    min_gamma: float = 0.0005


@dataclass(slots=True)
class CPRBreakoutStrategyConfig(EliteStrategyConfig):
    """CPR Breakout: Pivot width thresholds."""
    narrow_cpr_threshold: float = 0.25


@dataclass(slots=True)
class OrderFlowStrategyConfig(EliteStrategyConfig):
    """Order Flow: Depth imbalance thresholds."""
    imbalance_ratio_min: float = 1.5
    large_order_threshold_pct: float = 30.0


@dataclass(slots=True)
class BBSqueezeStrategyConfig(EliteStrategyConfig):
    """BB Squeeze: Volatility thresholds."""
    squeeze_threshold_pct: float = 0.5


@dataclass(slots=True)
class RSIDivergenceStrategyConfig(EliteStrategyConfig):
    """RSI Divergence: Oscillator periods."""
    rsi_period: int = 14


@dataclass(slots=True)
class ORBProStrategyConfig(EliteStrategyConfig):
    """ORB Pro: Time-based range thresholds."""
    orb_minutes: int = 15


@dataclass(slots=True)
class StraddleThetaStrategyConfig(EliteStrategyConfig):
    """Straddle/Theta: Option Greek thresholds."""
    adx_threshold: float = 25.0
    min_iv: float = 12.0


@dataclass(slots=True)
class EliteStrategiesSettings:
    """
    Aggregate configuration for the entire elite strategy suite.
    Production-Grade: Field names are mapped 1:1 to Builder Registry.
    """
    
    max_concurrent_strategies: int = 3
    position_size_pct: float = 2.0
    
    # Strategy-Specific Config Objects
    smc: SMCStrategyConfig = field(default_factory=SMCStrategyConfig)
    vwap: VWAPProStrategyConfig = field(default_factory=VWAPProStrategyConfig)
    oi_max_pain: OIMaxPainStrategyConfig = field(default_factory=OIMaxPainStrategyConfig)
    gamma_scalping: GammaScalpingStrategyConfig = field(default_factory=GammaScalpingStrategyConfig)
    cpr: CPRBreakoutStrategyConfig = field(default_factory=CPRBreakoutStrategyConfig)
    order_flow: OrderFlowStrategyConfig = field(default_factory=OrderFlowStrategyConfig)
    bb_squeeze: BBSqueezeStrategyConfig = field(default_factory=BBSqueezeStrategyConfig)
    rsi_div: RSIDivergenceStrategyConfig = field(default_factory=RSIDivergenceStrategyConfig)
    orb: ORBProStrategyConfig = field(default_factory=ORBProStrategyConfig)
    straddle: StraddleThetaStrategyConfig = field(default_factory=StraddleThetaStrategyConfig)

    def __post_init__(self) -> None:
        """Final aggregate validation."""
        if self.max_concurrent_strategies <= 0:
            self.max_concurrent_strategies = 1
        if self.position_size_pct < 0:
            self.position_size_pct = 0.0


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
