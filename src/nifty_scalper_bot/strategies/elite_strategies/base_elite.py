"""
Base abstractions and helpers for elite strategies.
Production-Grade: Optimized Dispatch (Zero-Reflection Runtime) & Type Safety.
Fixed: Resolves Abstract Method and Constructor Parameter mismatches.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Set, Optional, Mapping

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategyConfig,
)
from nifty_scalper_bot.strategies.signal_generator import Signal, Strategy
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


@dataclass(slots=True)
class EliteSignal:
    """
    Container for elite strategy signal output.
    Optimized with __slots__ for reduced memory footprint.
    """

    symbol: str
    signal: str  # Standardized name (was 'side')
    confidence: float
    entry_price: float
    stop_loss: float | None
    target: float | None
    quantity: int = 1
    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Backwards compatibility fields for execution engine
    take_profit_1: float | None = None 
    take_profit_2: float | None = None
    side: str = field(init=False) # Computed property for legacy support
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Normalize fields for backward compatibility."""
        # 1. Alias 'signal' to 'side' so old code works
        object.__setattr__(self, 'side', self.signal)
        
        # 2. Map 'target' to 'take_profit_1' if missing
        if self.target and not self.take_profit_1:
             object.__setattr__(self, 'take_profit_1', self.target)

    def to_payload(self) -> dict[str, Any]:
        """Return serializable representation for telemetry/logs."""
        return {
            "symbol": self.symbol,
            "side": self.signal,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "quantity": self.quantity,
            "strategy": self.strategy_name,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class EliteStrategy(Strategy):
    """
    Abstract base class for all Elite Strategies.
    Hybrid Architecture: Corrected to support Parent Class parameter requirements.
    """

    def __init__(self, config: EliteStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy logic and synchronize with core Strategy parent.
        """
        # 1. Detect name from config class
        name = config.__class__.__name__.replace("Config", "").replace("Strategy", "")
        
        # 2. ✅ FIX: Satisfy Strategy.__init__ requirement for 'parameters'
        # We convert the config dataclass to a dictionary to pass to the parent
        params = asdict(config) if hasattr(config, "__dataclass_fields__") else {}
        
        # 3. Call parent with both Name and Parameters
        super().__init__(name=name, parameters=params)
        
        self._config = config
        self._indicator_engine = indicator_engine
        self._last_signal_at: datetime | None = None
        self._signals_generated = 0
        self._last_signal: EliteSignal | None = None

        # PERFORMANCE OPTIMIZATION:
        # Inspect the signature ONCE at startup to decide execution path.
        sig = inspect.signature(self._evaluate_signal)
        self._is_legacy_signature = len(sig.parameters) == 0
        
        if self._is_legacy_signature:
            LOGGER.debug(f"⚠️ {self.name}: Running in Legacy Mode (Pull-Based)")
        else:
            LOGGER.debug(f"🚀 {self.name}: Running in Modern Mode (Push-Based)")

    def get_required_indicators(self) -> Set[str]:
        """Override this in subclasses to declare needed data."""
        return set()

    def generate_signal(
        self, 
        symbol: str, 
        indicators: Mapping[str, Any], 
        current_price: float, 
        position: Any | None = None
    ) -> Signal | None:
        """
        ✅ THE BRIDGE: Implements abstract method required by the parent 'Strategy' class.
        Resolves: 'Can't instantiate abstract class' error in logs.
        """
        if not self._config.enabled:
            return None

        # Route the call to our specific evaluation logic
        elite_signal = self._evaluate_signal(
            symbol=symbol, 
            indicators=indicators, 
            current_price=current_price, 
            position=position
        )

        if elite_signal:
            return self._process_signal(elite_signal)
            
        return None

    def evaluate(self) -> Signal | None:
        """Main entry point for polling-based runners."""
        if not self._config.enabled:
            return None

        # Cooldown check
        if self._last_signal_at:
            elapsed = (datetime.now(timezone.utc) - self._last_signal_at).total_seconds()
            if elapsed < self._config.cooldown_seconds:
                return None

        try:
            if self._is_legacy_signature:
                elite_signal = self._evaluate_signal() # type: ignore
                if elite_signal:
                    return self._process_signal(elite_signal)
            else:
                symbol = getattr(self._config, "symbol", None)
                if not symbol:
                    return None
                
                req_inds = self.get_required_indicators()
                indicators = self._indicator_engine.get_indicators(symbol, list(req_inds))
                ltp = float(indicators.get("ltp") or 0.0)
                
                return self.generate_signal(symbol, indicators, ltp)

        except Exception as e:
            LOGGER.error(f"Error evaluating {self.name}: {e}", exc_info=True)
        
        return None

    def _evaluate_signal(
        self, 
        symbol: str = "", 
        indicators: Dict[str, Any] = {}, 
        current_price: float = 0.0, 
        position: Any | None = None
    ) -> EliteSignal | None:
        """Abstract logic hook for child strategies."""
        raise NotImplementedError("Strategy must implement _evaluate_signal")

    def _process_signal(self, signal: EliteSignal) -> Signal:
        """Converts internal EliteSignal to core Signal format."""
        self._last_signal_at = signal.timestamp
        self._last_signal = signal
        self._signals_generated += 1

        metadata = signal.metadata.copy()
        metadata.update({
            "strategy": self.name,
            "mode": "Legacy" if self._is_legacy_signature else "Push",
            "quantity": signal.quantity
        })

        return Signal(
            action=signal.signal,
            symbol=signal.symbol,
            confidence=signal.confidence,
            price=signal.entry_price,
            tag=f"{self.name}",
            stop_loss=signal.stop_loss,
            take_profit=signal.target,
            metadata=metadata,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic statistics."""
        last_payload = self._last_signal.to_payload() if self._last_signal else None
        return {
            "strategy": self.name,
            "enabled": self._config.enabled,
            "signals_generated": self._signals_generated,
            "last_signal": last_payload,
            "mode": "Legacy" if self._is_legacy_signature else "Push"
        }

    @property
    def config(self) -> EliteStrategyConfig:
        return self._config


__all__ = ["EliteSignal", "EliteStrategy"]
