"""
Base abstractions and helpers for elite strategies.
Production-Grade: Optimized Dispatch & Cross-Component Compatibility.
Fixes: AttributeError 'strategy_name' in Signal execution logic.
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

# Initialize structured logger for production tracking
LOGGER = get_logger(__name__)


@dataclass(slots=True)
class EliteSignal:
    """
    High-performance container for elite strategy signal outputs.
    Using slots=True reduces memory overhead by ~40-50% for high-frequency ticks.
    """

    symbol: str
    signal: str  
    confidence: float
    entry_price: float
    stop_loss: float | None
    target: float | None
    quantity: int = 1
    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Backwards compatibility fields for the execution engine
    take_profit_1: float | None = None 
    take_profit_2: float | None = None
    side: str = field(init=False) 
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Normalize fields for core engine compatibility."""
        object.__setattr__(self, 'side', self.signal)
        if self.target and not self.take_profit_1:
             object.__setattr__(self, 'take_profit_1', self.target)

    def to_payload(self) -> dict[str, Any]:
        """Return serializable representation for telemetry."""
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
    World-Class Abstract Base Class for Elite Strategies.
    Implements a robust 'Bridge' pattern to resolve abstract instantiation errors.
    """

    def __init__(self, config: EliteStrategyConfig, indicator_engine: Any) -> None:
        """
        Initializes strategy and satisfies parent 'Strategy' requirements.
        """
        # Auto-detect name from the config class for zero-config naming
        name = config.__class__.__name__.replace("Config", "").replace("Strategy", "")
        
        # ✅ FIX: Map dataclass config to 'parameters' dict for parent __init__
        params = asdict(config) if hasattr(config, "__dataclass_fields__") else {}
        super().__init__(name=name, parameters=params)
        
        self._config = config
        self._indicator_engine = indicator_engine
        self._last_signal_at: datetime | None = None
        self._signals_generated = 0
        self._last_signal: EliteSignal | None = None

        # Detect signature once at startup to optimize per-tick dispatch
        sig = inspect.signature(self._evaluate_signal)
        self._is_legacy_signature = len(sig.parameters) == 0

    def generate_signal(
        self, 
        symbol: str, 
        indicators: Mapping[str, Any], 
        current_price: float, 
        position: Any | None = None
    ) -> Signal | None:
        """
        ✅ THE BRIDGE: Satisfies core engine abstract requirement.
        Resolves the 'Can't instantiate abstract class' error.
        """
        if not self._config.enabled:
            return None

        # Route the core engine call to the specific elite logic
        elite_signal = self._evaluate_signal(
            symbol=symbol, 
            indicators=indicators, 
            current_price=current_price, 
            position=position
        )

        return self._process_signal(elite_signal) if elite_signal else None

    def evaluate(self) -> Signal | None:
        """Fallback entry point for polling-based execution runners."""
        if not self._config.enabled:
            return None

        # Enforcement of strategy-level cooldowns
        if self._last_signal_at:
            elapsed = (datetime.now(timezone.utc) - self._last_signal_at).total_seconds()
            if elapsed < self._config.cooldown_seconds:
                return None

        try:
            if self._is_legacy_signature:
                elite_signal = self._evaluate_signal() # type: ignore
                return self._process_signal(elite_signal) if elite_signal else None
            
            # Modern Push path: resolve LTP and indicators for the bridge
            symbol = getattr(self._config, "symbol", None)
            if not symbol: return None
            
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
        """Abstract implementation hook for specific strategy files."""
        raise NotImplementedError("Strategy must implement _evaluate_signal")

    def _process_signal(self, elite_signal: EliteSignal) -> Signal:
        """
        Converts EliteSignal to core Signal and patches missing attributes.
        Fixes: 'Signal' object has no attribute 'strategy_name' crash.
        """
        self._last_signal_at = elite_signal.timestamp
        self._last_signal = elite_signal
        self._signals_generated += 1

        # Standard Core Signal
        core_signal = Signal(
            action=elite_signal.signal,
            symbol=elite_signal.symbol,
            confidence=elite_signal.confidence,
            price=elite_signal.entry_price,
            tag=f"{self.name}",
            stop_loss=elite_signal.stop_loss,
            take_profit=elite_signal.target,
            metadata=elite_signal.metadata.copy(),
        )

        # ✅ CRITICAL FIX: Inject strategy_name and quantity directly into the object
        # This allows the StrategyRunner to access them without an AttributeError.
        setattr(core_signal, "strategy_name", self.name)
        setattr(core_signal, "quantity", elite_signal.quantity)
        
        # Enrich metadata for logging transparency
        core_signal.metadata.update({
            "strategy": self.name,
            "mode": "Push",
            "quantity": elite_signal.quantity
        })

        return core_signal

    def get_stats(self) -> dict[str, Any]:
        """Return statistics for dashboard and health metrics."""
        last_payload = self._last_signal.to_payload() if self._last_signal else None
        return {
            "strategy": self.name,
            "enabled": self._config.enabled,
            "signals_generated": self._signals_generated,
            "last_signal": last_payload,
            "mode": "Legacy" if self._is_legacy_signature else "Push"
        }

    def get_required_indicators(self) -> Set[str]:
        """Override to declare indicator dependencies."""
        return set()

    @property
    def config(self) -> EliteStrategyConfig:
        return self._config


__all__ = ["EliteSignal", "EliteStrategy"]
