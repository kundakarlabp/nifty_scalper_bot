"""
Base abstractions and helpers for elite strategies.
Production-Grade: Optimized Dispatch (Zero-Reflection Runtime) & Type Safety.
Resolves: Can't instantiate abstract class errors.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
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
    Implements a Hybrid Architecture: Supports both Legacy (Pull) and Modern (Push) logic.
    """

    def __init__(self, config: EliteStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy logic.
        """
        # Auto-detect name from config class (e.g. SMCStrategyConfig -> SMC)
        name = config.__class__.__name__.replace("Config", "").replace("Strategy", "")
        super().__init__(name=name)
        
        self._config = config
        self._indicator_engine = indicator_engine
        self._last_signal_at: datetime | None = None
        self._signals_generated = 0
        self._last_signal: EliteSignal | None = None

        # PERFORMANCE OPTIMIZATION:
        # Inspect the signature ONCE at startup to decide execution path.
        sig = inspect.signature(self._evaluate_signal)
        # Legacy: _evaluate_signal(self) -> 0 params
        # Modern: _evaluate_signal(self, symbol, indicators, ...) -> >0 params
        self._is_legacy_signature = len(sig.parameters) == 0
        
        if self._is_legacy_signature:
            LOGGER.debug(f"⚠️ {self.name}: Running in Legacy Mode (Pull-Based)")
        else:
            LOGGER.debug(f"🚀 {self.name}: Running in Modern Mode (Push-Based)")

    def get_required_indicators(self) -> Set[str]:
        """
        Override this in subclasses to declare needed data.
        """
        return set()

    def generate_signal(
        self, 
        symbol: str, 
        indicators: Mapping[str, Any], 
        current_price: float, 
        position: Any | None = None
    ) -> Signal | None:
        """
        ✅ THE BRIDGE: Satisfies the abstract requirement of the parent 'Strategy' class.
        This resolves the 'Can't instantiate abstract class' error in your logs.
        """
        if not self._config.enabled:
            return None

        # Dispatch to the specific evaluation logic implemented in child classes
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
        """
        Main entry point called by the Strategy Runner loop.
        Acts as a 'Bridge' to handle data fetching automatically if needed.
        """
        if not self._config.enabled:
            return None

        # Cooldown check
        if self._last_signal_at:
            elapsed = (datetime.now(timezone.utc) - self._last_signal_at).total_seconds()
            if elapsed < self._config.cooldown_seconds:
                return None

        try:
            # Dispatch based on detected signature
            if self._is_legacy_signature:
                elite_signal = self._evaluate_signal() # type: ignore
                if elite_signal:
                    return self._process_signal(elite_signal)
            else:
                # Direct bridge to the modern signature with pre-fetched data
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
        """
        Abstract method to implement trading logic.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Strategy must implement _evaluate_signal")

    def _process_signal(self, signal: EliteSignal) -> Signal:
        """
        Converts internal EliteSignal to the bot's standard Signal format.
        """
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
        """Return diagnostic statistics for the strategy."""
        last_payload: dict[str, Any] | None = None
        if self._last_signal is not None:
            last_payload = self._last_signal.to_payload()
            
        return {
            "strategy": self.name,
            "enabled": self._config.enabled,
            "signals_generated": self._signals_generated,
            "last_signal": last_payload,
            "mode": "Legacy" if self._is_legacy_signature else "Push"
        }

    @property
    def config(self) -> EliteStrategyConfig:
        """Return strategy configuration reference."""
        return self._config


__all__ = ["EliteSignal", "EliteStrategy"]
