"""
Base abstractions and helpers for elite strategies.
Production-Grade: Optimized Dispatch & Attribute Injection.
Fixed: 'Signal' object has no attribute 'strategy_name' crash.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Mapping

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategyConfig,
)
from nifty_scalper_bot.strategies.signal_generator import Signal, Strategy
from nifty_scalper_bot.utils.logging import get_logger

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
    
    # Backwards compatibility fields
    take_profit_1: float | None = None 
    take_profit_2: float | None = None
    side: str = field(init=False) 
    action: str = field(init=False)
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, 'side', self.signal)
        object.__setattr__(self, 'action', self.signal)
        if self.target and not self.take_profit_1:
             object.__setattr__(self, 'take_profit_1', self.target)

    def to_payload(self) -> dict[str, Any]:
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
    World-Class Base Class for Elite Strategies.
    Implements the Bridge Pattern and Signal Patching.
    """

    def __init__(self, config: EliteStrategyConfig, indicator_engine: Any) -> None:
        # Auto-detect name from config class
        name = config.__class__.__name__.replace("Config", "").replace("Strategy", "")
        
        # ✅ FIX 1: Satisfy parent requirements
        params = asdict(config) if hasattr(config, "__dataclass_fields__") else {}
        super().__init__(name=name, parameters=params)
        
        self._config = config
        self._indicator_engine = indicator_engine
        self._last_signal_at: datetime | None = None
        self._signals_generated = 0
        self._last_signal: EliteSignal | None = None

        # Inspect signature once at startup
        sig = inspect.signature(self._evaluate_signal)
        self._is_legacy_signature = len(sig.parameters) == 0
        
        if self._is_legacy_signature:
            LOGGER.debug(f"⚠️ {self.name}: Running in Legacy Mode (Pull-Based)")
        else:
            LOGGER.debug(f"🚀 {self.name}: Running in Modern Mode (Push-Based)")

    def get_required_indicators(self) -> list[str]:
        """Args: none. Returns: list[str]. Raises: Exception."""
        LOGGER.debug('Entered EliteStrategy.get_required_indicators')
        try:
            LOGGER.info(
                'Condition met: using base indicator set',
                extra={'event': 'elite_strategy_required_indicators', 'strategy': self.name},
            )
            return []
        except Exception as exc:
            LOGGER.error('Failure in get_required_indicators: %s', exc, exc_info=exc)
            return []

    def generate_signal(
        self, 
        symbol: str, 
        indicators: Mapping[str, Any], 
        current_price: float, 
        position: Any | None = None
    ) -> Signal | None:
        """Args: symbol, indicators, price, pos. Returns: Signal|None. Raises: Err."""
        LOGGER.debug('Entered EliteStrategy.generate_signal')
        try:
            if not self._config.enabled:
                LOGGER.info(
                    'Condition met: strategy disabled',
                    extra={'event': 'elite_strategy_disabled', 'strategy': self.name},
                )
                return None

            if not symbol:
                LOGGER.info(
                    'Condition met: missing symbol',
                    extra={'event': 'elite_strategy_missing_symbol', 'strategy': self.name},
                )
                return None

            if current_price <= 0:
                LOGGER.info(
                    'Condition met: invalid current price',
                    extra={
                        'event': 'elite_strategy_invalid_price',
                        'strategy': self.name,
                        'price': current_price,
                    },
                )
                return None

            if indicators is None:
                LOGGER.info(
                    'Condition met: missing indicators payload',
                    extra={'event': 'elite_strategy_missing_indicators', 'strategy': self.name},
                )
                return None

            # ✅ Early exit if capital is exhausted (prevents wasted computation)
            if hasattr(self, '_orchestrator') and self._orchestrator:
                if not self._orchestrator._has_capital_headroom_quick():
                    LOGGER.info(
                        'Condition met: capital headroom exhausted',
                        extra={'event': 'elite_strategy_no_capital', 'strategy': self.name},
                    )
                    return None

            indicators_payload: dict[str, Any] = dict(indicators)
            elite_signal = self._evaluate_signal(
                symbol=symbol,
                indicators=indicators_payload,
                current_price=current_price,
                position=position,
            )

            if elite_signal:
                min_conf = max(0.0, min(float(self._config.min_confidence) / 100.0, 1.0))
                if float(elite_signal.confidence) < min_conf:
                    self._no_vote('below_strategy_min_confidence')
                    LOGGER.info(
                        'Condition met: below strategy min confidence',
                        extra={'event': 'elite_strategy_below_min_conf', 'strategy': self.name},
                    )
                    return None
                LOGGER.info(
                    'Condition met: elite signal generated',
                    extra={'event': 'elite_strategy_signal', 'strategy': self.name},
                )
                return self._process_signal(elite_signal)

            LOGGER.debug(
                'No signal generated',
                extra={'event': 'elite_strategy_no_signal', 'strategy': self.name},
            )
        except Exception as exc:
            LOGGER.error('Failure in generate_signal: %s', exc, exc_info=exc)

        return None

    def evaluate(self) -> Signal | None:
        """Args: none. Returns: Signal|None. Raises: Exception."""
        LOGGER.debug('Entered EliteStrategy.evaluate')
        try:
            if not self._config.enabled:
                LOGGER.info(
                    'Condition met: strategy disabled',
                    extra={'event': 'elite_strategy_disabled_poll', 'strategy': self.name},
                )
                return None

            if self._last_signal_at:
                elapsed = (datetime.now(timezone.utc) - self._last_signal_at).total_seconds()
                if elapsed < self._config.cooldown_seconds:
                    LOGGER.info(
                        'Condition met: cooldown active',
                        extra={
                            'event': 'elite_strategy_cooldown',
                            'strategy': self.name,
                            'elapsed': elapsed,
                        },
                    )
                    return None

            if self._is_legacy_signature:
                elite_signal = self._evaluate_signal()  # type: ignore
                if elite_signal:
                    LOGGER.info(
                        'Condition met: legacy signal generated',
                        extra={
                            'event': 'elite_strategy_legacy_signal',
                            'strategy': self.name,
                        },
                    )
                    return self._process_signal(elite_signal)
                LOGGER.debug(
                    'No legacy signal generated',
                    extra={'event': 'elite_strategy_legacy_no_signal', 'strategy': self.name},
                )
                return None

            symbol = getattr(self._config, 'symbol', None)
            if not symbol:
                LOGGER.info(
                    'Condition met: missing symbol',
                    extra={'event': 'elite_strategy_missing_symbol', 'strategy': self.name},
                )
                return None

            req_inds = self.get_required_indicators()
            indicators = self._indicator_engine.get_indicators(symbol, list(req_inds))
            ltp = float(indicators.get('ltp') or 0.0)

            return self.generate_signal(symbol, indicators, ltp)
        except Exception as exc:
            LOGGER.error('Failure in evaluate: %s', exc, exc_info=exc)

        return None


    def _no_vote(self, reason: str) -> None:
        """Record a single no-vote reason. Args: reason. Returns: None. Raises: none."""
        self.last_no_vote_reason = reason

    def _evaluate_signal(
        self, 
        symbol: str = "", 
        indicators: Dict[str, Any] | None = None, 
        current_price: float = 0.0, 
        position: Any | None = None
    ) -> EliteSignal | None:
        raise NotImplementedError("Strategy must implement _evaluate_signal")

    def _process_signal(self, elite_signal: EliteSignal) -> Signal:
        """
        Converts internal EliteSignal to core Signal format.
        Standard implementation (Runner now handles metadata extraction).
        """
        self._last_signal_at = elite_signal.timestamp
        self._last_signal = elite_signal
        self._signals_generated += 1

        # 1. Consolidate all extra data into metadata
        canonical_reason = elite_signal.strategy_name or self.name
        metadata = elite_signal.metadata.copy()
        metadata.update({
            "strategy": self.name,
            "mode": "Legacy" if self._is_legacy_signature else "Push",
            "quantity": elite_signal.quantity,
            "price": elite_signal.entry_price,        # Moved from argument
            "stop_loss": elite_signal.stop_loss,      # Moved from argument
            "take_profit": elite_signal.target,       # Moved from argument
            "tag": f"{self.name}"                     # Moved from argument
        })

        # 2. Create Signal using ONLY valid arguments
        # Valid args are: action, symbol, confidence, reason, metadata
        return Signal(
            action=elite_signal.signal,
            symbol=elite_signal.symbol,
            confidence=elite_signal.confidence,
            reason=canonical_reason,
            quantity=elite_signal.quantity,       # <--- Added mandatory arg
            stop_loss=elite_signal.stop_loss,     # <--- Added mandatory arg
            take_profit=elite_signal.target,
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
