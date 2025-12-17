"""
Base abstractions and helpers for elite strategies.
Production-Grade: Optimized Dispatch (Zero-Reflection Runtime) & Type Safety.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Set, Tuple

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategyConfig,
)
from nifty_scalper_bot.strategies.signal_generator import Signal, Strategy
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class EliteSignal:
    """Container for elite strategy signal output."""

    symbol: str
    side: str
    confidence: float
    entry_price: float
    stop_loss: float | None
    take_profit_1: float | None
    take_profit_2: float | None
    quantity: int = 1
    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_payload(self) -> dict[str, Any]:
        """Return serialisable representation of the signal."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "quantity": self.quantity,
            "strategy": self.strategy_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class EliteStrategy(Strategy):
    """Base class for high-probability elite setups."""

    def __init__(self, config: EliteStrategyConfig, indicator_engine: Any):
        """Initialize the elite strategy base with optimized dispatch."""
        
        # Initialize parent with correct signature (name + parameters dict)
        super().__init__(
            name=self.__class__.__name__,
            parameters=asdict(config) if is_dataclass(config) else {}
        )
        
        self._config = config
        self._indicator_engine = indicator_engine
        
        # State tracking
        self._last_signal_at: datetime | None = None
        self._last_signal: EliteSignal | None = None
        self._signals_generated: int = 0
        
        # ⚙️ PRODUCTION SAFETY SETTINGS
        self.min_oi = 50000
        self.max_spread_pct = 5.0
        self.min_delta = 0.30
        self.max_iv_percentile = 85.0

        # 🚀 PERFORMANCE OPTIMIZATION: Cache the dispatch decision ONCE.
        # We check the signature here so we don't slow down the trading loop later.
        sig = inspect.signature(self._evaluate_signal)
        # If it takes more than 0 params (excluding self), it's the new signature.
        self._is_legacy_signature = len(sig.parameters) == 0
        
        if self._is_legacy_signature:
            LOGGER.debug(f"{self.name}: Detected Legacy Signature (0 args)")
        else:
            LOGGER.debug(f"{self.name}: Detected Modern Signature (4 args)")

    def get_required_indicators(self) -> Set[str]:
        """
        Return the set of indicators required by this strategy.
        Elite strategies fetch their own data via _indicator_engine inside _evaluate_signal.
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
        Standard interface implementation bridging to elite logic.
        Uses cached dispatch logic for maximum performance.
        """
        if not self._config.enabled:
            return None

        try:
            # ✅ CONTEXT INJECTION: Inject symbol into config for legacy strategies
            # We use object.__setattr__ to bypass potential frozen/slots checks
            if hasattr(self._config, "symbol"):
                object.__setattr__(self._config, "symbol", symbol)

            # -----------------------------------------------------------
            # 🚀 OPTIMIZED DISPATCH (Zero Reflection)
            # -----------------------------------------------------------
            if self._is_legacy_signature:
                # The Old Way: Pass nothing (Strategy reads self._config.symbol)
                elite_signal = self._evaluate_signal()
            else:
                # The New Way: Pass arguments directly
                elite_signal = self._evaluate_signal(symbol, indicators, current_price, position)

            if elite_signal:
                # 🛡️ SAFETY GATE: Validate Trade Quality
                if not self.validate_option_health(elite_signal.symbol, elite_signal.side):
                    LOGGER.info(f"⛔ Rejected {elite_signal.symbol}: Failed Safety Check")
                    return None

                self._update_state(elite_signal)
                return self._convert_to_core_signal(elite_signal)
                
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(f"Strategy {self.name} failed evaluation: {exc}", exc_info=True)
            return None
        return None

    def _evaluate_signal(self, *args, **kwargs) -> EliteSignal | None:
        """Internal hook for strategy-specific logic."""
        raise NotImplementedError("Subclasses must implement _evaluate_signal")

    @property
    def name(self) -> str:
        """Return strategy identifier."""
        return self.__class__.__name__

    # ==========================================================================
    # 🛡️ SAFETY & RISK ENGINE
    # ==========================================================================

    def validate_option_health(self, symbol: str, direction: str) -> bool:
        """🛡️ GATEKEEPER: Stops the bot from trading 'Garbage Options'."""
        # Only validate Options contracts
        if "CE" not in symbol and "PE" not in symbol:
            return True

        quote = self._indicator_engine.get_quote(symbol)
        greeks = self._indicator_engine.get_indicators(symbol, ["delta", "theta", "iv_percentile"])
        
        if not quote:
            return False

        # 1. LIQUIDITY CHECK
        oi = quote.get('oi', 0) or 0
        if oi < self.min_oi:
            LOGGER.debug(f"⛔ {symbol}: Low Liquidity (OI: {oi}). Skip.")
            return False

        # 2. SPREAD CHECK
        bid = float(quote.get('bid', 0) or 0)
        ask = float(quote.get('ask', 0) or 0)
        if bid > 0:
            spread_pct = ((ask - bid) / bid) * 100
            if spread_pct > self.max_spread_pct:
                LOGGER.debug(f"⛔ {symbol}: Spread wide ({spread_pct:.2f}%). Skip.")
                return False

        # 3. GREEKS CHECK
        if greeks:
            delta = abs(float(greeks.get('delta') or 0.5))
            if delta < self.min_delta:
                LOGGER.debug(f"⛔ {symbol}: Weak Delta ({delta:.2f}). Skip.")
                return False
                
            theta = float(greeks.get('theta') or 0.0)
            if direction == "BUY" and theta < -20.0: 
                LOGGER.debug(f"⛔ {symbol}: High Theta Burn ({theta}). Skip.")
                return False
                
            iv_p = float(greeks.get('iv_percentile') or 50.0)
            if direction == "BUY" and iv_p > self.max_iv_percentile:
                LOGGER.debug(f"⛔ {symbol}: IV Expensive ({iv_p}). Skip.")
                return False

        return True

    def calculate_option_rr(self, premium: float, side: str = "BUY") -> Tuple[float, float]:
        """💰 RISK LOGIC: Calculates SL/TP based on Premium %."""
        # Standard Risk Profile
        SL_PCT = 0.15 
        TP_PCT = 0.30 

        if side == "BUY":
            sl_price = round(premium * (1 - SL_PCT), 1)
            tp_price = round(premium * (1 + TP_PCT), 1)
        else:
            sl_price = round(premium * (1 + SL_PCT), 1)
            tp_price = round(premium * (1 - TP_PCT), 1)
            
        # Bounds Check
        sl_price = max(0.05, sl_price)
        tp_price = max(0.05, tp_price)
            
        return sl_price, tp_price

    def _convert_to_core_signal(self, signal: EliteSignal) -> Signal:
        """Adapter converting elite signal to core signal format."""
        sl = signal.stop_loss
        tp = signal.take_profit_1
        
        # Auto-calculate Risk/Reward if not provided by strategy
        if not sl or not tp:
            calc_sl, calc_tp = self.calculate_option_rr(signal.entry_price, signal.side)
            if not sl: sl = calc_sl
            if not tp: tp = calc_tp

        metadata = signal.metadata.copy()
        metadata.update({
            "strategy": self.name,
            "elite_version": "2.0",
            "tp2": signal.take_profit_2,
            "quantity": signal.quantity
        })

        return Signal(
            action=signal.side, # type: ignore
            symbol=signal.symbol,
            confidence=signal.confidence,
            price=signal.entry_price,
            tag=f"{self.name} elite setup",
            stop_loss=sl,
            take_profit=tp,
            metadata=metadata,
        )

    def _update_state(self, signal: EliteSignal) -> None:
        """Persist bookkeeping fields for telemetry."""
        self._last_signal_at = signal.timestamp
        self._last_signal = signal
        self._signals_generated += 1

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic statistics for the strategy."""
        last_payload: dict[str, Any] | None = None
        if self._last_signal is not None:
            last_payload = self._last_signal.to_payload()
        return {
            "strategy": self.name,
            "enabled": self._config.enabled,
            "signals": self._signals_generated,
            "last_signal": last_payload,
        }

    @property
    def config(self) -> EliteStrategyConfig:
        """Return strategy configuration reference."""
        return self._config


__all__ = ["EliteSignal", "EliteStrategy"]
