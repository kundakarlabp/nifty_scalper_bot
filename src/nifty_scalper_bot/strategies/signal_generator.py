"""Advanced signal generation utilities."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Deque, Iterable, Literal, Mapping, MutableMapping, Protocol

from nifty_scalper_bot.utils.logging import get_logger

logger = get_logger(__name__)


class Position(Protocol):
    """Protocol describing the minimal position information required."""

    symbol: str
    side: Literal["LONG", "SHORT"]
    quantity: int
    entry_price: float


class IndicatorEngine(Protocol):
    """Protocol for indicator services used by the strategy manager."""

    def update_price(
        self,
        symbol: str,
        price: Any,
        *,
        volume: int = 0,
        timestamp: datetime | None = None,
    ) -> None:
        """Update the cached price for *symbol*."""

    def get_indicators(
        self, symbol: str, names: Iterable[str]
    ) -> Mapping[str, float | tuple[float, float, float] | None]:
        """Return the requested indicator snapshot for *symbol*."""


class PositionManager(Protocol):
    """Protocol describing the required position manager behaviour."""

    def get_position(self, symbol: str) -> Position | None:
        """Return the active position for *symbol*, if any."""


@dataclass(frozen=True)
class Signal:
    """Enhanced trading signal."""

    action: Literal["BUY", "SELL", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"]
    symbol: str
    quantity: int
    confidence: float  # 0.0 to 1.0
    reason: str  # Human-readable explanation
    stop_loss: float | None
    take_profit: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, **updates: Any) -> "Signal":
        """Return a new signal with ``metadata`` merged with *updates*."""

        merged = dict(self.metadata)
        merged.update(updates)
        return Signal(
            action=self.action,
            symbol=self.symbol,
            quantity=self.quantity,
            confidence=self.confidence,
            reason=self.reason,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            metadata=merged,
        )


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str, parameters: dict[str, Any]):
        """Initialize strategy with parameters."""

        self._name = name
        self._parameters = parameters
        self._description = parameters.get("description", self.__class__.__doc__ or "")
        if not self.validate_parameters():  # pragma: no cover - defensive
            msg = f"Invalid parameters supplied to strategy {name}: {parameters}"
            raise ValueError(msg)

    @abstractmethod
    def get_required_indicators(self) -> list[str]:
        """Return list of required indicators."""

    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""

        return True

    @property
    def name(self) -> str:
        """Strategy name."""

        return self._name

    @property
    def description(self) -> str:
        """Strategy description."""

        return self._description

    # ------------------------------------------------------------------
    # Utility helpers shared by concrete strategies
    # ------------------------------------------------------------------

    # ================= PASTE HERE =================
    def _validate_greeks(self, symbol: str, side: str) -> bool:
        """
        Safety Check: Rejects garbage options (Low Delta, High Theta).
        """
        # 1. Access Indicator Engine
        # Ensure we have access to indicators (fail safe if not)
        if not hasattr(self, "_indicators") or not self._indicators: 
            return True 

        # 2. Fetch Greeks
        # We request specific fields: delta, theta, iv_percentile
        greeks = self._indicators.get_indicators(symbol, ["delta", "theta", "iv_percentile"])
        
        # If greeks are missing (e.g., Index or new strike), default to Safe
        if not greeks: 
            return True 

        # 3. DELTA CHECK (Momentum)
        # Don't buy options that won't move (Deep OTM)
        # Delta ranges from 0 to 1. We want at least 0.30 (approx 30 delta)
        delta = abs(greeks.get("delta") or 0.5) 
        if delta < 0.30: 
            # self.logger.warning(f"⛔ Skipping {symbol}: Weak Delta {delta:.2f}")
            return False

        # 4. THETA CHECK (Time Decay)
        # Don't hold if time decay is burning too fast (e.g., > 15 pts/day)
        # Theta is usually negative. -20 is worse than -10.
        theta = greeks.get("theta") or 0.0
        if side == "BUY" and theta < -15.0:
            # self.logger.warning(f"⛔ Skipping {symbol}: High Theta Burn {theta:.2f}")
            return False

        # 5. IV PERCENTILE (Price)
        # Don't buy if IV is at yearly highs (Options are expensive)
        iv_pct = greeks.get("iv_percentile") or 0.0
        if side == "BUY" and iv_pct > 85.0:
             return False

        return True
    # ==============================================
    
    @staticmethod
    def _ensure_rr(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        current_price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> tuple[float, float] | None:
        """
        Calculate generic Stop Loss/Take Profit for Options (Premium-based).
        """
        if current_price <= 0: return None

        # 1. OPTION-SPECIFIC DEFAULTS (Percentage of Premium)
        # Risk 15% of the option premium to make 30%
        # This is standard for intraday scalping.
        SL_PCT = 0.15 
        TP_PCT = 0.30

        # 2. CALCULATE IF MISSING
        if not stop_loss:
            if side == "BUY":
                stop_loss = round(current_price * (1 - SL_PCT), 2)
            else: # Shorting options
                stop_loss = round(current_price * (1 + SL_PCT), 2)

        if not take_profit:
            if side == "BUY":
                take_profit = round(current_price * (1 + TP_PCT), 2)
            else:
                take_profit = round(current_price * (1 - TP_PCT), 2)

        # 3. SANITY CHECK (Don't allow negative prices)
        stop_loss = max(0.05, stop_loss)
        take_profit = max(0.05, take_profit)

        # 4. RR RATIO CHECK
        # Calculate Potential Loss vs Potential Gain
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit - current_price)

        if risk == 0: return None
        rr_ratio = reward / risk

        # Reject poor trades (Reward must be at least 1.5x Risk)
        if rr_ratio < 1.5:
            # logger.warning(f"Bad RR for {symbol}: {rr_ratio:.2f}")
            return None

        return stop_loss, take_profit

    @staticmethod
    def _bounded_confidence(value: float) -> float:
        """Clamp *value* to the inclusive range [0.0, 1.0]."""

        return max(0.0, min(1.0, value))


class RSIMeanReversionStrategy(Strategy):
    """RSI Mean Reversion Strategy.

    - Buy when RSI < oversold_threshold (default 30)
    - Sell when RSI > overbought_threshold (default 70)
    - Exit when RSI returns to neutral (50)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70,
        default_quantity: int = 1,
    ):
        """Initialize RSI strategy."""

        parameters = {
            "rsi_period": rsi_period,
            "oversold_threshold": oversold_threshold,
            "overbought_threshold": overbought_threshold,
            "default_quantity": default_quantity,
        }
        super().__init__("RSI Mean Reversion", parameters)

    def validate_parameters(self) -> bool:
        return (
            isinstance(self._parameters.get("rsi_period"), int)
            and 0 < self._parameters["rsi_period"] <= 200
            and 0
            < self._parameters["oversold_threshold"]
            < self._parameters["overbought_threshold"]
        )

    def get_required_indicators(self) -> list[str]:
        return ["rsi", "atr", "swing_low", "swing_high"]


class EMACrossoverStrategy(Strategy):
    """EMA Crossover Strategy.

    - Buy when fast EMA crosses above slow EMA (golden cross)
    - Sell when fast EMA crosses below slow EMA (death cross)
    """

    def __init__(
        self, fast_period: int = 9, slow_period: int = 21, default_quantity: int = 1
    ):
        """Initialize EMA crossover strategy."""

        parameters = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "default_quantity": default_quantity,
        }
        super().__init__("EMA Crossover", parameters)

    def validate_parameters(self) -> bool:
        fast = self._parameters.get("fast_period")
        slow = self._parameters.get("slow_period")
        return isinstance(fast, int) and isinstance(slow, int) and 0 < fast < slow

    def get_required_indicators(self) -> list[str]:
        return [
            "ema_fast",
            "ema_slow",
            "ema_fast_prev",
            "ema_slow_prev",
            "atr",
        ]

class MACDStrategy(Strategy):
    """MACD Strategy.

    - Buy when MACD line crosses above signal line and histogram > 0
    - Sell when MACD line crosses below signal line and histogram < 0
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        default_quantity: int = 1,
    ):
        """Initialize MACD strategy."""

        parameters = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
            "default_quantity": default_quantity,
        }
        super().__init__("MACD", parameters)

    def validate_parameters(self) -> bool:
        fast = self._parameters.get("fast_period")
        slow = self._parameters.get("slow_period")
        signal_period = self._parameters.get("signal_period")
        return (
            isinstance(fast, int)
            and isinstance(slow, int)
            and isinstance(signal_period, int)
            and 0 < fast < slow
            and 0 < signal_period < 100
        )

    def get_required_indicators(self) -> list[str]:
        return [
            "macd",
            "macd_signal",
            "macd_hist",
            "macd_prev",
            "macd_signal_prev",
            "atr",
            "support",
            "resistance",
        ]

    

class BollingerBandStrategy(Strategy):
    """Bollinger Band Strategy.

    - Buy when price touches lower band and RSI < 40
    - Sell when price touches upper band and RSI > 60
    - Exit when price returns to middle band
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        default_quantity: int = 1,
    ):
        """Initialize Bollinger Band strategy."""

        parameters = {
            "bb_period": bb_period,
            "bb_std": bb_std,
            "rsi_period": rsi_period,
            "default_quantity": default_quantity,
        }
        super().__init__("Bollinger Bands", parameters)

    def validate_parameters(self) -> bool:
        return (
            isinstance(self._parameters.get("bb_period"), int)
            and self._parameters["bb_period"] > 1
            and isinstance(self._parameters.get("bb_std"), (float, int))
            and self._parameters["bb_std"] > 0
        )

    def get_required_indicators(self) -> list[str]:
        return [
            "bb_upper",
            "bb_lower",
            "bb_middle",
            "rsi",
            "atr",
        ]



class OpeningRangeBreakoutStrategy(Strategy):
    """Opening range breakout strategy with NR7 and futures volume filters."""

    def __init__(
        self,
        *,
        opening_minutes: int = 30,
        volume_spike_ratio: float = 1.5,
        premium_stop_pct: float = 0.35,
        premium_target_rr: float = 2.5,
        default_quantity: int = 1,
    ) -> None:
        """Initialise the ORB strategy with breakout filters.

        Args:
            opening_minutes: Minutes defining the opening range window.
            volume_spike_ratio: Minimum futures volume spike ratio.
            premium_stop_pct: Stop placement as a fraction of option premium.
            premium_target_rr: Desired reward-to-risk multiple for the target.
            default_quantity: Default order quantity when no sizing hint exists.

        Returns:
            None.

        Raises:
            None.
        """

        parameters = {
            "opening_minutes": opening_minutes,
            "volume_spike_ratio": volume_spike_ratio,
            "premium_stop_pct": premium_stop_pct,
            "premium_target_rr": premium_target_rr,
            "default_quantity": default_quantity,
        }
        super().__init__("Opening Range Breakout", parameters)

    def get_required_indicators(self) -> list[str]:
        """Return indicator names required by the ORB strategy.

        Args:
            None.

        Returns:
            list[str]: Required indicator keys.

        Raises:
            None.
        """

        return [
            "orb_high",
            "orb_low",
            "orb_ready",
            "nr7",
            "nr7_range",
            "nr7_min_range",
            "futures_volume_ratio",
            "minutes_since_open",
            "minutes_until_close",
            "atr",
        ]


class VWAPMeanReversionStrategy(Strategy):
    """VWAP mean reversion strategy with RSI confirmation."""

    def __init__(
        self,
        *,
        deviation_pct: float = 0.004,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        premium_stop_pct: float = 0.25,
        premium_target_rr: float = 2.0,
        default_quantity: int = 1,
    ) -> None:
        """Initialise VWAP mean reversion parameters.

        Args:
            deviation_pct: Fractional deviation from VWAP to trigger entries.
            rsi_oversold: RSI threshold confirming long entries.
            rsi_overbought: RSI threshold confirming short entries.
            premium_stop_pct: Stop placement as a fraction of option premium.
            premium_target_rr: Target reward-to-risk multiple for premium exits.
            default_quantity: Default order quantity when no dynamic sizing exists.

        Returns:
            None.

        Raises:
            None.
        """

        parameters = {
            "deviation_pct": deviation_pct,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "premium_stop_pct": premium_stop_pct,
            "premium_target_rr": premium_target_rr,
            "default_quantity": default_quantity,
        }
        super().__init__("VWAP Mean Reversion", parameters)

    def get_required_indicators(self) -> list[str]:
        """Return indicator names used by the VWAP mean reversion strategy.

        Args:
            None.

        Returns:
            list[str]: Required indicator keys.

        Raises:
            None.
        """

        return [
            "vwap",
            "rsi",
            "atr",
            "minutes_until_close",
            "minutes_since_open",
            "futures_volume_ratio",
        ]

# ==============================================================================
# 3. THE STRATEGY MANAGER (Paste this at the BOTTOM of signal_generator.py)
# ==============================================================================

class StrategyManager:
    """
    The 'Brain' of the bot.
    Orchestrates the strategies defined above, validates physics (Greeks), and enforces Risk/Reward.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        indicator_engine: IndicatorEngine,
        position_manager: PositionManager,
        min_confidence: float = 0.60,  # Increased default safety from 0.35
        data_hub: Any | None = None,
        config: dict[str, Any] | None = None
    ):
        self._strategies = strategies
        self._indicators = indicator_engine
        self._positions = position_manager
        self._data_hub = data_hub
        
        # Try to infer config from the first strategy if not passed explicitly
        self._config = config if config else (strategies[0].config if strategies else {})
        
        self.logger = get_logger(__name__)
        
        # Load Risk Configuration (Safe Defaults)
        self.sl_pct = float(self._config.get("OPTION_SL_PCT", 0.15))  # 15% Max Loss
        self.tp_pct = float(self._config.get("OPTION_TP_PCT", 0.30))  # 30% Target
        
        # Load Physics Configuration
        self.min_delta = float(self._config.get("DATA__MIN_DELTA", 0.30))
        self.max_iv_percentile = float(self._config.get("DATA__MAX_IV_PERCENTILE", 85.0))
        self.max_spread_pct = float(self._config.get("MAX_BID_ASK_SPREAD", 5.0))

    def generate_signal(self) -> Signal | None:
        """
        MASTER EXECUTION LOOP
        1. Check Regime (VIX).
        2. Poll Strategies.
        3. Validate Greeks (Physics).
        4. Calculate Risk (Math).
        5. Rank & Return Best Signal.
        """
        try:
            # 1. REGIME CHECK (VIX)
            vix_data = self._indicators.get_indicators("NSE:INDIA VIX", ["ltp"])
            vix = float(vix_data.get("ltp", 15.0)) if vix_data else 15.0
            
            is_panic = vix > 24.0
            is_dead = vix < 11.0
            
            candidates: list[tuple[float, Signal]] = []

            # 2. STRATEGY POLLING
            for strategy in self._strategies:
                strategy_name = strategy.__class__.__name__
                
                # Optimization: Skip Momentum strategies in Dead markets
                if is_dead and ("Breakout" in strategy_name or "ORB" in strategy_name):
                    continue

                try:
                    # Run the strategy logic defined above in this file
                    raw_signal = strategy.generate_signal()
                except Exception as e:
                    self.logger.error(f"Strategy {strategy_name} crashed: {e}", extra={"strategy": strategy_name})
                    continue

                if not raw_signal:
                    continue

                # 3. PHYSICS GATE: VALIDATE GREEKS & LIQUIDITY
                if not self._validate_option_physics(raw_signal.symbol, raw_signal.action):
                    continue

                # 4. MATH GATE: CALCULATE PREMIUM-BASED RR
                rr_levels = self._calculate_premium_risk(
                    raw_signal.symbol, 
                    raw_signal.action,
                    raw_signal.price
                )
                
                if not rr_levels:
                    continue
                
                sl_price, tp_price = rr_levels

                # 5. CONFIDENCE SCALING
                final_confidence = raw_signal.confidence
                tag_suffix = ""
                
                if is_panic:
                    final_confidence *= 0.8
                    tag_suffix = "_HighVix"

                # 6. FINALIZE SIGNAL
                final_signal = Signal(
                    action=raw_signal.action,
                    symbol=raw_signal.symbol,
                    confidence=final_confidence,
                    price=raw_signal.price,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    metadata=raw_signal.metadata,
                    tag=f"{raw_signal.tag}{tag_suffix}" if raw_signal.tag else "Generated"
                )
                
                candidates.append((final_confidence, final_signal))

            # 7. SELECTION (Rank by Confidence)
            if not candidates:
                return None

            # Sort highest confidence first
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_signal = candidates[0][1]
            
            self.logger.info(
                f"✅ SIGNAL LOCKED: {best_signal.symbol} {best_signal.action}",
                extra={
                    "event": "signal_generated",
                    "symbol": best_signal.symbol,
                    "confidence": best_signal.confidence,
                    "sl": best_signal.stop_loss,
                    "tp": best_signal.take_profit
                }
            )
            return best_signal

        except Exception as e:
            self.logger.critical(f"🔥 StrategyManager Meltdown: {e}", exc_info=True)
            return None

    # ==========================================================================
    # SAFETY & MATH HELPERS
    # ==========================================================================

    def _validate_option_physics(self, symbol: str, side: str) -> bool:
        """Rejects 'Garbage Options' based on Greeks, Spread, and Liquidity."""
        # Skip for non-options (Futures/Stocks)
        if "CE" not in symbol and "PE" not in symbol:
            return True

        # A. LIQUIDITY CHECK (Spread)
        quote = self._indicators.get_quote(symbol)
        if quote:
            bid = float(quote.get('bid', 0) or 0)
            ask = float(quote.get('ask', 0) or 0)
            if bid > 0:
                spread_pct = ((ask - bid) / bid) * 100
                if spread_pct > self.max_spread_pct:
                    self.logger.warning(f"⛔ Rejected {symbol}: Spread {spread_pct:.2f}% too wide")
                    return False

        # B. GREEKS CHECK
        greeks = self._indicators.get_indicators(symbol, ["delta", "theta", "iv_percentile"])
        if not greeks:
            return True # Fail-open if no data (safety choice: could fail-close)

        # Delta (Momentum)
        delta = abs(float(greeks.get("delta") or 0.5))
        if delta < self.min_delta:
            self.logger.debug(f"⛔ Rejected {symbol}: Low Delta {delta:.2f}")
            return False

        # Theta (Decay) - Assuming negative theta
        theta = float(greeks.get("theta") or 0.0)
        if side == "BUY" and theta < -20.0:
            self.logger.debug(f"⛔ Rejected {symbol}: High Theta Burn {theta}")
            return False

        # IV Percentile (Price)
        iv_pct = float(greeks.get("iv_percentile") or 50.0)
        if side == "BUY" and iv_pct > self.max_iv_percentile:
            self.logger.debug(f"⛔ Rejected {symbol}: Expensive IV {iv_pct}")
            return False

        return True

    def _calculate_premium_risk(self, symbol: str, side: str, entry_price: float) -> tuple[float, float] | None:
        """Calculates Stop Loss and Take Profit based on Option Premium Percentage."""
        if entry_price <= 0: return None

        # 1. Calculate Prices based on Configured Percentages
        if side == "BUY":
            stop_loss = round(entry_price * (1 - self.sl_pct), 2)
            take_profit = round(entry_price * (1 + self.tp_pct), 2)
        else:
            # Short Option logic
            stop_loss = round(entry_price * (1 + self.sl_pct), 2)
            take_profit = round(entry_price * (1 - self.tp_pct), 2)

        # 2. Sanity Checks
        stop_loss = max(0.05, stop_loss)
        take_profit = max(0.05, take_profit)

        # Ensure minimal breathing room (5%)
        if abs(entry_price - stop_loss) < (entry_price * 0.05):
            self.logger.warning(f"⛔ Rejected {symbol}: SL too tight")
            return None

        return stop_loss, take_profit


__all__ = [
    "Signal",
    "Strategy",
    "RSIMeanReversionStrategy",
    "EMACrossoverStrategy",
    "MACDStrategy",
    "BollingerBandStrategy",
    "OpeningRangeBreakoutStrategy",
    "VWAPMeanReversionStrategy",
    "StrategyManager",
]

IndicatorMap = Mapping[str, Any]
