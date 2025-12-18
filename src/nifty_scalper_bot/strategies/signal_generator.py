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
    @property
    def deterministic_id(self) -> str:
        """
        Generate a restart-safe ID. 
        Signals for the same symbol/strategy within the same minute get the SAME ID.
        """
        # Round time to nearest minute string (e.g. 20251217_1430)
        # Assuming self.metadata['timestamp'] exists or using current time bucket
        ts = datetime.now().strftime("%Y%m%d%H%M") 
        
        # Combine core fields
        raw_str = f"{self.tag}:{self.symbol}:{self.action}:{ts}"
        
        # Create hash
        return hashlib.md5(raw_str.encode()).hexdigest()[:16]


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
    def generate_signal(
        self,
        symbol: str,
        indicators: IndicatorMap,
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        """Generate trading signal."""

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
    @staticmethod
    def _ensure_rr(
        side: Literal["BUY", "SELL"],
        current_price: float,
        stop_loss: float,
        desired_rr: float,
        min_rr: float = 2.0,
    ) -> tuple[float | None, float | None]:
        """Ensure the stop and take-profit respect the minimum RR constraint."""

        if math.isnan(stop_loss) or stop_loss <= 0:
            return None, None
        risk = current_price - stop_loss if side == "BUY" else stop_loss - current_price
        if risk <= 0:
            return None, None
        rr = max(desired_rr, min_rr)
        target_offset = risk * rr
        take_profit = (
            current_price + target_offset
            if side == "BUY"
            else current_price - target_offset
        )
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

    def generate_signal(
        self,
        symbol: str,
        indicators: IndicatorMap,
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        """Generate RSI-based signal."""

        rsi_raw = indicators.get("rsi")
        atr_raw = indicators.get("atr")
        if rsi_raw is None or atr_raw is None:
            return None
        rsi = float(rsi_raw)
        atr = float(atr_raw)

        oversold = float(self._parameters["oversold_threshold"])
        overbought = float(self._parameters["overbought_threshold"])
        qty = int(self._parameters.get("default_quantity", 1))
        swing_low_val = indicators.get("swing_low")
        swing_high_val = indicators.get("swing_high")
        swing_low = (
            float(swing_low_val) if swing_low_val is not None else current_price - atr
        )
        swing_high = (
            float(swing_high_val) if swing_high_val is not None else current_price + atr
        )

        metadata = {
            "strategy": self.name,
            "rsi": rsi,
            "atr": atr,
            "swing_low": swing_low,
            "swing_high": swing_high,
        }

        if position:
            if position.side == "LONG" and rsi >= 50:
                reason = f"RSI normalized to {rsi:.1f}, closing long"
                return Signal(
                    action="CLOSE_LONG",
                    symbol=symbol,
                    quantity=position.quantity,
                    confidence=self._bounded_confidence((rsi - 50) / 50),
                    reason=reason,
                    stop_loss=None,
                    take_profit=None,
                    metadata=metadata,
                )
            if position.side == "SHORT" and rsi <= 50:
                reason = f"RSI normalized to {rsi:.1f}, closing short"
                return Signal(
                    action="CLOSE_SHORT",
                    symbol=symbol,
                    quantity=position.quantity,
                    confidence=self._bounded_confidence((50 - rsi) / 50),
                    reason=reason,
                    stop_loss=None,
                    take_profit=None,
                    metadata=metadata,
                )

        if rsi < oversold and (position is None or position.side != "LONG"):
            stop = min(swing_low, current_price - atr)
            stop_loss, take_profit = self._ensure_rr(
                "BUY", current_price, stop, desired_rr=2.0
            )
            if stop_loss is None:
                return None
            confidence = self._bounded_confidence((oversold - rsi) / oversold)
            reason = f"RSI {rsi:.1f} below {oversold}, mean reversion long"
            return Signal(
                action="BUY",
                symbol=symbol,
                quantity=qty,
                confidence=confidence,
                reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        if rsi > overbought and (position is None or position.side != "SHORT"):
            stop = max(swing_high, current_price + atr)
            stop_loss, take_profit = self._ensure_rr(
                "SELL", current_price, stop, desired_rr=2.0
            )
            if stop_loss is None:
                return None
            confidence = self._bounded_confidence(
                (rsi - overbought) / (100 - overbought)
            )
            reason = f"RSI {rsi:.1f} above {overbought}, mean reversion short"
            return Signal(
                action="SELL",
                symbol=symbol,
                quantity=qty,
                confidence=confidence,
                reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        return None


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

    def generate_signal(
        self,
        symbol: str,
        indicators: IndicatorMap,
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        """Generate EMA crossover signal."""

        fast = indicators.get("ema_fast")
        slow = indicators.get("ema_slow")
        fast_prev = indicators.get("ema_fast_prev")
        slow_prev = indicators.get("ema_slow_prev")
        atr = indicators.get("atr")

        if (
            fast is None
            or slow is None
            or fast_prev is None
            or slow_prev is None
            or atr is None
        ):
            return None

        fast = float(fast)
        slow = float(slow)
        fast_prev = float(fast_prev)
        slow_prev = float(slow_prev)
        atr = float(atr)

        qty = int(self._parameters.get("default_quantity", 1))
        desired_rr = max(1.5, 2.0)
        metadata = {
            "strategy": self.name,
            "ema_fast": fast,
            "ema_slow": slow,
            "ema_fast_prev": fast_prev,
            "ema_slow_prev": slow_prev,
            "atr": atr,
        }

        crossed_up = fast_prev <= slow_prev and fast > slow
        crossed_down = fast_prev >= slow_prev and fast < slow

        if position:
            if position.side == "LONG" and crossed_down:
                confidence = self._bounded_confidence(abs(fast - slow) / atr)
                reason = "EMA death cross detected, exit long"
                return Signal(
                    action="CLOSE_LONG",
                    symbol=symbol,
                    quantity=position.quantity,
                    confidence=confidence,
                    reason=reason,
                    stop_loss=None,
                    take_profit=None,
                    metadata=metadata,
                )
            if position.side == "SHORT" and crossed_up:
                confidence = self._bounded_confidence(abs(fast - slow) / atr)
                reason = "EMA golden cross detected, exit short"
                return Signal(
                    action="CLOSE_SHORT",
                    symbol=symbol,
                    quantity=position.quantity,
                    confidence=confidence,
                    reason=reason,
                    stop_loss=None,
                    take_profit=None,
                    metadata=metadata,
                )

        if crossed_up and (position is None or position.side != "LONG"):
            stop = slow
            stop_loss, take_profit = self._ensure_rr(
                "BUY", current_price, stop, desired_rr
            )
            if stop_loss is None:
                return None
            confidence = self._bounded_confidence(abs(fast - slow) / max(atr, 1e-6))
            reason = "Fast EMA crossed above slow EMA"
            return Signal(
                action="BUY",
                symbol=symbol,
                quantity=qty,
                confidence=confidence,
                reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        if crossed_down and (position is None or position.side != "SHORT"):
            stop = slow
            stop_loss, take_profit = self._ensure_rr(
                "SELL", current_price, stop, desired_rr
            )
            if stop_loss is None:
                return None
            confidence = self._bounded_confidence(abs(fast - slow) / max(atr, 1e-6))
            reason = "Fast EMA crossed below slow EMA"
            return Signal(
                action="SELL",
                symbol=symbol,
                quantity=qty,
                confidence=confidence,
                reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        return None


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

    def generate_signal(
        self,
        symbol: str,
        indicators: IndicatorMap,
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        """Generate MACD-based signal."""

        macd = indicators.get("macd")
        signal_line = indicators.get("macd_signal")
        histogram = indicators.get("macd_hist")
        macd_prev = indicators.get("macd_prev")
        signal_prev = indicators.get("macd_signal_prev")
        atr = indicators.get("atr")

        if (
            macd is None
            or signal_line is None
            or histogram is None
            or macd_prev is None
            or signal_prev is None
            or atr is None
        ):
            return None

        macd = float(macd)
        signal_line = float(signal_line)
        histogram = float(histogram)
        macd_prev = float(macd_prev)
        signal_prev = float(signal_prev)
        atr = float(atr)

        qty = int(self._parameters.get("default_quantity", 1))
        metadata = {
            "strategy": self.name,
            "macd": macd,
            "macd_signal": signal_line,
            "macd_hist": histogram,
            "macd_prev": macd_prev,
            "macd_signal_prev": signal_prev,
            "atr": atr,
        }

        crossed_up = macd_prev <= signal_prev and macd > signal_line and histogram > 0
        crossed_down = macd_prev >= signal_prev and macd < signal_line and histogram < 0

        support_val = indicators.get("support")
        resistance_val = indicators.get("resistance")
        support = float(support_val) if support_val is not None else current_price - atr
        resistance = (
            float(resistance_val) if resistance_val is not None else current_price + atr
        )

        if position:
            if position.side == "LONG" and crossed_down:
                confidence = self._bounded_confidence(
                    abs(macd - signal_line) / max(abs(histogram), 1e-6)
                )
                reason = "MACD bearish crossover, exit long"
                return Signal(
                    action="CLOSE_LONG",
                    symbol=symbol,
                    quantity=position.quantity,
                    confidence=confidence,
                    reason=reason,
                    stop_loss=None,
                    take_profit=None,
                    metadata=metadata,
                )
            if position.side == "SHORT" and crossed_up:
                confidence = self._bounded_confidence(
                    abs(macd - signal_line) / max(abs(histogram), 1e-6)
                )
                reason = "MACD bullish crossover, exit short"
                return Signal(
                    action="CLOSE_SHORT",
                    symbol=symbol,
                    quantity=position.quantity,
                    confidence=confidence,
                    reason=reason,
                    stop_loss=None,
                    take_profit=None,
                    metadata=metadata,
                )

        if crossed_up and (position is None or position.side != "LONG"):
            stop = min(support, current_price - atr)
            stop_loss, take_profit = self._ensure_rr(
                "BUY", current_price, stop, desired_rr=2.0
            )
            if stop_loss is None:
                return None
            momentum = abs(histogram) / max(abs(signal_line), 1e-6)
            confidence = self._bounded_confidence(0.5 + 0.5 * min(momentum, 1.0))
            reason = "MACD bullish crossover confirmed"
            return Signal(
                action="BUY",
                symbol=symbol,
                quantity=qty,
                confidence=confidence,
                reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={**metadata, "support": support},
            )

        if crossed_down and (position is None or position.side != "SHORT"):
            stop = max(resistance, current_price + atr)
            stop_loss, take_profit = self._ensure_rr(
                "SELL", current_price, stop, desired_rr=2.0
            )
            if stop_loss is None:
                return None
            momentum = abs(histogram) / max(abs(signal_line), 1e-6)
            confidence = self._bounded_confidence(0.5 + 0.5 * min(momentum, 1.0))
            reason = "MACD bearish crossover confirmed"
            return Signal(
                action="SELL",
                symbol=symbol,
                quantity=qty,
                confidence=confidence,
                reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={**metadata, "resistance": resistance},
            )

        return None


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

    def generate_signal(
        self,
        symbol: str,
        indicators: IndicatorMap,
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        """Generate Bollinger Band signal."""

        upper = indicators.get("bb_upper")
        lower = indicators.get("bb_lower")
        middle = indicators.get("bb_middle")
        rsi = indicators.get("rsi")
        atr = indicators.get("atr")

        if (
            upper is None
            or lower is None
            or middle is None
            or rsi is None
            or atr is None
        ):
            return None

        upper = float(upper)
        lower = float(lower)
        middle = float(middle)
        rsi = float(rsi)
        atr = float(atr)

        qty = int(self._parameters.get("default_quantity", 1))
        metadata = {
            "strategy": self.name,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_middle": middle,
            "rsi": rsi,
            "atr": atr,
        }

        if position:
            if position.side == "LONG" and current_price >= middle:
                confidence = self._bounded_confidence(
                    (current_price - middle) / max(atr, 1e-6)
                )
                reason = "Price reverted to middle band, closing long"
                return Signal(
                    action="CLOSE_LONG",
                    symbol=symbol,
                    quantity=position.quantity,
                    confidence=confidence,
                    reason=reason,
                    stop_loss=None,
                    take_profit=None,
                    metadata=metadata,
                )
            if position.side == "SHORT" and current_price <= middle:
                confidence = self._bounded_confidence(
                    (middle - current_price) / max(atr, 1e-6)
                )
                reason = "Price reverted to middle band, closing short"
                return Signal(
                    action="CLOSE_SHORT",
                    symbol=symbol,
                    quantity=position.quantity,
                    confidence=confidence,
                    reason=reason,
                    stop_loss=None,
                    take_profit=None,
                    metadata=metadata,
                )

        if (
            current_price <= lower
            and rsi < 40
            and (position is None or position.side != "LONG")
        ):
            stop = min(lower - 0.5 * atr, current_price - atr)
            stop_loss, take_profit = self._ensure_rr(
                "BUY", current_price, stop, desired_rr=2.0
            )
            if stop_loss is None:
                return None
            distance = lower - current_price
            confidence = self._bounded_confidence(
                0.5 + 0.5 * min(abs(distance) / max(atr, 1e-6), 1.0)
            )
            reason = "Price at lower band with oversold RSI"
            return Signal(
                action="BUY",
                symbol=symbol,
                quantity=qty,
                confidence=confidence,
                reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        if (
            current_price >= upper
            and rsi > 60
            and (position is None or position.side != "SHORT")
        ):
            stop = max(upper + 0.5 * atr, current_price + atr)
            stop_loss, take_profit = self._ensure_rr(
                "SELL", current_price, stop, desired_rr=2.0
            )
            if stop_loss is None:
                return None
            distance = current_price - upper
            confidence = self._bounded_confidence(
                0.5 + 0.5 * min(abs(distance) / max(atr, 1e-6), 1.0)
            )
            reason = "Price at upper band with overbought RSI"
            return Signal(
                action="SELL",
                symbol=symbol,
                quantity=qty,
                confidence=confidence,
                reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        return None


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

    def generate_signal(
        self,
        symbol: str,
        indicators: IndicatorMap,
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        """Generate ORB trade signals after the opening range completes.

        Args:
            symbol: Canonical instrument identifier.
            indicators: Indicator snapshot for the instrument.
            current_price: Latest traded price from the runner.
            position: Existing position, if any.

        Returns:
            Signal | None: Trade intent when filters are satisfied otherwise ``None``.

        Raises:
            None.
        """

        logger.debug(
            "Entered OpeningRangeBreakoutStrategy.generate_signal",
            extra={"event": "orb_generate", "symbol": symbol},
        )
        try:
            orb_high = indicators.get("orb_high")
            orb_low = indicators.get("orb_low")
            orb_ready = indicators.get("orb_ready")
            atr = indicators.get("atr")
            nr7_flag = bool(indicators.get("nr7"))
            volume_ratio = indicators.get("futures_volume_ratio")
            minutes_since_open = indicators.get("minutes_since_open")
            minutes_until_close = indicators.get("minutes_until_close")
            if (
                orb_high is None
                or orb_low is None
                or not bool(orb_ready)
                or atr is None
                or minutes_since_open is None
                or minutes_until_close is None
            ):
                return None

            parameters = self._parameters
            min_volume_ratio = float(parameters["volume_spike_ratio"])
            if volume_ratio is None or float(volume_ratio) < min_volume_ratio:
                logger.debug(
                    "ORB volume filter blocked signal",
                    extra={"event": "orb_volume_block", "symbol": symbol},
                )
                return None
            if not nr7_flag:
                logger.debug(
                    "ORB NR7 filter blocked signal",
                    extra={"event": "orb_nr7_block", "symbol": symbol},
                )
                return None
            if float(minutes_since_open) < float(parameters["opening_minutes"]):
                return None
            if float(minutes_until_close) < 20.0:
                return None

            qty = int(parameters.get("default_quantity", 1))
            premium_stop_pct = float(parameters["premium_stop_pct"])
            premium_target_rr = float(parameters["premium_target_rr"])
            atr_value = float(atr)
            metadata = {
                "strategy": self.name,
                "orb_high": float(orb_high),
                "orb_low": float(orb_low),
                "nr7": nr7_flag,
                "volume_ratio": float(volume_ratio),
                "premium_stop_pct": premium_stop_pct,
                "premium_target_rr": premium_target_rr,
                "atr": atr_value,
            }

            if position is not None:
                if position.side == "LONG" and current_price < float(orb_low):
                    logger.info(
                        "Condition met: ORB exit long",
                        extra={"event": "orb_exit_long", "symbol": symbol},
                    )
                    return Signal(
                        action="CLOSE_LONG",
                        symbol=symbol,
                        quantity=position.quantity,
                        confidence=self._bounded_confidence(0.6),
                        reason="Price lost opening range low",
                        stop_loss=None,
                        take_profit=None,
                        metadata=metadata,
                    )
                if position.side == "SHORT" and current_price > float(orb_high):
                    logger.info(
                        "Condition met: ORB exit short",
                        extra={"event": "orb_exit_short", "symbol": symbol},
                    )
                    return Signal(
                        action="CLOSE_SHORT",
                        symbol=symbol,
                        quantity=position.quantity,
                        confidence=self._bounded_confidence(0.6),
                        reason="Price reclaimed opening range high",
                        stop_loss=None,
                        take_profit=None,
                        metadata=metadata,
                    )

            if current_price > float(orb_high) and (
                position is None or position.side != "LONG"
            ):
                distance = current_price - float(orb_high)
                confidence = self._bounded_confidence(distance / max(atr_value, 1.0))
                logger.info(
                    "Condition met: ORB breakout long",
                    extra={"event": "orb_long", "symbol": symbol},
                )
                return Signal(
                    action="BUY",
                    symbol=symbol,
                    quantity=qty,
                    confidence=confidence,
                    reason="ORB breakout above opening high",
                    stop_loss=None,
                    take_profit=None,
                    metadata=metadata,
                )

            if current_price < float(orb_low) and (
                position is None or position.side != "SHORT"
            ):
                distance = float(orb_low) - current_price
                confidence = self._bounded_confidence(distance / max(atr_value, 1.0))
                logger.info(
                    "Condition met: ORB breakdown short",
                    extra={"event": "orb_short", "symbol": symbol},
                )
                return Signal(
                    action="SELL",
                    symbol=symbol,
                    quantity=qty,
                    confidence=confidence,
                    reason="ORB breakdown below opening low",
                    stop_loss=None,
                    take_profit=None,
                    metadata=metadata,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in OpeningRangeBreakoutStrategy.generate_signal: %s",
                exc,
                extra={"event": "orb_generate_failure", "symbol": symbol},
            )
        return None


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

    def generate_signal(
        self,
        symbol: str,
        indicators: IndicatorMap,
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        """Generate VWAP mean reversion entries with RSI confirmation.

        Args:
            symbol: Canonical instrument identifier.
            indicators: Indicator snapshot for the instrument.
            current_price: Latest traded price from the runner.
            position: Existing position, if any.

        Returns:
            Signal | None: Trade intent when setup detected else ``None``.

        Raises:
            None.
        """

        logger.debug(
            "Entered VWAPMeanReversionStrategy.generate_signal",
            extra={"event": "vwap_generate", "symbol": symbol},
        )
        try:
            vwap = indicators.get("vwap")
            rsi = indicators.get("rsi")
            atr = indicators.get("atr")
            minutes_until_close = indicators.get("minutes_until_close")
            minutes_since_open = indicators.get("minutes_since_open")
            if (
                vwap is None
                or rsi is None
                or atr is None
                or minutes_until_close is None
                or minutes_since_open is None
            ):
                return None

            parameters = self._parameters
            deviation_pct = float(parameters["deviation_pct"])
            oversold = float(parameters["rsi_oversold"])
            overbought = float(parameters["rsi_overbought"])
            premium_stop_pct = float(parameters["premium_stop_pct"])
            premium_target_rr = float(parameters["premium_target_rr"])
            qty = int(parameters.get("default_quantity", 1))
            atr_value = float(atr)

            if float(minutes_since_open) < 20.0 or float(minutes_until_close) < 15.0:
                return None

            metadata = {
                "strategy": self.name,
                "vwap": float(vwap),
                "rsi": float(rsi),
                "premium_stop_pct": premium_stop_pct,
                "premium_target_rr": premium_target_rr,
                "atr": atr_value,
            }

            upper_threshold = float(vwap) * (1.0 + deviation_pct)
            lower_threshold = float(vwap) * (1.0 - deviation_pct)

            if position is not None:
                if position.side == "LONG" and current_price >= float(vwap):
                    logger.info(
                        "Condition met: VWAP exit long",
                        extra={"event": "vwap_exit_long", "symbol": symbol},
                    )
                    return Signal(
                        action="CLOSE_LONG",
                        symbol=symbol,
                        quantity=position.quantity,
                        confidence=self._bounded_confidence(0.55),
                        reason="Price reverted to VWAP",
                        stop_loss=None,
                        take_profit=None,
                        metadata=metadata,
                    )
                if position.side == "SHORT" and current_price <= float(vwap):
                    logger.info(
                        "Condition met: VWAP exit short",
                        extra={"event": "vwap_exit_short", "symbol": symbol},
                    )
                    return Signal(
                        action="CLOSE_SHORT",
                        symbol=symbol,
                        quantity=position.quantity,
                        confidence=self._bounded_confidence(0.55),
                        reason="Price reverted to VWAP",
                        stop_loss=None,
                        take_profit=None,
                        metadata=metadata,
                    )

            if (
                current_price < lower_threshold
                and float(rsi) <= oversold
                and (position is None or position.side != "LONG")
            ):
                distance = float(vwap) - current_price
                confidence = self._bounded_confidence(distance / max(atr_value, 1.0))
                logger.info(
                    "Condition met: VWAP long entry",
                    extra={"event": "vwap_long", "symbol": symbol},
                )
                return Signal(
                    action="BUY",
                    symbol=symbol,
                    quantity=qty,
                    confidence=confidence,
                    reason="Price below VWAP with oversold RSI",
                    stop_loss=None,
                    take_profit=None,
                    metadata=metadata,
                )

            if (
                current_price > upper_threshold
                and float(rsi) >= overbought
                and (position is None or position.side != "SHORT")
            ):
                distance = current_price - float(vwap)
                confidence = self._bounded_confidence(distance / max(atr_value, 1.0))
                logger.info(
                    "Condition met: VWAP short entry",
                    extra={"event": "vwap_short", "symbol": symbol},
                )
                return Signal(
                    action="SELL",
                    symbol=symbol,
                    quantity=qty,
                    confidence=confidence,
                    reason="Price above VWAP with overbought RSI",
                    stop_loss=None,
                    take_profit=None,
                    metadata=metadata,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in VWAPMeanReversionStrategy.generate_signal: %s",
                exc,
                extra={"event": "vwap_generate_failure", "symbol": symbol},
            )
        return None


class StrategyManager:
    """
    The 'Brain' of the bot. 
    Orchestrates strategies, validates physics (Greeks), and enforces Risk/Reward.
    State-of-the-Art implementation for NIFTY/BANKNIFTY Options.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        indicator_engine: IndicatorEngine,
        position_manager: PositionManager,
        min_confidence: float = 0.60,
        data_hub: Any | None = None,
        orchestrator: Any | None = None,
        futures_symbol: str | None = None,
        config: Any | None = None  # Changed type hint to Any to accept objects
    ):
        self._strategies = strategies
        self._indicator_engine = indicator_engine
        self._position_manager = position_manager
        self._min_confidence = min_confidence
        self._data_hub = data_hub
        self._orchestrator = orchestrator
        self._futures_symbol = (futures_symbol or "NIFTY").strip().upper()
        self._futures_volume_history: Deque[float] = deque(maxlen=120)
        
        # 1. Resolve Config Source
        # Priority: Passed config -> First Strategy's config -> Empty dict
        raw_config = config if config else (strategies[0].config if strategies else {})
        self._config = raw_config # Store raw for debugging

        # 2. Robust Config Getter Helper
        def get_cfg(key: str, default: Any) -> Any:
            # Try dictionary access
            if isinstance(raw_config, dict):
                return raw_config.get(key, default)
            # Try attribute access (for Settings objects)
            if hasattr(raw_config, key):
                return getattr(raw_config, key)
            # Try lowercase attribute
            if hasattr(raw_config, key.lower()):
                return getattr(raw_config, key.lower())
            return default

        # 3. Load Settings using Helper
        try:
            self.sl_pct = float(get_cfg("OPTION_SL_PCT", 0.15))
            self.tp_pct = float(get_cfg("OPTION_TP_PCT", 0.30))
            self.min_delta = float(get_cfg("DATA__MIN_DELTA", 0.30))
            self.max_iv_percentile = float(get_cfg("DATA__MAX_IV_PERCENTILE", 85.0))
            self.max_spread_pct = float(get_cfg("MAX_BID_ASK_SPREAD", 5.0))
        except (ValueError, TypeError):
            self.logger.warning("Failed to parse risk settings, using defaults.")
            self.sl_pct = 0.15
            self.tp_pct = 0.30
            self.min_delta = 0.30
            self.max_iv_percentile = 85.0
            self.max_spread_pct = 5.0

    def generate_signal(self, symbol: str, current_price: float) -> Signal | None:
        """
        MASTER EXECUTION LOOP:
        1. Context Augmentation (Futures/VIX)
        2. Strategy Polling
        3. Consensus Building
        4. Physics Gate (Greeks/Liquidity)
        5. Risk Gate (Premium Calculation)
        """
        logger.debug(
            "Entered StrategyManager.generate_signal",
            extra={"event": "strategy_manager_generate", "symbol": symbol},
        )

        # 1. MARKET CONTEXT & REGIME
        required: set[str] = {"volume", "avg_volume", "minutes_since_open", "minutes_until_close"}
        for strategy in self._strategies:
            required.update(strategy.get_required_indicators())

        indicators_raw = self._indicator_engine.get_indicators(symbol, required)
        indicators: dict[str, Any] = dict(indicators_raw)
        self._augment_futures_metrics(indicators)
        
        # VIX Check (Regime)
        vix_data = self._indicator_engine.get_indicators("NSE:INDIA VIX", ["ltp"])
        vix = float(vix_data.get("ltp", 15.0)) if vix_data else 15.0
        regime_factor = self._get_regime_modifier(vix)

        position = self._position_manager.get_position(symbol)
        signals: list[Signal] = []

        # 2. STRATEGY POLLING
        for strategy in self._strategies:
            try:
                # Optimization: Skip Momentum strategies in Dead markets (Low VIX)
                if vix < 12.0 and ("Breakout" in strategy.name or "ORB" in strategy.name):
                    continue

                signal = strategy.generate_signal(symbol, indicators, current_price, position)
                
                if signal:
                    # Apply Regime Factor (Downgrade confidence in Panic)
                    if regime_factor < 1.0:
                        signal = dataclasses.replace(
                            signal, 
                            confidence=signal.confidence * regime_factor,
                            tag=f"{signal.tag}_VixAdj" if signal.tag else "VixAdj"
                        )
                    signals.append(signal.with_metadata(indicators=indicators))
                    
            except Exception as exc: 
                logger.exception("Strategy %s failed: %s", strategy.name, exc)
                continue

        if not signals:
            return None

        # 3. CONSENSUS BUILDING
        combined = self._combine_signals(signals)
        if not combined:
            return None

        # 4. PHYSICS GATE: VALIDATE GREEKS & LIQUIDITY
        # Only apply to Options (CE/PE)
        if ("CE" in symbol or "PE" in symbol) and not self._validate_option_physics(symbol, combined.action):
            logger.info(f"⛔ Rejected {symbol}: Failed Physics Check (Greeks/Liq)")
            return None

        # 5. RISK GATE: CALCULATE PREMIUM-BASED RR
        # If strategy didn't set specific SL/TP, calculate it based on premium
        if not combined.stop_loss or not combined.take_profit:
            rr_levels = self._calculate_premium_risk(symbol, combined.action, current_price)
            if rr_levels:
                combined = dataclasses.replace(
                    combined, 
                    stop_loss=rr_levels[0], 
                    take_profit=rr_levels[1]
                )
            else:
                logger.info(f"⛔ Rejected {symbol}: Invalid Risk/Reward Profile")
                return None

        # 6. FINAL FILTERING
        if self._filter_signal(combined):
            if self._orchestrator:
                try:
                    combined = self._orchestrator.filter_signal(combined, indicators, self._position_manager)
                except Exception as exc:
                    logger.error("Failure in orchestrator.filter_signal: %s", exc)
                    return None
            
            if combined:
                logger.info(
                    "✅ Signal Validated & Locked",
                    extra={
                        "event": "strategy_manager_signal_ready",
                        "symbol": symbol,
                        "action": combined.action,
                        "conf": combined.confidence,
                        "sl": combined.stop_loss
                    },
                )
                return combined
        
        return None

    # --- SAFETY ENGINES ---

    def _get_regime_modifier(self, vix: float) -> float:
        """Returns confidence multiplier based on Volatility."""
        if vix > 24.0: return 0.8  # Panic: Reduce size/confidence
        if vix < 11.0: return 0.9  # Dead: Slight reduction
        return 1.0  # Normal

    def _validate_option_physics(self, symbol: str, action: str) -> bool:
        """Rejects 'Garbage Options' based on Greeks, Spread, and Liquidity."""
        # 1. Spread Check
        quote = self._indicator_engine.get_quote(symbol)
        if quote:
            bid = float(quote.get('bid', 0) or 0)
            ask = float(quote.get('ask', 0) or 0)
            if bid > 0:
                spread = ((ask - bid) / bid) * 100
                if spread > self.max_spread_pct: return False

        # 2. Greeks Check
        greeks = self._indicator_engine.get_indicators(symbol, ["delta", "theta", "iv_percentile"])
        if not greeks: return True # Fail-open if no data

        delta = abs(float(greeks.get("delta") or 0.5))
        if delta < self.min_delta: return False

        theta = float(greeks.get("theta") or 0.0)
        # Avoid high theta burn on long positions
        if action == "BUY" and theta < -20.0: return False

        return True

    def _calculate_premium_risk(self, symbol: str, side: str, price: float) -> tuple[float, float] | None:
        """Calculates Standardized Option Risk (15% SL / 30% TP)."""
        if price <= 0: return None
        
        if side == "BUY":
            sl = round(price * (1 - self.sl_pct), 2)
            tp = round(price * (1 + self.tp_pct), 2)
        else:
            sl = round(price * (1 + self.sl_pct), 2)
            tp = round(price * (1 - self.tp_pct), 2)
            
        # Sanity: Ensure SL is not too close (min 5%)
        if abs(price - sl) < (price * 0.05): return None
        return (sl, tp)

    # --- EXISTING HELPER METHODS (Preserved) ---

    def _augment_futures_metrics(self, indicators: MutableMapping[str, Any]) -> None:
        """Augment indicator snapshot with futures volume context."""
        indicators.setdefault("futures_volume", None)
        indicators.setdefault("futures_volume_avg", None)
        indicators.setdefault("futures_volume_ratio", None)
        data_hub = self._data_hub
        if data_hub is None: return
        
        try:
            quote = data_hub.get_quote(self._futures_symbol)
        except Exception: return
        
        volume = self._extract_float(quote, ("volume_traded_today", "volume", "last_quantity"))
        if volume is None: return
        
        self._futures_volume_history.append(volume)
        indicators["futures_volume"] = volume
        if self._futures_volume_history:
            avg = sum(self._futures_volume_history) / len(self._futures_volume_history)
            indicators["futures_volume_avg"] = avg
            if avg > 0: indicators["futures_volume_ratio"] = volume / avg

    @staticmethod
    def _extract_float(payload: Mapping[str, Any] | None, keys: Iterable[str]) -> float | None:
        if payload is None: return None
        for key in keys:
            val = payload.get(key)
            if val is not None:
                try: return float(val)
                except (TypeError, ValueError): continue
        return None

    def _combine_signals(self, signals: list[Signal]) -> Signal | None:
        """Combine multiple signals into one consensus signal."""
        by_action = defaultdict(list)
        for sig in signals:
            if sig.action != "HOLD": by_action[sig.action].append(sig)

        if not by_action: return None

        # Priority: Exits first
        for exit_act in ("CLOSE_LONG", "CLOSE_SHORT"):
            if exit_act in by_action:
                return max(by_action[exit_act], key=lambda s: s.confidence)

        # Conflict Resolution
        if len(by_action) > 1 and {"BUY", "SELL"}.issubset(by_action.keys()):
            buy_conf = sum(s.confidence for s in by_action["BUY"]) / len(by_action["BUY"])
            sell_conf = sum(s.confidence for s in by_action["SELL"]) / len(by_action["SELL"])
            if abs(buy_conf - sell_conf) < 0.15: return None # Indecisive

        best_action = max(by_action.items(), key=lambda i: sum(s.confidence for s in i[1])/len(i[1]))
        selected_list = best_action[1]
        best_signal = max(selected_list, key=lambda s: s.confidence)

        if len(selected_list) == 1: return best_signal

        # Blend
        avg_conf = sum(s.confidence for s in selected_list) / len(selected_list)
        # Use simple bounded logic instead of calling Strategy._bounded_confidence if not available
        final_conf = min(1.0, max(0.0, avg_conf + 0.05))
        
        # Merge metadata
        reasons = ", ".join(s.reason for s in selected_list if s.reason)
        meta = dict(best_signal.metadata)
        meta["confirming_strategies"] = [s.metadata.get("strategy") for s in selected_list if s.metadata.get("strategy")]
        
        return dataclasses.replace(
            best_signal,
            confidence=final_conf,
            reason=f"Consensus: {reasons}",
            metadata=meta
        )

    def _filter_signal(self, signal: Signal) -> bool:
        """Filter signal using confidence, volume and time windows."""
        if signal.confidence < self._min_confidence: return False

        indicators = signal.metadata.get("indicators", {}) if isinstance(signal.metadata, dict) else {}

        # Volume Filter
        vol = indicators.get("volume")
        avg_vol = indicators.get("avg_volume")
        if isinstance(vol, (int, float)) and isinstance(avg_vol, (int, float)) and avg_vol > 0:
            if float(vol) < 0.75 * float(avg_vol): return False

        # Time Filter (Hardcoded 9:30 - 15:15)
        m_time = indicators.get("market_time")
        if isinstance(m_time, datetime): m_time = m_time.time()
        if isinstance(m_time, time):
            if m_time < time(9, 30) or m_time > time(15, 15): return False

        return True


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
