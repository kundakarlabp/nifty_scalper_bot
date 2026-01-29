"""Advanced signal generation utilities with hardened observability."""

from __future__ import annotations

import math
import hashlib
import dataclasses
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
        return dataclasses.replace(self, metadata=merged)

    @property
    def deterministic_id(self) -> str:
        """
        Generate a stable, restart-safe ID for this signal.
        Logic: HASH(Strategy + Symbol + Action + MinuteTimestamp)
        """
        ts_raw = self.metadata.get("timestamp")
        
        if isinstance(ts_raw, str):
            ts_str = ts_raw[:16] 
        elif isinstance(ts_raw, (int, float)):
            ts_str = str(int(ts_raw / 60)) 
        elif isinstance(ts_raw, datetime):
            ts_str = ts_raw.strftime("%Y%m%d%H%M")
        else:
            now = datetime.now()
            ts_str = now.strftime("%Y%m%d%H%M")

        strategy = self.metadata.get("strategy", "manual")
        raw_sig = f"{strategy}:{self.symbol}:{self.action}:{ts_str}"
        return hashlib.md5(raw_sig.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "action": str(self.action),
            "symbol": str(self.symbol),
            "quantity": int(self.quantity),
            "confidence": float(self.confidence),
            "reason": str(self.reason) if self.reason else None,
        }
        if self.stop_loss is not None:
            result["stop_loss"] = float(self.stop_loss)
        if self.take_profit is not None:
            result["take_profit"] = float(self.take_profit)
        if self.metadata:
            result["metadata"] = self._sanitize_for_json(self.metadata)
        return result

    @staticmethod
    def _sanitize_for_json(data: Any) -> Any:
        """Recursively sanitize data for JSON serialization."""
        if data is None: return None
        if isinstance(data, (str, int, float, bool)): return data
        if isinstance(data, datetime): return data.isoformat()
        if hasattr(data, "item"): return float(data) # numpy scalar support
        if isinstance(data, dict):
            return {str(k): Signal._sanitize_for_json(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [Signal._sanitize_for_json(item) for item in data]
        try:
            return str(data)
        except:
            return "<unserializable>"


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str, parameters: dict[str, Any]):
        """Initialize strategy with parameters."""
        self._name = name
        self._parameters = parameters
        self._description = parameters.get("description", self.__class__.__doc__ or "")
        if not self.validate_parameters():
            msg = f"Invalid parameters supplied to strategy {name}: {parameters}"
            raise ValueError(msg)

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        """Generate trading signal."""

        # ✅ FIX: Block futures at strategy entry, not execution
        # This saves CPU cycles and prevents misleading signals
        if "FUT" in symbol.upper():
            return None

    @abstractmethod
    def get_required_indicators(self) -> list[str]:
        """Return list of required indicators."""

    def validate_parameters(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def config(self) -> dict[str, Any]:
        return self._parameters

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
    """RSI Mean Reversion Strategy."""

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70,
        default_quantity: int = 1,
    ):
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
        indicators: Mapping[str, Any],
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        logger.debug(f"EVAL {self.name}: {symbol}")

        rsi_raw = indicators.get("rsi")
        atr_raw = indicators.get("atr")
        if rsi_raw is None or atr_raw is None:
            logger.debug(f"SKIP {self.name}: data_missing (rsi={rsi_raw}, atr={atr_raw}) | {symbol}")
            return None
        rsi = float(rsi_raw)
        atr = float(atr_raw)

        oversold = float(self._parameters["oversold_threshold"])
        overbought = float(self._parameters["overbought_threshold"])
        qty = int(self._parameters.get("default_quantity", 1))
        
        swing_low = float(indicators.get("swing_low") or (current_price - atr))
        swing_high = float(indicators.get("swing_high") or (current_price + atr))

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
                logger.info(f"SIGNAL {self.name}: CLOSE_LONG | {reason}")
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
                logger.info(f"SIGNAL {self.name}: CLOSE_SHORT | {reason}")
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
                logger.debug(f"SKIP {self.name}: risk_invalid (SL calculation failed) | {symbol}")
                return None
            
            confidence = self._bounded_confidence((oversold - rsi) / oversold)
            reason = f"RSI {rsi:.1f} below {oversold}, mean reversion long"
            logger.info(f"SIGNAL {self.name}: BUY | {reason}")
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
                logger.debug(f"SKIP {self.name}: risk_invalid (SL calculation failed) | {symbol}")
                return None
            
            confidence = self._bounded_confidence((rsi - overbought) / (100 - overbought))
            reason = f"RSI {rsi:.1f} above {overbought}, mean reversion short"
            logger.info(f"SIGNAL {self.name}: SELL | {reason}")
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

        logger.debug(f"SKIP {self.name}: neutral (rsi={rsi:.2f}) | {symbol}")
        return None


class EMACrossoverStrategy(Strategy):
    """EMA Crossover Strategy."""

    def __init__(
        self, fast_period: int = 9, slow_period: int = 21, default_quantity: int = 1
    ):
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
        return ["ema_fast", "ema_slow", "ema_fast_prev", "ema_slow_prev", "atr"]

    def generate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        logger.debug(f"EVAL {self.name}: {symbol}")

        fast = indicators.get("ema_fast")
        slow = indicators.get("ema_slow")
        fast_prev = indicators.get("ema_fast_prev")
        slow_prev = indicators.get("ema_slow_prev")
        atr = indicators.get("atr")

        if (fast is None or slow is None or fast_prev is None or slow_prev is None or atr is None):
            logger.debug(f"SKIP {self.name}: data_missing | {symbol}")
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
            "atr": atr,
        }

        crossed_up = fast_prev <= slow_prev and fast > slow
        crossed_down = fast_prev >= slow_prev and fast < slow

        if position:
            if position.side == "LONG" and crossed_down:
                confidence = self._bounded_confidence(abs(fast - slow) / atr)
                reason = "EMA death cross detected, exit long"
                logger.info(f"SIGNAL {self.name}: CLOSE_LONG | {reason}")
                return Signal("CLOSE_LONG", symbol, position.quantity, confidence, reason, None, None, metadata)
            
            if position.side == "SHORT" and crossed_up:
                confidence = self._bounded_confidence(abs(fast - slow) / atr)
                reason = "EMA golden cross detected, exit short"
                logger.info(f"SIGNAL {self.name}: CLOSE_SHORT | {reason}")
                return Signal("CLOSE_SHORT", symbol, position.quantity, confidence, reason, None, None, metadata)

        if crossed_up and (position is None or position.side != "LONG"):
            stop = slow
            stop_loss, take_profit = self._ensure_rr("BUY", current_price, stop, desired_rr)
            if stop_loss is None:
                logger.debug(f"SKIP {self.name}: risk_invalid (SL {stop:.2f}) | {symbol}")
                return None
            
            confidence = self._bounded_confidence(abs(fast - slow) / max(atr, 1e-6))
            reason = "Fast EMA crossed above slow EMA"
            logger.info(f"SIGNAL {self.name}: BUY | {reason}")
            return Signal("BUY", symbol, qty, confidence, reason, stop_loss, take_profit, metadata)

        if crossed_down and (position is None or position.side != "SHORT"):
            stop = slow
            stop_loss, take_profit = self._ensure_rr("SELL", current_price, stop, desired_rr)
            if stop_loss is None:
                logger.debug(f"SKIP {self.name}: risk_invalid (SL {stop:.2f}) | {symbol}")
                return None
            
            confidence = self._bounded_confidence(abs(fast - slow) / max(atr, 1e-6))
            reason = "Fast EMA crossed below slow EMA"
            logger.info(f"SIGNAL {self.name}: SELL | {reason}")
            return Signal("SELL", symbol, qty, confidence, reason, stop_loss, take_profit, metadata)

        logger.debug(f"SKIP {self.name}: neutral (no cross) | {symbol}")
        return None


class MACDStrategy(Strategy):
    """MACD Strategy."""

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        default_quantity: int = 1,
    ):
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
            "macd", "macd_signal", "macd_hist",
            "macd_prev", "macd_signal_prev",
            "atr", "support", "resistance",
        ]

    def generate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        logger.debug(f"EVAL {self.name}: {symbol}")

        macd = indicators.get("macd")
        signal_line = indicators.get("macd_signal")
        histogram = indicators.get("macd_hist")
        macd_prev = indicators.get("macd_prev")
        signal_prev = indicators.get("macd_signal_prev")
        atr = indicators.get("atr")

        if (macd is None or signal_line is None or histogram is None or 
            macd_prev is None or signal_prev is None or atr is None):
            logger.debug(f"SKIP {self.name}: data_missing | {symbol}")
            return None

        macd = float(macd)
        signal_line = float(signal_line)
        histogram = float(histogram)
        macd_prev = float(macd_prev)
        signal_prev = float(signal_prev)
        atr = float(atr)

        qty = int(self._parameters.get("default_quantity", 1))
        metadata = {"strategy": self.name, "macd": macd, "atr": atr}

        crossed_up = macd_prev <= signal_prev and macd > signal_line and histogram > 0
        crossed_down = macd_prev >= signal_prev and macd < signal_line and histogram < 0

        support = float(indicators.get("support") or (current_price - atr))
        resistance = float(indicators.get("resistance") or (current_price + atr))

        if position:
            if position.side == "LONG" and crossed_down:
                confidence = self._bounded_confidence(abs(macd - signal_line) / max(abs(histogram), 1e-6))
                reason = "MACD bearish crossover, exit long"
                logger.info(f"SIGNAL {self.name}: CLOSE_LONG | {reason}")
                return Signal("CLOSE_LONG", symbol, position.quantity, confidence, reason, None, None, metadata)
            
            if position.side == "SHORT" and crossed_up:
                confidence = self._bounded_confidence(abs(macd - signal_line) / max(abs(histogram), 1e-6))
                reason = "MACD bullish crossover, exit short"
                logger.info(f"SIGNAL {self.name}: CLOSE_SHORT | {reason}")
                return Signal("CLOSE_SHORT", symbol, position.quantity, confidence, reason, None, None, metadata)

        if crossed_up and (position is None or position.side != "LONG"):
            stop = min(support, current_price - atr)
            stop_loss, take_profit = self._ensure_rr("BUY", current_price, stop, desired_rr=2.0)
            if stop_loss is None:
                logger.debug(f"SKIP {self.name}: risk_invalid (SL {stop:.2f}) | {symbol}")
                return None
            
            momentum = abs(histogram) / max(abs(signal_line), 1e-6)
            confidence = self._bounded_confidence(0.5 + 0.5 * min(momentum, 1.0))
            reason = "MACD bullish crossover confirmed"
            logger.info(f"SIGNAL {self.name}: BUY | {reason}")
            return Signal("BUY", symbol, qty, confidence, reason, stop_loss, take_profit, metadata)

        if crossed_down and (position is None or position.side != "SHORT"):
            stop = max(resistance, current_price + atr)
            stop_loss, take_profit = self._ensure_rr("SELL", current_price, stop, desired_rr=2.0)
            if stop_loss is None:
                logger.debug(f"SKIP {self.name}: risk_invalid (SL {stop:.2f}) | {symbol}")
                return None
            
            momentum = abs(histogram) / max(abs(signal_line), 1e-6)
            confidence = self._bounded_confidence(0.5 + 0.5 * min(momentum, 1.0))
            reason = "MACD bearish crossover confirmed"
            logger.info(f"SIGNAL {self.name}: SELL | {reason}")
            return Signal("SELL", symbol, qty, confidence, reason, stop_loss, take_profit, metadata)

        logger.debug(f"SKIP {self.name}: neutral (no cross) | {symbol}")
        return None


class BollingerBandStrategy(Strategy):
    """Bollinger Band Strategy."""

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        default_quantity: int = 1,
    ):
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
        return ["bb_upper", "bb_lower", "bb_middle", "rsi", "atr"]

    def generate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        logger.debug(f"EVAL {self.name}: {symbol}")

        upper = indicators.get("bb_upper")
        lower = indicators.get("bb_lower")
        middle = indicators.get("bb_middle")
        rsi = indicators.get("rsi")
        atr = indicators.get("atr")

        if (upper is None or lower is None or middle is None or rsi is None or atr is None):
            logger.debug(f"SKIP {self.name}: data_missing | {symbol}")
            return None

        upper = float(upper)
        lower = float(lower)
        middle = float(middle)
        rsi = float(rsi)
        atr = float(atr)

        qty = int(self._parameters.get("default_quantity", 1))
        metadata = {"strategy": self.name, "bb_upper": upper, "bb_lower": lower, "rsi": rsi}

        if position:
            if position.side == "LONG" and current_price >= middle:
                confidence = self._bounded_confidence((current_price - middle) / max(atr, 1e-6))
                reason = "Price reverted to middle band, closing long"
                logger.info(f"SIGNAL {self.name}: CLOSE_LONG | {reason}")
                return Signal("CLOSE_LONG", symbol, position.quantity, confidence, reason, None, None, metadata)
            
            if position.side == "SHORT" and current_price <= middle:
                confidence = self._bounded_confidence((middle - current_price) / max(atr, 1e-6))
                reason = "Price reverted to middle band, closing short"
                logger.info(f"SIGNAL {self.name}: CLOSE_SHORT | {reason}")
                return Signal("CLOSE_SHORT", symbol, position.quantity, confidence, reason, None, None, metadata)

        if current_price <= lower and rsi < 40 and (position is None or position.side != "LONG"):
            stop = min(lower - 0.5 * atr, current_price - atr)
            stop_loss, take_profit = self._ensure_rr("BUY", current_price, stop, desired_rr=2.0)
            if stop_loss is None:
                logger.debug(f"SKIP {self.name}: risk_invalid (SL {stop:.2f}) | {symbol}")
                return None
            
            distance = lower - current_price
            confidence = self._bounded_confidence(0.5 + 0.5 * min(abs(distance) / max(atr, 1e-6), 1.0))
            reason = "Price at lower band with oversold RSI"
            logger.info(f"SIGNAL {self.name}: BUY | {reason}")
            return Signal("BUY", symbol, qty, confidence, reason, stop_loss, take_profit, metadata)

        if current_price >= upper and rsi > 60 and (position is None or position.side != "SHORT"):
            stop = max(upper + 0.5 * atr, current_price + atr)
            stop_loss, take_profit = self._ensure_rr("SELL", current_price, stop, desired_rr=2.0)
            if stop_loss is None:
                logger.debug(f"SKIP {self.name}: risk_invalid (SL {stop:.2f}) | {symbol}")
                return None
            
            distance = current_price - upper
            confidence = self._bounded_confidence(0.5 + 0.5 * min(abs(distance) / max(atr, 1e-6), 1.0))
            reason = "Price at upper band with overbought RSI"
            logger.info(f"SIGNAL {self.name}: SELL | {reason}")
            return Signal("SELL", symbol, qty, confidence, reason, stop_loss, take_profit, metadata)

        logger.debug(f"SKIP {self.name}: neutral | {symbol}")
        return None


class OpeningRangeBreakoutStrategy(Strategy):
    """Opening Range Breakout Strategy."""

    def __init__(
        self,
        *,
        opening_minutes: int = 30,
        volume_spike_ratio: float = 1.5,
        premium_stop_pct: float = 0.35,
        premium_target_rr: float = 2.5,
        default_quantity: int = 1,
    ) -> None:
        parameters = {
            "opening_minutes": opening_minutes,
            "volume_spike_ratio": volume_spike_ratio,
            "premium_stop_pct": premium_stop_pct,
            "premium_target_rr": premium_target_rr,
            "default_quantity": default_quantity,
        }
        super().__init__("Opening Range Breakout", parameters)

    def get_required_indicators(self) -> list[str]:
        return [
            "orb_high", "orb_low", "orb_ready",
            "nr7", "nr7_range", "nr7_min_range",
            "futures_volume_ratio",
            "minutes_since_open", "minutes_until_close",
            "atr",
        ]

    def generate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        logger.debug(f"EVAL {self.name}: {symbol}")

        try:
            orb_high = indicators.get("orb_high")
            orb_low = indicators.get("orb_low")
            orb_ready = indicators.get("orb_ready")
            atr = indicators.get("atr")
            nr7_flag = bool(indicators.get("nr7"))
            volume_ratio = indicators.get("futures_volume_ratio")
            minutes_since_open = indicators.get("minutes_since_open")
            minutes_until_close = indicators.get("minutes_until_close")

            if (orb_high is None or orb_low is None or not bool(orb_ready) or atr is None
                or minutes_since_open is None or minutes_until_close is None):
                logger.debug(f"SKIP {self.name}: data_missing | {symbol}")
                return None

            parameters = self._parameters
            min_volume_ratio = float(parameters["volume_spike_ratio"])
            
            if volume_ratio is None or float(volume_ratio) < min_volume_ratio:
                logger.debug(f"SKIP {self.name}: volume_filter ({volume_ratio} < {min_volume_ratio}) | {symbol}")
                return None
            if not nr7_flag:
                logger.debug(f"SKIP {self.name}: nr7_filter | {symbol}")
                return None
            if float(minutes_since_open) < float(parameters["opening_minutes"]):
                logger.debug(f"SKIP {self.name}: pre_window | {symbol}")
                return None
            if float(minutes_until_close) < 20.0:
                logger.debug(f"SKIP {self.name}: near_close | {symbol}")
                return None

            qty = int(parameters.get("default_quantity", 1))
            atr_value = float(atr)
            metadata = {
                "strategy": self.name,
                "orb_high": float(orb_high),
                "orb_low": float(orb_low),
                "atr": atr_value,
                "premium_stop_pct": float(parameters["premium_stop_pct"]),
                "premium_target_rr": float(parameters["premium_target_rr"]),
            }

            if position is not None:
                if position.side == "LONG" and current_price < float(orb_low):
                    reason = "Price lost opening range low"
                    logger.info(f"SIGNAL {self.name}: CLOSE_LONG | {reason}")
                    return Signal("CLOSE_LONG", symbol, position.quantity, self._bounded_confidence(0.6), reason, None, None, metadata)
                
                if position.side == "SHORT" and current_price > float(orb_high):
                    reason = "Price reclaimed opening range high"
                    logger.info(f"SIGNAL {self.name}: CLOSE_SHORT | {reason}")
                    return Signal("CLOSE_SHORT", symbol, position.quantity, self._bounded_confidence(0.6), reason, None, None, metadata)

            if current_price > float(orb_high) and (position is None or position.side != "LONG"):
                distance = current_price - float(orb_high)
                confidence = self._bounded_confidence(distance / max(atr_value, 1.0))
                reason = "ORB breakout above opening high"
                logger.info(f"SIGNAL {self.name}: BUY | {reason}")
                return Signal("BUY", symbol, qty, confidence, reason, None, None, metadata)

            if current_price < float(orb_low) and (position is None or position.side != "SHORT"):
                distance = float(orb_low) - current_price
                confidence = self._bounded_confidence(distance / max(atr_value, 1.0))
                reason = "ORB breakdown below opening low"
                logger.info(f"SIGNAL {self.name}: SELL | {reason}")
                return Signal("SELL", symbol, qty, confidence, reason, None, None, metadata)

        except Exception as exc:
            logger.error(f"Failure in {self.name}.generate_signal: {exc}", extra={"symbol": symbol})
        
        logger.debug(f"SKIP {self.name}: neutral | {symbol}")
        return None


class VWAPMeanReversionStrategy(Strategy):
    """VWAP Mean Reversion Strategy."""

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
        return [
            "vwap", "rsi", "atr",
            "minutes_until_close", "minutes_since_open",
            "futures_volume_ratio",
        ]

    def generate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        logger.debug(f"EVAL {self.name}: {symbol}")

        try:
            vwap = indicators.get("vwap")
            rsi = indicators.get("rsi")
            atr = indicators.get("atr")
            minutes_until_close = indicators.get("minutes_until_close")
            minutes_since_open = indicators.get("minutes_since_open")

            if (vwap is None or rsi is None or atr is None or 
                minutes_until_close is None or minutes_since_open is None):
                logger.debug(f"SKIP {self.name}: data_missing | {symbol}")
                return None

            if float(minutes_since_open) < 20.0 or float(minutes_until_close) < 15.0:
                logger.debug(f"SKIP {self.name}: time_window | {symbol}")
                return None

            parameters = self._parameters
            deviation_pct = float(parameters["deviation_pct"])
            oversold = float(parameters["rsi_oversold"])
            overbought = float(parameters["rsi_overbought"])
            qty = int(parameters.get("default_quantity", 1))
            atr_value = float(atr)

            metadata = {
                "strategy": self.name,
                "vwap": float(vwap),
                "rsi": float(rsi),
                "premium_stop_pct": float(parameters["premium_stop_pct"]),
                "premium_target_rr": float(parameters["premium_target_rr"]),
                "atr": atr_value,
            }

            upper_threshold = float(vwap) * (1.0 + deviation_pct)
            lower_threshold = float(vwap) * (1.0 - deviation_pct)

            if position is not None:
                if position.side == "LONG" and current_price >= float(vwap):
                    reason = "Price reverted to VWAP"
                    logger.info(f"SIGNAL {self.name}: CLOSE_LONG | {reason}")
                    return Signal("CLOSE_LONG", symbol, position.quantity, self._bounded_confidence(0.55), reason, None, None, metadata)
                
                if position.side == "SHORT" and current_price <= float(vwap):
                    reason = "Price reverted to VWAP"
                    logger.info(f"SIGNAL {self.name}: CLOSE_SHORT | {reason}")
                    return Signal("CLOSE_SHORT", symbol, position.quantity, self._bounded_confidence(0.55), reason, None, None, metadata)

            if current_price < lower_threshold and float(rsi) <= oversold and (position is None or position.side != "LONG"):
                distance = float(vwap) - current_price
                confidence = self._bounded_confidence(distance / max(atr_value, 1.0))
                reason = "Price below VWAP with oversold RSI"
                logger.info(f"SIGNAL {self.name}: BUY | {reason}")
                return Signal("BUY", symbol, qty, confidence, reason, None, None, metadata)

            if current_price > upper_threshold and float(rsi) >= overbought and (position is None or position.side != "SHORT"):
                distance = current_price - float(vwap)
                confidence = self._bounded_confidence(distance / max(atr_value, 1.0))
                reason = "Price above VWAP with overbought RSI"
                logger.info(f"SIGNAL {self.name}: SELL | {reason}")
                return Signal("SELL", symbol, qty, confidence, reason, None, None, metadata)

        except Exception as exc:
            logger.error(f"Failure in {self.name}.generate_signal: {exc}", extra={"symbol": symbol})
        
        logger.debug(f"SKIP {self.name}: neutral | {symbol}")
        return None


class StrategyManager:
    """Coordinating manager for multiple strategies."""

    def __init__(
        self,
        strategies: list[Strategy],
        indicator_engine: IndicatorEngine,
        position_manager: PositionManager,
        min_confidence: float = 0.60,
        data_hub: Any | None = None,
        orchestrator: Any | None = None,
        futures_symbol: str | None = None,
        config: Any | None = None
    ):
        self._strategies = strategies
        self._indicator_engine = indicator_engine
        self._position_manager = position_manager
        self._min_confidence = min_confidence
        self._data_hub = data_hub
        self._orchestrator = orchestrator
        self._futures_symbol = (futures_symbol or "NIFTY").strip().upper()
        self._futures_volume_history: Deque[float] = deque(maxlen=120)
        
        raw_config = config if config else (strategies[0].config if strategies else {})
        self._config = raw_config

        def get_cfg(key: str, default: Any) -> Any:
            if isinstance(raw_config, dict):
                return raw_config.get(key, default)
            if hasattr(raw_config, key):
                return getattr(raw_config, key)
            if hasattr(raw_config, key.lower()):
                return getattr(raw_config, key.lower())
            return default

        try:
            self.sl_pct = float(get_cfg("OPTION_SL_PCT", 0.15))
            self.tp_pct = float(get_cfg("OPTION_TP_PCT", 0.30))
            self.min_delta = float(get_cfg("DATA__MIN_DELTA", 0.30))
            self.max_iv_percentile = float(get_cfg("DATA__MAX_IV_PERCENTILE", 85.0))
            self.max_spread_pct = float(get_cfg("MAX_BID_ASK_SPREAD", 5.0))
        except (ValueError, TypeError):
            logger.warning("Failed to parse risk settings, using defaults.")
            self.sl_pct = 0.15
            self.tp_pct = 0.30
            self.min_delta = 0.30
            self.max_iv_percentile = 85.0
            self.max_spread_pct = 5.0

    def generate_signal(self, symbol: str, current_price: float) -> Signal | None:
        """
        MASTER EXECUTION LOOP.
        """
        logger.debug(
            "Entered StrategyManager.generate_signal",
            extra={"event": "strategy_manager_generate", "symbol": symbol},
        )

        required: set[str] = {"volume", "avg_volume", "minutes_since_open", "minutes_until_close"}
        # ✅ FIX: Evaluate ALL strategies, collect ALL signals (no short-circuit)
        all_signals: list[Signal] = []
        eval_count = 0
        skip_count = 0

        # Get bar count for strategy-specific gating
        bar_count = int(indicators.get('bar_count', 100))

        for strategy in self._strategies:
            # ✅ FIX: Check strategy-specific min_bars requirement
            strategy_min_bars = getattr(strategy, 'MIN_BARS_REQUIRED', 3)
            
            if bar_count < strategy_min_bars:
                logger.debug(
                    f"SKIP {strategy.name}: need {strategy_min_bars} bars, have {bar_count} | {symbol}"
                )
                skip_count += 1
                continue
            
            # Check required indicators
            reqs = strategy.get_required_indicators()
            missing = [k for k in reqs if k not in indicators or indicators[k] is None]
            
            if missing:
                logger.debug(f"SKIP {strategy.name}: missing_indicators {missing} | {symbol}")
                skip_count += 1
                continue

            try:
                # Skip momentum strategies in dead markets
                if vix < 12.0 and ("Breakout" in strategy.name or "ORB" in strategy.name):
                    logger.debug(f"SKIP {strategy.name}: low_vix_filter | {symbol}")
                    skip_count += 1
                    continue

                eval_count += 1
                signal = strategy.generate_signal(symbol, indicators, current_price, position)
                
                if signal:
                    # Apply regime factor
                    if regime_factor < 1.0:
                        signal = dataclasses.replace(
                            signal, 
                            confidence=signal.confidence * regime_factor,
                        )
                    
                    # ✅ Collect ALL signals with strategy attribution
                    all_signals.append(signal.with_metadata(
                        indicators=indicators,
                        source_strategy=strategy.name,
                    ))
                    
                    logger.info(
                        f"📊 Signal from {strategy.name}: {signal.action} | conf={signal.confidence:.2f}",
                        extra={
                            "event": "strategy_signal_generated",
                            "strategy": strategy.name,
                            "symbol": symbol,
                            "action": signal.action,
                            "confidence": signal.confidence,
                        }
                    )
                    
            except Exception as exc: 
                logger.exception(f"Strategy {strategy.name} failed: {exc}")
                continue

        logger.debug(
            f"StrategyManager Stats: Evaluated={eval_count}, Skipped={skip_count}, "
            f"Signals={len(all_signals)} | {symbol}"
        )

        if not all_signals:
            return None

        # ✅ Combine ALL signals using ensemble voting
        combined = self._combine_signals_ensemble(all_signals)
                    
            except Exception as exc: 
                logger.exception("Strategy %s failed: %s", strategy.name, exc)
                continue

        if not signals:
            logger.debug(f"StrategyManager Stats: Evaluated={eval_count}, Skipped={skip_count}, Signals=0 | {symbol}")
            return None

        combined = self._combine_signals(signals)
        if not combined:
            return None

        if ("CE" in symbol or "PE" in symbol) and not self._validate_option_physics(symbol, combined.action):
            logger.info(f"⛔ Rejected {symbol}: Failed Physics Check (Greeks/Liq)")
            return None

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

    def _combine_signals_ensemble(self, signals: list[Signal]) -> Signal | None:
        """
        ✅ NEW: Combine multiple strategy signals using weighted voting.
        
        Ensures all strategies get fair consideration instead of VWAP dominance.
        """
        if not signals:
            return None
        
        if len(signals) == 1:
            return signals[0]
        
        # Separate by action
        buy_signals = [s for s in signals if s.action == "BUY"]
        sell_signals = [s for s in signals if s.action == "SELL"]
        
        # Calculate weighted votes
        buy_weight = sum(s.confidence for s in buy_signals)
        sell_weight = sum(s.confidence for s in sell_signals)
        
        logger.info(
            f"📊 Ensemble: BUY={len(buy_signals)} ({buy_weight:.1f}), "
            f"SELL={len(sell_signals)} ({sell_weight:.1f})"
        )
        
        # Determine winner
        if buy_weight > sell_weight and buy_signals:
            # Use the highest confidence BUY signal as base
            best = max(buy_signals, key=lambda s: s.confidence)
            # Average confidence across all BUY signals
            avg_conf = buy_weight / len(buy_signals)
            return dataclasses.replace(
                best, 
                confidence=min(avg_conf, 99.0),
                metadata={
                    **(best.metadata or {}),
                    "ensemble_count": len(buy_signals),
                    "ensemble_weight": round(buy_weight, 2),
                }
            )
        elif sell_signals:
            best = max(sell_signals, key=lambda s: s.confidence)
            avg_conf = sell_weight / len(sell_signals)
            return dataclasses.replace(
                best, 
                confidence=min(avg_conf, 99.0),
                metadata={
                    **(best.metadata or {}),
                    "ensemble_count": len(sell_signals),
                    "ensemble_weight": round(sell_weight, 2),
                }
            )
        
        return None
        

    def _get_regime_modifier(self, vix: float) -> float:
        if vix > 24.0: return 0.8
        if vix < 11.0: return 0.9
        return 1.0

    def _validate_option_physics(self, symbol: str, action: str) -> bool:
        """Rejects 'Garbage Options' based on Greeks, Spread, and Liquidity."""
        quote = self._indicator_engine.get_indicators(symbol, ["bid", "ask"])
        if quote:
            bid = float(quote.get('bid', 0) or 0)
            ask = float(quote.get('ask', 0) or 0)
            if bid > 0:
                spread = ((ask - bid) / bid) * 100
                if spread > self.max_spread_pct:
                    logger.debug(f"Physics REJECT {symbol}: Spread {spread:.2f}% > {self.max_spread_pct}%")
                    return False

        greeks = self._indicator_engine.get_indicators(symbol, ["delta", "theta", "iv_percentile"])
        if not greeks: return True

        delta = abs(float(greeks.get("delta") or 0.5))
        if delta < self.min_delta:
            logger.debug(f"Physics REJECT {symbol}: Delta {delta:.2f} < {self.min_delta}")
            return False

        theta = float(greeks.get("theta") or 0.0)
        if action == "BUY" and theta < -20.0:
            logger.debug(f"Physics REJECT {symbol}: Theta Burn {theta:.2f} < -20.0")
            return False

        return True

    def _calculate_premium_risk(self, symbol: str, side: str, price: float) -> tuple[float, float] | None:
        if price <= 0: return None
        
        if side == "BUY":
            sl = round(price * (1 - self.sl_pct), 2)
            tp = round(price * (1 + self.tp_pct), 2)
        else:
            sl = round(price * (1 + self.sl_pct), 2)
            tp = round(price * (1 - self.tp_pct), 2)
            
        if abs(price - sl) < (price * 0.05):
            logger.debug(f"Risk REJECT {symbol}: SL too close (<5%)")
            return None
        return (sl, tp)

    def _augment_futures_metrics(self, indicators: MutableMapping[str, Any]) -> None:
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
        by_action = defaultdict(list)

        # Collect actionable signals only
        for sig in signals:
            if sig.action != "HOLD":
                by_action[sig.action].append(sig)

        # Nothing actionable at all
        if not by_action:
            logger.debug("No actionable signals after filtering HOLD")
            return None

        # 1️⃣ Hard priority: exit signals always win
        for exit_act in ("CLOSE_LONG", "CLOSE_SHORT"):
            if exit_act in by_action:
                chosen = max(by_action[exit_act], key=lambda s: s.confidence)
                logger.debug(f"Exit signal selected: {exit_act} @ {chosen.confidence:.2f}")
                return chosen

        # 2️⃣ BUY vs SELL conflict resolution
        if {"BUY", "SELL"}.issubset(by_action.keys()):
            buy_list = by_action["BUY"]
            sell_list = by_action["SELL"]

            buy_conf = sum(s.confidence for s in buy_list) / len(buy_list)
            sell_conf = sum(s.confidence for s in sell_list) / len(sell_list)
            diff = abs(buy_conf - sell_conf)

            if diff < 0.15:
                # 🔴 CRITICAL FIX: Return None (SKIP) instead of crashing on self.symbol
                # StrategyManager does not have self.symbol/self.name attributes.
                logger.debug(
                    f"Consensus WEAK -> SKIP "
                    f"(Buy:{buy_conf:.2f} Sell:{sell_conf:.2f} Diff:{diff:.2f})"
                )
                return None

        # 3️⃣ Select best action by average confidence
        best_action, selected_list = max(
            by_action.items(),
            key=lambda i: sum(s.confidence for s in i[1]) / len(i[1]),
        )

        best_signal = max(selected_list, key=lambda s: s.confidence)

        # Single strategy -> return as-is
        if len(selected_list) == 1:
            logger.debug(
                f"Single signal selected: {best_signal.action} @ {best_signal.confidence:.2f}"
            )
            return best_signal

        # 4️⃣ Multi-strategy consensus boost (bounded)
        avg_conf = sum(s.confidence for s in selected_list) / len(selected_list)
        final_conf = min(1.0, max(0.0, avg_conf + 0.05))

        reasons = ", ".join(s.reason for s in selected_list if s.reason)
        meta = dict(best_signal.metadata or {})
        
        # Safely extract confirming strategies
        confirming = []
        for s in selected_list:
            if s.metadata and "strategy" in s.metadata:
                confirming.append(s.metadata["strategy"])
        
        if confirming:
            meta["confirming_strategies"] = confirming

        logger.debug(
            f"Consensus signal: {best_signal.action} "
            f"Avg:{avg_conf:.2f} Final:{final_conf:.2f} "
            f"Strategies:{len(selected_list)}"
        )

        return dataclasses.replace(
            best_signal,
            confidence=final_conf,
            reason=f"Consensus: {reasons}" if reasons else "Consensus signal",
            metadata=meta,
        )
    def _filter_signal(self, signal: Signal) -> bool:
        if signal.confidence < self._min_confidence:
            logger.debug(f"Filter REJECT {signal.symbol}: Low Confidence ({signal.confidence:.2f} < {self._min_confidence})")
            return False

        indicators = signal.metadata.get("indicators", {}) if isinstance(signal.metadata, dict) else {}

        vol = indicators.get("volume")
        avg_vol = indicators.get("avg_volume")
        if isinstance(vol, (int, float)) and isinstance(avg_vol, (int, float)) and avg_vol > 0:
            if float(vol) < 0.75 * float(avg_vol):
                logger.debug(f"Filter REJECT {signal.symbol}: Low Volume")
                return False

        m_time = indicators.get("market_time")
        if isinstance(m_time, datetime): m_time = m_time.time()
        if isinstance(m_time, time):
            if m_time < time(9, 30) or m_time > time(15, 15):
                logger.debug(f"Filter REJECT {signal.symbol}: Time Window")
                return False

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
