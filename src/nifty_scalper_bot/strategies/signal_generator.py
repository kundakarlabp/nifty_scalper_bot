"""
Advanced signal generation utilities.
Production-Grade implementation with Greeks validation, Regime awareness, and Premium-based Risk.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Deque, Iterable, Literal, Mapping, MutableMapping, Protocol, List, Tuple, Optional

from nifty_scalper_bot.utils.logging import get_logger

logger = get_logger(__name__)


# ==============================================================================
# 1. CORE PROTOCOLS & DATA STRUCTURES
# ==============================================================================

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
    ) -> Mapping[str, float | tuple[float, ...] | None]:
        """Return the requested indicator snapshot for *symbol*."""
    
    def get_quote(self, symbol: str) -> dict[str, Any] | None:
        """Return full market depth/quote."""


class PositionManager(Protocol):
    """Protocol describing the required position manager behaviour."""

    def get_position(self, symbol: str) -> Position | None:
        """Return the active position for *symbol*, if any."""


@dataclass(frozen=True)
class Signal:
    """Enhanced trading signal."""
    action: Literal["BUY", "SELL", "CLOSE_LONG", "CLOSE_SHORT"]
    symbol: str
    confidence: float  # 0.0 to 1.0
    price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tag: str | None = None


class Strategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, config: dict[str, Any], indicator_engine: IndicatorEngine):
        self.config = config
        self.indicators = indicator_engine

    @abstractmethod
    def generate_signal(self) -> Signal | None:
        """Evaluate market data and return a signal if conditions met."""


# ==============================================================================
# 2. INDIVIDUAL STRATEGY LOGIC (Preserved from Original)
# ==============================================================================

class RSIMeanReversionStrategy(Strategy):
    """RSI Mean Reversion Strategy."""

    def generate_signal(self) -> Signal | None:
        symbol = self.config.get("symbol", "NIFTY")
        indicators = self.indicators.get_indicators(symbol, ["rsi", "ltp"])
        rsi = indicators.get("rsi")
        ltp = indicators.get("ltp")

        if not isinstance(rsi, (int, float)) or not isinstance(ltp, (int, float)):
            return None

        rsi_period = int(self.config.get("rsi_period", 14))
        oversold = float(self.config.get("rsi_oversold", 30))
        overbought = float(self.config.get("rsi_overbought", 70))

        if rsi < oversold:
            return Signal(
                action="BUY",
                symbol=symbol,
                confidence=min(1.0, (oversold - rsi) / 10 + 0.5),
                price=float(ltp),
                metadata={"strategy": "rsi_mean_reversion", "rsi": rsi},
                tag="RSI_Oversold"
            )
        elif rsi > overbought:
            return Signal(
                action="SELL",
                symbol=symbol,
                confidence=min(1.0, (rsi - overbought) / 10 + 0.5),
                price=float(ltp),
                metadata={"strategy": "rsi_mean_reversion", "rsi": rsi},
                tag="RSI_Overbought"
            )
        return None


class EMACrossoverStrategy(Strategy):
    """EMA Crossover Strategy."""

    def generate_signal(self) -> Signal | None:
        symbol = self.config.get("symbol", "NIFTY")
        short_window = int(self.config.get("ema_short", 9))
        long_window = int(self.config.get("ema_long", 21))
        
        indicators = self.indicators.get_indicators(
            symbol, [f"ema_{short_window}", f"ema_{long_window}", "ltp"]
        )
        ema_short = indicators.get(f"ema_{short_window}")
        ema_long = indicators.get(f"ema_{long_window}")
        ltp = indicators.get("ltp")

        if (
            not isinstance(ema_short, (int, float))
            or not isinstance(ema_long, (int, float))
            or not isinstance(ltp, (int, float))
        ):
            return None

        # Determine crossover logic (simplified for example)
        # In a real implementation, you'd check previous values to confirm the "cross"
        if ema_short > ema_long * 1.001:  # 0.1% buffer
            return Signal(
                action="BUY",
                symbol=symbol,
                confidence=0.7,
                price=float(ltp),
                metadata={"strategy": "ema_crossover", "ema_diff": ema_short - ema_long},
                tag="EMA_Cross_Bull"
            )
        elif ema_short < ema_long * 0.999:
            return Signal(
                action="SELL",
                symbol=symbol,
                confidence=0.7,
                price=float(ltp),
                metadata={"strategy": "ema_crossover", "ema_diff": ema_long - ema_short},
                tag="EMA_Cross_Bear"
            )
        return None


class MACDStrategy(Strategy):
    """MACD Strategy."""

    def generate_signal(self) -> Signal | None:
        symbol = self.config.get("symbol", "NIFTY")
        indicators = self.indicators.get_indicators(symbol, ["macd", "macd_signal", "ltp"])
        macd = indicators.get("macd")
        signal_line = indicators.get("macd_signal")
        ltp = indicators.get("ltp")

        if (
            not isinstance(macd, (int, float))
            or not isinstance(signal_line, (int, float))
            or not isinstance(ltp, (int, float))
        ):
            return None

        if macd > signal_line:
            return Signal(
                action="BUY",
                symbol=symbol,
                confidence=0.6,
                price=float(ltp),
                metadata={"strategy": "macd", "hist": macd - signal_line},
                tag="MACD_Bull"
            )
        elif macd < signal_line:
            return Signal(
                action="SELL",
                symbol=symbol,
                confidence=0.6,
                price=float(ltp),
                metadata={"strategy": "macd", "hist": macd - signal_line},
                tag="MACD_Bear"
            )
        return None


class BollingerBandStrategy(Strategy):
    """Bollinger Band Mean Reversion Strategy."""

    def generate_signal(self) -> Signal | None:
        symbol = self.config.get("symbol", "NIFTY")
        indicators = self.indicators.get_indicators(
            symbol, ["bb_upper", "bb_lower", "ltp"]
        )
        upper = indicators.get("bb_upper")
        lower = indicators.get("bb_lower")
        ltp = indicators.get("ltp")

        if (
            not isinstance(upper, (int, float))
            or not isinstance(lower, (int, float))
            or not isinstance(ltp, (int, float))
        ):
            return None

        if ltp < lower:
            return Signal(
                action="BUY",
                symbol=symbol,
                confidence=0.8,
                price=float(ltp),
                metadata={"strategy": "bb_reversion", "deviation": lower - ltp},
                tag="BB_Lower_Bounce"
            )
        elif ltp > upper:
            return Signal(
                action="SELL",
                symbol=symbol,
                confidence=0.8,
                price=float(ltp),
                metadata={"strategy": "bb_reversion", "deviation": ltp - upper},
                tag="BB_Upper_Reject"
            )
        return None


class OpeningRangeBreakoutStrategy(Strategy):
    """Opening Range Breakout (ORB) Strategy."""

    def __init__(self, config: dict[str, Any], indicator_engine: IndicatorEngine):
        super().__init__(config, indicator_engine)
        self.orb_high: float | None = None
        self.orb_low: float | None = None
        self.orb_period_minutes = int(self.config.get("orb_period_minutes", 15))
        self.orb_start_time = time(9, 15)
        
        # Calculate end time based on start + minutes
        # Simplified: defaulting to 9:30 for 15 min ORB
        self.orb_end_time = time(9, 15 + self.orb_period_minutes) 

    def generate_signal(self) -> Signal | None:
        symbol = self.config.get("symbol", "NIFTY")
        
        # NOTE: In a real implementation, 'ltp' would come from tick updates
        # Here we assume indicator engine has the latest tick
        indicators = self.indicators.get_indicators(symbol, ["ltp", "market_time"])
        ltp = indicators.get("ltp")
        market_time = indicators.get("market_time") # Expecting datetime object

        if not isinstance(ltp, (int, float)) or not isinstance(market_time, datetime):
            return None

        current_time = market_time.time()

        # 1. Define Range
        if self.orb_start_time <= current_time <= self.orb_end_time:
            if self.orb_high is None or ltp > self.orb_high:
                self.orb_high = float(ltp)
            if self.orb_low is None or ltp < self.orb_low:
                self.orb_low = float(ltp)
            return None

        if self.orb_high is None or self.orb_low is None:
            return None

        # 2. Check Breakout
        # Filter: Don't take ORB signals late in the day (e.g. after 10:30)
        if current_time > time(10, 30):
            return None

        if ltp > self.orb_high:
            return Signal(
                action="BUY",
                symbol=symbol,
                confidence=0.9,
                price=float(ltp),
                metadata={"strategy": "orb", "breakout": "high"},
                tag="ORB_High_Break"
            )
        elif ltp < self.orb_low:
            return Signal(
                action="SELL",
                symbol=symbol,
                confidence=0.9,
                price=float(ltp),
                metadata={"strategy": "orb", "breakout": "low"},
                tag="ORB_Low_Break"
            )
        return None


class VWAPMeanReversionStrategy(Strategy):
    """VWAP Mean Reversion Strategy."""

    def generate_signal(self) -> Signal | None:
        symbol = self.config.get("symbol", "NIFTY")
        indicators = self.indicators.get_indicators(symbol, ["vwap", "ltp"])
        vwap = indicators.get("vwap")
        ltp = indicators.get("ltp")

        if not isinstance(vwap, (int, float)) or not isinstance(ltp, (int, float)):
            return None

        # Logic: If price deviates significantly from VWAP, bet on return
        deviation_threshold = 0.005 # 0.5%
        
        if ltp < vwap * (1 - deviation_threshold):
            return Signal(
                action="BUY",
                symbol=symbol,
                confidence=0.65,
                price=float(ltp),
                metadata={"strategy": "vwap_reversion", "dist": vwap - ltp},
                tag="VWAP_Oversold"
            )
        elif ltp > vwap * (1 + deviation_threshold):
            return Signal(
                action="SELL",
                symbol=symbol,
                confidence=0.65,
                price=float(ltp),
                metadata={"strategy": "vwap_reversion", "dist": ltp - vwap},
                tag="VWAP_Overbought"
            )
        return None


# ==============================================================================
# 3. THE STRATEGY MANAGER (Production Grade Orchestrator)
# ==============================================================================

class StrategyManager:
    """
    The 'Brain' of the bot.
    Orchestrates the strategies, validates physics (Greeks), and enforces Risk/Reward.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        indicator_engine: IndicatorEngine,
        position_manager: PositionManager,
        min_confidence: float = 0.60,
        data_hub: Any | None = None,
        config: dict[str, Any] | None = None
    ):
        self._strategies = strategies
        self._indicators = indicator_engine
        self._positions = position_manager
        self._min_confidence = min_confidence
        self._data_hub = data_hub
        
        # Try to infer config from the first strategy if not passed explicitly
        self._config = config if config else (strategies[0].config if strategies else {})
        
        self.logger = logger
        
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
            vix_val = vix_data.get("ltp")
            
            # Default to neutral if VIX data missing
            vix = float(vix_val) if isinstance(vix_val, (int, float)) else 15.0
            
            is_panic = vix > 24.0
            is_dead = vix < 11.0
            
            candidates: List[Tuple[float, Signal]] = []

            # 2. STRATEGY POLLING
            for strategy in self._strategies:
                strategy_name = strategy.__class__.__name__
                
                # Optimization: Skip Momentum strategies in Dead markets
                if is_dead and ("Breakout" in strategy_name or "ORB" in strategy_name):
                    continue

                try:
                    # Run the strategy logic
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

                # Filter low confidence signals immediately
                if final_confidence < self._min_confidence:
                    continue

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
                    "tp": best_signal.take_profit,
                    "tag": best_signal.tag
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
                    self.logger.warning(
                        f"⛔ Rejected {symbol}: Spread {spread_pct:.2f}% too wide",
                        extra={"symbol": symbol, "spread": spread_pct}
                    )
                    return False

        # B. GREEKS CHECK
        greeks = self._indicators.get_indicators(symbol, ["delta", "theta", "iv_percentile"])
        if not greeks:
            return True # Fail-open if no data (safety choice)

        # Delta (Momentum)
        delta_val = greeks.get("delta")
        delta = abs(float(delta_val)) if delta_val is not None else 0.5
        
        if delta < self.min_delta:
            self.logger.debug(f"⛔ Rejected {symbol}: Low Delta {delta:.2f}")
            return False

        # Theta (Decay) - Assuming negative theta
        theta_val = greeks.get("theta")
        theta = float(theta_val) if theta_val is not None else 0.0
        
        if side == "BUY" and theta < -20.0:
            self.logger.debug(f"⛔ Rejected {symbol}: High Theta Burn {theta}")
            return False

        # IV Percentile (Price)
        iv_val = greeks.get("iv_percentile")
        iv_pct = float(iv_val) if iv_val is not None else 50.0
        
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
            self.logger.warning(
                f"⛔ Rejected {symbol}: SL too tight",
                extra={"symbol": symbol, "entry": entry_price, "sl": stop_loss}
            )
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
