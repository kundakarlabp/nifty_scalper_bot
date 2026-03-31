"""
Gamma Scalping Strategy.
World-Class implementation with Greeks Validation (Gamma/Theta Efficiency) and Momentum.
Refactored for Push-Based Architecture (Zero-Latency).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    GammaScalpingStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class GammaScalpingStrategy(EliteStrategy):
    """
    Trade directional gamma edges.
    Captures explosive moves where Gamma (Acceleration) justifies the Theta (Decay) cost.
    Entry: High Momentum + Positive Gamma Environment.
    """

    MIN_BARS_REQUIRED = 3

    # ✅ OPTIMIZATION: Use slots for memory efficiency
    __slots__ = ("_gamma_config",)

    def __init__(
        self, config: GammaScalpingStrategyConfig, indicator_engine: Any
    ) -> None:
        """
        Initialize strategy with configuration and engine.

        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._gamma_config = config

    def get_required_indicators(self) -> set[str]:
        """
        Declare which indicators this strategy needs pre-calculated.
        The StrategyManager will inject these into _evaluate_signal.
        """
        return {
            "gamma",
            "theta",
            "delta",
            "ltp",
            "volume",
            "avg_volume",
            "macd",  # Momentum Trigger
            "macd_signal",  # Signal Line
            "atr",  # Volatility for stops
            "futures_vwap",
        }

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        LOGGER.debug(
            "Entered GammaScalpingStrategy._evaluate_signal",
            extra={"event": "gamma_scalping_enter", "symbol": symbol},
        )
        try:
            # 1. Safe Data Extraction (Fast Path)
            gamma = float(indicators.get("gamma") or 0.0)
            theta = float(indicators.get("theta") or 0.0)
            delta = float(indicators.get("delta") or 0.0)

            macd = float(indicators.get("macd") or 0.0)
            signal_line = float(indicators.get("macd_signal") or 0.0)

            atr = float(indicators.get("atr") or 0.0)
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("avg_volume") or 1.0)
            futures_vwap = float(
                indicators.get("futures_vwap")
                or indicators.get("nifty_fut_vwap")
                or indicators.get("nifty_index_vwap")
                or 0.0
            )

            # Sanity Checks
            if current_price <= 0 or futures_vwap <= 0:
                return None

            # 2. Logic: Gamma Filter (Acceleration)
            # We are looking for "Long Gamma" setups (Buying explosive moves).
            # Gamma must be positive and significant enough to drive price.
            # (Note: Gamma is usually small, e.g., 0.001 to 0.05)
            min_gamma = getattr(self._gamma_config, "min_gamma", 0.0005)
            if gamma < min_gamma:
                return None

            # 3. Logic: Theta Efficiency (Cost of Time)
            # Don't buy if Theta is burning too hard relative to the move.
            # Theta is usually negative for long options.
            # If Theta < -10 (burning fast) AND Gamma is not super high, skip.
            if theta < -15.0 and gamma < 0.002:
                # Too expensive to hold this position
                return None

            # Bullish Crossover: MACD crosses above Signal
            bullish_momentum = (macd > signal_line) and (macd - signal_line) > 0.5

            # Bearish Crossover: MACD crosses below Signal
            bearish_momentum = (macd < signal_line) and (signal_line - macd) > 0.5

            # Skip if no momentum in either direction
            if not bullish_momentum and not bearish_momentum:
                return None

            # 5. Logic: Volume Confirmation
            # Acceleration needs fuel.
            vol_ratio = vol / avg_vol
            if vol_ratio < 1.0:  # At least average volume
                return None

            # 6. Construct Signal — always long-only (BUY options, never sell/short)
            # Bullish momentum → BUY CE (Call). Bearish momentum → BUY PE (Put).
            # SELL was only reachable if OPTIONS_LONG_ONLY=false env var was set,
            # which was a footgun: Zerodha needs margin to sell options.
            option_type = None
            if bullish_momentum:
                side = "BUY"
                option_type = "CE"
            else:
                # Bearish momentum → buy PE option (long put, not short call)
                side = "BUY"
                option_type = "PE"

            # Fallback ATR
            if atr <= 0:
                atr = current_price * 0.01

            risk_mult = 1.0 if vol_ratio <= 1.5 else 1.2
            risk = max(atr * risk_mult, current_price * 0.004)
            rr_1 = 1.4
            rr_2 = 2.8

            stop_loss = current_price - risk
            tp1 = current_price + (risk * rr_1)
            tp2 = current_price + (risk * rr_2)

            if stop_loss >= current_price or tp1 <= current_price:
                LOGGER.info(
                    "Condition met: invalid gamma scalping brackets",
                    extra={
                        "event": "gamma_scalping_invalid_bracket",
                        "symbol": symbol,
                        "side": side,
                        "entry": current_price,
                        "stop_loss": stop_loss,
                        "tp1": tp1,
                        "tp2": tp2,
                    },
                )
                return None

            # 7. Confidence Scoring
            # Base 65% (Scalping is noisy).
            confidence = 0.65

            # Boost if Gamma is high (Acceleration is likely)
            if gamma > 0.002:
                confidence += 0.15

            # Boost if Volume is Absorbing (>2x)
            if vol_ratio > 2.0:
                confidence += 0.10

            LOGGER.info(
                "Condition met: gamma_scalping_signal",
                extra={
                    "event": "gamma_scalping_signal",
                    "symbol": symbol,
                    "gamma": gamma,
                    "theta": theta,
                    "vol_ratio": vol_ratio,
                    "detail": (
                        f"⚡ Gamma Scalp: {symbol} {side} | Gamma: {gamma:.4f} | "
                        f"Theta: {theta:.2f} | MACD Diff: {(macd - signal_line):.2f}"
                    ),
                },
            )

            return EliteSignal(
                symbol=symbol,
                signal=side,
                confidence=min(confidence, 0.99),
                entry_price=current_price,
                stop_loss=stop_loss,
                target=tp2,
                take_profit_1=tp1,
                take_profit_2=tp2,
                quantity=self._gamma_config.quantity or 1,
                strategy_name="Gamma_Scalp_Pro",
                metadata={
                    "type": "Momentum_Acceleration",
                    "gamma_efficiency": f"{gamma:.4f}/{theta:.1f}",
                    "momentum": "MACD_Bullish",
                    "vol_ratio": round(vol_ratio, 2),
                    "option_type": option_type,
                    "tp1": tp1,
                    "tp2": tp2,
                    "sl_atr_mult": risk_mult,
                    "tp1_rr": rr_1,
                    "tp2_rr": rr_2,
                    "tp1_qty_pct": 0.5,
                },
            )

        except Exception as e:
            LOGGER.error(f"Gamma Strategy Error on {symbol}: {e}", exc_info=True)
            return None


__all__ = ["GammaScalpingStrategy"]
