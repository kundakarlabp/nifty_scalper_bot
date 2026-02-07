"""
OI Max Pain / VWAP Reversion Strategy.
World-Class implementation with Gravity Pull (Max Pain), Momentum Trigger (MACD), and Time Guards.
Refactored for Push-Based Architecture (Zero-Latency).
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, Optional

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    OIMaxPainStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class OIMaxPainStrategy(EliteStrategy):
    """
    Mean Reversion Strategy based on 'Option Pain'.
    Thesis: As expiry approaches, Price is pulled towards the Max Pain strike (High OI).
    Entry: Price is overextended from VWAP/MaxPain + MACD indicates momentum slowing.
    """

    # ✅ OPTIMIZATION: Use slots for memory efficiency
    __slots__ = ("_oi_config",)

    def __init__(self, config: OIMaxPainStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.

        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._oi_config = config

    def get_required_indicators(self) -> set[str]:
        """
        Declare which indicators this strategy needs pre-calculated.
        The StrategyManager will inject these into _evaluate_signal.
        """
        return {
            "max_pain",  # The strike with highest total OI (Magnet)
            "vwap",  # Fair value reference
            "macd_hist",  # Momentum Reversal Trigger (12, 26, 9)
            "atr",  # Volatility for stops
            "ltp",
            "timestamp",  # To avoid EOD trading
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
            "Entered OIMaxPainStrategy._evaluate_signal",
            extra={"event": "oi_max_pain_enter", "symbol": symbol},
        )
        try:
            # 1. Safe Data Extraction (Fast Path)
            max_pain = float(indicators.get("max_pain") or 0.0)
            vwap = float(indicators.get("vwap") or 0.0)
            hist = float(indicators.get("macd_hist") or 0.0)
            atr = float(indicators.get("atr") or 0.0)

            # Timestamp check
            ts = indicators.get("timestamp")
            if not ts:
                return None

            # 2. Time Guard
            # Avoid Mean Reversion in last 30 mins (Gamma moves can destroy reversals)
            now = ts if isinstance(ts, datetime) else datetime.now()
            if now.time() > time(15, 0):
                return None

            # 3. Sanity Checks
            # If Max Pain is 0 (data missing), we cannot trade this strategy.
            if max_pain <= 0 or vwap <= 0:
                return None

            # 4. Logic: Calculate "Extension" (Distance)
            # How far is price from the 'Magnet' (Max Pain) and Fair Value (VWAP)?
            dist_mp_pct = (current_price - max_pain) / max_pain * 100
            dist_vwap_pct = (current_price - vwap) / vwap * 100

            # Configurable threshold for "Overextended" (e.g., 0.5% away)
            min_dist = getattr(self._oi_config, "min_deviation_pct", 0.5)

            # 5. Signal Detection
            side = ""
            reason = ""

            # --- Bullish Reversion (Long) ---
            # Condition 1: Price is significantly BELOW Max Pain (Magnet is above)
            # Condition 2: Price is BELOW VWAP (Oversold)
            # Condition 3: MACD Histogram is Positive/Tick-up (Momentum shifting up)
            if (dist_mp_pct < -min_dist) and (current_price < vwap) and (hist > 0):
                side = "BUY"
                reason = "Oversold_Magnet_Pull"

            # --- Bearish Reversion (Short) ---
            # Condition 1: Price is significantly ABOVE Max Pain (Magnet is below)
            # Condition 2: Price is ABOVE VWAP (Overbought)
            # Condition 3: MACD Histogram is Negative/Tick-down (Momentum shifting down)
            elif (dist_mp_pct > min_dist) and (current_price > vwap) and (hist < 0):
                side = "SELL"
                reason = "Overbought_Magnet_Pull"

            if not side:
                return None

            # 6. Risk Management
            # Reversion trades need room to breathe, but hard stops if momentum accelerates.
            if atr <= 0:
                atr = current_price * 0.005

            risk = atr * 1.3
            if side == "BUY":
                stop_loss = current_price - risk
                candidates = [
                    level for level in (vwap, max_pain) if level > current_price
                ]
            else:
                stop_loss = current_price + risk
                candidates = [
                    level for level in (vwap, max_pain) if level < current_price
                ]

            candidates_sorted = sorted(
                candidates, key=lambda level: abs(level - current_price)
            )
            if len(candidates_sorted) >= 2:
                tp1, tp2 = candidates_sorted[:2]
            elif len(candidates_sorted) == 1:
                tp1 = candidates_sorted[0]
                tp2 = (
                    current_price + (risk * 2.6)
                    if side == "BUY"
                    else current_price - (risk * 2.6)
                )
            else:
                tp1 = (
                    current_price + (risk * 1.3)
                    if side == "BUY"
                    else current_price - (risk * 1.3)
                )
                tp2 = (
                    current_price + (risk * 2.6)
                    if side == "BUY"
                    else current_price - (risk * 2.6)
                )
            if (
                side == "BUY" and (stop_loss >= current_price or tp1 <= current_price)
            ) or (
                side == "SELL" and (stop_loss <= current_price or tp1 >= current_price)
            ):
                LOGGER.info(
                    "Condition met: invalid max pain brackets",
                    extra={
                        "event": "oi_max_pain_invalid_bracket",
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
            # Base 70%.
            confidence = 0.70

            # Boost if Price is BOTH far from VWAP and Max Pain (Double Confluence)
            if abs(dist_mp_pct) > (min_dist * 2):
                confidence += 10.0

            # Boost if we are very close to expiry (Max Pain gravity is stronger)
            # (Logic implies manual adjustment or external day check, simplified here)

            # 8. Construct Signal
            LOGGER.info(
                "Condition met: oi_max_pain_signal",
                extra={
                    "event": "oi_max_pain_signal",
                    "symbol": symbol,
                    "max_pain": max_pain,
                    "vwap": vwap,
                    "macd_hist": hist,
                    "detail": (
                        f"🧲 Max Pain Reversion: {symbol} {side} | Pain: "
                        f"{max_pain} | Dist: {dist_mp_pct:.2f}%"
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
                quantity=self._oi_config.quantity or 1,
                strategy_name="OI_Max_Pain_Revert",
                metadata={
                    "type": reason,
                    "max_pain_strike": max_pain,
                    "distance_pct": round(dist_mp_pct, 2),
                    "momentum_confirm": "MACD_Hist",
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp1_rr": 1.3,
                    "tp2_rr": 2.6,
                },
            )

        except Exception as e:
            LOGGER.error(f"Max Pain Strategy Error on {symbol}: {e}", exc_info=True)
            return None


__all__ = ["OIMaxPainStrategy"]
