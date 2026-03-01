"""
Opening Range Breakout (ORB) Pro Strategy.
Long-only options-buying mode: BUY CE breakouts only.
Uses pre-computed orb_high/orb_low from indicator engine (correct, not option H/L).
Uses market_time/minutes_since_open from indicator engine (always present).
"""

from __future__ import annotations

from typing import Any, Dict

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    ORBProStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class ORBProStrategy(EliteStrategy):
    """
    Trade validated Opening Range Breakouts (ORB) with volume confirmation.
    Long-only (BUY only on bullish breakout): options buying mode, never shorts.
    Uses orb_high/orb_low/orb_ready pre-computed by the indicator engine after 9:30 AM.
    """

    MIN_BARS_REQUIRED = 5

    __slots__ = ("_orb_config",)

    def __init__(self, config: ORBProStrategyConfig, indicator_engine: Any) -> None:
        """
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._orb_config = config

    def get_required_indicators(self) -> set[str]:
        """
        orb_high, orb_low, orb_ready, market_time, minutes_since_open are all
        pre-computed by the indicator engine — no manual H/L cache needed.
        """
        return {
            "orb_high",
            "orb_low",
            "orb_ready",
            "market_time",
            "minutes_since_open",
            "volume",
            "avg_volume",
            "vwap",
            "atr",
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
            "Entered ORBProStrategy._evaluate_signal",
            extra={"event": "orb_pro_enter", "symbol": symbol},
        )
        try:
            # 1. ORB range from indicator engine (computed from first 15 mins of session)
            orb_ready = bool(indicators.get("orb_ready"))
            if not orb_ready:
                return None  # Range still forming — wait

            orb_high = float(indicators.get("orb_high") or 0.0)
            orb_low = float(indicators.get("orb_low") or 0.0)
            if orb_high <= 0 or orb_low <= 0:
                return None

            range_width = orb_high - orb_low
            if range_width < (orb_high * 0.001):  # < 0.1% is flatline/bad data
                return None

            # 2. Time guard: ORB signals lose edge after 90 minutes post open
            minutes_since_open = float(indicators.get("minutes_since_open") or 0.0)
            if minutes_since_open > 90:
                return None  # Past 10:45 AM — other dynamics dominate

            # 3. Safe data extraction
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("avg_volume") or 1.0)
            vwap = float(indicators.get("vwap") or 0.0)
            _raw_atr = float(indicators.get("atr") or 0.0)
            # ATR floor: 1% of premium, min Rs1 — prevents near-zero SL on cheap options
            atr = max(_raw_atr, current_price * 0.01, 1.0)

            # 4. Long-only: Only bullish ORB breakout (price above range high)
            if current_price <= orb_high:
                return None  # No breakout yet

            # VWAP alignment: must be above VWAP for genuine bullish trend
            if vwap > 0 and current_price < vwap:
                LOGGER.debug(
                    "ORB blocked: price below VWAP",
                    extra={"event": "orb_vwap_fail", "symbol": symbol,
                           "price": current_price, "vwap": vwap},
                )
                return None

            # Volume: breakout must have participation (at least average volume)
            if avg_vol > 0 and (vol / avg_vol) < 1.0:
                return None

            # 5. Risk management — long option BUY
            risk = max(range_width * 0.6, atr * 1.2)
            stop_loss = max(orb_low, current_price - risk)
            tp1 = current_price + (risk * 1.5)
            tp2 = current_price + (risk * 3.0)

            if stop_loss >= current_price or tp1 <= current_price:
                LOGGER.debug(
                    "ORB invalid SL/TP",
                    extra={"event": "orb_invalid_bracket", "symbol": symbol,
                           "entry": current_price, "sl": stop_loss, "tp1": tp1},
                )
                return None

            # 6. Confidence
            vol_ratio = (vol / avg_vol) if avg_vol > 0 else 0.0
            confidence = 0.80
            if vol_ratio > 2.0:
                confidence += 0.10

            LOGGER.info(
                "Condition met: orb_breakout_signal",
                extra={
                    "event": "orb_breakout_signal",
                    "symbol": symbol,
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "vwap": vwap,
                    "vol_ratio": round(vol_ratio, 2),
                    "minutes_since_open": minutes_since_open,
                },
            )

            return EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=min(confidence, 0.99),
                entry_price=current_price,
                stop_loss=stop_loss,
                target=tp2,
                take_profit_1=tp1,
                take_profit_2=tp2,
                quantity=self._orb_config.quantity or 1,
                strategy_name="ORB_Pro_Breakout",
                metadata={
                    "type": "Opening_Range_Breakout",
                    "range_width": round(range_width, 2),
                    "vwap_confirmation": True,
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp1_rr": 1.5,
                    "tp2_rr": 3.0,
                },
            )

        except Exception as e:
            LOGGER.error(f"ORB Strategy Error on {symbol}: {e}", exc_info=True)
            return None


__all__ = ["ORBProStrategy"]
