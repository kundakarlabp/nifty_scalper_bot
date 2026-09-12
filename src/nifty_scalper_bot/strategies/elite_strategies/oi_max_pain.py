from __future__ import annotations

from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import OIMaxPainStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class OIMaxPainStrategy(EliteStrategy):
    """OI/max-pain context using underlying/strike-domain prices only."""

    def __init__(self, config: OIMaxPainStrategyConfig, indicator_engine: Any) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config

    def get_required_indicators(self) -> set[str]:
        return {
            "max_pain",
            "call_oi_wall",
            "put_oi_wall",
            "spot_price",
            "direction_bias",
        }

    @staticmethod
    def _positive_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 and number == number else None

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        del position
        try:
            max_pain = self._positive_float(indicators.get("max_pain"))
            call_wall = self._positive_float(indicators.get("call_oi_wall"))
            put_wall = self._positive_float(indicators.get("put_oi_wall"))
            spot_price = self._positive_float(indicators.get("spot_price"))
            option_symbol = str(symbol or "").upper().endswith(("CE", "PE"))

            if max_pain is None:
                self._no_vote("missing_oi_data")
                return None
            if spot_price is None:
                if option_symbol:
                    self._no_vote("missing_underlying_spot_price")
                    return None
                spot_price = self._positive_float(current_price)
            if spot_price is None:
                self._no_vote("missing_underlying_spot_price")
                return None

            min_deviation_pct = max(0.0, float(self._cfg.min_deviation_pct))
            max_pain_deviation_pct = abs(spot_price - max_pain) / spot_price * 100.0
            if max_pain_deviation_pct < min_deviation_pct:
                self._no_vote("max_pain_deviation_too_small")
                return None

            context_bias = "CE" if spot_price < max_pain else "PE"
            direction = str(indicators.get("direction_bias") or "").upper()
            side = direction if direction in {"CE", "PE"} else context_bias

            wall = call_wall if side == "CE" else put_wall
            wall_ahead = bool(
                wall is not None
                and (
                    (side == "CE" and wall > spot_price)
                    or (side == "PE" and wall < spot_price)
                )
            )
            distance_to_oi_wall = (
                abs(spot_price - wall) if wall_ahead and wall is not None else None
            )
            distance_to_oi_wall_pct = (
                distance_to_oi_wall / spot_price * 100.0
                if distance_to_oi_wall is not None
                else None
            )

            score = 2.0
            reasons = ["oi_context_bias_only"]
            if side == context_bias:
                score += 1.0
                reasons.append("max_pain_alignment")
            if (
                distance_to_oi_wall_pct is not None
                and distance_to_oi_wall_pct < min_deviation_pct
            ):
                score -= 1.5
                reasons.append("near_oi_wall_against_trade")

            strategy_score = max(0.0, min(4.0, score))
            if strategy_score <= 0:
                self._no_vote("no_max_pain_edge")
                return None

            metadata = {
                "strategy": "OIMaxPain",
                "strategy_name": "OIMaxPain",
                "role": "context",
                "can_trigger": False,
                "source_domain": "underlying_strike",
                "side": side,
                "trade_side": side,
                "direction_bias": side,
                "strategy_score": strategy_score,
                "context_score": strategy_score,
                "setup_quality": strategy_score,
                "setup_type": "oi_context",
                "required_data_present": True,
                "stale_data_used": bool(indicators.get("stale_data_used")),
                "candidate_symbol": symbol,
                "score_reasons": reasons,
                "rejection_reasons": [],
                "underlying_reference_price": spot_price,
                "underlying_reference_source": "spot_price",
                "max_pain_level": max_pain,
                "max_pain_deviation_pct": round(max_pain_deviation_pct, 4),
                "min_deviation_pct": min_deviation_pct,
                "call_oi_wall": call_wall,
                "put_oi_wall": put_wall,
                "distance_to_oi_wall": (
                    round(distance_to_oi_wall, 3)
                    if distance_to_oi_wall is not None
                    else None
                ),
                "distance_to_oi_wall_pct": (
                    round(distance_to_oi_wall_pct, 4)
                    if distance_to_oi_wall_pct is not None
                    else None
                ),
                "context_bias": context_bias,
                "invalidation_level": None,
            }
            LOGGER.info(
                "STRATEGY_VOTE strategy=OIMaxPain side=%s score=%.2f",
                side,
                strategy_score,
            )
            return EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=max(0.05, min(0.45, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=None,
                target=None,
                quantity=self._cfg.quantity or 1,
                strategy_name="OIMaxPain",
                metadata=metadata,
            )
        except Exception as exc:
            LOGGER.error(
                "Failure in OIMaxPainStrategy._evaluate_signal: %s", exc, exc_info=exc
            )
            return None


__all__ = ["OIMaxPainStrategy"]
