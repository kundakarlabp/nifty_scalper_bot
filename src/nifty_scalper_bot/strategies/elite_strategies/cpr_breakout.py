from __future__ import annotations

from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import CPRBreakoutStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class CPRBreakoutStrategy(EliteStrategy):
    """Narrow-CPR breakout context provider with independent evidence scoring."""

    MIN_BARS_REQUIRED = 2

    def __init__(self, config: CPRBreakoutStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: indicators set. Raises: Exception."""
        return {
            "pivot",
            "bc",
            "tc",
            "r1",
            "s1",
            "close",
            "atr",
            "direction_bias",
            "retest_confirmed",
        }

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None."""
        del position
        try:
            self._no_vote("stale_or_invalid_data")
            if symbol.upper().endswith(("CE", "PE")) and not indicators.get("source_symbol"):
                self._no_vote("invalid_price_domain")
                return None

            cpr_bottom = float(indicators.get("bc") or 0.0)
            cpr_top = float(indicators.get("tc") or 0.0)
            pivot = float(indicators.get("pivot") or 0.0)
            r1 = float(indicators.get("r1") or 0.0)
            s1 = float(indicators.get("s1") or 0.0)
            atr = max(float(indicators.get("atr") or 0.0), current_price * 0.01, 1.0)
            direction = str(indicators.get("direction_bias") or "").upper()

            if min(cpr_bottom, cpr_top, pivot) <= 0 or cpr_top <= cpr_bottom:
                self._no_vote("invalid_cpr_levels")
                return None

            width_threshold_pct = float(self._cfg.narrow_cpr_threshold)
            if width_threshold_pct <= 0.0:
                self._no_vote("invalid_cpr_width_threshold")
                return None
            cpr_width_pct = ((cpr_top - cpr_bottom) / pivot) * 100.0
            if cpr_width_pct > width_threshold_pct:
                self._no_vote("cpr_not_narrow")
                return None

            if cpr_bottom <= current_price <= cpr_top:
                self._no_vote("inside_cpr")
                return None

            side = "CE" if current_price > cpr_top else "PE"
            breakout_level = cpr_top if side == "CE" else cpr_bottom
            breakout_quality = abs(current_price - breakout_level) / atr

            nearest_level_distance: float | None = None
            if side == "CE" and r1 > current_price:
                nearest_level_distance = r1 - current_price
            elif side == "PE" and 0 < s1 < current_price:
                nearest_level_distance = current_price - s1
            if nearest_level_distance is not None and nearest_level_distance < 0.5 * atr:
                self._no_vote("nearby_level")
                return None

            retest_confirmed = bool(indicators.get("retest_confirmed"))
            score = 3.0
            reasons = ["narrow_cpr", "clean_break_beyond_cpr"]

            if direction in {"CE", "PE"} and direction == side:
                score += 2.0
                reasons.append("direction_alignment")
            if retest_confirmed:
                score += 2.0
                reasons.append("retest_confirmed")
            if nearest_level_distance is not None and nearest_level_distance >= atr:
                score += 1.0
                reasons.append("adequate_distance_to_next_level")
            if breakout_quality >= 1.0:
                score += 2.0
                reasons.append(f"momentum_strong_{breakout_quality:.1f}")
            elif breakout_quality >= 0.6:
                score += 1.0
                reasons.append(f"momentum_moderate_{breakout_quality:.1f}")
            else:
                reasons.append(f"momentum_weak_{breakout_quality:.1f}")

            strategy_score = max(0.0, min(10.0, score))
            metadata = {
                "strategy": "CPRBreakout",
                "strategy_name": "CPRBreakout",
                "role": "context",
                "can_trigger": False,
                "requires_feature_set": "cpr_levels",
                "signal_family": "directional_context",
                "trade_side": side,
                "side": side,
                "direction_bias": side,
                "preliminary_only": True,
                "requires_runner_final_score": True,
                "direction_score": strategy_score,
                "strategy_score": strategy_score,
                "context_score": strategy_score,
                "data_score": 8.0,
                "setup_quality": strategy_score,
                "setup_type": "cpr_breakout_context",
                "required_data_present": True,
                "stale_data_used": bool(indicators.get("stale_data_used")),
                "candidate_symbol": symbol,
                "score_reasons": reasons,
                "rejection_reasons": [],
                "cpr_top": cpr_top,
                "cpr_bottom": cpr_bottom,
                "pivot": pivot,
                "cpr_width_pct": round(cpr_width_pct, 4),
                "narrow_cpr_threshold_pct": width_threshold_pct,
                "relation_to_cpr": "above" if side == "CE" else "below",
                "breakout_quality": round(breakout_quality, 3),
                "nearest_level_distance": (
                    round(nearest_level_distance, 3)
                    if nearest_level_distance is not None
                    else None
                ),
                "retest_confirmed": retest_confirmed,
                "underlying_invalidation_level": (
                    cpr_bottom if side == "CE" else cpr_top
                ),
            }
            LOGGER.info(
                "STRATEGY_CONTEXT strategy=CPRBreakout side=%s score=%.2f",
                side,
                strategy_score,
            )
            return EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=max(0.1, min(0.88, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=None,
                target=None,
                quantity=self._cfg.quantity or 1,
                strategy_name="CPRBreakout",
                metadata=metadata,
            )
        except Exception as exc:
            LOGGER.error(
                "Failure in CPRBreakoutStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
            )
            return None


__all__ = ["CPRBreakoutStrategy"]
