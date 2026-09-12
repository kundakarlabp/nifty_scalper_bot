from __future__ import annotations

import os
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import GammaScalpingStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.smart_symbol import (
    WEEKLY_EXPIRY_WEEKDAY,
    get_actual_expiry_date,
    now_ist,
)

LOGGER = get_logger(__name__)


def _is_expiry_session(indicators: dict[str, Any]) -> bool:
    raw_days = indicators.get("days_to_expiry")
    if raw_days is not None:
        try:
            return float(raw_days) <= 0.0
        except (TypeError, ValueError):
            return False
    today = now_ist().date()
    return get_actual_expiry_date(today, WEEKLY_EXPIRY_WEEKDAY) == today


class GammaScalpingStrategy(EliteStrategy):
    """Long-premium expiry-gamma strategy gated by explicit mode flags."""

    MIN_BARS_REQUIRED = 3

    def __init__(self, config: GammaScalpingStrategyConfig, indicator_engine: Any) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config

    def get_required_indicators(self) -> set[str]:
        return {
            "gamma",
            "theta",
            "atr",
            "direction_bias",
            "volatility_expansion_confirmed",
            "days_to_expiry",
        }

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        del position
        try:
            strategy_mode = str(os.getenv("STRATEGY_MODE", "directional_scalp")).lower()
            gamma_enabled = str(
                os.getenv("ALLOW_EXPIRY_GAMMA_STRATEGIES", "false")
            ).lower() in {"1", "true", "yes", "on"}
            if not (strategy_mode == "expiry_gamma" and gamma_enabled):
                self._no_vote("gamma_mode_disabled")
                return None
            if not _is_expiry_session(indicators):
                self._no_vote("not_expiry_session")
                return None

            gamma = float(indicators.get("gamma") or 0.0)
            theta = float(indicators.get("theta") or 0.0)
            atr = max(
                float(indicators.get("atr") or 0.0), current_price * 0.01, 1.0
            )
            direction = str(indicators.get("direction_bias") or "").upper()
            min_gamma = float(self._cfg.min_gamma)

            if gamma <= min_gamma:
                self._no_vote("gamma_below_minimum")
                return None
            if theta >= 0:
                self._no_vote("theta_not_negative")
                return None
            if direction not in {"CE", "PE"}:
                self._no_vote("missing_direction_context")
                return None

            side = direction
            premium_decay_risk = min(1.0, abs(theta) / max(current_price, 1.0))
            vol_exp = bool(
                indicators.get("volatility_expansion_confirmed") or gamma > 0.0015
            )

            score = 4.0
            reasons = ["expiry_gamma_core", "direction_context"]
            if vol_exp:
                score += 2.0
                reasons.append("volatility_expansion_confirmed")
            if gamma >= max(0.0015, 2.0 * min_gamma):
                score += 1.0
                reasons.append("gamma_strong")
            strategy_score = max(0.0, min(10.0, score))

            # CE and PE are both long-premium positions. Premium risk is always
            # below entry; underlying direction must not invert premium SL/TP.
            stop_loss = current_price - atr
            target = current_price + (1.8 * atr)
            metadata = {
                "strategy": "GammaScalping",
                "strategy_name": "GammaScalping",
                "role": "trigger",
                "side": side,
                "trade_side": side,
                "direction_bias": side,
                "strategy_score": strategy_score,
                "setup_quality": strategy_score,
                "setup_type": "expiry_gamma",
                "required_data_present": True,
                "stale_data_used": bool(indicators.get("stale_data_used")),
                "candidate_symbol": symbol,
                "score_reasons": reasons,
                "rejection_reasons": [],
                "expiry_day": True,
                "days_to_expiry": indicators.get("days_to_expiry"),
                "gamma_mode_enabled": True,
                "premium_decay_risk": round(premium_decay_risk, 4),
                "volatility_expansion_confirmed": vol_exp,
                "invalidation_level": stop_loss,
                "premium_stop_distance": atr,
                "premium_target_rr": 1.8,
            }
            LOGGER.info(
                "STRATEGY_VOTE strategy=GammaScalping side=%s score=%.2f",
                side,
                strategy_score,
            )
            return EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=max(0.1, min(0.85, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=stop_loss,
                target=target,
                quantity=self._cfg.quantity or 1,
                strategy_name="GammaScalping",
                metadata=metadata,
            )
        except Exception as exc:
            LOGGER.error(
                "Failure in GammaScalpingStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
            )
            return None


__all__ = ["GammaScalpingStrategy"]
