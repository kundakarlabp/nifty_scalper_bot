"""Tuesday-expiry long-premium gamma strategy.

Direction comes only from the canonical fresh underlying context. Position sizing,
daily-loss limits and open-position admission remain owned by the centralized risk
and execution layers.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    TuesdayGammaBuyerStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.smart_symbol import (
    WEEKLY_EXPIRY_WEEKDAY,
    get_actual_expiry_date,
    now_ist,
)

LOGGER = get_logger(__name__)

BaseEliteStrategy = EliteStrategy


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0 or parsed != parsed:
        return None
    return parsed


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _aligned(side: str, bullish: bool) -> bool:
    return bullish if side == "CE" else not bullish


class EliteTuesdayGammaBuyer(BaseEliteStrategy):
    """Expiry-day long-premium trigger using canonical underlying context."""

    def __init__(
        self,
        config: TuesdayGammaBuyerStrategyConfig,
        indicator_engine: Any,
    ) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._config = config

    def get_required_indicators(self) -> set[str]:
        """Return only option-local indicators required before context enrichment."""
        return {"atr", "direction_bias"}

    def _now_local(self):
        clock = getattr(self, "clock", None)
        if clock and callable(getattr(clock, "now_local", None)):
            return clock.now_local()
        return now_ist()

    @staticmethod
    def _underlying_context(
        indicators: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], str] | None:
        """Resolve one coherent underlying price/indicator domain."""
        spot = _mapping(indicators.get("spot_context"))
        futures = _mapping(indicators.get("futures_context"))
        for context, source in ((spot, "spot_context"), (futures, "futures_context")):
            price = _positive_float(
                context.get("ltp") or context.get("close") or context.get("price")
            )
            if price is not None:
                return context, source
        return None

    @staticmethod
    def _expiry_session(indicators: Mapping[str, Any], trading_date) -> bool:
        raw_days = indicators.get("days_to_expiry")
        if raw_days is not None:
            try:
                return float(raw_days) <= 0.0
            except (TypeError, ValueError):
                return False
        return (
            get_actual_expiry_date(trading_date, WEEKLY_EXPIRY_WEEKDAY)
            == trading_date
        )

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        del position
        try:
            strategy_mode = str(
                os.getenv("STRATEGY_MODE", "directional_scalp")
            ).strip().lower()
            gamma_enabled = str(
                os.getenv("ALLOW_EXPIRY_GAMMA_STRATEGIES", "false")
            ).strip().lower() in {"1", "true", "yes", "on"}
            if not (strategy_mode == "expiry_gamma" and gamma_enabled):
                self._no_vote("gamma_mode_disabled")
                return None

            now_local = self._now_local()
            if not self._expiry_session(indicators, now_local.date()):
                self._no_vote("not_expiry_session")
                return None
            minute_of_day = now_local.hour * 60 + now_local.minute
            if minute_of_day < 9 * 60 + 20 or minute_of_day >= 14 * 60 + 45:
                self._no_vote("outside_expiry_gamma_window")
                return None

            if not bool(indicators.get("context_fresh")):
                self._no_vote("underlying_context_stale")
                return None
            side = str(
                indicators.get("underlying_direction_bias")
                or indicators.get("direction_bias")
                or ""
            ).upper()
            if side not in {"CE", "PE"}:
                self._no_vote("underlying_direction_unresolved")
                return None

            resolved = self._underlying_context(indicators)
            if resolved is None:
                self._no_vote("underlying_context_unavailable")
                return None
            context, context_source = resolved
            underlying_price = _positive_float(
                context.get("ltp") or context.get("close") or context.get("price")
            )
            if underlying_price is None:
                self._no_vote("underlying_context_unavailable")
                return None

            option_atr = _positive_float(indicators.get("atr"))
            if option_atr is None:
                self._no_vote("atr_unavailable")
                return None

            vwap = _positive_float(context.get("vwap"))
            ema_fast = _positive_float(context.get("ema_fast"))
            ema_slow = _positive_float(context.get("ema_slow"))
            price_alignment: bool | None = None
            ema_alignment: bool | None = None
            if vwap is not None:
                price_alignment = _aligned(side, underlying_price > vwap)
            if ema_fast is not None and ema_slow is not None:
                ema_alignment = _aligned(side, ema_fast > ema_slow)

            observed_alignment = [
                item for item in (price_alignment, ema_alignment) if item is not None
            ]
            if not observed_alignment:
                self._no_vote("underlying_trend_evidence_unavailable")
                return None
            if not all(observed_alignment):
                self._no_vote("underlying_trend_conflict")
                return None

            raw_return = context.get("spot_return_3min")
            if raw_return is None:
                raw_return = indicators.get("underlying_spot_return_3min")
            if raw_return is not None:
                try:
                    if abs(float(raw_return)) > 0.006:
                        self._no_vote("underlying_move_overextended")
                        return None
                except (TypeError, ValueError):
                    pass

            score = 4.0
            reasons = ["expiry_session", "underlying_direction_context"]
            if price_alignment is True:
                score += 2.0
                reasons.append("underlying_vwap_alignment")
            if ema_alignment is True:
                score += 2.0
                reasons.append("underlying_ema_alignment")

            futures = _mapping(indicators.get("futures_context"))
            futures_volume = _positive_float(futures.get("volume"))
            futures_avg_volume = _positive_float(futures.get("avg_volume"))
            volume_expansion = bool(
                futures_volume is not None
                and futures_avg_volume is not None
                and futures_volume >= 1.2 * futures_avg_volume
            )
            if volume_expansion:
                score += 1.0
                reasons.append("futures_volume_expansion")

            underlying_atr = _positive_float(context.get("atr"))
            underlying_atr_ma = _positive_float(context.get("atr_ma"))
            volatility_expansion = bool(
                underlying_atr is not None
                and underlying_atr_ma is not None
                and underlying_atr > underlying_atr_ma
            )
            if volatility_expansion:
                score += 1.0
                reasons.append("underlying_volatility_expansion")

            strategy_score = max(0.0, min(10.0, score))
            setup_min = 6.0
            if strategy_score < setup_min:
                self._no_vote("expiry_gamma_quality_below_minimum")
                return None

            atr_multiplier = max(float(self._config.atr_multiplier), 0.1)
            target_multiplier = max(float(self._config.target_multiplier), 0.1)
            stop_distance = option_atr * atr_multiplier
            target_distance = option_atr * target_multiplier
            stop_loss = max(0.05, current_price - stop_distance)
            target = current_price + target_distance

            metadata = {
                "strategy": "EliteTuesdayGammaBuyer",
                "strategy_name": "EliteTuesdayGammaBuyer",
                "role": "trigger",
                "side": side,
                "trade_side": side,
                "contract_side": side,
                "direction_bias": side,
                "source_domain": "underlying_context_plus_option_premium_risk",
                "underlying_context_source": context_source,
                "underlying_reference_symbol": context.get("symbol"),
                "underlying_reference_price": underlying_price,
                "strategy_score": strategy_score,
                "raw_setup_score": strategy_score,
                "setup_score": strategy_score,
                "setup_min": setup_min,
                "setup_pass": True,
                "setup_quality": strategy_score,
                "setup_type": "expiry_gamma",
                "preliminary_only": True,
                "requires_runner_final_score": True,
                "required_data_present": True,
                "stale_data_used": bool(indicators.get("stale_data_used")),
                "candidate_symbol": symbol,
                "score_reasons": reasons,
                "rejection_reasons": [],
                "expiry_day": True,
                "days_to_expiry": indicators.get("days_to_expiry"),
                "gamma_mode_enabled": True,
                "underlying_vwap_alignment": price_alignment,
                "underlying_ema_alignment": ema_alignment,
                "futures_volume_expansion": volume_expansion,
                "volatility_expansion_confirmed": volatility_expansion,
                "sizing_owner": "RiskManager",
                "premium_stop_distance": stop_distance,
                "premium_target_distance": target_distance,
                "premium_target_rr": target_distance / max(stop_distance, 1e-9),
                "bracket_type": "VIRTUAL",
                "sl_mode": "ATR_TRAIL",
                "enable_trailing": True,
                "sl_atr_mult": atr_multiplier,
                "trailing_atr_mult": atr_multiplier,
                "tp2_atr_mult": target_multiplier,
                "atr_multiplier": atr_multiplier,
                "target_multiplier": target_multiplier,
            }
            LOGGER.info(
                "STRATEGY_VOTE strategy=EliteTuesdayGammaBuyer side=%s score=%.2f source=%s",
                side,
                strategy_score,
                context_source,
            )
            return EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=max(0.1, min(0.9, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=stop_loss,
                target=target,
                quantity=self._config.quantity or 1,
                strategy_name="EliteTuesdayGammaBuyer",
                metadata=metadata,
            )
        except Exception as exc:
            LOGGER.error(
                "Failure in EliteTuesdayGammaBuyer._evaluate_signal: %s",
                exc,
                exc_info=exc,
            )
            return None


__all__ = ["EliteTuesdayGammaBuyer"]
