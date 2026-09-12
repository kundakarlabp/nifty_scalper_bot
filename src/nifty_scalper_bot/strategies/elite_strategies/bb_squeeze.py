from __future__ import annotations

from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import BBSqueezeStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class BBSqueezeStrategy(EliteStrategy):
    """Bollinger squeeze-to-expansion context provider."""

    MIN_BARS_REQUIRED = 25

    def __init__(self, config: BBSqueezeStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: required indicators. Raises: Exception."""
        return {
            "bollinger_upper",
            "bollinger_lower",
            "bollinger_middle",
            "close",
            "open",
            "volume",
            "avg_volume",
            "direction_bias",
            "atr",
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
                self._no_vote("domain_skip_option_symbol")
                return None

            upper = float(indicators.get("bollinger_upper") or 0.0)
            lower = float(indicators.get("bollinger_lower") or 0.0)
            mid = float(indicators.get("bollinger_middle") or 0.0)
            close = float(indicators.get("close") or current_price)
            open_price = float(indicators.get("open") or current_price)
            atr = float(indicators.get("atr") or 0.0)
            if not atr > 0.0:
                self._no_vote("atr_unavailable")
                return None
            direction = str(indicators.get("direction_bias") or "").upper()
            volume = float(indicators.get("volume") or 0.0)
            avg_volume = float(indicators.get("avg_volume") or 0.0)

            if min(upper, lower, mid) <= 0 or upper <= lower:
                self._no_vote("invalid_bb_levels")
                return None

            squeeze_threshold = float(self._cfg.squeeze_threshold_pct) / 100.0
            if squeeze_threshold <= 0.0:
                self._no_vote("invalid_squeeze_threshold")
                return None

            bb_width = (upper - lower) / mid
            if bb_width > squeeze_threshold:
                self._no_vote("no_squeeze")
                return None

            breakout_side = "CE" if close > upper else "PE" if close < lower else "UNKNOWN"
            if breakout_side == "UNKNOWN":
                self._no_vote("no_breakout")
                return None

            expansion_confirmed = abs(close - open_price) >= 0.35 * atr
            if not expansion_confirmed:
                self._no_vote("weak_expansion")
                return None

            score = 5.0
            reasons = ["configured_squeeze", "expansion_confirmed", "breakout_candle"]
            if direction in {"CE", "PE"} and direction == breakout_side:
                score += 2.0
                reasons.append("direction_alignment")

            momentum_confirmed = avg_volume > 0 and volume >= avg_volume
            if momentum_confirmed:
                score += 1.0
                reasons.append("volume_confirmation")

            tightness = bb_width / squeeze_threshold
            if tightness <= 0.6:
                score += 2.0
                reasons.append("squeeze_very_tight")
            elif tightness <= 0.85:
                score += 1.0
                reasons.append("squeeze_tight")
            else:
                reasons.append("squeeze_marginal")

            percentile = indicators.get("bb_width_percentile")
            try:
                percentile_value = float(percentile) if percentile is not None else None
            except (TypeError, ValueError):
                percentile_value = None
            if percentile_value is not None and 0.0 <= percentile_value <= 1.0:
                percentile_value *= 100.0
            if percentile_value is not None and 0.0 <= percentile_value <= 20.0:
                score += 0.5
                reasons.append("historically_tight_bandwidth")

            strategy_score = max(0.0, min(10.0, score))
            metadata = {
                "strategy": "BBSqueeze",
                "strategy_name": "BBSqueeze",
                "role": "context",
                "can_trigger": False,
                "signal_family": "directional_context",
                "trade_side": breakout_side,
                "side": breakout_side,
                "direction_bias": breakout_side,
                "preliminary_only": True,
                "requires_runner_final_score": True,
                "direction_score": strategy_score,
                "strategy_score": strategy_score,
                "context_score": strategy_score,
                "data_score": 8.0,
                "setup_quality": strategy_score,
                "setup_type": "squeeze_expansion_context",
                "required_data_present": True,
                "stale_data_used": bool(indicators.get("stale_data_used")),
                "candidate_symbol": symbol,
                "score_reasons": reasons,
                "rejection_reasons": [],
                "bb_width": round(bb_width, 6),
                "bb_width_percentile": percentile_value,
                "squeeze_threshold_pct": self._cfg.squeeze_threshold_pct,
                "squeeze_detected": True,
                "expansion_confirmed": True,
                "breakout_side": breakout_side,
                "momentum_confirmed": momentum_confirmed,
            }
            LOGGER.info(
                "STRATEGY_CONTEXT strategy=BBSqueeze side=%s score=%.2f",
                breakout_side,
                strategy_score,
            )
            return EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=max(0.1, min(0.9, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=None,
                target=None,
                quantity=self._cfg.quantity or 1,
                strategy_name="BBSqueeze",
                metadata=metadata,
            )
        except Exception as exc:
            LOGGER.error(
                "Failure in BBSqueezeStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
            )
            return None


__all__ = ["BBSqueezeStrategy"]
