from __future__ import annotations

from typing import Any

from nifty_scalper_bot.config.regime_ontology import MarketRegime, normalize_regime
from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import RSIDivergenceStrategyConfig
from nifty_scalper_bot.strategies.signal_quality import resolve_signal_domain
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _confirmed_swing_pairs(
    indicators: dict[str, Any],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return confirmed (price, RSI) lows and highs from completed-bar evidence."""
    lows: list[tuple[float, float]] = []
    highs: list[tuple[float, float]] = []
    raw_points = indicators.get("rsi_swing_points")
    if not isinstance(raw_points, (list, tuple)):
        return lows, highs

    for point in raw_points:
        if not isinstance(point, dict) or not bool(point.get("confirmed")):
            continue
        kind = str(point.get("kind") or point.get("type") or "").strip().lower()
        if kind not in {"low", "high"}:
            continue
        try:
            price = float(point.get("price"))
            rsi = float(point.get("rsi"))
        except (TypeError, ValueError):
            continue
        if price <= 0.0 or not 0.0 <= rsi <= 100.0:
            continue
        if kind == "low":
            lows.append((price, rsi))
        else:
            highs.append((price, rsi))
    return lows, highs


class RSIDivergenceStrategy(EliteStrategy):
    """Confirmed-swing RSI divergence context provider."""

    MIN_BARS_REQUIRED = 20

    def __init__(self, config: RSIDivergenceStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: required indicators. Raises: Exception."""
        return {
            "rsi",
            "close",
            "atr",
            "regime",
            "direction_bias",
            "confirmation_candle",
            "rsi_swing_points",
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
            close = float(indicators.get("close") or current_price)
            atr = max(float(indicators.get("atr") or 0.0), current_price * 0.01, 1.0)
            regime = normalize_regime(indicators.get("regime"))
            direction = str(indicators.get("direction_bias") or "").upper()

            lows, highs = _confirmed_swing_pairs(indicators)
            bullish_pair = lows[-2:] if len(lows) >= 2 else []
            bearish_pair = highs[-2:] if len(highs) >= 2 else []

            bullish_div = False
            bearish_div = False
            selected_pair: list[tuple[float, float]] = []
            if len(bullish_pair) == 2:
                (p1, r1), (p2, r2) = bullish_pair
                bullish_div = p2 < p1 and r2 > r1 and abs(p2 - p1) >= 0.25 * atr
            if len(bearish_pair) == 2:
                (hp1, hr1), (hp2, hr2) = bearish_pair
                bearish_div = hp2 > hp1 and hr2 < hr1 and abs(hp2 - hp1) >= 0.25 * atr

            if not bullish_div and not bearish_div:
                self._no_vote("no_confirmed_swing_divergence")
                return None

            contract_side, option_premium_domain, _ = resolve_signal_domain(symbol, indicators)
            if option_premium_domain:
                if not bullish_div:
                    self._no_vote("premium_no_bullish_divergence")
                    return None
                side = contract_side
                selected_pair = bullish_pair
                divergence_type = "bullish"
            elif bullish_div:
                side = "CE"
                selected_pair = bullish_pair
                divergence_type = "bullish"
            else:
                side = "PE"
                selected_pair = bearish_pair
                divergence_type = "bearish"

            if side not in {"CE", "PE"}:
                self._no_vote("unresolved_trade_side")
                return None

            confirmation = bool(indicators.get("confirmation_candle"))
            if not confirmation:
                self._no_vote("no_confirmation")
                return None

            (p1, r1), (p2, r2) = selected_pair
            score = 4.0
            reasons = ["confirmed_swing_divergence", "structure_confirmation"]
            if regime in {MarketRegime.RANGE, MarketRegime.LOW_ACTIVITY}:
                score += 2.0
                reasons.append("regime_support")
            elif regime is MarketRegime.TREND:
                score -= 1.5
                reasons.append("strong_trend_penalty")
            if direction in {"CE", "PE"} and direction == side:
                score += 1.0
                reasons.append("direction_context")

            rsi_delta = abs(r2 - r1)
            if rsi_delta >= 5.0:
                score += 2.0
                reasons.append(f"divergence_strong_{rsi_delta:.1f}")
            elif rsi_delta >= 2.5:
                score += 1.0
                reasons.append(f"divergence_moderate_{rsi_delta:.1f}")
            else:
                reasons.append(f"divergence_weak_{rsi_delta:.1f}")

            strategy_score = max(0.0, min(10.0, score))
            if strategy_score < 3.5:
                self._no_vote("low_score")
                return None

            source_symbol = str(indicators.get("source_symbol") or "").strip()
            source_domain = "underlying_price" if source_symbol else "option_premium"
            metadata = {
                "strategy": "RSIDivergence",
                "strategy_name": "RSIDivergence",
                "role": "context",
                "can_trigger": False,
                "source_domain": "option_premium" if option_premium_domain else source_domain,
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
                "setup_type": "rsi_confirmed_swing_reversal",
                "required_data_present": True,
                "stale_data_used": bool(indicators.get("stale_data_used")),
                "candidate_symbol": symbol,
                "score_reasons": reasons,
                "rejection_reasons": [],
                "divergence_type": divergence_type,
                "swing_points": selected_pair,
                "confirmation_candle": True,
                "trend_regime": regime.value,
                "reversal_quality": round(strategy_score / 10.0, 3),
            }
            LOGGER.info(
                "STRATEGY_CONTEXT strategy=RSIDivergence side=%s score=%.2f",
                side,
                strategy_score,
            )
            return EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=max(0.1, min(0.82, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=None,
                target=None,
                quantity=self._cfg.quantity or 1,
                strategy_name="RSIDivergence",
                metadata=metadata,
            )
        except Exception as exc:
            LOGGER.error(
                "Failure in RSIDivergenceStrategy._evaluate_signal: %s",
                exc,
                exc_info=exc,
            )
            return None


__all__ = ["RSIDivergenceStrategy"]
