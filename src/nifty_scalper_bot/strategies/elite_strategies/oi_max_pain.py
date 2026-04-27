from __future__ import annotations

from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import OIMaxPainStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class OIMaxPainStrategy(EliteStrategy):
    """OI/max-pain contextual vote with conservative score cap."""

    def __init__(self, config: OIMaxPainStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: indicators set. Raises: Exception."""
        return {'max_pain', 'call_oi_wall', 'put_oi_wall', 'close', 'atr', 'direction_bias'}

    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        try:
            max_pain = float(indicators.get('max_pain') or 0.0)
            call_wall = float(indicators.get('call_oi_wall') or 0.0)
            put_wall = float(indicators.get('put_oi_wall') or 0.0)
            atr = max(float(indicators.get('atr') or 0.0), current_price * 0.01, 1.0)
            direction = str(indicators.get('direction_bias') or '').upper()
            if max_pain <= 0:
                LOGGER.debug('STRATEGY_NO_VOTE strategy=OIMaxPain reason=missing_oi')
                return None

            context_bias = 'CE' if current_price < max_pain else 'PE'
            side = direction if direction in {'CE', 'PE'} else context_bias
            distance_to_oi_wall = abs(current_price - (call_wall if side == 'CE' and call_wall > 0 else put_wall if put_wall > 0 else max_pain))

            score = 2.0
            reasons = ['oi_context_bias_only']
            if side == context_bias:
                score += 1.0
            if distance_to_oi_wall < 0.5 * atr:
                score -= 1.5
                reasons.append('near_oi_wall_against_trade')
            strategy_score = max(0.0, min(4.0, score))
            if strategy_score <= 0:
                return None

            metadata = {
                'strategy': 'OIMaxPain',
                'side': side,
                'direction_bias': side,
                'strategy_score': strategy_score,
                'setup_quality': strategy_score,
                'setup_type': 'oi_context',
                'required_data_present': True,
                'stale_data_used': bool(indicators.get('stale_data_used')),
                'candidate_symbol': symbol,
                'score_reasons': reasons,
                'rejection_reasons': [],
                'max_pain_level': max_pain,
                'call_oi_wall': call_wall,
                'put_oi_wall': put_wall,
                'distance_to_oi_wall': round(distance_to_oi_wall, 3),
                'context_bias': context_bias,
                'invalidation_level': None,
            }
            LOGGER.info('STRATEGY_VOTE strategy=OIMaxPain side=%s score=%.2f', side, strategy_score)
            return EliteSignal(
                symbol=symbol,
                signal='BUY',
                confidence=max(0.05, min(0.45, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=current_price - atr,
                target=current_price + atr,
                quantity=self._cfg.quantity or 1,
                strategy_name='OIMaxPain',
                metadata=metadata,
            )
        except Exception as e:
            LOGGER.error('Failure in OIMaxPainStrategy._evaluate_signal: %s', e, exc_info=e)
            return None


__all__ = ['OIMaxPainStrategy']
