from __future__ import annotations

from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import BBSqueezeStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class BBSqueezeStrategy(EliteStrategy):
    """Bollinger squeeze-to-expansion strategy vote emitter."""

    MIN_BARS_REQUIRED = 25

    def __init__(self, config: BBSqueezeStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: required indicators. Raises: Exception."""
        return {'bollinger_upper', 'bollinger_lower', 'bollinger_middle', 'close', 'open', 'volume', 'avg_volume', 'direction_bias', 'atr'}

    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        try:
            self._no_vote("stale_or_invalid_data")
            upper = float(indicators.get('bollinger_upper') or 0.0)
            lower = float(indicators.get('bollinger_lower') or 0.0)
            mid = float(indicators.get('bollinger_middle') or 0.0)
            close = float(indicators.get('close') or current_price)
            open_price = float(indicators.get('open') or current_price)
            atr = max(float(indicators.get('atr') or 0.0), current_price * 0.01, 1.0)
            direction = str(indicators.get('direction_bias') or '').upper()
            vol = float(indicators.get('volume') or 0.0)
            avg_vol = float(indicators.get('avg_volume') or 0.0)

            if min(upper, lower, mid) <= 0 or upper <= lower:
                self._no_vote('invalid_bb_levels')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=BBSqueeze reason=invalid_bb_levels')
                return None
            bb_width = (upper - lower) / mid
            squeeze_threshold = float(getattr(self._cfg, 'squeeze_threshold_pct', 0.5)) / 100.0
            squeeze_detected = bb_width <= max(squeeze_threshold, 0.008)
            if not squeeze_detected:
                self._no_vote('no_squeeze')
                return None

            breakout_side = 'CE' if close > upper else 'PE' if close < lower else 'UNKNOWN'
            if breakout_side == 'UNKNOWN':
                self._no_vote('no_breakout')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=BBSqueeze reason=no_breakout')
                return None
            expansion_confirmed = abs(close - open_price) >= 0.35 * atr
            if not expansion_confirmed:
                self._no_vote('weak_expansion')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=BBSqueeze reason=no_expansion')
                return None

            score = 1.0 + 2.0 + 2.0
            reasons = ['contraction', 'expansion_confirmed', 'breakout_candle']
            if direction in {'CE', 'PE'} and direction == breakout_side:
                score += 2.0
                reasons.append('direction_alignment')
            momentum_confirmed = avg_vol > 0 and vol >= avg_vol
            if momentum_confirmed:
                score += 1.0
                reasons.append('volume_confirmation')
            score += 2.0
            reasons.append('clean_rr')

            strategy_score = max(0.0, min(10.0, score))
            metadata = {
                'strategy': 'BBSqueeze',
                'side': breakout_side,
                'direction_bias': breakout_side,
                'strategy_score': strategy_score,
                'setup_quality': strategy_score,
                'setup_type': 'squeeze_expansion_breakout',
                'required_data_present': True,
                'stale_data_used': bool(indicators.get('stale_data_used')),
                'candidate_symbol': symbol,
                'score_reasons': reasons,
                'rejection_reasons': [],
                'bb_width': round(bb_width, 4),
                'bb_width_percentile': indicators.get('bb_width_percentile', None),
                'squeeze_detected': squeeze_detected,
                'expansion_confirmed': expansion_confirmed,
                'breakout_side': breakout_side,
                'momentum_confirmed': momentum_confirmed,
                'invalidation_level': mid,
            }
            LOGGER.info('STRATEGY_VOTE strategy=BBSqueeze side=%s score=%.2f', breakout_side, strategy_score)
            return EliteSignal(
                symbol=symbol,
                signal='BUY',
                confidence=max(0.1, min(0.9, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=mid,
                target=current_price + (2.0 * atr),
                quantity=self._cfg.quantity or 1,
                strategy_name='BBSqueeze',
                metadata=metadata,
            )
        except Exception as e:
            LOGGER.error('Failure in BBSqueezeStrategy._evaluate_signal: %s', e, exc_info=e)
            return None


__all__ = ['BBSqueezeStrategy']
