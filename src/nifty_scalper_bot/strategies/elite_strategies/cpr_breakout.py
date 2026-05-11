from __future__ import annotations

import os
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import CPRBreakoutStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class CPRBreakoutStrategy(EliteStrategy):
    """CPR breakout context strategy with level-distance safeguards."""

    MIN_BARS_REQUIRED = 2

    def __init__(self, config: CPRBreakoutStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config
        self._require_valid_levels = str(os.getenv('CPR_REQUIRE_VALID_LEVELS', 'true')).strip().lower() in {'1', 'true', 'yes', 'on'}
        self._invalid_levels_logged = False

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: indicators set. Raises: Exception."""
        return {'pivot', 'bc', 'tc', 'r1', 's1', 'close', 'open', 'atr', 'direction_bias', 'retest_confirmed'}

    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        try:
            self._no_vote("stale_or_invalid_data")
            if symbol.upper().endswith(('CE', 'PE')) and not indicators.get('source_symbol'):
                self._no_vote('invalid_price_domain')
                return None
            cpr_bottom = float(indicators.get('bc') or 0.0)
            cpr_top = float(indicators.get('tc') or 0.0)
            pivot = float(indicators.get('pivot') or 0.0)
            r1 = float(indicators.get('r1') or 0.0)
            s1 = float(indicators.get('s1') or 0.0)
            atr = max(float(indicators.get('atr') or 0.0), current_price * 0.01, 1.0)
            direction = str(indicators.get('direction_bias') or '').upper()

            if self._require_valid_levels and (min(cpr_bottom, cpr_top, pivot) <= 0 or cpr_top <= cpr_bottom):
                if not self._invalid_levels_logged:
                    self._no_vote('invalid_cpr_levels')
                    self._invalid_levels_logged = True
                return None
            if cpr_bottom <= current_price <= cpr_top:
                self._no_vote('inside_cpr')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=CPRBreakout reason=inside_cpr')
                return None

            side = 'CE' if current_price > cpr_top else 'PE'
            nearest_level_distance = (r1 - current_price) if side == 'CE' and r1 > 0 else (current_price - s1) if side == 'PE' and s1 > 0 else atr
            if nearest_level_distance < 0.5 * atr:
                self._no_vote('nearby_level')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=CPRBreakout reason=nearby_level')
                return None

            retest_confirmed = bool(indicators.get('retest_confirmed'))
            breakout_quality = abs(current_price - (cpr_top if side == 'CE' else cpr_bottom)) / atr
            score = 2.0
            reasons = ['clean_break_beyond_cpr']
            if direction in {'CE', 'PE'} and direction == side:
                score += 2.0
                reasons.append('direction_alignment')
            if retest_confirmed or breakout_quality >= 0.5:
                score += 2.0
                reasons.append('retest_hold')
            if nearest_level_distance >= atr:
                score += 2.0
                reasons.append('adequate_distance_to_next_level')
            score += 2.0
            reasons.append('candidate_quality')

            strategy_score = max(0.0, min(10.0, score))
            metadata = {
                'strategy': 'CPRBreakout',
                'strategy_name': 'CPRBreakout',
                'role': 'trigger',
                'requires_feature_set': 'cpr_levels',
                'signal_family': 'directional_trigger',
                'trade_side': side,
                'side': side,
                'direction_bias': side,
                'preliminary_only': True,
                'requires_runner_final_score': True,
                'direction_score': strategy_score,
                'strategy_score': strategy_score,
                'data_score': 8.0,
                'setup_quality': strategy_score,
                'setup_type': 'cpr_breakout',
                'required_data_present': True,
                'stale_data_used': bool(indicators.get('stale_data_used')),
                'candidate_symbol': symbol,
                'score_reasons': reasons,
                'rejection_reasons': [],
                'cpr_top': cpr_top,
                'cpr_bottom': cpr_bottom,
                'pivot': pivot,
                'relation_to_cpr': 'above' if side == 'CE' else 'below',
                'breakout_quality': round(breakout_quality, 3),
                'nearest_level_distance': round(nearest_level_distance, 3),
                'underlying_invalidation_level': cpr_bottom if side == 'CE' else cpr_top,
                'premium_stop_distance': max(atr * 0.9, current_price * 0.02, 1.0),
                'premium_target_rr': 2.0,
            }
            LOGGER.info('STRATEGY_VOTE strategy=CPRBreakout side=%s score=%.2f', side, strategy_score)
            return EliteSignal(
                symbol=symbol,
                signal='BUY',
                confidence=max(0.1, min(0.88, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=None,
                target=None,
                quantity=self._cfg.quantity or 1,
                strategy_name='CPRBreakout',
                metadata=metadata,
            )
        except Exception as e:
            LOGGER.error('Failure in CPRBreakoutStrategy._evaluate_signal: %s', e, exc_info=e)
            return None


__all__ = ['CPRBreakoutStrategy']
