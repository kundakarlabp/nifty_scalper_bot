from __future__ import annotations

from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import SMCStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class SMCStrategy(EliteStrategy):
    """SMC liquidity sweep strategy producing structured votes only."""

    MIN_BARS_REQUIRED = 15

    def __init__(self, config: SMCStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: indicators set. Raises: Exception."""
        return {'high', 'low', 'close', 'open', 'atr', 'direction_bias', 'bos_confirmed', 'choch_confirmed', 'retest_confirmed'}

    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        try:
            self._no_vote("stale_or_invalid_data")
            high = float(indicators.get('high') or current_price)
            low = float(indicators.get('low') or current_price)
            close = float(indicators.get('close') or current_price)
            open_price = float(indicators.get('open') or current_price)
            atr = max(float(indicators.get('atr') or 0.0), current_price * 0.01, 1.0)
            direction = str(indicators.get('direction_bias') or '').upper()
            stale_data = bool(indicators.get('stale_data_used')) or float(indicators.get('data_age_seconds') or 0.0) > 120.0

            if stale_data or current_price <= 0:
                self._no_vote('no_liquidity_sweep')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=SMC reason=stale_or_invalid_data')
                return None

            body = abs(close - open_price)
            displacement_score = body / atr
            bullish_sweep = low < (open_price - 0.3 * atr) and close > open_price
            bearish_sweep = high > (open_price + 0.3 * atr) and close < open_price
            if not bullish_sweep and not bearish_sweep:
                self._no_vote('no_liquidity_sweep')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=SMC reason=no_sweep')
                return None

            side = 'CE' if bullish_sweep else 'PE'
            sweep_level = low if bullish_sweep else high
            structure_confirmed = bool(indicators.get('bos_confirmed') or indicators.get('choch_confirmed') or displacement_score >= 0.6)
            retest_confirmed = bool(indicators.get('retest_confirmed') or indicators.get('mitigation_confirmed'))

            score = 2.0
            reasons = ['liquidity_sweep']
            if displacement_score >= 0.7:
                score += 2.0
                reasons.append('displacement')
            if structure_confirmed:
                score += 2.0
                reasons.append('structure_confirmation')
            if retest_confirmed:
                score += 1.0
                reasons.append('retest_mitigation')
            if direction in {'CE', 'PE'} and direction == side:
                score += 2.0
                reasons.append('direction_alignment')
            if displacement_score >= 0.9:
                score += 1.0
                reasons.append('clean_invalidation_rr')

            strategy_score = max(0.0, min(10.0, score))
            if strategy_score < 4.0:
                self._no_vote('no_liquidity_sweep')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=SMC reason=low_quality')
                return None

            invalidation_level = sweep_level - 0.2 * atr if side == 'CE' else sweep_level + 0.2 * atr
            metadata = {
                'strategy': 'SMC',
                'side': side,
                'direction_bias': side,
                'strategy_score': strategy_score,
                'score_reasons': reasons,
                'setup_quality': strategy_score,
                'setup_type': 'liquidity_sweep_reclaim',
                'required_data_present': True,
                'stale_data_used': stale_data,
                'candidate_symbol': symbol,
                'rejection_reasons': [],
                'sweep_level': sweep_level,
                'displacement_score': round(displacement_score, 3),
                'structure_confirmed': structure_confirmed,
                'retest_confirmed': retest_confirmed,
                'invalidation_level': invalidation_level,
            }
            LOGGER.info('STRATEGY_VOTE strategy=SMC side=%s score=%.2f', side, strategy_score)
            return EliteSignal(
                symbol=symbol,
                signal='BUY',
                confidence=max(0.1, min(0.88, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=invalidation_level,
                target=current_price + (2.0 * atr),
                quantity=self._cfg.quantity or 1,
                strategy_name='SMC',
                metadata=metadata,
            )
        except Exception as e:
            LOGGER.error('Failure in SMCStrategy._evaluate_signal: %s', e, exc_info=e)
            return None


__all__ = ['SMCStrategy']
