from __future__ import annotations

import os
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import SMCStrategyConfig
from nifty_scalper_bot.strategies.signal_quality import resolve_signal_domain
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class SMCStrategy(EliteStrategy):
    """SMC liquidity sweep strategy producing structured votes only."""

    MIN_BARS_REQUIRED = 15
    ROLE = 'trigger'
    TRIGGER_KEY = 'smc_lite'

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
            prior_swing_low = float(indicators.get('prior_swing_low') or low)
            prior_swing_high = float(indicators.get('prior_swing_high') or high)
            bullish_sweep = bool(indicators.get('liquidity_sweep_confirmed')) or (low <= prior_swing_low and close > open_price)
            bearish_sweep = bool(indicators.get('liquidity_sweep_confirmed_bear')) or (high >= prior_swing_high and close < open_price)
            if not bullish_sweep and not bearish_sweep:
                self._no_vote('no_liquidity_sweep')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=SMC reason=no_sweep')
                return None

            contract_side, option_premium_domain, _ = resolve_signal_domain(symbol, indicators)
            if option_premium_domain:
                premium_reversal = bool(bullish_sweep or indicators.get('premium_reclaim') or indicators.get('bullish_reversal'))
                structure_flip = bool(indicators.get('choch_confirmed') or indicators.get('bos_confirmed'))
                if not premium_reversal and not structure_flip:
                    self._no_vote('premium_not_reversing_up')
                    return None
                side = contract_side
            else:
                side = 'CE' if bullish_sweep else 'PE'
            sweep_level = low if bullish_sweep else high
            choch_confirmed = bool(indicators.get('choch_confirmed'))
            bos_confirmed = bool(indicators.get('bos_confirmed'))
            premium_reclaim = bool(indicators.get('premium_reclaim'))
            structure_confirmed = bool(bos_confirmed or choch_confirmed)
            retest_confirmed = bool(indicators.get('retest_confirmed') or indicators.get('mitigation_confirmed'))

            score = 2.0
            reasons = ['liquidity_sweep']
            if displacement_score >= 0.7:
                score += 2.0
                reasons.append('displacement')
            if structure_confirmed:
                score += 2.0
                reasons.append('structure_confirmation')
            elif displacement_score >= 0.6:
                score += 1.0
                reasons.append('displacement_only')
            if retest_confirmed:
                score += 1.0
                reasons.append('retest_mitigation')
            direction_aligned = direction in {'CE', 'PE'} and direction == side
            if direction_aligned:
                score += 2.0
                reasons.append('direction_alignment')
            elif premium_reclaim:
                score += 1.0
                reasons.append('premium_reclaim_support')
            if displacement_score >= 0.9:
                score += 1.0
                reasons.append('clean_invalidation_rr')

            strategy_score = max(0.0, min(10.0, score))
            execution_mode = str(os.getenv('EXECUTION_MODE', 'SHADOW') or 'SHADOW').strip().upper()
            is_live = execution_mode == 'LIVE'
            min_score = float(os.getenv('SMC_MIN_SCORE_LIVE', '6.5') if is_live else os.getenv('SMC_MIN_SCORE_SHADOW', '4.5'))
            require_structure_live = str(os.getenv('SMC_REQUIRE_STRUCTURE_CONFIRMATION_LIVE', 'true')).lower() in {'1', 'true', 'yes', 'on'}
            if is_live and require_structure_live and not structure_confirmed:
                self._no_vote('smc_structure_required_live')
                return None
            if not (bullish_sweep or bearish_sweep or premium_reclaim) or not (displacement_score >= 0.6 or structure_confirmed) or not (direction_aligned or premium_reclaim):
                self._no_vote('smc_quality_gate_failed')
                return None
            if strategy_score < min_score:
                self._no_vote('no_liquidity_sweep')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=SMC reason=low_quality')
                return None

            metadata = {
                'strategy': 'SMC',
                'strategy_name': 'SMC',
                'role': 'trigger',
                'signal_family': 'directional_trigger',
                'trade_side': side,
                'side': side,
                'direction_bias': side,
                'source_domain': 'option_premium' if option_premium_domain else 'underlying_price',
                'preliminary_only': True,
                'requires_runner_final_score': True,
                'direction_score': strategy_score,
                'strategy_score': strategy_score,
                'data_score': 8.0 if not stale_data else 3.0,
                'score_reasons': reasons,
                'setup_quality': strategy_score,
                'setup_type': 'liquidity_sweep_retest',
                'required_data_present': True,
                'stale_data_used': stale_data,
                'candidate_symbol': symbol,
                'rejection_reasons': [],
                'sweep_level': sweep_level,
                'displacement_score': round(displacement_score, 3),
                'structure_confirmed': structure_confirmed,
                'smc_sweep_type': 'bullish' if bullish_sweep else 'bearish',
                'structure_confirmation_used': structure_confirmed,
                'premium_reclaim_used': premium_reclaim,
                'smc_quality_score': strategy_score,
                'smc_block_reason': '',
                'retest_confirmed': retest_confirmed,
                'underlying_invalidation_level': sweep_level,
                'premium_stop_distance': max(atr * 1.2, current_price * 0.025, 1.0),
                'premium_target_rr': 2.0,
                'premium_reversal_gate_mode': 'hard' if option_premium_domain and not premium_reclaim and not structure_confirmed else 'soft',
            }
            LOGGER.info('STRATEGY_VOTE strategy=SMC side=%s score=%.2f', side, strategy_score)
            return EliteSignal(
                symbol=symbol,
                signal='BUY',
                confidence=max(0.1, min(0.88, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=None,
                target=None,
                quantity=self._cfg.quantity or 1,
                strategy_name='SMC',
                metadata=metadata,
            )
        except Exception as e:
            LOGGER.error('Failure in SMCStrategy._evaluate_signal: %s', e, exc_info=e)
            return None


__all__ = ['SMCStrategy']
