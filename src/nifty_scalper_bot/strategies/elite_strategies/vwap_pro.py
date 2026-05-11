from __future__ import annotations

import os
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import VWAPProStrategyConfig
from nifty_scalper_bot.strategies.signal_quality import resolve_signal_domain
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class VWAPProStrategy(EliteStrategy):
    """VWAP continuation/pullback strategy emitting scored strategy votes."""

    MIN_BARS_REQUIRED = 10
    ROLE = 'trigger'
    TRIGGER_KEY = 'vwap_pro'

    def __init__(self, config: VWAPProStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config
        self._allow_pullback = str(os.getenv('VWAP_ALLOW_PULLBACK_ENTRY', '1')).lower() in {'1', 'true', 'yes', 'on'}
        self._slack_atr_mult = float(os.getenv('VWAP_SLACK_ATR_MULT', '1.5') or 1.5)
        self._max_distance_pct = float(os.getenv('VWAP_MAX_OPTION_DISTANCE_PCT', '0.18') or 0.18)
        self._max_atr_distance_mult = float(os.getenv('VWAP_MAX_ATR_DISTANCE_MULT', '1.5') or 1.5)

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: indicators set. Raises: Exception."""
        return {'vwap', 'atr', 'close', 'open', 'high', 'low', 'volume', 'avg_volume', 'direction_bias'}

    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        try:
            self._no_vote("stale_or_invalid_data")
            vwap = float(indicators.get('vwap') or indicators.get('exchange_vwap') or 0.0)
            atr = float(indicators.get('atr') or 0.0)
            close = float(indicators.get('close') or current_price)
            open_price = float(indicators.get('open') or current_price)
            high = float(indicators.get('high') or current_price)
            low = float(indicators.get('low') or current_price)
            vol = float(indicators.get('volume') or 0.0)
            avg_vol = float(indicators.get('avg_volume') or 0.0)
            spread_pct = float(indicators.get('spread_pct') or 0.0)
            direction = str(indicators.get('direction_bias') or '').upper()
            if current_price <= 0 or vwap <= 0:
                self._no_vote('missing_vwap')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=VWAPPro reason=missing_vwap')
                return None

            required_data_present = bool(vwap > 0 and atr >= 0)
            max_data_age = 45.0 if str(os.getenv('EXECUTION_MODE','SHADOW')).strip().upper() == 'LIVE' else 120.0
            stale_data = bool(indicators.get('stale_data_used')) or float(indicators.get('data_age_seconds') or 0.0) > max_data_age
            distance_pct = abs(current_price - vwap) / max(vwap, 1e-9)
            allowed_distance = max(self._max_distance_pct, self._max_atr_distance_mult * atr / max(vwap, 1e-9))
            overextended = distance_pct > allowed_distance
            if overextended:
                self._no_vote('distance_outside_band')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=VWAPPro reason=overextended')
                return None
            if stale_data:
                self._no_vote('stale_data')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=VWAPPro reason=stale_data')
                return None
            max_spread_pct = 12.0 if str(os.getenv('EXECUTION_MODE','SHADOW')).strip().upper() == 'LIVE' else 28.0

            if spread_pct > max_spread_pct:
                self._no_vote('wide_spread')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=VWAPPro reason=wide_spread')
                return None

            score = 0.0
            reasons: list[str] = []
            contract_side, option_premium_domain, _ = resolve_signal_domain(symbol, indicators)
            trend_alignment = False
            pullback_flag = False
            continuation_confirmed = False

            if contract_side not in {'CE', 'PE'}:
                fallback_side = str(indicators.get('direction_bias') or '').upper()
                if fallback_side in {'CE', 'PE'}:
                    contract_side = fallback_side
                else:
                    self._no_vote('unknown_contract_side')
                    LOGGER.debug('STRATEGY_NO_VOTE strategy=VWAPPro reason=unknown_contract_side')
                    return None
            if not option_premium_domain:
                self._no_vote('invalid_price_domain')
                return None
            premium_above_vwap = current_price >= vwap
            if premium_above_vwap:
                score += 2.0
                reasons.append('premium_above_vwap')

            candle_body = abs(close - open_price)
            atr_safe = max(atr, current_price * 0.01, 1.0)
            continuation_confirmed = candle_body >= (0.35 * atr_safe)
            if continuation_confirmed:
                score += 1.5
                reasons.append('premium_continuation')

            reclaim_from_below = low <= (vwap - (atr_safe * self._slack_atr_mult * 0.2)) and close >= vwap
            if self._allow_pullback and reclaim_from_below:
                pullback_flag = True
                score += 1.5
                reasons.append('premium_reclaim_vwap')

            if distance_pct <= self._max_distance_pct * 0.7:
                score += 1.0
                reasons.append('not_overextended')

            if avg_vol > 0 and vol >= 0.6 * avg_vol:
                score += 1.0

            if direction in {'CE', 'PE'}:
                trend_alignment = direction == contract_side
                if trend_alignment:
                    score += 2.0
                    reasons.append('trend_alignment')
                else:
                    score -= 2.0
                    reasons.append('direction_conflict')

            if score < 5.5:
                self._no_vote('weak_score')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=VWAPPro reason=low_score')
                return None

            strategy_score = max(0.0, min(10.0, score))
            confidence = max(0.45, min(0.85, strategy_score / 10.0))
            metadata = {
                'strategy': 'VWAPPro',
                'strategy_name': 'VWAPPro',
                'role': 'trigger',
                'source_domain': 'option_premium',
                'signal_family': 'directional_trigger',
                'trade_side': contract_side,
                'side': contract_side,
                'contract_side': contract_side,
                'premium_above_vwap': premium_above_vwap,
                'direction_bias': contract_side,
                'preliminary_only': True,
                'requires_runner_final_score': True,
                'direction_score': strategy_score,
                'strategy_score': strategy_score,
                'data_score': 8.0 if not stale_data else 3.0,
                'score_reasons': reasons,
                'setup_type': 'continuation_pullback',
                'setup_quality': strategy_score,
                'required_data_present': required_data_present,
                'stale_data_used': stale_data,
                'candidate_symbol': symbol,
                'rejection_reasons': [],
                'vwap': vwap,
                'distance_pct': round(distance_pct, 4),
                'allowed_distance_pct': round(allowed_distance, 4),
                'atr': atr_safe,
                'pullback_flag': pullback_flag,
                'trend_alignment': trend_alignment,
                'continuation_confirmed': continuation_confirmed,
                'setup_invalidation_premium': current_price - atr_safe,
                'premium_stop_distance': atr_safe,
                'premium_target_rr': 2.0,
                'underlying_invalidation_level': (vwap - atr_safe) if contract_side == 'CE' else (vwap + atr_safe),
            }
            LOGGER.info('STRATEGY_VOTE strategy=VWAPPro side=%s score=%.2f', contract_side, strategy_score)
            return EliteSignal(
                symbol=symbol,
                signal='BUY',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=None,
                target=None,
                quantity=self._cfg.quantity or 1,
                strategy_name='VWAPPro',
                metadata=metadata,
            )
        except Exception as e:
            LOGGER.error('Failure in VWAPProStrategy._evaluate_signal: %s', e, exc_info=e)
            return None


__all__ = ['VWAPProStrategy']
