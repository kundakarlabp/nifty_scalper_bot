from __future__ import annotations

import os
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import OrderFlowStrategyConfig
from nifty_scalper_bot.strategies.signal_quality import resolve_signal_domain
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


def safe_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return float(default)


class OrderFlowStrategy(EliteStrategy):
    """Order-flow vote using spread, depth imbalance and tick direction."""

    MIN_BARS_REQUIRED = 5

    def __init__(self, config: OrderFlowStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: indicator keys. Raises: Exception."""
        return {'bid', 'ask', 'depth', 'tick_direction', 'buy_qty', 'sell_qty', 'direction_bias', 'spread_pct', 'atr'}

    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        try:
            self._no_vote('stale_or_invalid_data')
            bid = float(indicators.get('bid') or 0.0)
            ask = float(indicators.get('ask') or 0.0)
            depth = indicators.get('depth') or {}
            tick_direction = str(indicators.get('tick_direction') or '').upper()
            direction = str(indicators.get('direction_bias') or '').upper()
            contract_side, option_premium_domain, _ = resolve_signal_domain(symbol, indicators)
            atr = max(float(indicators.get('atr') or 0.0), current_price * 0.01, 1.0)
            execution_mode = str(os.getenv('EXECUTION_MODE', 'SHADOW') or 'SHADOW').strip().upper()
            is_live_mode = execution_mode == 'LIVE'
            allow_orderflow_trigger = str(os.getenv('ORDERFLOW_ALLOW_LIVE_TRIGGER' if is_live_mode else 'ORDERFLOW_ALLOW_TRIGGER_ROLE', 'true')).strip().lower() in {'1', 'true', 'yes', 'on'}
            allow_ltp_trigger = str(os.getenv('ORDERFLOW_ALLOW_LTP_FALLBACK_TRIGGER', os.getenv('ORDERFLOW_ALLOW_LTP_FALLBACK_TRIGGER', 'false'))).strip().lower() in {'1', 'true', 'yes', 'on'}
            trigger_min_score = safe_float_env('ORDERFLOW_MIN_SCORE_LIVE', 8.0) if is_live_mode else safe_float_env('ORDERFLOW_TRIGGER_MIN_SCORE', 5.0)
            trigger_max_spread_pct = safe_float_env('ORDERFLOW_MAX_SPREAD_PCT', 0.75) if is_live_mode else safe_float_env('ORDERFLOW_TRIGGER_MAX_SPREAD_PCT', 12.0)
            context_min_score = float(os.getenv('ORDERFLOW_CONTEXT_MIN_SCORE', '4.0') or '4.0')
            require_tradable_quote_live = str(os.getenv('ORDERFLOW_REQUIRE_TRADABLE_QUOTE_LIVE', 'true')).strip().lower() in {'1', 'true', 'yes', 'on'}
            tradable_quote = bool(indicators.get('tradable_quote', True))
            quote_depth_valid = bool(indicators.get('quote_depth_valid', True))
            tick_age_ms = float(indicators.get('tick_age_ms') or 0.0)
            max_tick_age_ms = float(os.getenv('LIVE_MAX_TICK_AGE_MS', '2500') or '2500')

            if bid <= 0 or ask <= 0 or ask <= bid:
                self._no_vote('ltp_only_no_depth' if not depth else 'missing_bid_ask')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=OrderFlow reason=missing_bid_ask')
                return None
            spread_pct = float(indicators.get('spread_pct') or (((ask - bid) / ((ask + bid) / 2.0)) * 100.0))
            if spread_pct > 28.0:
                self._no_vote('wide_spread')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=OrderFlow reason=wide_spread')
                return None

            bids = depth.get('buy', []) if isinstance(depth, dict) else []
            asks = depth.get('sell', []) if isinstance(depth, dict) else []
            depth_available = bool(bids and asks)
            total_bid = sum(float(level.get('quantity', 0.0)) for level in bids[:5]) if depth_available else 0.0
            total_ask = sum(float(level.get('quantity', 0.0)) for level in asks[:5]) if depth_available else 0.0

            if total_bid + total_ask <= 0:
                allow_fallback = str(os.getenv('ORDERFLOW_ALLOW_LTP_TICK_FALLBACK', 'false')).strip().lower() in {'1', 'true', 'yes', 'on'}
                strict_spread_required = str(os.getenv('ORDERFLOW_REQUIRE_SPREAD_IN_STRICT_MODE', 'true')).strip().lower() in {'1', 'true', 'yes', 'on'}
                stale_age_s = float(indicators.get('data_age_seconds') or 0.0)
                if not allow_fallback:
                    self._no_vote('missing_depth')
                    return None
                if stale_age_s > 2.0:
                    self._no_vote('stale_tick_for_ltp_fallback')
                    return None
                if strict_spread_required and spread_pct <= 0:
                    self._no_vote('spread_unavailable_for_ltp_fallback')
                    return None

                side = contract_side if option_premium_domain else ('CE' if tick_direction in {'UP', 'BUY'} else 'PE')
                tick_supports = tick_direction in {'UP', 'BUY'} if option_premium_domain else ((side == 'CE' and tick_direction in {'UP', 'BUY'}) or (side == 'PE' and tick_direction in {'DOWN', 'SELL'}))
                fallback_score = 2.0 + (2.0 if spread_pct <= 12.0 else 0.0)
                if direction in {'CE', 'PE'} and direction == side:
                    fallback_score += 1.5
                if tick_direction in {'UP', 'DOWN', 'BUY', 'SELL'}:
                    fallback_score += 1.0
                strategy_score = min(5.5, max(0.0, fallback_score))
                if strategy_score < context_min_score:
                    self._no_vote('weak_tick_confirmation')
                    return None
                trigger_conditions_met = bool(
                    allow_ltp_trigger
                    and not is_live_mode
                    and strategy_score >= trigger_min_score
                    and spread_pct > 0
                    and spread_pct <= trigger_max_spread_pct
                    and stale_age_s <= 2.0
                )
                trigger_block_reason = '' if trigger_conditions_met else ('ltp_fallback_not_allowed_live' if is_live_mode else 'ltp_trigger_disabled')
                if not trigger_conditions_met and strategy_score < trigger_min_score:
                    trigger_block_reason = 'score_below_trigger_min'
                elif not trigger_conditions_met and (spread_pct <= 0 or spread_pct > trigger_max_spread_pct):
                    trigger_block_reason = 'spread_above_trigger_max'
                elif not trigger_conditions_met and stale_age_s > 2.0:
                    trigger_block_reason = 'stale_tick_for_ltp_trigger'
                metadata = {
                    'orderflow_depth_source': 'ltp_tick_fallback',
                    'risk_label': 'ltp_only_orderflow_reduced_confidence',
                    'strategy': 'OrderFlow',
                    'strategy_name': 'OrderFlow',
                    'role': 'trigger' if trigger_conditions_met else 'context',
                    'source_domain': 'market_microstructure',
                    'context_score': strategy_score,
                    'side': side,
                    'trade_side': side,
                    'contract_side': side,
                    'direction_bias': direction if direction in {'CE', 'PE'} else None,
                    'strategy_score': strategy_score,
                    'setup_quality': strategy_score,
                    'spread_pct': round(spread_pct, 3),
                    'depth_imbalance': 0.0,
                    'tick_direction': tick_direction,
                    'score_reasons': ['ltp_fallback', 'reduced_confidence'],
                    'trigger_min_score': trigger_min_score,
                    'trigger_max_spread_pct': trigger_max_spread_pct,
                    'trigger_conditions_met': trigger_conditions_met,
                    'trigger_block_reason': trigger_block_reason,
                    'quote_depth_valid': False,
                    'can_trigger': bool(trigger_conditions_met),
                    'spread_score': 2.0 if spread_pct <= trigger_max_spread_pct else 0.0,
                    'depth_score': 0.0,
                    'tick_score': 2.0 if tick_supports else 0.0,
                    'direction_alignment_score': 2.0 if (direction in {'CE', 'PE'} and direction == side) else 0.0,
                    'premium_stop_distance': max(0.8 * atr, current_price * 0.02, 1.0),
                    'premium_target_rr': 1.8,
                    'tradable_quote': tradable_quote,
                    'depth_available': depth_available,
                    'premium_flow_direction': tick_direction,
                    'liquidity_score': 1.0,
                    'tick_age_ms': tick_age_ms,
                    'quote_update_version': indicators.get('quote_update_version'),
                }
                side_aligns = direction in {'CE', 'PE'} and direction == side
                metadata.update({'context_role': 'confirmation', 'context_bonus_score': strategy_score if side_aligns else 0.0, 'context_veto_score': strategy_score if (direction in {'CE', 'PE'} and direction != side) else 0.0, 'tick_supports_direction': tick_supports})
                return EliteSignal(symbol=symbol, signal='BUY', confidence=max(0.1, min(0.55, strategy_score / 10.0)), entry_price=current_price, stop_loss=None, target=None, quantity=self._cfg.quantity or 1, strategy_name='OrderFlow', metadata=metadata)

            depth_imbalance = (total_bid - total_ask) / max(total_bid + total_ask, 1.0)
            side = contract_side if option_premium_domain else ('CE' if depth_imbalance > 0 else 'PE')
            clear_adverse_flow = bool(option_premium_domain and quote_depth_valid and depth_available and depth_imbalance <= -0.10 and tick_direction not in {'UP', 'BUY'})
            if clear_adverse_flow:
                self._no_vote('negative_premium_flow')
                return None

            score = 0.0
            reasons: list[str] = []
            if spread_pct <= 12.0:
                score += 2.0
                reasons.append('tight_spread')
            if abs(depth_imbalance) >= 0.15:
                score += 2.0
                reasons.append('depth_imbalance')
            if abs(depth_imbalance) >= 0.50:
                score += 1.0
                reasons.append('strong_depth_imbalance')
            tick_supports = tick_direction in {'UP', 'BUY'} if option_premium_domain else ((side == 'CE' and tick_direction in {'UP', 'BUY'}) or (side == 'PE' and tick_direction in {'DOWN', 'SELL'}))
            if tick_supports:
                score += 2.0
                reasons.append('tick_direction_alignment')
            if direction in {'CE', 'PE'} and direction == side:
                score += 2.0
                reasons.append('direction_context_alignment')
            if not bool(indicators.get('stale_data_used')):
                score += 1.0
            score += 1.0
            strategy_score = max(0.0, min(10.0, score))
            if strategy_score < context_min_score:
                self._no_vote('low_score')
                return None

            side_aligns = direction in {'CE', 'PE'} and direction == side
            side_alignment_ok = direction not in {'CE', 'PE'} or side_aligns
            tick_direction_missing = tick_direction not in {'UP', 'DOWN', 'BUY', 'SELL'}
            direction_context_missing = direction not in {'CE', 'PE'}
            allow_without_direction_live = str(os.getenv('ORDERFLOW_ALLOW_TRIGGER_WITHOUT_DIRECTION_LIVE', 'false')).strip().lower() in {'1', 'true', 'yes', 'on'}
            direction_context_ok = (direction in {'CE', 'PE'}) or (not is_live_mode) or allow_without_direction_live
            max_context_age = safe_float_env('ORDERFLOW_MAX_CONTEXT_AGE_SECONDS', 5.0)
            age_raw = indicators.get('context_age_seconds')
            try:
                context_age_ok = age_raw is not None and float(age_raw) <= max_context_age
            except (TypeError, ValueError):
                context_age_ok = False
            trigger_conditions_met = bool(
                allow_orderflow_trigger
                and quote_depth_valid
                and (tradable_quote or not (is_live_mode and require_tradable_quote_live))
                and bid > 0.0
                and ask > 0.0
                and depth_available
                and strategy_score >= trigger_min_score
                and spread_pct <= trigger_max_spread_pct
                and side_alignment_ok
                and direction_context_ok
                and context_age_ok
                and tick_supports
                and tick_age_ms <= max_tick_age_ms
            )
            near_atm_threshold = safe_float_env('STRATEGY_NEAR_ATM_THRESHOLD_POINTS', 50.0)
            selected_meta_available = any(indicators.get(name) is not None for name in ('is_selected_option', 'strike_distance_from_atm', 'selected_ce', 'selected_pe'))
            selected_or_near_atm = bool(indicators.get('is_selected_option'))
            if not selected_or_near_atm and indicators.get('strike_distance_from_atm') is not None:
                try:
                    selected_or_near_atm = float(indicators.get('strike_distance_from_atm')) <= near_atm_threshold
                except (TypeError, ValueError):
                    selected_or_near_atm = False
            if is_live_mode and not selected_meta_available:
                selected_or_near_atm = False
            conflict_override_requested = bool(
                not side_alignment_ok
                and strategy_score >= float(os.getenv('ORDERFLOW_CONFLICT_OVERRIDE_MIN_SCORE', '9.0') or '9.0')
                and bool(quote_depth_valid)
                and bool(depth_available)
                and spread_pct <= trigger_max_spread_pct
                and bool(context_age_ok)
                and bool(tick_supports)
                and tick_age_ms <= max_tick_age_ms
                and bool(selected_or_near_atm)
            )
            conflict_override_applied = False
            if conflict_override_requested and not trigger_conditions_met:
                trigger_conditions_met = bool(
                    allow_orderflow_trigger
                    and (tradable_quote or not (is_live_mode and require_tradable_quote_live))
                    and bid > 0.0
                    and ask > 0.0
                    and direction_context_ok
                )
                conflict_override_applied = bool(trigger_conditions_met)
                if conflict_override_applied:
                    LOGGER.info(
                        'ORDERFLOW_HIGH_CONVICTION_OVERRIDE symbol=%s side=%s score=%.2f spread_pct=%.2f context_age=%s',
                        symbol,
                        side,
                        strategy_score,
                        spread_pct,
                        age_raw,
                        extra={'event': 'ORDERFLOW_HIGH_CONVICTION_OVERRIDE', 'symbol': symbol, 'side': side, 'score': strategy_score, 'spread_pct': spread_pct, 'context_age': age_raw},
                    )
            trigger_block_reason = ''
            if not allow_orderflow_trigger:
                trigger_block_reason = 'trigger_role_disabled'
            elif not quote_depth_valid or not depth_available:
                trigger_block_reason = 'quote_depth_missing'
            elif is_live_mode and require_tradable_quote_live and not tradable_quote:
                trigger_block_reason = 'tradable_quote_false'
            elif tick_age_ms > max_tick_age_ms:
                trigger_block_reason = 'tick_stale'
            elif tick_direction_missing:
                trigger_block_reason = 'tick_direction_missing_or_neutral'
            elif not direction_context_ok:
                trigger_block_reason = 'direction_context_missing_live'
            elif age_raw is None:
                trigger_block_reason = 'context_age_missing'
            elif not context_age_ok:
                trigger_block_reason = 'context_stale'
            elif not side_alignment_ok and not conflict_override_applied:
                trigger_block_reason = 'direction_bias_conflict'
            elif spread_pct > trigger_max_spread_pct:
                trigger_block_reason = 'spread_too_wide'
            elif strategy_score < trigger_min_score:
                trigger_block_reason = 'score_below_live_trigger_min' if is_live_mode else 'score_below_trigger_min'
            elif not tick_supports:
                trigger_block_reason = 'negative_premium_flow'

            metadata = {
                'strategy': 'OrderFlow', 'strategy_name': 'OrderFlow', 'role': 'trigger' if trigger_conditions_met else 'context',
                'source_domain': 'market_microstructure', 'context_score': strategy_score, 'side': side, 'trade_side': side,
                'contract_side': side, 'direction_bias': direction if direction in {'CE', 'PE'} else None,
                'strategy_score': strategy_score, 'setup_quality': strategy_score, 'setup_type': 'microstructure_imbalance',
                'required_data_present': depth_available, 'stale_data_used': bool(indicators.get('stale_data_used')), 'candidate_symbol': symbol,
                'score_reasons': reasons, 'rejection_reasons': [] if depth_available else ['depth_missing'], 'bid': bid, 'ask': ask,
                'spread_pct': round(spread_pct, 3), 'depth_imbalance': round(depth_imbalance, 4), 'tick_direction': tick_direction,
                'liquidity_ok': spread_pct <= 12.0, 'premium_stop_distance': max(0.8 * atr, current_price * 0.02, 1.0),
                'premium_target_rr': 1.8, 'can_trigger': bool(trigger_conditions_met), 'trigger_min_score': trigger_min_score,
                'trigger_max_spread_pct': trigger_max_spread_pct, 'trigger_conditions_met': trigger_conditions_met,
                'trigger_block_reason': trigger_block_reason, 'quote_depth_valid': bool(quote_depth_valid),
                'tick_direction_missing': tick_direction_missing, 'direction_context_missing': direction_context_missing,
                'direction_context_ok': direction_context_ok,
                'trigger_eligible': bool(trigger_conditions_met),
                'trigger_disqualified_by': trigger_block_reason or None,
                'liquidity_score': 2.0 if spread_pct <= 12.0 else 0.5,
                'spread_score': 2.0 if spread_pct <= trigger_max_spread_pct else 0.0,
                'depth_score': 2.0 if depth_available else 0.0,
                'tick_score': 2.0 if tick_supports else 0.0,
                'direction_alignment_score': 2.0 if side_alignment_ok else 0.0,
                'tradable_quote': tradable_quote,
                'depth_available': depth_available,
                'premium_flow_direction': tick_direction,
                'negative_premium_flow_mode': 'hard' if clear_adverse_flow else 'soft',
                'tick_age_ms': tick_age_ms,
                'quote_update_version': indicators.get('quote_update_version'),
                'selected_or_near_atm': selected_or_near_atm,
                'orderflow_conflict_override_requested': conflict_override_requested,
                'orderflow_conflict_override_applied': conflict_override_applied,
                'orderflow_conflict_override': conflict_override_applied,
                'conflict_override_reason': 'high_conviction_depth_spread_tick_near_atm' if conflict_override_applied else '',
            }
            if trigger_conditions_met:
                metadata['approval_candidate'] = 'orderflow_live_depth_trigger'
            metadata.update({'context_role': 'confirmation', 'context_bonus_score': strategy_score if side_aligns else 0.0, 'context_veto_score': strategy_score if (direction in {'CE', 'PE'} and direction != side) else 0.0})
            LOGGER.info('ORDERFLOW_TRIGGER_DECISION symbol=%s side=%s trigger_conditions_met=%s trigger_block_reason=%s score=%.2f spread_pct=%.2f context_age_seconds=%s', symbol, side, trigger_conditions_met, trigger_block_reason, strategy_score, spread_pct, indicators.get('context_age_seconds'), extra={'event':'ORDERFLOW_TRIGGER_DECISION','symbol':symbol,'side':side,'trigger_conditions_met':trigger_conditions_met,'trigger_block_reason':trigger_block_reason}); return EliteSignal(symbol=symbol, signal='BUY', confidence=max(0.1, min(0.85, strategy_score / 10.0)), entry_price=current_price, stop_loss=None, target=None, quantity=self._cfg.quantity or 1, strategy_name='OrderFlow', metadata=metadata)
        except Exception as e:
            LOGGER.error('Failure in OrderFlowStrategy._evaluate_signal: %s', e, exc_info=e)
            return None


__all__ = ['OrderFlowStrategy']
