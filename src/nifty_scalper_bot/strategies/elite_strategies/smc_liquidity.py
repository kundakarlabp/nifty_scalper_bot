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

    MIN_BARS_REQUIRED = 30
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
            direction = str(indicators.get("direction_bias") or "").upper()
            stale_data = bool(indicators.get('stale_data_used')) or float(indicators.get('data_age_seconds') or 0.0) > 120.0
            try:
                context_age_seconds = float(indicators.get("context_age_seconds") or 999.0)
            except (TypeError, ValueError):
                context_age_seconds = 999.0
            underlying_direction = str(indicators.get("underlying_direction_bias") or "").upper()
            effective_direction = underlying_direction or direction
            if str(os.getenv('EXECUTION_MODE', 'SHADOW') or 'SHADOW').strip().upper() == 'LIVE' and not effective_direction:
                self._no_vote("direction_context_not_ready")
                LOGGER.warning(
                    "STRATEGY_NO_VOTE strategy=SMC symbol=%s reason=direction_context_not_ready direction_bias=%s underlying_direction_bias=%s",
                    symbol,
                    direction,
                    underlying_direction,
                    extra={"event": "STRATEGY_NO_VOTE", "strategy": "SMC", "symbol": symbol, "reason": "direction_context_not_ready"},
                )
                return None
            min_bars_required = int(os.getenv("SMC_MIN_BARS_REQUIRED", "30") or "30")
            execution_mode = str(os.getenv('EXECUTION_MODE', 'SHADOW') or 'SHADOW').strip().upper()
            is_live = execution_mode == 'LIVE'
            history_domain_used = str(indicators.get("history_domain_used") or "unknown")
            option_history_count = indicators.get("option_history_count")
            underlying_history_count = indicators.get("underlying_history_count")
            spot_history_count = indicators.get("spot_history_count")
            indicator_history_count = indicators.get("indicator_history_count")
            raw_history_count = indicators.get("history_count")
            resolved_history_count = (
                option_history_count if history_domain_used == "options"
                else underlying_history_count if history_domain_used == "underlying"
                else raw_history_count if raw_history_count is not None else indicator_history_count
            )
            if is_live and resolved_history_count is None:
                self._no_vote("smc_history_count_missing")
                LOGGER.warning(
                    "STRATEGY_NO_VOTE strategy=SMC symbol=%s reason=smc_history_count_missing min_bars=%s history_domain_used=%s",
                    symbol,
                    min_bars_required,
                    history_domain_used,
                    extra={
                        "event": "STRATEGY_NO_VOTE",
                        "strategy": "SMC",
                        "symbol": symbol,
                        "reason": "smc_history_count_missing",
                        "min_bars_required": min_bars_required,
                        "resolved_history_count": resolved_history_count,
                        "history_domain_used": history_domain_used,
                        "option_history_count": option_history_count,
                        "underlying_history_count": underlying_history_count,
                        "spot_history_count": spot_history_count,
                        "indicator_history_count": indicator_history_count,
                        "history_source": indicators.get("history_source"),
                    },
                )
                return None
            history_count = int(resolved_history_count or 0)
            if is_live and history_count < min_bars_required:
                self._no_vote("smc_insufficient_history")
                LOGGER.warning(
                    "STRATEGY_NO_VOTE strategy=SMC symbol=%s reason=smc_insufficient_history history_count=%s min_bars=%s",
                    symbol,
                    history_count,
                    min_bars_required,
                    extra={
                        "event": "STRATEGY_NO_VOTE",
                        "strategy": "SMC",
                        "symbol": symbol,
                        "reason": "smc_insufficient_history",
                        "history_count": history_count,
                        "min_bars_required": min_bars_required,
                        "resolved_history_count": resolved_history_count,
                        "history_domain_used": history_domain_used,
                        "option_history_count": option_history_count,
                        "underlying_history_count": underlying_history_count,
                        "spot_history_count": spot_history_count,
                        "indicator_history_count": indicator_history_count,
                        "history_source": indicators.get("history_source"),
                    },
                )
                return None

            if stale_data or current_price <= 0:
                self._no_vote("stale_or_invalid_data")
                LOGGER.debug(
                    "STRATEGY_NO_VOTE strategy=SMC reason=stale_or_invalid_data symbol=%s",
                    symbol,
                    extra={
                        "event": "STRATEGY_NO_VOTE",
                        "strategy": "SMC",
                        "symbol": symbol,
                        "reason": "stale_or_invalid_data",
                        "stale_data": stale_data,
                        "current_price": current_price,
                    },
                )
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
                    LOGGER.info(
                        "STRATEGY_NO_VOTE strategy=SMC symbol=%s reason=premium_not_reversing_up "
                        "bullish_sweep=%s bearish_sweep=%s premium_reclaim=%s bullish_reversal=%s "
                        "choch_confirmed=%s bos_confirmed=%s direction=%s underlying_direction=%s "
                        "context_age_seconds=%.2f",
                        symbol,
                        bullish_sweep,
                        bearish_sweep,
                        bool(indicators.get("premium_reclaim")),
                        bool(indicators.get("bullish_reversal")),
                        bool(indicators.get("choch_confirmed")),
                        bool(indicators.get("bos_confirmed")),
                        direction,
                        underlying_direction,
                        context_age_seconds,
                        extra={
                            "event": "STRATEGY_NO_VOTE",
                            "strategy": "SMC",
                            "symbol": symbol,
                            "reason": "premium_not_reversing_up",
                            "bullish_sweep": bullish_sweep,
                            "bearish_sweep": bearish_sweep,
                            "premium_reclaim": bool(indicators.get("premium_reclaim")),
                            "bullish_reversal": bool(indicators.get("bullish_reversal")),
                            "choch_confirmed": bool(indicators.get("choch_confirmed")),
                            "bos_confirmed": bool(indicators.get("bos_confirmed")),
                            "direction": direction,
                            "underlying_direction": underlying_direction,
                            "context_age_seconds": context_age_seconds,
                        },
                    )
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
            direction_aligned = effective_direction in {'CE', 'PE'} and effective_direction == side
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
            min_score = float(os.getenv('SMC_MIN_SCORE_LIVE', '6.5') if is_live else os.getenv('SMC_MIN_SCORE_SHADOW', '4.5'))
            require_structure_live = str(os.getenv('SMC_REQUIRE_STRUCTURE_CONFIRMATION_LIVE', 'true')).lower() in {'1', 'true', 'yes', 'on'}
            allow_momentum_without_structure = str(os.getenv("SMC_ALLOW_MOMENTUM_WITHOUT_STRUCTURE_LIVE", "true")).lower() in {"1", "true", "yes", "on"}
            momentum_confirmed = (
                allow_momentum_without_structure
                and displacement_score >= float(os.getenv("SMC_MOMENTUM_DISPLACEMENT_MIN", "0.80") or "0.80")
                and direction_aligned
                and (premium_reclaim or retest_confirmed or str(os.getenv("SMC_MOMENTUM_REQUIRE_PREMIUM_RECLAIM_OR_RETEST", "true")).lower() not in {"1", "true", "yes", "on"})
            )
            if is_live and require_structure_live and not (structure_confirmed or momentum_confirmed):
                self._no_vote('smc_structure_required_live')
                premium_reclaim_source = indicators.get("premium_reclaim_source")
                premium_current = indicators.get("premium_current")
                premium_prev_close = indicators.get("premium_prev_close")
                premium_vwap = indicators.get("premium_vwap")
                retest_reason = indicators.get("retest_reason")
                LOGGER.info(
                    "STRATEGY_NO_VOTE strategy=SMC symbol=%s reason=smc_structure_required_live "
                    "choch_confirmed=%s bos_confirmed=%s retest_confirmed=%s displacement_score=%.3f "
                    "premium_reclaim=%s direction_aligned=%s direction=%s underlying_direction=%s "
                    "context_age_seconds=%.2f",
                    symbol,
                    choch_confirmed,
                    bos_confirmed,
                    retest_confirmed,
                    displacement_score,
                    premium_reclaim,
                    direction_aligned,
                    direction,
                    underlying_direction,
                    context_age_seconds,
                    extra={
                        "event": "STRATEGY_NO_VOTE",
                        "strategy": "SMC",
                        "symbol": symbol,
                        "reason": "smc_structure_required_live",
                        "choch_confirmed": choch_confirmed,
                        "bos_confirmed": bos_confirmed,
                        "retest_confirmed": retest_confirmed,
                        "displacement_score": displacement_score,
                        "premium_reclaim": premium_reclaim,
                        "direction_aligned": direction_aligned,
                        "direction": direction,
                        "underlying_direction": underlying_direction,
                        "context_age_seconds": context_age_seconds,
                        "momentum_confirmed": momentum_confirmed,
                        "effective_direction": effective_direction,
                        "min_score": min_score,
                        "strategy_score": strategy_score,
                        "SMC_MOMENTUM_DISPLACEMENT_MIN": float(os.getenv("SMC_MOMENTUM_DISPLACEMENT_MIN", "0.80") or "0.80"),
                        "SMC_MOMENTUM_REQUIRE_PREMIUM_RECLAIM_OR_RETEST": str(os.getenv("SMC_MOMENTUM_REQUIRE_PREMIUM_RECLAIM_OR_RETEST", "true")).lower() in {"1", "true", "yes", "on"},
                    },
                )
                LOGGER.info(
                    "SMC_RECLAIM_DIAGNOSTICS symbol=%s premium_reclaim=%s premium_reclaim_source=%s premium_current=%s premium_prev_close=%s premium_vwap=%s retest_confirmed=%s retest_reason=%s choch_confirmed=%s bos_confirmed=%s displacement_score=%.3f direction_aligned=%s direction=%s underlying_direction=%s reason=smc_structure_required_live",
                    symbol,
                    premium_reclaim,
                    premium_reclaim_source if premium_reclaim_source is not None else "unavailable",
                    premium_current if premium_current is not None else "unavailable",
                    premium_prev_close if premium_prev_close is not None else "unavailable",
                    premium_vwap if premium_vwap is not None else "unavailable",
                    retest_confirmed,
                    retest_reason if retest_reason is not None else "unavailable",
                    choch_confirmed,
                    bos_confirmed,
                    displacement_score,
                    direction_aligned,
                    direction or "unavailable",
                    underlying_direction or "unavailable",
                    extra={
                        "event": "SMC_RECLAIM_DIAGNOSTICS",
                        "symbol": symbol,
                        "premium_reclaim": premium_reclaim,
                        "premium_reclaim_source": premium_reclaim_source,
                        "premium_current": premium_current,
                        "premium_prev_close": premium_prev_close,
                        "premium_vwap": premium_vwap,
                        "retest_confirmed": retest_confirmed,
                        "retest_reason": retest_reason,
                        "choch_confirmed": choch_confirmed,
                        "bos_confirmed": bos_confirmed,
                        "displacement_score": displacement_score,
                        "direction_aligned": direction_aligned,
                        "direction": direction,
                        "underlying_direction": underlying_direction,
                        "reason": "smc_structure_required_live",
                        "effective_direction": effective_direction,
                        "momentum_confirmed": momentum_confirmed,
                        "min_score": min_score,
                        "strategy_score": strategy_score,
                        "SMC_MOMENTUM_DISPLACEMENT_MIN": float(os.getenv("SMC_MOMENTUM_DISPLACEMENT_MIN", "0.80") or "0.80"),
                        "SMC_MOMENTUM_REQUIRE_PREMIUM_RECLAIM_OR_RETEST": str(os.getenv("SMC_MOMENTUM_REQUIRE_PREMIUM_RECLAIM_OR_RETEST", "true")).lower() in {"1", "true", "yes", "on"},
                    },
                )
                return None
            if not (bullish_sweep or bearish_sweep or premium_reclaim) or not (displacement_score >= 0.6 or structure_confirmed) or not (direction_aligned or premium_reclaim):
                self._no_vote('smc_quality_gate_failed')
                LOGGER.info(
                    "STRATEGY_NO_VOTE strategy=SMC symbol=%s reason=smc_quality_gate_failed "
                    "bullish_sweep=%s bearish_sweep=%s premium_reclaim=%s "
                    "displacement_score=%.3f structure_confirmed=%s direction_aligned=%s",
                    symbol,
                    bullish_sweep,
                    bearish_sweep,
                    premium_reclaim,
                    displacement_score,
                    structure_confirmed,
                    direction_aligned,
                    extra={
                        "event": "STRATEGY_NO_VOTE",
                        "strategy": "SMC",
                        "symbol": symbol,
                        "reason": "smc_quality_gate_failed",
                        "bullish_sweep": bullish_sweep,
                        "bearish_sweep": bearish_sweep,
                        "premium_reclaim": premium_reclaim,
                        "displacement_score": displacement_score,
                        "structure_confirmed": structure_confirmed,
                        "direction_aligned": direction_aligned,
                    },
                )
                return None
            if strategy_score < min_score:
                self._no_vote("smc_low_score")
                LOGGER.info(
                    "STRATEGY_NO_VOTE strategy=SMC symbol=%s reason=smc_low_score "
                    "score=%.2f min_score=%.2f reasons=%s direction=%s underlying_direction=%s",
                    symbol,
                    strategy_score,
                    min_score,
                    reasons,
                    direction,
                    underlying_direction,
                    extra={
                        "event": "STRATEGY_NO_VOTE",
                        "strategy": "SMC",
                        "symbol": symbol,
                        "reason": "smc_low_score",
                        "score": strategy_score,
                        "min_score": min_score,
                        "score_reasons": reasons,
                        "direction": direction,
                        "underlying_direction": underlying_direction,
                        "context_age_seconds": context_age_seconds,
                    },
                )
                return None

            metadata = {
                'strategy': 'SMC',
                'strategy_name': 'SMC',
                'role': 'trigger',
                'signal_family': 'directional_trigger',
                'trade_side': side,
                'side': side,
                'direction_bias': side,
                "underlying_direction_bias": underlying_direction if underlying_direction in {"CE", "PE"} else None,
                "context_age_seconds": context_age_seconds,
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
                'momentum_confirmed': momentum_confirmed,
                'structure_or_momentum_confirmed': bool(structure_confirmed or momentum_confirmed),
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
            LOGGER.error(
                "Failure in SMCStrategy._evaluate_signal symbol=%s last_no_vote_reason=%s error=%s",
                symbol,
                getattr(self, "last_no_vote_reason", None),
                e,
                exc_info=e,
            )
            return None


__all__ = ['SMCStrategy']
