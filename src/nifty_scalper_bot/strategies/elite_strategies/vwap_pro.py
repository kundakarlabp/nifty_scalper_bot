from __future__ import annotations

import os
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import VWAPProStrategyConfig
from nifty_scalper_bot.strategies.signal_quality import resolve_signal_domain
from nifty_scalper_bot.strategies.runtime_context_contract import resolve_context_age_seconds
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _fmt_optional(value: float | None, digits: int) -> str:
    """Format an optional metric for logging without crashing on None.

    futures_vwap_slope / futures_volume_ratio are legitimately None when the
    futures context is unavailable (they are no longer conflated with 0.0).
    Feeding None to a %.4f/%.2f specifier raised
    "TypeError: must be real number, not NoneType" inside the STRATEGY_NO_VOTE
    log call, which the broad handler swallowed as a strategy failure -- so
    VWAPPro silently produced no vote on every evaluation.
    Args: value, digits. Returns: formatted str or "unavailable". Raises: none.
    """
    if value is None:
        return "unavailable"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "unavailable"


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
        self._futures_slope_neutral_eps = float(os.getenv("VWAP_FUTURES_SLOPE_NEUTRAL_EPS", "0.00005") or 0.00005)
        self._allow_early_trend_pullback = str(
            os.getenv("VWAP_PRO_ALLOW_EARLY_TREND_PULLBACK_LIVE", "false")
            if str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper() == "LIVE"
            else os.getenv("VWAP_PRO_ALLOW_EARLY_TREND_PULLBACK", "false")
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._early_trend_min_context_conf = float(os.getenv("VWAP_EARLY_TREND_MIN_CONTEXT_CONF", "0.90") or 0.90)
        self._early_trend_max_context_age = float(os.getenv("VWAP_EARLY_TREND_MAX_CONTEXT_AGE", "2.0") or 2.0)

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: indicators set. Raises: Exception."""
        return {
            "vwap",
            "exchange_vwap",
            "atr",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "avg_volume",
            "direction_bias",
            "underlying_direction_bias",
            "underlying_direction_confidence",
            "context_age_seconds",
            "futures_vwap_slope",
            "futures_volume_ratio",
            "spread_pct",
            "spot_context",
            "futures_context",
            "stale_data_used",
            "data_age_seconds",
        }

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
            spot_ctx = indicators.get("spot_context") if isinstance(indicators.get("spot_context"), dict) else {}
            fut_ctx = indicators.get("futures_context") if isinstance(indicators.get("futures_context"), dict) else {}
            direction = str(
                indicators.get("direction_bias")
                or indicators.get("underlying_direction_bias")
                or spot_ctx.get("direction_bias")
                or fut_ctx.get("direction_bias")
                or ""
            ).upper()
            underlying_direction = str(
                indicators.get("underlying_direction_bias")
                or indicators.get("direction_bias")
                or spot_ctx.get("underlying_direction_bias")
                or spot_ctx.get("direction_bias")
                or fut_ctx.get("underlying_direction_bias")
                or fut_ctx.get("direction_bias")
                or ""
            ).upper()
            def _optional_float(value: Any) -> float | None:
                try:
                    return None if value is None else float(value)
                except (TypeError, ValueError):
                    return None

            futures_vwap_slope = _optional_float(indicators.get('futures_vwap_slope'))
            futures_volume_ratio = _optional_float(indicators.get('futures_volume_ratio'))
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
            conflict_penalty_applied = 0.0
            contract_side, option_premium_domain, _ = resolve_signal_domain(symbol, indicators)
            trend_alignment = False
            pullback_flag = False
            continuation_confirmed = False

            if contract_side not in {'CE', 'PE'}:
                fallback_side = str(indicators.get('direction_bias') or '').upper()
                if fallback_side in {'CE', 'PE'}:
                    contract_side = fallback_side
                elif symbol.upper().endswith('CE'):
                    contract_side = 'CE'
                    score -= 1.0
                    reasons.append('symbol_side_fallback')
                elif symbol.upper().endswith('PE'):
                    contract_side = 'PE'
                    score -= 1.0
                    reasons.append('symbol_side_fallback')
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

            vol_support = avg_vol > 0 and vol >= 0.6 * avg_vol
            fut_vol_support = futures_volume_ratio is not None and futures_volume_ratio >= 1.0
            if vol_support or fut_vol_support:
                score += 1.0
                reasons.append('volume_confirmation')

            bias = direction if direction in {'CE','PE'} else underlying_direction
            if bias in {'CE', 'PE'}:
                trend_alignment = bias == contract_side
                if trend_alignment:
                    score += 2.0
                    reasons.append('trend_alignment')
                else:
                    score -= 2.0
                    reasons.append('direction_conflict')

            if futures_vwap_slope is None:
                slope_support = False
                reasons.append('futures_slope_unavailable')
            elif abs(futures_vwap_slope) <= self._futures_slope_neutral_eps:
                slope_support = False
                reasons.append('futures_slope_neutral')
            else:
                slope_support = (contract_side == 'CE' and futures_vwap_slope > 0) or (contract_side == 'PE' and futures_vwap_slope < 0)
                if slope_support:
                    score += 1.0
                    reasons.append('futures_slope_alignment')
                else:
                    reasons.append('futures_slope_conflict')
            if not trend_alignment and bias in {'CE', 'PE'}:
                if 'direction_conflict' not in reasons:
                    conflict_penalty_applied = float(os.getenv('VWAP_PRO_CONFLICT_PENALTY', '2.0') or '2.0')
                    score -= conflict_penalty_applied
                    reasons.append('conflict_penalty')

            execution_mode = str(os.getenv('EXECUTION_MODE', 'SHADOW') or 'SHADOW').strip().upper()
            is_live = execution_mode == 'LIVE'
            require_alignment_live = str(os.getenv('VWAP_PRO_REQUIRE_UNDERLYING_ALIGNMENT_LIVE', 'true')).lower() in {'1', 'true', 'yes', 'on'}
            require_alignment_shadow = str(os.getenv('VWAP_PRO_REQUIRE_UNDERLYING_ALIGNMENT_SHADOW', 'false')).lower() in {'1', 'true', 'yes', 'on'}
            min_score_default = '5.8' if is_live else '5.0'
            min_trend_score_default = '5.5' if is_live else '4.5'
            min_score = float(os.getenv('VWAP_PRO_MIN_TREND_ALIGNED_SCORE', min_trend_score_default) if trend_alignment else os.getenv('VWAP_PRO_MIN_SCORE', min_score_default))
            context_age_seconds = resolve_context_age_seconds(indicators)
            try:
                underlying_direction_confidence = float(
                    indicators.get("underlying_direction_confidence")
                    or spot_ctx.get("underlying_direction_confidence")
                    or fut_ctx.get("underlying_direction_confidence")
                    or 0.0
                )
            except (TypeError, ValueError):
                underlying_direction_confidence = 0.0
            context_fresh = context_age_seconds <= float(os.getenv('VWAP_CONTEXT_MAX_AGE_SECONDS', '120') or '120')
            context_strong = underlying_direction_confidence >= float(os.getenv('VWAP_CONTEXT_MIN_CONFIDENCE', '0.75') or '0.75')
            hard_conflict = bool(((is_live and require_alignment_live) or ((not is_live) and require_alignment_shadow)) and not trend_alignment and context_fresh and context_strong)
            if hard_conflict:
                self._no_vote('underlying_direction_conflict')
                return None
            early_trend_pullback = bool(
                self._allow_early_trend_pullback
                and trend_alignment
                and context_fresh
                and context_age_seconds <= self._early_trend_max_context_age
                and underlying_direction_confidence >= self._early_trend_min_context_conf
                and not premium_above_vwap
                and not pullback_flag
                and distance_pct <= min(self._max_distance_pct * 0.15, 0.03)
                and spread_pct <= float(os.getenv("VWAP_EARLY_TREND_MAX_SPREAD_PCT", "8.0") or 8.0)
            )
            if early_trend_pullback:
                score += 1.2
                reasons.append("early_trend_pullback_context")
            if (
                trend_alignment
                and context_fresh
                and underlying_direction_confidence >= float(os.getenv("VWAP_CONTEXT_BOOST_MIN_CONFIDENCE", "0.90") or "0.90")
                and spread_pct <= float(os.getenv("LIVE_MAX_SPREAD_PCT", "0.75") or "0.75")
                and premium_above_vwap
            ):
                boost = float(os.getenv("VWAP_PRO_TREND_CONTEXT_BOOST", "0.5") or "0.5")
                score = min(10.0, score + boost)
                reasons.append("trend_context_boost")
            threshold_source = 'trend_aligned' if trend_alignment else 'base'

            if score < min_score:
                self._no_vote('weak_score')
                LOGGER.info(
                    "STRATEGY_NO_VOTE strategy=VWAPPro reason=weak_score "
                    "symbol=%s score=%.2f min_score=%.2f threshold_source=%s "
                    "direction=%s underlying_direction=%s contract_side=%s current_price=%.2f "
                    "vwap=%.2f distance_pct=%.4f allowed_distance=%.4f atr=%.2f "
                    "volume=%.2f avg_volume=%.2f trend_alignment=%s pullback_flag=%s "
                    "context_age_seconds=%.2f context_fresh=%s context_confidence=%.2f "
                    "futures_vwap_slope=%s futures_volume_ratio=%s slope_support=%s "
                    "premium_above_vwap=%s continuation_confirmed=%s reasons=%s",
                    symbol,
                    score,
                    min_score,
                    threshold_source,
                    direction,
                    underlying_direction,
                    contract_side,
                    current_price,
                    vwap,
                    distance_pct,
                    allowed_distance,
                    atr_safe,
                    vol,
                    avg_vol,
                    trend_alignment,
                    pullback_flag,
                    context_age_seconds,
                    context_fresh,
                    underlying_direction_confidence,
                    _fmt_optional(futures_vwap_slope, 4),
                    _fmt_optional(futures_volume_ratio, 2),
                    slope_support,
                    premium_above_vwap,
                    continuation_confirmed,
                    reasons,
                    extra={
                        "event": "STRATEGY_NO_VOTE",
                        "strategy": "VWAPPro",
                        "reason": "weak_score",
                        "symbol": symbol,
                        "score": score,
                        "min_score": min_score,
                        "threshold_source": threshold_source,
                        "direction": direction,
                        "underlying_direction": underlying_direction,
                        "contract_side": contract_side,
                        "score_reasons": reasons,
                        "context_age_seconds": context_age_seconds,
                        "context_fresh": context_fresh,
                        "underlying_direction_confidence": underlying_direction_confidence,
                        "futures_vwap_slope": futures_vwap_slope,
                        "futures_volume_ratio": futures_volume_ratio,
                        "slope_support": slope_support,
                        "premium_above_vwap": premium_above_vwap,
                        "continuation_confirmed": continuation_confirmed,
                        "trend_alignment": trend_alignment,
                        "pullback_flag": pullback_flag,
                        "early_trend_pullback": early_trend_pullback,
                    },
                )
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
                'direction_bias': direction if direction in {'CE', 'PE'} else None,
                "underlying_direction_bias": underlying_direction if underlying_direction in {"CE", "PE"} else None,
                "underlying_direction_confidence": underlying_direction_confidence,
                "context_age_seconds": context_age_seconds,
                "context_fresh": context_fresh,
                "context_direction_used": direction if direction in {"CE", "PE"} else underlying_direction if underlying_direction in {"CE", "PE"} else None,
                'preliminary_only': True,
                'requires_runner_final_score': True,
                'raw_setup_score': strategy_score,
                'setup_score': strategy_score,
                'setup_min': min_score,
                'setup_pass': True,
                'execution_required': True,
                'regime_required': True,
                'strategy_family': 'vwap_continuation_pullback',
                'context_required': False,
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
                'threshold_source': threshold_source,
                'underlying_context_used': bool(indicators.get('spot_context') or indicators.get('underlying_direction_bias')),
                'futures_context_used': bool(indicators.get('futures_context') or indicators.get('futures_vwap') is not None),
                'futures_vwap_slope': futures_vwap_slope,
                'vwap_domain': 'option_premium',
                'underlying_alignment': trend_alignment,
                'futures_alignment': slope_support,
                'conflict_penalty_applied': conflict_penalty_applied,
                'final_vwap_score': strategy_score,
                'futures_volume_ratio': futures_volume_ratio,
                'trigger_block_reason': '' if strategy_score >= min_score else 'weak_score',
                'continuation_confirmed': continuation_confirmed,
                'early_trend_pullback': early_trend_pullback,
                'setup_invalidation_premium': current_price - atr_safe,
                'premium_stop_distance': atr_safe,
                'premium_target_rr': 2.0,
                'underlying_invalidation_level': (vwap - atr_safe) if contract_side == 'CE' else (vwap + atr_safe),
                'no_vote_conflict_mode': 'hard' if hard_conflict else 'soft',
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
