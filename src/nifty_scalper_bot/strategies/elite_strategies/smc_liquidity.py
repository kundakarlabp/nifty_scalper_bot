from __future__ import annotations

import os
from typing import Any

from nifty_scalper_bot.execution.readiness import HistoryReadinessPolicy
from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import SMCStrategyConfig
from nifty_scalper_bot.strategies.signal_quality import resolve_signal_domain
from nifty_scalper_bot.utils.logging import get_logger, log_throttled

LOGGER = get_logger(__name__)


def _safe_history_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_not_none(*values: int | None) -> int | None:
    return next((value for value in values if value is not None), None)


def safe_float_env(name: str, default: float) -> float:
    from nifty_scalper_bot.config.env_utils import parse_float_env
    return parse_float_env(os.getenv(name), default)


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
                log_throttled(
                    LOGGER,
                    f"smc_direction_context_not_ready:{symbol}",
                    "STRATEGY_NO_VOTE strategy=SMC symbol=%s reason=direction_context_not_ready direction_bias=%s underlying_direction_bias=%s",
                    symbol,
                    direction,
                    underlying_direction,
                    interval_sec=float(os.getenv("SMC_DIRECTION_CONTEXT_NO_VOTE_LOG_THROTTLE_SECONDS", "45") or "45"),
                    level=30,
                    extra={"event": "STRATEGY_NO_VOTE", "strategy": "SMC", "symbol": symbol, "reason": "direction_context_not_ready"},
                )
                return None
            min_bars_required = HistoryReadinessPolicy.from_env().smc_min_bars
            execution_mode = str(os.getenv('EXECUTION_MODE', 'SHADOW') or 'SHADOW').strip().upper()
            is_live = execution_mode == 'LIVE'
            history_domain_used = str(indicators.get("history_domain_used") or "unknown").lower()
            option_history_count = _safe_history_int(indicators.get("option_history_count"))
            underlying_history_count = _safe_history_int(indicators.get("underlying_history_count"))
            spot_history_count = _safe_history_int(indicators.get("spot_history_count"))
            indicator_history_count = _safe_history_int(indicators.get("indicator_history_count"))
            raw_history_count = _safe_history_int(indicators.get("history_count"))
            resolved_count = _safe_history_int(indicators.get("history_resolved_count"))
            if history_domain_used == "options":
                resolved_history_count = _first_not_none(
                    option_history_count,
                    resolved_count,
                    raw_history_count,
                    indicator_history_count,
                )
            elif history_domain_used == "spot":
                resolved_history_count = _first_not_none(
                    spot_history_count,
                    underlying_history_count,
                    resolved_count,
                    raw_history_count,
                    indicator_history_count,
                )
            elif history_domain_used == "underlying":
                resolved_history_count = _first_not_none(
                    underlying_history_count,
                    spot_history_count,
                    resolved_count,
                    raw_history_count,
                    indicator_history_count,
                )
            else:
                resolved_history_count = _first_not_none(
                    resolved_count,
                    raw_history_count,
                    indicator_history_count,
                    option_history_count,
                    underlying_history_count,
                    spot_history_count,
                )
            if is_live and resolved_history_count is None:
                self._no_vote("smc_history_count_missing")
                log_throttled(
                    LOGGER,
                    f"smc_history_no_vote:{symbol}:smc_history_count_missing:{history_domain_used}",
                    "STRATEGY_NO_VOTE strategy=SMC symbol=%s reason=smc_history_count_missing min_bars=%s history_domain_used=%s",
                    symbol,
                    min_bars_required,
                    history_domain_used,
                    interval_sec=float(os.getenv("SMC_HISTORY_NO_VOTE_LOG_THROTTLE_SECONDS", "60") or "60"),
                    level=30,
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
                        "history_symbol_key": indicators.get("history_symbol_key"),
                        "oldest_bar_ts": indicators.get("oldest_bar_ts"),
                        "latest_bar_ts": indicators.get("latest_bar_ts"),
                        "data_phase": indicators.get("data_phase"),
                        "eval_id": indicators.get("eval_id"),
                    },
                )
                return None
            history_count = int(resolved_history_count or 0)
            if is_live and history_count < min_bars_required:
                self._no_vote("smc_insufficient_history")
                log_throttled(
                    LOGGER,
                    f"smc_history_no_vote:{symbol}:smc_insufficient_history:{history_domain_used}",
                    "STRATEGY_NO_VOTE strategy=SMC symbol=%s reason=smc_insufficient_history history_count=%s min_bars=%s",
                    symbol,
                    history_count,
                    min_bars_required,
                    interval_sec=float(os.getenv("SMC_HISTORY_NO_VOTE_LOG_THROTTLE_SECONDS", "60") or "60"),
                    level=30,
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
                        "history_symbol_key": indicators.get("history_symbol_key"),
                        "oldest_bar_ts": indicators.get("oldest_bar_ts"),
                        "latest_bar_ts": indicators.get("latest_bar_ts"),
                        "data_phase": indicators.get("data_phase"),
                        "eval_id": indicators.get("eval_id"),
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
            required_features = ("premium_reclaim", "bullish_reversal", "choch_confirmed", "bos_confirmed", "retest_confirmed")
            present_features = [name for name in required_features if indicators.get(name) is not None]
            feature_completeness = float(len(present_features)) / float(len(required_features))
            feature_threshold = safe_float_env("SMC_FEATURE_COMPLETENESS_THRESHOLD", 0.6)
            missing_features = [name for name in required_features if name not in present_features]
            history_count = indicators.get("history_count")
            missing_feature_sources = {
                "bos_confirmed": {"missing": indicators.get("bos_confirmed") is None, "source": "structure_detector", "required_inputs": ["swing_high", "swing_low", "close", "history_count"], "history_count": history_count, "swing_high": indicators.get("swing_high"), "swing_low": indicators.get("swing_low")},
                "choch_confirmed": {"missing": indicators.get("choch_confirmed") is None, "source": "structure_detector", "required_inputs": ["swing_high", "swing_low", "close", "history_count"], "history_count": history_count, "swing_high": indicators.get("swing_high"), "swing_low": indicators.get("swing_low")},
                "retest_confirmed": {"missing": indicators.get("retest_confirmed") is None and indicators.get("mitigation_confirmed") is None, "source": "retest_detector", "required_inputs": ["retest_confirmed|mitigation_confirmed", "history_count"], "history_count": history_count},
                "premium_reclaim": {"missing": indicators.get("premium_reclaim") is None, "source": "premium_flow", "required_inputs": ["premium_current", "premium_vwap", "premium_prev_close"], "premium_current": indicators.get("premium_current"), "premium_vwap": indicators.get("premium_vwap"), "premium_prev_close": indicators.get("premium_prev_close")},
                "bullish_reversal": {"missing": indicators.get("bullish_reversal") is None, "source": "reversal_detector", "required_inputs": ["open", "close", "high", "low", "atr"], "context_age_seconds": indicators.get("context_age_seconds")},
            }
            feature_ready = feature_completeness >= feature_threshold
            log_throttled(
                LOGGER,
                f"smc_feature_readiness:{symbol}",
                "SMC_FEATURE_READINESS symbol=%s feature_completeness=%.3f missing_features=%s missing_feature_sources=%s live_enabled=%s vote_allowed=%s hard_veto=%s",
                symbol, feature_completeness, ",".join(missing_features), missing_feature_sources, is_live, feature_ready, False,
                interval_sec=30.0,
                extra={"event": "SMC_FEATURE_READINESS", "symbol": symbol, "feature_completeness": feature_completeness, "missing_features": missing_features, "missing_feature_sources": missing_feature_sources, "live_enabled": is_live, "vote_allowed": feature_ready, "hard_veto": False})
            if option_premium_domain:
                premium_reversal = bool(bullish_sweep or indicators.get('premium_reclaim') or indicators.get('bullish_reversal'))
                structure_flip = bool(indicators.get('choch_confirmed') or indicators.get('bos_confirmed'))
                partial_features_used = False
                if not feature_ready:
                    fallback_enabled = str(os.getenv("SMC_ALLOW_DERIVED_FEATURE_FALLBACK_LIVE", "false")).lower() in {"1", "true", "yes", "on"}
                    momentum_min = float(os.getenv("SMC_MOMENTUM_DISPLACEMENT_MIN", "0.80") or "0.80")
                    fallback_signal_present = bool(
                        indicators.get("premium_reclaim")
                        or indicators.get("bos_confirmed")
                        or indicators.get("choch_confirmed")
                        or indicators.get("retest_confirmed")
                        or (bullish_sweep and displacement_score >= 0.9)
                    )
                    fallback_allowed = bool(
                        is_live
                        and fallback_enabled
                        and effective_direction in {"CE", "PE"}
                        and displacement_score >= momentum_min
                        and fallback_signal_present
                    )
                    if fallback_allowed:
                        partial_features_used = True
                        LOGGER.info(
                            "SMC_DERIVED_FEATURE_FALLBACK symbol=%s feature_completeness=%.3f displacement_score=%.3f effective_direction=%s",
                            symbol,
                            feature_completeness,
                            displacement_score,
                            effective_direction,
                            extra={"event": "SMC_DERIVED_FEATURE_FALLBACK", "symbol": symbol, "feature_completeness": feature_completeness, "displacement_score": displacement_score, "effective_direction": effective_direction},
                        )
                    else:
                        self._no_vote('strategy_feature_unavailable')
                        LOGGER.info(
                            "SMC_FEATURE_UNAVAILABLE symbol=%s missing_features=%s missing_feature_sources=%s",
                            symbol,
                            ",".join(missing_features),
                            missing_feature_sources,
                            extra={"event": "SMC_FEATURE_UNAVAILABLE", "symbol": symbol, "missing_features": missing_features, "missing_feature_sources": missing_feature_sources},
                        )
                        return None
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
                partial_features_used = False
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
            if partial_features_used:
                strategy_score = max(0.0, strategy_score - 0.5)
                reasons.append('derived_feature_fallback')
            min_score = float(os.getenv('SMC_MIN_SCORE_LIVE', '6.5') if is_live else os.getenv('SMC_MIN_SCORE_SHADOW', '4.5'))
            require_structure_live = str(os.getenv('SMC_REQUIRE_STRUCTURE_CONFIRMATION_LIVE', 'true')).lower() in {'1', 'true', 'yes', 'on'}
            allow_momentum_without_structure = str(os.getenv("SMC_ALLOW_MOMENTUM_WITHOUT_STRUCTURE_LIVE", "true")).lower() in {"1", "true", "yes", "on"}
            momentum_confirmed = (
                allow_momentum_without_structure
                and displacement_score >= float(os.getenv("SMC_MOMENTUM_DISPLACEMENT_MIN", "0.80") or "0.80")
                and direction_aligned
                and (premium_reclaim or retest_confirmed or str(os.getenv("SMC_MOMENTUM_REQUIRE_PREMIUM_RECLAIM_OR_RETEST", "true")).lower() not in {"1", "true", "yes", "on"})
            )
            if is_live and require_structure_live and feature_ready and not (structure_confirmed or momentum_confirmed):
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
                LOGGER.info(
                    "SMC_SCORE_BREAKDOWN symbol=%s raw_score=%.2f min_score=%.2f bos_confirmed=%s choch_confirmed=%s retest_confirmed=%s premium_reclaim=%s bullish_reversal=%s bearish_reversal=%s history_count=%s",
                    symbol,
                    strategy_score,
                    min_score,
                    bos_confirmed,
                    choch_confirmed,
                    retest_confirmed,
                    premium_reclaim,
                    bool(indicators.get("bullish_reversal")),
                    bool(indicators.get("bearish_reversal")),
                    indicators.get("history_count") or indicators.get("indicator_history_count"),
                    extra={
                        "event": "SMC_SCORE_BREAKDOWN",
                        "symbol": symbol,
                        "raw_score": strategy_score,
                        "min_score": min_score,
                        "bos_confirmed": bos_confirmed,
                        "choch_confirmed": choch_confirmed,
                        "retest_confirmed": retest_confirmed,
                        "premium_reclaim": premium_reclaim,
                        "bullish_reversal": bool(indicators.get("bullish_reversal")),
                        "bearish_reversal": bool(indicators.get("bearish_reversal")),
                        "history_count": indicators.get("history_count") or indicators.get("indicator_history_count"),
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
                'partial_features_used': partial_features_used,
                'feature_completeness': feature_completeness,
                'smc_fallback_reason': 'derived_feature_fallback' if partial_features_used else '',
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
