"""Signal quality scoring helpers for runner gating."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

REQUIRED_SCORE_COMPONENTS: tuple[str, ...] = (
    'direction_score',
    'strategy_score',
    'option_score',
    'data_score',
    'rr_score',
)


def infer_option_side(symbol: str, metadata: dict[str, object] | None = None) -> str:
    """Args: symbol + metadata. Returns: CE/PE/UNKNOWN side. Raises: none."""
    upper = str(symbol or '').upper()
    if upper.endswith('CE'):
        return 'CE'
    if upper.endswith('PE'):
        return 'PE'
    payload = dict(metadata or {})
    return str(payload.get('direction_bias', 'UNKNOWN')).upper()


def resolve_signal_domain(symbol: str, metadata: dict[str, object] | None = None) -> tuple[str, bool, bool]:
    """Return (contract_side, option_premium_domain, underlying_domain)."""
    payload = dict(metadata or {})
    contract_side = infer_option_side(symbol, payload)
    source_symbol = str(payload.get("source_symbol") or "").strip().upper()
    option_symbol = str(symbol or "").upper().endswith(("CE", "PE"))
    option_premium_domain = bool(option_symbol and not source_symbol)
    underlying_domain = bool(source_symbol)
    return contract_side, option_premium_domain, underlying_domain


def canonical_max_spread_pct() -> float:
    """Return the single live entry spread limit used by strategy and execution."""
    for name in ("ORDER_MAX_SPREAD_PCT", "SPREAD_MAX_PCT"):
        raw = os.getenv(name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 10.0


def build_trade_quality_evidence(
    indicators: Mapping[str, object] | None,
    *,
    side: str,
) -> dict[str, object]:
    """Derive canonical quality fields from evidence already owned by the engine."""
    payload = dict(indicators or {})
    resolved_side = str(side or "").strip().upper()
    direction = str(
        payload.get("underlying_direction_bias")
        or payload.get("direction_bias")
        or ""
    ).strip().upper()

    bid = ask = 0.0
    try:
        bid = float(payload.get("bid") or 0.0)
        ask = float(payload.get("ask") or 0.0)
    except (TypeError, ValueError):
        pass
    bid_ask_valid = bid > 0.0 and ask >= bid

    spread_observed = payload.get("spread_pct") is not None
    try:
        spread_pct = float(payload.get("spread_pct")) if spread_observed else 999.0
    except (TypeError, ValueError):
        spread_pct = 999.0
    if not spread_observed and bid_ask_valid:
        midpoint = (bid + ask) / 2.0
        if midpoint > 0:
            spread_pct = ((ask - bid) / midpoint) * 100.0
            spread_observed = True

    spread_limit = canonical_max_spread_pct()
    spread_pass = bool(not spread_observed or spread_pct <= spread_limit)
    depth_valid = bool(payload.get("quote_depth_valid"))
    tradable_quote = bool(payload.get("tradable_quote"))
    quote_valid = bool(bid_ask_valid or tradable_quote)
    if depth_valid and quote_valid and spread_pass:
        liquidity_score = 2.0
    elif quote_valid and spread_pass:
        liquidity_score = 1.0
    else:
        liquidity_score = 0.0

    regime = str(payload.get("regime") or payload.get("market_regime") or "").upper()
    regime_score = 1.0 if regime and regime != "CHOPPY" else 0.0

    return {
        "direction_alignment_score": (
            2.0
            if resolved_side in {"CE", "PE"} and direction == resolved_side
            else 0.0
        ),
        "liquidity_score": liquidity_score,
        "regime_time_suitability_score": regime_score,
        "quality_spread_observed": spread_observed,
        "quality_spread_pass": spread_pass,
        "quality_spread_pct": spread_pct if spread_observed else None,
        "quality_spread_limit_pct": spread_limit,
    }


@dataclass(slots=True)
class SignalQualityScore:
    """Args: score components. Returns: normalized score object. Raises: none."""

    final_score: float
    direction_score: float
    strategy_score: float
    option_score: float
    data_score: float
    rr_score: float
    allowed: bool
    reasons: list[str]
    components: dict[str, float] = field(default_factory=dict)


def missing_score_components(metadata: dict[str, object] | None) -> list[str]:
    """Args: score metadata dict. Returns: missing score keys. Raises: none."""
    payload = dict(metadata or {})
    return [key for key in REQUIRED_SCORE_COMPONENTS if payload.get(key) is None]


def normalize_strategy_name(strategy_name: str | None) -> str:
    """Args: strategy_name. Returns: canonical strategy key. Raises: none."""
    raw = str(strategy_name or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "smc": "smc_lite",
        "smc_liquidity": "smc_lite",
        "smc_lite": "smc_lite",
        "smc_liquidity_sweep_lite": "smc_lite",
        "premium_momentum": "premium_squeeze",
        "premium_momentum_squeeze": "premium_squeeze",
        "premium_squeeze": "premium_squeeze",
        "rsidivergence": "rsi_divergence",
        "rsi_divergence": "rsi_divergence",
        "cprbreakout": "cpr_breakout",
        "cpr_breakout": "cpr_breakout",
        "bbsqueeze": "bb_squeeze",
        "bb_squeeze": "bb_squeeze",
        "orderflow": "order_flow",
        "order_flow": "order_flow",
        "orbpro": "orb_pro",
        "orb_pro": "orb_pro",
        "vwappro": "vwap_pro",
        "vwap_pro": "vwap_pro",
    }
    return aliases.get(raw, raw)


def _parse_score_threshold(raw: object) -> float | None:
    """Normalize configured threshold values to the internal 0..10 scale."""
    try:
        token = str(raw).strip()
        if not token:
            return None
        if token.endswith('%'):
            token = token[:-1].strip()
        value = float(token)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value <= 1.0:
        value *= 10.0
    elif value > 10.0 and value <= 100.0:
        value /= 10.0
    elif value > 100.0:
        return None
    return max(0.0, min(10.0, round(value, 3)))


def _normalise_score_threshold(raw: object, default: float) -> float:
    parsed = _parse_score_threshold(raw)
    if parsed is None:
        return max(0.0, min(10.0, round(float(default), 3)))
    return parsed


def _env_score_threshold(primary_env: str, legacy_env: str | None, default_value: float) -> float:
    value = os.getenv(primary_env)
    if value is None and legacy_env:
        value = os.getenv(legacy_env)
    if value is None:
        return max(0.0, min(10.0, round(float(default_value), 3)))
    return _normalise_score_threshold(value, default_value)


def _global_score_floor() -> float | None:
    raw = os.getenv('GLOBAL_MIN_SIGNAL_CONFIDENCE')
    if raw is None:
        return None
    return _parse_score_threshold(raw)


def trigger_threshold(strategy_name: str | None, mode: str | None = None) -> float:
    """Args: strategy_name/mode. Returns: trigger threshold on 0..10 scale. Raises: none."""
    effective_mode = str(mode or os.getenv('EXECUTION_MODE', 'SHADOW')).strip().upper()
    strategy_key = normalize_strategy_name(strategy_name)
    is_live = effective_mode == 'LIVE'
    defaults = {
        'vwap_pro': (7.5, 6.5, 'TRIGGER_VWAP_PRO_LIVE_MIN', 'SIGNAL_MIN_SCORE_LIVE_VWAP_PRO'),
        'premium_squeeze': (7.4, 6.4, 'TRIGGER_PREMIUM_SQUEEZE_LIVE_MIN', 'SIGNAL_MIN_SCORE_LIVE_PREMIUM_SQUEEZE'),
        'rsi_divergence': (7.6, 6.6, 'TRIGGER_RSI_DIVERGENCE_LIVE_MIN', 'SIGNAL_MIN_SCORE_LIVE_RSI_DIVERGENCE'),
        'smc_lite': (7.0, 6.0, 'TRIGGER_SMC_LIVE_MIN', None),
        'cpr_breakout': (7.2, 6.2, 'TRIGGER_CPR_BREAKOUT_LIVE_MIN', None),
        'bb_squeeze': (7.3, 6.3, 'TRIGGER_BB_SQUEEZE_LIVE_MIN', None),
        'order_flow': (7.1, 6.1, 'TRIGGER_ORDER_FLOW_LIVE_MIN', None),
        'orb_pro': (7.4, 6.4, 'TRIGGER_ORB_PRO_LIVE_MIN', None),
    }
    live_default, paper_default, primary_env, legacy_env = defaults.get(strategy_key, (8.0, 6.5, 'SIGNAL_MIN_SCORE_LIVE', None))
    default_value = live_default if is_live else paper_default
    threshold = _env_score_threshold(primary_env, legacy_env, default_value)
    global_floor = _global_score_floor()
    if global_floor is not None:
        threshold = max(threshold, global_floor)
    return max(0.0, min(10.0, round(threshold, 3)))


def context_boost_cap(strategy_name: str | None = None) -> float:
    """Args: strategy_name. Returns: absolute context boost cap. Raises: none."""
    _ = strategy_name
    return float(os.getenv('CONTEXT_BOOST_CAP', '1.25') or 1.25)


def rejection_cooldown(reason_family: str) -> int:
    """Args: reason_family. Returns: cooldown seconds. Raises: none."""
    family = str(reason_family or 'score').strip().lower()
    default_map = {'score': 60, 'candidate': 20, 'spread': 15, 'stale': 10, 'risk': 120, 'infra': 5}
    return int(float(os.getenv(f'SIGNAL_REJECT_COOLDOWN_{family.upper()}_SECONDS', str(default_map.get(family, 60))) or default_map.get(family, 60)))


def compute_context_boost(context_scores: list[float], *, strategy_name: str | None = None) -> float:
    """Args: context scores list. Returns: bounded context boost. Raises: none."""
    if not context_scores:
        return 0.0
    mean_centered = sum(float(s) - 5.0 for s in context_scores) / float(len(context_scores))
    boost = mean_centered / 2.5
    cap = context_boost_cap(strategy_name)
    return max(-cap, min(cap, boost))


def compute_final_execution_score(*, trigger_score: float, context_score_effective: float, candidate_score: float, data_score: float, rr_score: float) -> float:
    """Args: score parts. Returns: final execution score 0..10. Raises: none."""
    score = 0.45 * float(trigger_score) + 0.15 * float(context_score_effective) + 0.20 * float(candidate_score) + 0.10 * float(data_score) + 0.10 * float(rr_score)
    return max(0.0, min(10.0, round(score, 3)))


def score_signal_quality(
    *,
    direction_score: float,
    strategy_score: float,
    option_score: float,
    data_score: float,
    rr_score: float,
    strategy_name: str | None = None,
) -> SignalQualityScore:
    """Args: normalized components. Returns: weighted quality score. Raises: none."""
    direction = max(0.0, min(10.0, float(direction_score)))
    strategy = max(0.0, min(10.0, float(strategy_score)))
    option = max(0.0, min(10.0, float(option_score)))
    data = max(0.0, min(10.0, float(data_score)))
    rr = max(0.0, min(10.0, float(rr_score)))

    final = (
        0.30 * direction
        + 0.25 * strategy
        + 0.20 * option
        + 0.15 * data
        + 0.10 * rr
    )
    normalized_strategy_name = normalize_strategy_name(strategy_name)
    threshold = trigger_threshold(strategy_name=normalized_strategy_name)
    reasons: list[str] = []
    if final < threshold:
        reasons.append('score_below_threshold')
    if direction < 6.0:
        reasons.append('direction_below_minimum')
    return SignalQualityScore(
        final_score=round(final, 3),
        direction_score=direction,
        strategy_score=strategy,
        option_score=option,
        data_score=data,
        rr_score=rr,
        allowed=(final >= threshold and direction >= 6.0),
        reasons=reasons,
        components={
            'direction_score': direction,
            'strategy_score': strategy,
            'option_score': option,
            'data_score': data,
            'rr_score': rr,
            'final_score': round(final, 3),
            'threshold': threshold,
            'strategy_name': strategy_name or '',
            'normalized_strategy_name': normalized_strategy_name,
        },
    )