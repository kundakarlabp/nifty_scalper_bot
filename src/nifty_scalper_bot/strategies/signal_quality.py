"""Signal quality scoring helpers for runner gating."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from nifty_scalper_bot.config.entry_policy import resolve_entry_policy
from nifty_scalper_bot.config.regime_ontology import MarketRegime, normalize_regime

TRADABLE_REGIMES: frozenset[MarketRegime] = frozenset(
    {MarketRegime.TREND, MarketRegime.RANGE, MarketRegime.VOLATILE}
)

REQUIRED_SCORE_COMPONENTS: tuple[str, ...] = (
    "direction_score",
    "strategy_score",
    "option_score",
    "data_score",
    "rr_score",
)

CONTEXT_ONLY_STRATEGIES: frozenset[str] = frozenset(
    {
        "oi_max_pain",
        "order_flow",
        "bb_squeeze",
        "cpr_breakout",
        "rsi_divergence",
    }
)


def infer_option_side(symbol: str, metadata: dict[str, object] | None = None) -> str:
    """Args: symbol + metadata. Returns: CE/PE/UNKNOWN side. Raises: none."""
    upper = str(symbol or "").upper()
    if upper.endswith("CE"):
        return "CE"
    if upper.endswith("PE"):
        return "PE"
    payload = dict(metadata or {})
    return str(payload.get("direction_bias", "UNKNOWN")).upper()


def resolve_signal_domain(
    symbol: str,
    metadata: dict[str, object] | None = None,
) -> tuple[str, bool, bool]:
    """Return (contract_side, option_premium_domain, underlying_domain)."""
    payload = dict(metadata or {})
    contract_side = infer_option_side(symbol, payload)
    source_symbol = str(payload.get("source_symbol") or "").strip().upper()
    option_symbol = str(symbol or "").upper().endswith(("CE", "PE"))
    option_premium_domain = bool(option_symbol and not source_symbol)
    underlying_domain = bool(source_symbol)
    return contract_side, option_premium_domain, underlying_domain


def canonical_max_spread_pct() -> float:
    """Return the spread limit quality evidence is judged against."""
    return resolve_entry_policy().execution_max_spread_pct


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
        spread_pct = float(payload.get("spread_pct")) if spread_observed else 0.0
    except (TypeError, ValueError):
        spread_pct = 0.0
        spread_observed = False
    if not spread_observed and bid_ask_valid:
        midpoint = (bid + ask) / 2.0
        if midpoint > 0:
            spread_pct = ((ask - bid) / midpoint) * 100.0
            spread_observed = True

    spread_limit = canonical_max_spread_pct()
    spread_pass: bool | None = (
        spread_pct <= spread_limit if spread_observed else None
    )
    spread_status = (
        "unknown"
        if spread_pass is None
        else "pass"
        if spread_pass
        else "fail"
    )

    depth_valid = bool(payload.get("quote_depth_valid"))
    tradable_quote = bool(payload.get("tradable_quote"))
    quote_valid = bool(bid_ask_valid or tradable_quote)
    if depth_valid and quote_valid and spread_pass is True:
        liquidity_score = 2.0
    elif quote_valid and spread_pass is True:
        liquidity_score = 1.0
    elif tradable_quote and spread_pass is None:
        # Unknown spread remains non-blocking, but is not positive spread evidence.
        liquidity_score = 0.5
    else:
        liquidity_score = 0.0

    regime = normalize_regime(payload.get("regime") or payload.get("market_regime"))
    regime_score = 1.0 if regime in TRADABLE_REGIMES else 0.0

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
        "quality_spread_status": spread_status,
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
        "oimaxpain": "oi_max_pain",
        "oi_max_pain": "oi_max_pain",
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
        if token.endswith("%"):
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


def _env_score_threshold(
    primary_env: str,
    legacy_env: str | None,
    default_value: float,
) -> float:
    value = os.getenv(primary_env)
    if value is None and legacy_env:
        value = os.getenv(legacy_env)
    if value is None:
        return max(0.0, min(10.0, round(float(default_value), 3)))
    return _normalise_score_threshold(value, default_value)


def _global_score_floor() -> float | None:
    raw = os.getenv("GLOBAL_MIN_SIGNAL_CONFIDENCE")
    if raw is None:
        return None
    return _parse_score_threshold(raw)


def trigger_threshold(strategy_name: str | None, mode: str | None = None) -> float:
    """Args: strategy_name/mode. Returns: trigger threshold on 0..10 scale. Raises: none."""
    effective_mode = str(mode or os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper()
    strategy_key = normalize_strategy_name(strategy_name)
    is_live = effective_mode == "LIVE"
    defaults = {
        "vwap_pro": (
            7.5,
            6.5,
            "TRIGGER_VWAP_PRO_LIVE_MIN",
            "SIGNAL_MIN_SCORE_LIVE_VWAP_PRO",
        ),
        "premium_squeeze": (
            7.4,
            6.4,
            "TRIGGER_PREMIUM_SQUEEZE_LIVE_MIN",
            "SIGNAL_MIN_SCORE_LIVE_PREMIUM_SQUEEZE",
        ),
        "smc_lite": (7.0, 6.0, "TRIGGER_SMC_LIVE_MIN", None),
        "orb_pro": (7.4, 6.4, "TRIGGER_ORB_PRO_LIVE_MIN", None),
    }
    live_default, paper_default, primary_env, legacy_env = defaults.get(
        strategy_key,
        (8.0, 6.5, "SIGNAL_MIN_SCORE_LIVE", None),
    )
    default_value = live_default if is_live else paper_default
    threshold = _env_score_threshold(primary_env, legacy_env, default_value)
    global_floor = _global_score_floor()
    if global_floor is not None:
        threshold = max(threshold, global_floor)
    return max(0.0, min(10.0, round(threshold, 3)))


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
    # Direction + native setup quality are the alpha evidence. Option
    # microstructure, data readiness and R:R validate executability but must
    # not rescue a weak directional thesis into an entry.
    alpha_score = 0.55 * direction + 0.45 * strategy
    normalized_strategy_name = normalize_strategy_name(strategy_name)
    threshold = trigger_threshold(strategy_name=normalized_strategy_name)
    context_only = normalized_strategy_name in CONTEXT_ONLY_STRATEGIES
    alpha_floor_required = normalized_strategy_name == "vwap_pro"
    reasons: list[str] = []
    if context_only:
        reasons.append("context_only_strategy")
    if final < threshold:
        reasons.append("score_below_threshold")
    if alpha_floor_required and alpha_score < threshold:
        reasons.append("alpha_below_threshold")
    if direction < 6.0:
        reasons.append("direction_below_minimum")
    return SignalQualityScore(
        final_score=round(final, 3),
        direction_score=direction,
        strategy_score=strategy,
        option_score=option,
        data_score=data,
        rr_score=rr,
        allowed=(
            not context_only
            and final >= threshold
            and (not alpha_floor_required or alpha_score >= threshold)
            and direction >= 6.0
        ),
        reasons=reasons,
        components={
            "direction_score": direction,
            "strategy_score": strategy,
            "option_score": option,
            "data_score": data,
            "rr_score": rr,
            "final_score": round(final, 3),
            "alpha_score": round(alpha_score, 3),
            "alpha_floor_required": alpha_floor_required,
            "threshold": threshold,
            "strategy_name": strategy_name or "",
            "normalized_strategy_name": normalized_strategy_name,
        },
    )
