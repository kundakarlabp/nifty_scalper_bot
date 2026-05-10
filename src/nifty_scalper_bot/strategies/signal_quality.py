"""Signal quality scoring helpers for runner gating."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

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


def trigger_threshold(strategy_name: str | None, mode: str | None = None) -> float:
    """Args: strategy_name/mode. Returns: trigger threshold. Raises: none."""
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
    value = os.getenv(primary_env)
    if value is None and legacy_env:
        value = os.getenv(legacy_env)
    return float(value or default_value)


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
        components={'threshold': threshold, 'strategy_name': strategy_name or '', 'normalized_strategy_name': normalized_strategy_name},
    )
