"""Signal quality scoring helpers for runner gating."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

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
        "premium_momentum": "premium_squeeze",
        "premium_momentum_squeeze": "premium_squeeze",
        "premium_squeeze": "premium_squeeze",
        "rsidivergence": "rsi_divergence",
        "rsi_divergence": "rsi_divergence",
        "vwappro": "vwap_pro",
        "vwap_pro": "vwap_pro",
    }
    return aliases.get(raw, raw)

def _mode_threshold(strategy_name: str | None = None) -> float:
    mode = (os.getenv('EXECUTION_MODE', 'SHADOW') or 'SHADOW').strip().upper()
    strategy_key = normalize_strategy_name(strategy_name)
    if mode == 'LIVE':
        if strategy_key == 'premium_squeeze':
            return float(os.getenv('SIGNAL_MIN_SCORE_LIVE_PREMIUM_SQUEEZE', '7.4') or 7.4)
        if strategy_key == 'rsi_divergence':
            return float(os.getenv('SIGNAL_MIN_SCORE_LIVE_RSI_DIVERGENCE', '7.6') or 7.6)
        if strategy_key == 'vwap_pro':
            return float(os.getenv('SIGNAL_MIN_SCORE_LIVE_VWAP_PRO', '7.5') or 7.5)
        return float(os.getenv('SIGNAL_MIN_SCORE_LIVE', '8.0') or 8.0)
    if mode in {'PAPER', 'DRY_RUN'}:
        return float(os.getenv('SIGNAL_MIN_SCORE_PAPER', '6.5') or 6.5)
    return float(os.getenv('SIGNAL_MIN_SCORE_SHADOW', '6.5') or 6.5)


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
    threshold = _mode_threshold(strategy_name=normalized_strategy_name)
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
