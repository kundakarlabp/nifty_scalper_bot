"""Canonical post-strategy quality normalization for elite strategy votes.

The strategy engines remain responsible for detecting their native market
structures.  This module owns the cross-strategy quality contract immediately
before arbitration so a numeric score has comparable meaning across SMC,
VWAPPro, ORBPro and OrderFlow.

Design rules:
- never create a trigger that the native strategy did not create;
- OrderFlow remains context-only and can only change confirmation strength;
- option-premium evidence never authorizes underlying direction;
- live quality gates may reject weak/overextended triggers, never bypass risk;
- exposed strategy configuration is consumed here rather than existing as a
  misleading no-op tuning surface.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
from typing import Any, Mapping

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)
_PATCHED = False
_CONTRACT_VERSION = "strategy-quality-v2"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _env_float(name: str, default: float) -> float:
    return _safe_float(os.getenv(name), default)


def _config_float(config: Any, name: str, default: float) -> float:
    if config is None:
        return float(default)
    return _safe_float(getattr(config, name, default), default)


def _confidence_fraction(value: Any) -> float:
    number = _safe_float(value, 0.0)
    if number > 1.0:
        number /= 100.0
    return _clamp(number, 0.0, 1.0)


def _metadata_strategy(signal: Any, fallback: str = "") -> str:
    metadata = dict(getattr(signal, "metadata", {}) or {})
    return str(
        metadata.get("strategy_name")
        or metadata.get("strategy")
        or fallback
        or ""
    ).strip()


def _replace_signal(signal: Any, *, metadata: dict[str, Any], confidence: float) -> Any:
    """Return the same signal type with immutable fields safely replaced."""
    try:
        return dataclasses.replace(
            signal,
            metadata=metadata,
            confidence=_clamp(confidence, 0.0, 1.0),
        )
    except (TypeError, ValueError):
        with_metadata = getattr(signal, "with_metadata", None)
        if callable(with_metadata):
            updated = with_metadata(**metadata)
            try:
                object.__setattr__(updated, "confidence", _clamp(confidence, 0.0, 1.0))
            except Exception:  # pragma: no cover - compatibility fallback
                pass
            return updated
        return signal


def _raw_score(metadata: Mapping[str, Any]) -> float:
    for key in ("raw_setup_score", "setup_score", "strategy_score", "setup_quality", "context_score"):
        if metadata.get(key) is not None:
            return _clamp(_safe_float(metadata.get(key), 0.0), 0.0, 10.0)
    return 0.0


def _stamp_common(
    metadata: dict[str, Any],
    *,
    raw_score: float,
    quality_score: float,
    minimum: float | None,
    passed: bool,
) -> None:
    metadata["quality_contract_version"] = _CONTRACT_VERSION
    metadata["raw_strategy_score_pre_quality"] = round(raw_score, 4)
    metadata["quality_score"] = round(quality_score, 4)
    metadata["raw_setup_score"] = round(quality_score, 4)
    metadata["setup_score"] = round(quality_score, 4)
    metadata["strategy_score"] = round(quality_score, 4)
    metadata["setup_quality"] = round(quality_score, 4)
    if minimum is not None:
        metadata["setup_min"] = round(float(minimum), 4)
    metadata["setup_pass"] = bool(passed)


def _normalise_smc(
    signal: Any,
    *,
    metadata: dict[str, Any],
    live_mode: bool,
) -> tuple[Any | None, str | None]:
    reasons = {str(value) for value in metadata.get("score_reasons", [])}
    raw_score = _raw_score(metadata)

    # Core sweep/reclaim/displacement proves that the pattern exists.  It no
    # longer grants five points by itself; at least one independent quality
    # domain is needed for a normal live admission.
    score = 4.0
    if "direction_alignment" in reasons:
        score += 1.5
    if "volume_confirmation" in reasons:
        score += 1.0
    if "structure_confirmation" in reasons:
        score += 1.0
    if {"retest_mitigation", "premium_reclaim_support"} & reasons:
        score += 0.5
    if "balanced_sweep_depth" in reasons:
        score += 0.5
    score = _clamp(score, 0.0, 10.0)

    minimum = _env_float(
        "SMC_MIN_SCORE_LIVE" if live_mode else "SMC_MIN_SCORE_SHADOW",
        6.5 if live_mode else 4.5,
    )
    passed = score >= minimum
    _stamp_common(
        metadata,
        raw_score=raw_score,
        quality_score=score,
        minimum=minimum,
        passed=passed,
    )
    metadata["smc_quality_score"] = round(score, 4)
    metadata["smc_quality_independent_confirmation"] = bool(
        {
            "volume_confirmation",
            "structure_confirmation",
            "retest_mitigation",
            "premium_reclaim_support",
            "balanced_sweep_depth",
        }
        & reasons
    )

    if live_mode and not passed:
        metadata["trigger_block_reason"] = "smc_quality_below_minimum"
        return None, "smc_quality_below_minimum"
    confidence = _clamp(score / 10.0, 0.10, 0.88)
    return _replace_signal(signal, metadata=metadata, confidence=confidence), None


def _normalise_vwap(
    signal: Any,
    *,
    config: Any,
    metadata: dict[str, Any],
    current_price: float,
    live_mode: bool,
) -> tuple[Any | None, str | None]:
    raw_score = _raw_score(metadata)
    reasons = {str(value) for value in metadata.get("score_reasons", [])}
    vwap = _safe_float(metadata.get("vwap"), 0.0)
    atr = max(_safe_float(metadata.get("atr"), 0.0), max(current_price, 0.0) * 0.01, 1e-9)
    price = _safe_float(metadata.get("price"), current_price)
    distance_points = abs(price - vwap) if vwap > 0 else float("inf")
    distance_atr = distance_points / atr if math.isfinite(distance_points) else float("inf")
    distance_fraction = distance_points / vwap if vwap > 0 else float("inf")

    # VWAP_PROXIMITY_PCT is expressed as percentage points (0.15 == 0.15%).
    # It was previously loaded into config but never consumed by VWAPPro.
    configured_proximity_pct = max(0.0, _config_float(config, "proximity_pct", 0.15))
    configured_proximity_fraction = configured_proximity_pct / 100.0
    near_configured_vwap = bool(
        math.isfinite(distance_fraction)
        and distance_fraction <= configured_proximity_fraction
    )

    max_distance_atr = max(0.5, _env_float("VWAP_QUALITY_MAX_DISTANCE_ATR", 2.0))
    metadata["vwap_distance_atr"] = round(distance_atr, 4) if math.isfinite(distance_atr) else None
    metadata["vwap_configured_proximity_pct"] = configured_proximity_pct
    metadata["vwap_configured_proximity_pass"] = near_configured_vwap
    metadata["vwap_quality_max_distance_atr"] = max_distance_atr

    if live_mode and (not math.isfinite(distance_atr) or distance_atr > max_distance_atr):
        _stamp_common(
            metadata,
            raw_score=raw_score,
            quality_score=0.0,
            minimum=_safe_float(metadata.get("setup_min"), 5.8),
            passed=False,
        )
        metadata["trigger_block_reason"] = "vwap_quality_overextended_atr"
        return None, "vwap_quality_overextended_atr"

    # Independent domains: confirmed event, VWAP location, underlying trend,
    # futures slope, participation and normalized proximity.  Continuation,
    # reclaim and penetration are alternative event types and are deliberately
    # not stacked as independent evidence.
    score = 0.0
    if bool(metadata.get("vwap_event_confirmed")):
        score += 2.0
    if bool(metadata.get("premium_above_vwap")):
        score += 1.5
    if bool(metadata.get("trend_alignment")):
        score += 2.0
    if bool(metadata.get("futures_alignment")):
        score += 1.0
    if "volume_confirmation" in reasons:
        score += 1.0
    if math.isfinite(distance_atr):
        if distance_atr <= 1.0:
            score += 1.0
        elif distance_atr <= 1.5:
            score += 0.5
    if near_configured_vwap:
        score += 0.5
    score = _clamp(score, 0.0, 10.0)

    minimum = _safe_float(
        metadata.get("setup_min"),
        5.5 if bool(metadata.get("trend_alignment")) and live_mode else 5.8 if live_mode else 5.0,
    )
    passed = score >= minimum
    _stamp_common(
        metadata,
        raw_score=raw_score,
        quality_score=score,
        minimum=minimum,
        passed=passed,
    )
    metadata["final_vwap_score"] = round(score, 4)
    if live_mode and not passed:
        metadata["trigger_block_reason"] = "vwap_quality_below_minimum"
        return None, "vwap_quality_below_minimum"
    confidence = _clamp(score / 10.0, 0.10, 0.85)
    return _replace_signal(signal, metadata=metadata, confidence=confidence), None


def _normalise_orb(
    signal: Any,
    *,
    metadata: dict[str, Any],
    live_mode: bool,
) -> tuple[Any | None, str | None]:
    raw_score = _raw_score(metadata)
    reasons = {str(value) for value in metadata.get("score_reasons", [])}
    orb_high = _safe_float(metadata.get("opening_range_high"), 0.0)
    orb_low = _safe_float(metadata.get("opening_range_low"), 0.0)
    underlying_atr = max(_safe_float(metadata.get("underlying_atr"), 0.0), 1e-9)
    range_atr = (
        (orb_high - orb_low) / underlying_atr
        if orb_high > orb_low and underlying_atr > 0
        else float("inf")
    )
    min_balanced = max(0.05, _env_float("ORB_BALANCED_RANGE_MIN_ATR", 0.25))
    max_balanced = max(min_balanced, _env_float("ORB_BALANCED_RANGE_MAX_ATR", 1.75))
    balanced_range = bool(
        math.isfinite(range_atr) and min_balanced <= range_atr <= max_balanced
    )

    # Fresh breakout + a confirmed momentum/retest branch establishes the setup
    # but does not automatically earn a high score.  Direction, participation,
    # penetration, slope and range geometry are independent quality domains.
    score = 5.0
    if "underlying_direction_alignment" in reasons:
        score += 1.0
    if "underlying_volume_confirmation" in reasons:
        score += 1.0
    if "normalized_breakout_penetration" in reasons:
        score += 1.0
    if "futures_vwap_slope_alignment" in reasons:
        score += 1.0
    if balanced_range:
        score += 1.0
    score = _clamp(score, 0.0, 10.0)
    minimum = _env_float(
        "ORB_QUALITY_MIN_SCORE_LIVE" if live_mode else "ORB_QUALITY_MIN_SCORE_SHADOW",
        6.0 if live_mode else 5.0,
    )
    passed = score >= minimum
    _stamp_common(
        metadata,
        raw_score=raw_score,
        quality_score=score,
        minimum=minimum,
        passed=passed,
    )
    metadata["opening_range_width_atr"] = round(range_atr, 4) if math.isfinite(range_atr) else None
    metadata["opening_range_balanced"] = balanced_range
    metadata["opening_range_balanced_min_atr"] = min_balanced
    metadata["opening_range_balanced_max_atr"] = max_balanced
    if live_mode and not passed:
        metadata["trigger_block_reason"] = "orb_quality_below_minimum"
        return None, "orb_quality_below_minimum"
    confidence = _clamp(score / 10.0, 0.10, 0.90)
    return _replace_signal(signal, metadata=metadata, confidence=confidence), None


def _signed_depth_support(depth_imbalance: float, *, side: str, option_domain: bool, threshold: float) -> bool:
    if option_domain:
        return depth_imbalance >= threshold
    if side == "CE":
        return depth_imbalance >= threshold
    if side == "PE":
        return depth_imbalance <= -threshold
    return False


def _normalise_orderflow(
    signal: Any,
    *,
    config: Any,
    metadata: dict[str, Any],
    live_mode: bool,
) -> tuple[Any | None, str | None]:
    raw_score = _raw_score(metadata)
    symbol = str(getattr(signal, "symbol", "") or "").upper()
    side = str(metadata.get("contract_side") or metadata.get("trade_side") or metadata.get("side") or "").upper()
    option_domain = symbol.endswith(("CE", "PE"))
    depth_imbalance = _safe_float(metadata.get("depth_imbalance"), 0.0)
    spread_pct = _safe_float(metadata.get("spread_pct"), float("inf"))
    spread_limit = max(0.01, _safe_float(metadata.get("trigger_max_spread_pct"), 0.75 if live_mode else 12.0))

    # Wire the two historically exposed-but-unused OrderFlow config knobs.
    support_threshold = _clamp(
        _config_float(config, "large_order_threshold_pct", 15.0) / 100.0,
        0.05,
        0.50,
    )
    ratio_min = max(1.0, _config_float(config, "imbalance_ratio_min", 2.8))
    ratio_threshold = (ratio_min - 1.0) / (ratio_min + 1.0) if ratio_min > 1.0 else 0.0
    strong_threshold = _clamp(max(support_threshold, ratio_threshold), support_threshold, 0.85)

    depth_support = _signed_depth_support(
        depth_imbalance,
        side=side,
        option_domain=option_domain,
        threshold=support_threshold,
    )
    strong_depth_support = _signed_depth_support(
        depth_imbalance,
        side=side,
        option_domain=option_domain,
        threshold=strong_threshold,
    )
    tick_support = bool(metadata.get("tick_supports_direction"))
    aligned = bool(metadata.get("effective_context_alignment"))
    conflict = bool(metadata.get("effective_context_conflict"))
    context_eligible = bool(metadata.get("context_quality_eligible"))
    stale = bool(metadata.get("stale_data_used"))

    spread_score = 2.0 if spread_pct <= spread_limit else 1.0 if spread_pct <= (2.0 * spread_limit) else 0.0
    depth_score = (2.0 if depth_support else 0.0) + (1.0 if strong_depth_support else 0.0)
    tick_score = 2.0 if tick_support else 0.0
    direction_score = 1.0 if aligned else 0.0
    freshness_score = 1.0 if (not stale and context_eligible) else 0.0
    score = _clamp(
        spread_score + depth_score + tick_score + direction_score + freshness_score,
        0.0,
        10.0,
    )

    context_min = max(0.0, _env_float("ORDERFLOW_CONTEXT_MIN_SCORE", 4.0))
    evidence = max(0.0, score - context_min)
    bonus = 0.5 * evidence if context_eligible and aligned else 0.0
    veto = score if context_eligible and conflict else 0.0

    _stamp_common(
        metadata,
        raw_score=raw_score,
        quality_score=score,
        minimum=context_min,
        passed=score >= context_min,
    )
    metadata.update(
        {
            "context_score": round(score, 4),
            "context_evidence_score": round(evidence, 4),
            "context_bonus_score": round(bonus, 4),
            "context_veto_score": round(veto, 4),
            "depth_supports_side": depth_support,
            "strong_depth_supports_side": strong_depth_support,
            "depth_support_threshold": round(support_threshold, 4),
            "strong_depth_support_threshold": round(strong_threshold, 4),
            "orderflow_config_imbalance_ratio_min": ratio_min,
            "orderflow_config_large_order_threshold_pct": _config_float(
                config, "large_order_threshold_pct", 15.0
            ),
            "spread_score": spread_score,
            "depth_score": depth_score,
            "tick_score": tick_score,
            "direction_alignment_score": direction_score,
            "freshness_score": freshness_score,
            "orderflow_unconditional_score_removed": True,
        }
    )
    confidence = _clamp(score / 10.0, 0.10, 0.85)
    return _replace_signal(signal, metadata=metadata, confidence=confidence), None


def normalise_strategy_signal(
    signal: Any,
    *,
    strategy_name: str,
    config: Any,
    indicators: Mapping[str, Any],
    current_price: float,
    live_mode: bool,
) -> tuple[Any | None, str | None]:
    """Apply the canonical quality contract to one already-generated signal."""
    del indicators  # Reserved for future evidence families; no new direction source.
    metadata = dict(getattr(signal, "metadata", {}) or {})
    name = _metadata_strategy(signal, strategy_name).lower().replace("_", "")
    if name in {"smc", "smcliquidity"}:
        return _normalise_smc(signal, metadata=metadata, live_mode=live_mode)
    if name in {"vwappro", "vwap"}:
        return _normalise_vwap(
            signal,
            config=config,
            metadata=metadata,
            current_price=current_price,
            live_mode=live_mode,
        )
    if name in {"orbpro", "orb"}:
        return _normalise_orb(signal, metadata=metadata, live_mode=live_mode)
    if name in {"orderflow", "orderflowstrategy"}:
        return _normalise_orderflow(
            signal,
            config=config,
            metadata=metadata,
            live_mode=live_mode,
        )
    metadata.setdefault("quality_contract_version", _CONTRACT_VERSION)
    return _replace_signal(
        signal,
        metadata=metadata,
        confidence=_safe_float(getattr(signal, "confidence", 0.0), 0.0),
    ), None


def _install_elite_strategy_quality_contract() -> None:
    from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteStrategy

    current = EliteStrategy.generate_signal
    if getattr(current, "_strategy_quality_contract_patch", False):
        return

    def generate_signal(
        self: Any,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> Any:
        signal = current(self, symbol, indicators, current_price, position)
        if signal is None:
            return None
        live_mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper() == "LIVE"
        updated, rejection_reason = normalise_strategy_signal(
            signal,
            strategy_name=str(getattr(self, "name", "") or ""),
            config=getattr(self, "_config", None),
            indicators=indicators,
            current_price=current_price,
            live_mode=live_mode,
        )
        if updated is None:
            no_vote = getattr(self, "_no_vote", None)
            if callable(no_vote):
                no_vote(rejection_reason or "strategy_quality_rejected")
            LOGGER.info(
                "STRATEGY_QUALITY_REJECT strategy=%s symbol=%s reason=%s",
                getattr(self, "name", "unknown"),
                symbol,
                rejection_reason,
                extra={
                    "event": "STRATEGY_QUALITY_REJECT",
                    "strategy": str(getattr(self, "name", "unknown")),
                    "symbol": symbol,
                    "reason": rejection_reason,
                },
            )
            return None

        min_conf = _confidence_fraction(getattr(getattr(self, "_config", None), "min_confidence", 0.0))
        role = str((getattr(updated, "metadata", {}) or {}).get("role") or "trigger").lower()
        if role != "context" and float(getattr(updated, "confidence", 0.0) or 0.0) < min_conf:
            if callable(getattr(self, "_no_vote", None)):
                self._no_vote("below_strategy_min_confidence_after_quality")
            return None
        return updated

    generate_signal.__name__ = getattr(current, "__name__", "generate_signal")
    generate_signal.__doc__ = getattr(current, "__doc__", None)
    setattr(generate_signal, "_strategy_quality_contract_patch", True)
    setattr(generate_signal, "_original", current)
    EliteStrategy.generate_signal = generate_signal  # type: ignore[assignment]


def apply_patches() -> None:
    """Install the post-strategy quality contract exactly once."""
    global _PATCHED
    if _PATCHED:
        return
    _install_elite_strategy_quality_contract()
    _PATCHED = True


__all__ = ["apply_patches", "normalise_strategy_signal"]
