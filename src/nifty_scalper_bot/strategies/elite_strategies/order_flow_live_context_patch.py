"""Live direction-context proof adapter for OrderFlow.

This patch is deliberately narrow.  It does not invent an underlying CE/PE bias.
It only prevents a live OrderFlow candidate from being blocked solely because
``direction_bias`` is absent when fresh spot/futures context proof is present and
all other execution gates already passed.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from nifty_scalper_bot.execution.quote_readiness import resolve_tick_age_ms
from nifty_scalper_bot.strategies.runtime_context_contract import live_direction_context_has_proof

_PATCH_ATTR = "_live_direction_context_proof_patch_installed"
_ORIGINAL_ATTR = "_live_direction_context_proof_original_evaluate_signal"
_TRUTHY = {"1", "true", "yes", "y", "on"}


def _env_live() -> bool:
    return str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper() == "LIVE"


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in _TRUTHY


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _can_upgrade_direction_context_block(metadata: Mapping[str, Any], indicators: Mapping[str, Any]) -> bool:
    if str(metadata.get("trigger_block_reason") or "") != "direction_context_missing_live":
        return False
    if metadata.get("raw_direction_bias") in {"CE", "PE"} or metadata.get("direction_bias") in {"CE", "PE"}:
        return False
    context = dict(indicators or {})
    context.update(dict(metadata or {}))
    if not live_direction_context_has_proof(context):
        return False
    score = _safe_float(metadata.get("strategy_score") or metadata.get("setup_quality"))
    min_score = _safe_float(metadata.get("trigger_min_score"))
    spread = _safe_float(metadata.get("spread_pct"))
    max_spread = _safe_float(metadata.get("trigger_max_spread_pct"))
    tick_age_ms = resolve_tick_age_ms(metadata)
    max_tick_age_ms = _safe_float(os.getenv("LIVE_MAX_TICK_AGE_MS", "2500") or 2500)
    if score is None or min_score is None or score < min_score:
        return False
    if spread is None or max_spread is None or spread > max_spread:
        return False
    if tick_age_ms is None or max_tick_age_ms is None or tick_age_ms > max_tick_age_ms:
        return False
    if _boolish(metadata.get("tick_direction_missing")):
        return False
    if not _boolish(metadata.get("quote_readiness_allowed")):
        return False
    if not _boolish(metadata.get("quote_depth_valid")) or not _boolish(metadata.get("depth_available")):
        return False
    if not _boolish(metadata.get("tradable_quote")):
        return False
    if not _boolish(metadata.get("selected_or_near_atm")):
        return False
    return True


def apply_orderflow_live_context_patch(orderflow_cls: type[Any]) -> bool:
    """Patch OrderFlowStrategy once. Returns True when installed."""

    if bool(getattr(orderflow_cls, _PATCH_ATTR, False)):
        return False
    original = getattr(orderflow_cls, "_evaluate_signal", None)
    if not callable(original):
        return False

    def _evaluate_signal_with_live_context_proof(self: Any, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> Any:
        signal = original(self, symbol, indicators, current_price, position)
        if signal is None or not _env_live():
            return signal
        metadata = dict(getattr(signal, "metadata", {}) or {})
        if not _can_upgrade_direction_context_block(metadata, indicators or {}):
            return signal
        metadata.update(
            {
                "role": "trigger",
                "can_trigger": True,
                "trigger_conditions_met": True,
                "trigger_eligible": True,
                "trigger_block_reason": "",
                "trigger_disqualified_by": None,
                "direction_context_ok": True,
                "direction_context_live_proof": True,
                "approval_candidate": "orderflow_live_depth_trigger",
            }
        )
        signal.metadata = metadata
        return signal

    setattr(_evaluate_signal_with_live_context_proof, _ORIGINAL_ATTR, original)
    setattr(orderflow_cls, "_evaluate_signal", _evaluate_signal_with_live_context_proof)
    setattr(orderflow_cls, _PATCH_ATTR, True)
    return True


__all__ = ["apply_orderflow_live_context_patch"]
