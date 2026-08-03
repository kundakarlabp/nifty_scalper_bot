"""Live direction-context proof and quote identity for OrderFlow."""

from __future__ import annotations

import os
from typing import Any, Mapping

from nifty_scalper_bot.execution.quote_readiness import resolve_tick_age_ms
from nifty_scalper_bot.strategies.runtime_context_contract import (
    live_direction_context_has_proof,
)

_TRUTHY = {"1", "true", "yes", "y", "on"}


def _env_live() -> bool:
    return (
        str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper() == "LIVE"
    )


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


def _stamp_quote_update_identity(
    metadata: dict[str, Any], indicators: Mapping[str, Any]
) -> None:
    """Preserve the real version, or a stable observed-quote fingerprint."""
    for source in (metadata, indicators):
        for key in (
            "quote_update_version",
            "update_version",
            "tick_version",
            "last_tick_ts_ms",
            "timestamp_ms",
            "last_tick_timestamp",
        ):
            value = source.get(key)
            if value not in (None, "", 0, 0.0):
                metadata["quote_update_version"] = value
                metadata.setdefault("quote_update_version_source", key)
                return

    bid = _safe_float(metadata.get("bid") or indicators.get("bid"))
    ask = _safe_float(metadata.get("ask") or indicators.get("ask"))
    imbalance = _safe_float(
        metadata.get("depth_imbalance") or indicators.get("depth_imbalance")
    )
    tick_direction = str(
        metadata.get("tick_direction") or indicators.get("tick_direction") or ""
    ).upper()
    if bid is None and ask is None and imbalance is None and not tick_direction:
        return
    metadata["quote_update_version"] = (
        f"micro:{bid if bid is not None else 'na'}:"
        f"{ask if ask is not None else 'na'}:"
        f"{imbalance if imbalance is not None else 'na'}:{tick_direction or 'na'}"
    )
    metadata["quote_update_version_source"] = "microstructure_fingerprint"


def _can_upgrade_direction_context_block(
    metadata: Mapping[str, Any], indicators: Mapping[str, Any]
) -> bool:
    if (
        str(metadata.get("trigger_block_reason") or "")
        != "direction_context_missing_live"
    ):
        return False
    if metadata.get("raw_direction_bias") in {"CE", "PE"} or metadata.get(
        "direction_bias"
    ) in {"CE", "PE"}:
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
    if not _boolish(metadata.get("quote_depth_valid")) or not _boolish(
        metadata.get("depth_available")
    ):
        return False
    if not _boolish(metadata.get("tradable_quote")):
        return False
    if not _boolish(metadata.get("selected_or_near_atm")):
        return False
    return True


def apply_orderflow_live_context_proof(
    signal: Any,
    indicators: Mapping[str, Any],
) -> Any:
    """Preserve quote identity and upgrade only fully proven live context."""

    if signal is None:
        return signal
    metadata = dict(getattr(signal, "metadata", {}) or {})
    _stamp_quote_update_identity(metadata, indicators or {})
    signal.metadata = metadata
    if not _env_live() or not _can_upgrade_direction_context_block(
        metadata, indicators or {}
    ):
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


__all__ = ["apply_orderflow_live_context_proof"]
