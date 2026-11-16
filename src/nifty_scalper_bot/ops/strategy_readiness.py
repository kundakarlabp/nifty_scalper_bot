"""Read-only, evidence-qualified per-strategy dashboard readiness."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _dominant_reason(signals: Mapping[str, Any], strategy: str) -> str | None:
    windows = _mapping(signals.get("no_vote_reason_counts_by_strategy_window"))
    reasons = _mapping(windows.get(strategy))
    valid = [(str(reason), int(n)) for reason, n in reasons.items() if isinstance(n, int) and n > 0]
    return max(valid, key=lambda item: item[1])[0] if valid else None


def build_strategy_readiness(
    runtime: Mapping[str, Any], signals: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Avoid conflating execution arm state with strategy trigger eligibility.

    ORB/SMC labels are explicitly about recent evidence, not instantaneous
    market state. Missing evidence is UNKNOWN rather than an invented signal.
    """
    raw_required = os.getenv("VWAP_EMA_PERIOD", "50")
    try:
        required = max(10, int(float(raw_required)))
    except (TypeError, ValueError):
        required = 50
    history = _mapping(runtime.get("history"))
    selected = _mapping(runtime.get("selected"))
    pairs: dict[str, dict[str, Any]] = {}
    for side in ("ce", "pe"):
        counts = _mapping(history.get(side))
        values = [counts.get(name) for name in ("mdm", "runner", "indicator")]
        usable = min(int(v) for v in values) if all(isinstance(v, (int, float)) and v >= 0 for v in values) else None
        pairs[side] = {
            "symbol": selected.get(side),
            "usable_bars": usable,
            "required_bars": required,
            "history_ready": usable >= required if usable is not None and selected.get(side) else None,
        }
    known = [x["history_ready"] for x in pairs.values()]
    if not selected.get("ce") or not selected.get("pe") or any(v is None for v in known):
        vwap_status = "UNKNOWN"
    else:
        vwap_status = "HISTORY_READY" if all(known) else "WARMING"
    observed = _mapping(signals.get("no_vote_reason_counts_by_strategy_window"))
    orb_window = _mapping(observed.get("ORBPro"))
    past_full_range = any(
        int(orb_window.get(reason, 0) or 0) > 0
        for reason in ("no_fresh_underlying_breakout", "orb_bar_already_evaluated", "awaiting_orb_retest")
    )
    orb_status = "RANGE_EVALUATED_RECENTLY" if past_full_range else (
        "AWAITING_RANGE_OR_EVIDENCE" if orb_window else "UNKNOWN"
    )
    lifecycle = _mapping(signals.get("setup_lifecycle"))
    active = lifecycle.get("active") or []
    smc_confirming = any(
        isinstance(row, Mapping)
        and str(row.get("strategy", "")).upper() == "SMC"
        and row.get("stage") == "CONFIRMING"
        for row in active if isinstance(active, list)
    )
    smc_reason = _dominant_reason(signals, "SMC")
    smc_status = (
        "AWAITING_CONFIRMATION" if smc_confirming
        else "RECENT_CONFIRMATION_WAIT" if smc_reason in {"smc_awaiting_confirmation", "smc_trend_corroboration_missing"}
        else "NO_ACTIVE_SETUP_OBSERVED" if smc_reason
        else "UNKNOWN"
    )
    return {
        "VWAPPro": {
            "state": vwap_status,
            "ce": pairs["ce"], "pe": pairs["pe"],
            "required_bars": required,
            "recent_no_vote_reason": _dominant_reason(signals, "VWAPPro"),
            "basis": "minimum_cached_mdm_runner_indicator_bars",
        },
        "ORBPro": {
            "state": orb_status,
            "recent_no_vote_reason": _dominant_reason(signals, "ORBPro"),
            "basis": "recent_strategy_evaluations_not_current_breakout",
        },
        "SMC": {
            "state": smc_status,
            "recent_no_vote_reason": smc_reason,
            "basis": "active_observability_lifecycle_or_recent_evaluations_not_execution",
        },
    }
