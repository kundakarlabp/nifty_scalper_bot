"""Execution subsystem assessment helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

_ALLOWED_SOFT_REASONS = {"COOLDOWN", "STALE"}


def assess_order_guard_passes_when_override_and_soft_state(
    order_mgr: Any, risk: Any
) -> Tuple[bool, str, Dict[str, Any]]:
    """Ensure soft risk states do not escalate to hard guard failures."""

    _ = order_mgr  # placeholder for symmetry with future probes
    allowed, raw_reasons = risk.risk_gate_should_trade()
    reasons = tuple(raw_reasons or ())
    ok = bool(allowed) or not reasons or _reasons_soft_only(reasons)
    detail = "ok" if ok else f"reasons={list(reasons)}"
    return ok, detail, {"reasons": list(reasons), "allowed": bool(allowed)}


def _reasons_soft_only(reasons: Iterable[str]) -> bool:
    return all(reason in _ALLOWED_SOFT_REASONS for reason in reasons)


__all__ = ["assess_order_guard_passes_when_override_and_soft_state"]
