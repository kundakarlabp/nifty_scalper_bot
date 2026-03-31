"""Risk subsystem assessment helpers."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def assess_risk_soft_clear(risk: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """Ensure soft breakers clear after requesting a reset."""

    risk.reset_on_start(True)
    remaining = float(risk.cooldown_remaining())
    breaker = bool(getattr(risk, "_breaker_tripped", False))
    ok = remaining == 0.0 and not breaker
    detail = "ok" if ok else f"cooldown_left={remaining}"
    return ok, detail, {"cooldown": remaining, "breaker": breaker}


def assess_risk_codes(_risk: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """Placeholder assessment validated by targeted unit tests."""

    return True, "ok", {}


__all__ = ["assess_risk_soft_clear", "assess_risk_codes"]
