"""Tests for execution assessment helpers."""

from __future__ import annotations

from nifty_scalper_bot.execution.assess_orders import (
    assess_order_guard_passes_when_override_and_soft_state,
)


class _StubRisk:
    def __init__(self, allowed: bool, reasons: tuple[str, ...]) -> None:
        self._allowed = allowed
        self._reasons = reasons

    def risk_gate_should_trade(self) -> tuple[bool, tuple[str, ...]]:
        return self._allowed, self._reasons


def test_guard_allows_soft_reasons() -> None:
    risk = _StubRisk(False, ("COOLDOWN", "STALE"))
    ok, detail, meta = assess_order_guard_passes_when_override_and_soft_state(
        object(),
        risk,
    )
    assert ok
    assert detail == "ok"
    assert meta["allowed"] is False
    assert meta["reasons"] == ["COOLDOWN", "STALE"]


def test_guard_flags_hard_reason() -> None:
    risk = _StubRisk(False, ("BREAKER",))
    ok, detail, meta = assess_order_guard_passes_when_override_and_soft_state(
        object(),
        risk,
    )
    assert not ok
    assert detail == "reasons=['BREAKER']"
    assert meta["allowed"] is False
    assert meta["reasons"] == ["BREAKER"]
