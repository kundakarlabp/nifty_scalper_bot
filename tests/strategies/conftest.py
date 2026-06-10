"""Strategy test fixtures: keep tests deterministic w.r.t. the real clock."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_expiry_theta_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the wall-clock expiry gate so selector tests pass on any day.

    The gate itself is tested explicitly in tests/risk/test_expiry_gate.py.
    """
    monkeypatch.setenv("EXPIRY_THETA_GATE_ENABLED", "false")


@pytest.fixture(autouse=True)
def _disable_midday_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the wall-clock midday pause so selector tests pass at any hour."""
    monkeypatch.setenv("MIDDAY_PAUSE_ENABLED", "false")
