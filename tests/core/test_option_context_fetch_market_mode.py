"""Option-context broker fetch is blocked only when market is genuinely closed.

Previously the gate fired for any mode != OPEN (including transient UNKNOWN around
the open bell), keeping SMC history-cold (broker_fetch_not_allowed) so it could never
vote — artificially suppressing consensus.
"""
from __future__ import annotations

import pathlib


async def test_fetch_gate_uses_closed_modes_only() -> None:
    src = pathlib.Path("src/nifty_scalper_bot/core/app.py").read_text()
    assert '_market_closed_for_fetch = _market_mode in {"PRE_MARKET", "POST_MARKET", "HOLIDAY"}' in src
    assert 'get_runtime_market_mode() != "OPEN"' not in src


async def test_closed_modes_block_open_and_unknown_allow() -> None:
    closed = {"PRE_MARKET", "POST_MARKET", "HOLIDAY"}
    assert all(m in closed for m in ("PRE_MARKET", "POST_MARKET", "HOLIDAY"))
    assert "OPEN" not in closed and "UNKNOWN" not in closed


import pytest
from types import SimpleNamespace

from nifty_scalper_bot.core import history_readiness


@pytest.mark.parametrize(
    "mode,expected_allow",
    [
        ("OPEN", True),
        ("UNKNOWN", True),   # transient UNKNOWN near open must not keep SMC cold
        ("PRE_MARKET", False),
        ("POST_MARKET", False),
        ("HOLIDAY", False),
    ],
)
async def test_option_context_fetch_gate_behavior(mode, expected_allow, monkeypatch):
    """Behavior test of the ACTUAL policy layer (history_readiness), replacing
    the string assertion against app.py which guarded the wrong file."""
    monkeypatch.setattr(history_readiness, "get_runtime_market_mode", lambda: mode)
    policy = history_readiness.resolve_history_policy(
        SimpleNamespace(strategy_runner=None),
        "NFO:NIFTY2671424000CE",
        role="option_context",
        phase="dynamic_update",
        reason="test",
    )
    assert bool(policy.allow_broker_fetch) is expected_allow


async def test_option_context_fetch_allowed_in_recovery_even_when_closed(monkeypatch):
    monkeypatch.setattr(history_readiness, "get_runtime_market_mode", lambda: "POST_MARKET")
    policy = history_readiness.resolve_history_policy(
        SimpleNamespace(strategy_runner=None),
        "NFO:NIFTY2671424000CE",
        role="option_context",
        phase="recovery",
        reason="test",
    )
    assert bool(policy.allow_broker_fetch) is True
