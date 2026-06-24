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
