"""Post-close reconcile gating: don't hammer the broker all night when flat."""
from __future__ import annotations

import types
from unittest.mock import patch

from nifty_scalper_bot.core.app import _should_reconcile_now


def _ctx(open_positions):
    pm = types.SimpleNamespace(get_open_positions=lambda: open_positions)
    return types.SimpleNamespace(position_manager=pm)


async def test_skips_when_closed_and_flat(monkeypatch):
    monkeypatch.setenv("HEALTH_RECONCILE_SKIP_WHEN_CLOSED", "true")
    with patch("nifty_scalper_bot.risk.session_gate._is_market_open", return_value=False):
        assert _should_reconcile_now(_ctx([])) is False


async def test_reconciles_when_closed_but_position_open(monkeypatch):
    monkeypatch.setenv("HEALTH_RECONCILE_SKIP_WHEN_CLOSED", "true")
    with patch("nifty_scalper_bot.risk.session_gate._is_market_open", return_value=False):
        assert _should_reconcile_now(_ctx([{"symbol": "X"}])) is True


async def test_reconciles_during_market_hours(monkeypatch):
    monkeypatch.setenv("HEALTH_RECONCILE_SKIP_WHEN_CLOSED", "true")
    with patch("nifty_scalper_bot.risk.session_gate._is_market_open", return_value=True):
        assert _should_reconcile_now(_ctx([])) is True


async def test_disabled_always_reconciles(monkeypatch):
    monkeypatch.setenv("HEALTH_RECONCILE_SKIP_WHEN_CLOSED", "false")
    with patch("nifty_scalper_bot.risk.session_gate._is_market_open", return_value=False):
        assert _should_reconcile_now(_ctx([])) is True
