"""Post-close reconcile gating: don't hammer the broker all night when flat."""
from __future__ import annotations

import types
from unittest.mock import patch

from nifty_scalper_bot.core.app import (
    _reconciliation_sleep_seconds,
    _should_reconcile_now,
)


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


def test_flat_market_open_reconciliation_uses_relaxed_cadence(monkeypatch):
    monkeypatch.delenv("HEALTH_FLAT_RECONCILE_INTERVAL_SEC", raising=False)
    ctx = types.SimpleNamespace(
        position_reconciliation_failed=False,
        unresolved_reconciliation_symbols=set(),
        unprotected_broker_positions=set(),
        unprotected_broker_position=False,
        position_manager=types.SimpleNamespace(
            get_open_positions=lambda: [], get_pending_orders=lambda: []
        ),
    )

    assert _reconciliation_sleep_seconds(ctx, market_open=True) == 60.0


def test_reconciliation_stays_rapid_for_exposure_pending_orders_or_uncertainty():
    def _ctx_for(*, positions=(), pending=(), failed=False):
        return types.SimpleNamespace(
            position_reconciliation_failed=failed,
            unresolved_reconciliation_symbols=set(),
            unprotected_broker_positions=set(),
            unprotected_broker_position=False,
            position_manager=types.SimpleNamespace(
                get_open_positions=lambda: list(positions),
                get_pending_orders=lambda: list(pending),
            ),
        )

    assert _reconciliation_sleep_seconds(
        _ctx_for(positions=({"symbol": "NFO:NIFTYCE"},)), market_open=True
    ) == 15.0
    assert _reconciliation_sleep_seconds(
        _ctx_for(pending=({"order_id": "1"},)), market_open=True
    ) == 15.0
    assert _reconciliation_sleep_seconds(
        _ctx_for(failed=True), market_open=True
    ) == 15.0
