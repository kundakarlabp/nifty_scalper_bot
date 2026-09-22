"""Regression tests for native polling supervisor installation identity."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from nifty_scalper_bot.core import polling_failover_runtime as runtime
from nifty_scalper_bot.core.polling_failover_runtime import apply_app_patch


def test_apply_app_patch_replaces_unversioned_supervisor():
    async def legacy_supervisor(*args, **kwargs):
        return None, None

    module = SimpleNamespace(
        _polling_failover_supervisor_iteration=legacy_supervisor,
        _polling_fallback_degraded=lambda **kwargs: False,
    )

    assert apply_app_patch(module) is True
    installed = module._polling_failover_supervisor_iteration
    assert installed is not legacy_supervisor
    assert getattr(installed, "_nifty_polling_supervisor_version", None) == 2


def test_apply_app_patch_is_idempotent_for_versioned_supervisor():
    async def installed_supervisor(*args, **kwargs):
        return None, None

    setattr(installed_supervisor, "_nifty_polling_supervisor_version", 2)
    module = SimpleNamespace(
        _polling_failover_supervisor_iteration=installed_supervisor,
        _polling_fallback_degraded=lambda **kwargs: False,
    )

    assert apply_app_patch(module) is False
    assert module._polling_failover_supervisor_iteration is installed_supervisor


def test_healthy_decision_uses_state_change_logging(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runtime,
        "log_on_change",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    ctx = SimpleNamespace(
        is_market_open_now=lambda: True,
        websocket_manager=SimpleNamespace(is_connected=lambda: True),
        market_data_manager=SimpleNamespace(
            trading_feed_health=lambda: {
                "lagging": False,
                "futures_fresh": True,
                "options_fresh": True,
            },
            data_age_ms=lambda: 100.0,
        ),
    )
    fallback = SimpleNamespace(is_running=lambda: False)

    asyncio.run(
        runtime._polling_failover_supervisor_iteration(
            ctx,
            fallback,
            quote_stale_ms=2_000,
            degraded_since=None,
            recovered_since=None,
            activate_after=5,
            recover_cooldown=10,
        )
    )

    assert len(calls) == 1
    assert calls[0]["key"] == "polling_fallback_decision"
    assert calls[0]["state"][0] is False
    assert calls[0]["reminder_seconds"] == 60.0
    assert "activate=False" in calls[0]["message"]
