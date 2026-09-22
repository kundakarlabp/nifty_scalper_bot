from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

from nifty_scalper_bot.core import app


def test_polling_failover_supervisor_is_owned_by_core_app() -> None:
    supervisor = app._polling_failover_supervisor_iteration

    assert callable(supervisor)
    assert supervisor.__module__ == "nifty_scalper_bot.core.app"
    assert "polling_failover_runtime" not in inspect.getsource(supervisor)


def test_healthy_native_polling_decision_uses_change_logging(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        app,
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
        app._polling_failover_supervisor_iteration(
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
    assert "activate=False" in str(calls[0]["message"])


def test_core_runtime_hardening_does_not_install_polling_adapter() -> None:
    import nifty_scalper_bot.core as core

    source = inspect.getsource(core)

    assert "polling_failover_runtime" not in source
    assert "_polling_adapter" not in source
