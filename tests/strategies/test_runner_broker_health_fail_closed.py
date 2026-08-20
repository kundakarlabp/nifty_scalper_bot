"""Broker health must be affirmative before live entry is eligible."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.utils.logging import get_logger


def _runner(order_manager: object) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._order_manager = order_manager
    runner._logger = get_logger(__name__)
    runner._order_manager_kill_switch_status_for_entry = lambda: (False, {})
    return runner


@pytest.mark.parametrize(
    "health_surface",
    (
        {},
        {"get_broker_health_snapshot": lambda: None},
        {"get_broker_health_snapshot": lambda: {}},
        {
            "get_broker_health_snapshot": lambda: (_ for _ in ()).throw(
                RuntimeError("unavailable")
            )
        },
    ),
)
def test_unknown_broker_health_blocks_entry(health_surface: dict[str, object]) -> None:
    manager = SimpleNamespace(is_live_mode=lambda: True, **health_surface)

    allowed, reason, details = _runner(
        manager
    )._resolve_order_manager_health_for_entry()

    assert allowed is False
    assert reason == "broker_health_unknown"
    assert details["broker_ready"] is False
    assert details["broker_ready_assumed"] is False


def test_affirmative_broker_health_allows_entry() -> None:
    manager = SimpleNamespace(
        is_live_mode=lambda: True,
        get_broker_health_snapshot=lambda: {
            "broker_connected": True,
            "order_api_available": True,
            "trading_allowed_effect": "none",
        },
    )

    allowed, reason, details = _runner(
        manager
    )._resolve_order_manager_health_for_entry()

    assert allowed is True
    assert reason == "broker_health_ready"
    assert details["broker_ready"] is True
