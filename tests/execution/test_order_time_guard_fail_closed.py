from __future__ import annotations

from typing import Any

import pytest

from nifty_scalper_bot.execution import order_manager_core
from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType


class _Broker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.auth_invalid = False

    def place_order(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {"order_id": "OID-1", "status": "OPEN"}

    def get_orders(self) -> list[dict[str, Any]]:
        return []


class _Positions:
    def current_pnl_reconciliation_blocker(self) -> None:
        return None

    def has_open_position(self, _symbol: str) -> bool:
        return False

    def get_open_positions(self) -> list[Any]:
        return []


@pytest.fixture
def live_manager(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> tuple[OrderManager, _Broker]:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _Broker()
    manager = OrderManager(
        broker_client=broker,
        position_manager=_Positions(),  # type: ignore[arg-type]
        rate_limiter=object(),  # type: ignore[arg-type]
        history_path=tmp_path / "orders.json",
    )
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 75)
    return manager, broker


def test_entry_fails_closed_when_market_time_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
    live_manager: tuple[OrderManager, _Broker],
) -> None:
    manager, broker = live_manager

    def _broken_time_status() -> tuple[bool, str]:
        raise RuntimeError("clock unavailable")

    monkeypatch.setattr(order_manager_core, "get_time_status", _broken_time_status)

    order_id = manager.place_order(
        symbol="NFO:NIFTY25APR25000CE",
        side="BUY",
        quantity=75,
        order_type=OrderType.LIMIT,
        price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        signal_id="time-guard-entry",
        intent="ENTRY",
        check_risk=False,
    )

    assert order_id is None
    assert broker.calls == []
    assert manager.consume_skip_reason() == "time_guard_error"
