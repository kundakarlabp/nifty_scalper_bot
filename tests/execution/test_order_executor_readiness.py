from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.config.base import RiskConfig
from nifty_scalper_bot.execution.order_executor import OrderExecutor


class _Broker:
    def __init__(self, *, connected: bool = True, margin: float = 10_000.0) -> None:
        self._connected = connected
        self._margin = margin

    def is_connected(self) -> bool:
        return self._connected

    def available_margin(self) -> float:
        return self._margin


def test_is_execution_ready_reports_disconnected_broker() -> None:
    executor = OrderExecutor(_Broker(connected=False), RiskConfig(), SimpleNamespace())
    ready, reason = executor.is_execution_ready("NFO:NIFTY24APR25000CE")
    assert ready is False
    assert reason == "broker_disconnected"


def test_is_execution_ready_reports_invalid_symbol() -> None:
    resolver = SimpleNamespace(lookup=lambda _: None)
    mdm = SimpleNamespace(resolver=resolver)
    executor = OrderExecutor(_Broker(), RiskConfig(), mdm)
    ready, reason = executor.is_execution_ready("NFO:NIFTY24APR25000CE")
    assert ready is False
    assert reason == "invalid_instrument"
