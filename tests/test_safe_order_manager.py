from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nifty_scalper_bot.config.settings import OrderSettings
from nifty_scalper_bot.execution.safe_order_manager import SafeOrderManager


@dataclass
class DummyOrderManager:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    monitoring_started: bool = False
    _throttled: int = 2
    _rejections: int = 3

    def place_order(self, *args: Any, **kwargs: Any) -> str:
        self.calls.append((args, dict(kwargs)))
        return "order-1"

    def start_monitoring(self) -> None:
        self.monitoring_started = True

    def stop_monitoring(self) -> None:
        self.monitoring_started = False

    def consume_skip_reason(self) -> str:
        return "canonical_skip"

    def canonical_only_method(self) -> str:
        return "delegated"


def _adapter() -> tuple[SafeOrderManager, DummyOrderManager]:
    manager = DummyOrderManager()
    adapter = SafeOrderManager(
        order_manager=manager,  # type: ignore[arg-type]
        settings=OrderSettings(),
    )
    return adapter, manager


def test_adapter_delegates_order_once_without_mutation_or_retry() -> None:
    adapter, manager = _adapter()
    result = adapter.place_order(
        symbol="NFO:NIFTY26JUN24000CE",
        side="BUY",
        quantity=65,
        order_type="LIMIT",
        price=100.0,
        tag="runner_entry",
    )
    assert result == "order-1"
    assert len(manager.calls) == 1
    assert manager.calls[0][1]["price"] == 100.0
    assert manager.calls[0][1]["tag"] == "runner_entry"


def test_adapter_has_no_independent_execution_state() -> None:
    adapter, manager = _adapter()
    assert adapter.canonical_only_method() == "delegated"
    assert adapter.throttled_count() == 2
    assert adapter.rejection_count() == 3
    assert adapter.consume_skip_reason() == "canonical_skip"
    adapter.start_monitoring()
    assert manager.monitoring_started is True
    adapter.stop_monitoring()
    assert manager.monitoring_started is False


def test_live_toggle_updates_only_shared_settings() -> None:
    adapter, _manager = _adapter()
    adapter.set_live_enabled(True)
    assert adapter.settings.enable_live is True
    adapter.set_live_enabled(False)
    assert adapter.settings.enable_live is False
