from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from nifty_scalper_bot.execution import order_manager_core
from nifty_scalper_bot.execution.runtime_order_manager import RuntimeOrderManager
from nifty_scalper_bot.strategies.signal_identity_patch import (
    _deterministic_id,
    current_order_setup_metadata,
)


class _Logger:
    def critical(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Provider:
    def has_unresolved_exit(self) -> bool:
        return False


def _manager() -> RuntimeOrderManager:
    manager = object.__new__(RuntimeOrderManager)
    manager._logger = _Logger()
    manager._last_order_decision = {}
    manager._unresolved_exit_provider = None
    manager._unresolved_exit_guard_installed = False
    return manager


def test_runtime_order_scopes_exact_setup_identity_to_core_call(monkeypatch) -> None:
    manager = _manager()
    setup_timestamp = "2026-08-11T09:46:00+05:30"
    signal = SimpleNamespace(
        symbol="NFO:NIFTY2681124500PE",
        action="BUY",
        metadata={
            "strategy": "VWAPPro",
            "role": "trigger",
            "contract_side": "PE",
            "setup_id": f"vwap:PE:{setup_timestamp}",
            "setup_candle_timestamp": setup_timestamp,
        },
    )
    signal_id = _deterministic_id(signal)
    observed: dict[str, Any] = {}

    def core_place_order(self: Any, *args: Any, **kwargs: Any) -> str:
        observed.update(current_order_setup_metadata())
        return "OID-SETUP"

    monkeypatch.setattr(order_manager_core.OrderManager, "place_order", core_place_order)

    order_id = manager.place_order(
        symbol=signal.symbol,
        side="BUY",
        quantity=65,
        signal_id=signal_id,
        intent="ENTRY",
        check_risk=False,
    )

    assert order_id == "OID-SETUP"
    assert observed["setup_candle_timestamp"] == setup_timestamp
    assert observed["setup_id"] == signal.metadata["setup_id"]
    assert current_order_setup_metadata() == {}


def test_context_only_vote_does_not_become_rearm_provenance(monkeypatch) -> None:
    manager = _manager()
    signal = SimpleNamespace(
        symbol="NFO:NIFTY2681124500PE",
        action="BUY",
        metadata={
            "strategy": "OrderFlow",
            "role": "context",
            "contract_side": "PE",
            "setup_candle_timestamp": "2026-08-11T09:47:00+05:30",
        },
    )
    signal_id = _deterministic_id(signal)
    observed: dict[str, Any] = {"sentinel": True}

    def core_place_order(self: Any, *args: Any, **kwargs: Any) -> str:
        observed.clear()
        observed.update(current_order_setup_metadata())
        return "OID-CONTEXT"

    monkeypatch.setattr(order_manager_core.OrderManager, "place_order", core_place_order)

    manager.place_order(
        symbol=signal.symbol,
        side="BUY",
        quantity=65,
        signal_id=signal_id,
        intent="ENTRY",
        check_risk=False,
    )

    assert observed == {}
