from __future__ import annotations

import importlib
import json
import os
from types import SimpleNamespace
import subprocess
import sys
from typing import Any

import nifty_scalper_bot.execution as execution
from nifty_scalper_bot.execution import order_manager
from nifty_scalper_bot.execution import order_manager_core
from nifty_scalper_bot.execution.native_entry_gate import NO_BLOCK
from nifty_scalper_bot.execution.runtime_order_manager import RuntimeOrderManager


class _Logger:
    def critical(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Provider:
    def __init__(self, unresolved: bool = True) -> None:
        self.unresolved = unresolved

    def has_unresolved_exit(self) -> bool:
        return self.unresolved

    def get_first_unresolved_exit_bracket_id(self) -> str:
        return "entry-1"


def _manager(provider: Any | None = None) -> RuntimeOrderManager:
    manager = object.__new__(RuntimeOrderManager)
    manager._logger = _Logger()
    manager._last_order_decision = {}
    manager._unresolved_exit_provider = provider
    manager._unresolved_exit_guard_installed = provider is not None
    return manager


def test_public_order_import_has_one_stable_runtime_identity() -> None:
    assert order_manager.OrderManager is RuntimeOrderManager
    assert execution.OrderManager is RuntimeOrderManager
    assert issubclass(RuntimeOrderManager, order_manager_core.OrderManager)
    assert not hasattr(order_manager, "LegacyOrderManager")
    assert RuntimeOrderManager.submit_trade_plan_result.__module__ == (
        "nifty_scalper_bot.execution.runtime_order_manager"
    )
    assert RuntimeOrderManager._update_from_response.__module__ == (
        "nifty_scalper_bot.execution.runtime_order_manager"
    )
    assert not hasattr(RuntimeOrderManager, "_canonical_entry_recovery_installed")


def test_importing_package_does_not_replace_order_methods() -> None:
    before = order_manager.OrderManager.submit_trade_plan_result
    imported = importlib.import_module("nifty_scalper_bot.execution")
    assert imported.OrderManager is order_manager.OrderManager
    assert order_manager.OrderManager.submit_trade_plan_result is before


def test_native_gate_blocks_new_entry_without_calling_base_engine() -> None:
    manager = _manager(_Provider(True))
    result = RuntimeOrderManager.submit_trade_plan_result(
        manager,
        SimpleNamespace(symbol="NFO:NIFTY26JUN24000CE"),
    )
    assert result.accepted is False
    assert result.reason == "unresolved_exit_position"
    assert result.broker_attempted is False
    assert manager._last_order_decision["bracket_id"] == "entry-1"


def test_native_gate_allows_protective_exit_but_blocks_normal_order() -> None:
    manager = _manager(_Provider(True))
    protective = manager._blocked(
        "place_order",
        (),
        {
            "symbol": "NFO:NIFTY26JUN24000CE",
            "side": "SELL",
            "quantity": 50,
            "tag": "EXIT_HARD_SL",
            "reduce_only": True,
        },
    )
    normal = manager._blocked(
        "place_order",
        (),
        {
            "symbol": "NFO:NIFTY26JUN24000CE",
            "side": "BUY",
            "quantity": 50,
            "tag": "runner_entry",
        },
    )
    assert protective is NO_BLOCK
    assert normal is None


def test_unresolved_exit_provider_is_canonical_reconciliation_owner() -> None:
    manager = _manager(None)
    provider = _Provider(True)

    manager.set_unresolved_exit_provider(provider)

    assert manager._unresolved_exit_provider is provider
    assert manager._bracket_manager is provider

    manager.set_unresolved_exit_provider(None)

    assert manager._unresolved_exit_provider is None
    assert manager._bracket_manager is None


def test_managed_order_preserves_approved_strategy_name(monkeypatch) -> None:
    manager = _manager(None)
    captured: dict[str, Any] = {}

    def core_place_order(self: Any, *args: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "OID-1"

    def core_managed(self: Any, *args: Any, **kwargs: Any) -> Any:
        order_id = self.place_order(
            symbol=kwargs["symbol"],
            side=kwargs["side"],
            quantity=kwargs["quantity"],
            signal_id=kwargs.get("signal_id"),
            intent="ENTRY",
        )
        return SimpleNamespace(
            accepted=bool(order_id),
            order_id=order_id,
            reason="accepted",
            details={},
            broker_attempted=True,
        )

    monkeypatch.setattr(order_manager_core.OrderManager, "place_order", core_place_order)
    monkeypatch.setattr(
        order_manager_core.OrderManager,
        "place_managed_order_result",
        core_managed,
    )

    result = manager.place_managed_order_result(
        symbol="NFO:NIFTY26JUL23950PE",
        side="BUY",
        quantity=65,
        strategy_name="OrderFlow",
        signal_id="sig-1",
    )

    assert result.accepted is True
    assert captured["strategy_name"] == "OrderFlow"


def test_live_env_normalizes_per_trade_risk_to_seven_percent(monkeypatch) -> None:
    from nifty_scalper_bot.config.env_utils import normalise_live_env_defaults

    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("RISK__PER_TRADE_RISK_PCT", "4.0")
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "4.0")

    normalise_live_env_defaults()

    assert os.environ["RISK__PER_TRADE_RISK_PCT"] == "7.0"
    assert os.environ["RISK_PER_TRADE_PCT"] == "7.0"


def test_order_module_is_safe_when_imported_before_package() -> None:
    code = r'''
import importlib
import json
om = importlib.import_module("nifty_scalper_bot.execution.order_manager")
before_class = id(om.OrderManager)
before_method = id(om.OrderManager.submit_trade_plan_result)
execution = importlib.import_module("nifty_scalper_bot.execution")
print(json.dumps({
    "before_class": before_class,
    "after_class": id(om.OrderManager),
    "package_class": id(execution.OrderManager),
    "before_method": before_method,
    "after_method": id(om.OrderManager.submit_trade_plan_result),
    "module": om.OrderManager.__module__,
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, ["src", os.environ.get("PYTHONPATH", "")])
            ),
        },
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["before_class"] == payload["after_class"] == payload["package_class"]
    assert payload["before_method"] == payload["after_method"]
    assert payload["module"] == "nifty_scalper_bot.execution.runtime_order_manager"
