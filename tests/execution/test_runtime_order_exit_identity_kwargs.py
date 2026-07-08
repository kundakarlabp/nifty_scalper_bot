from __future__ import annotations

from typing import Any

from nifty_scalper_bot.execution import order_manager_core as core
from nifty_scalper_bot.execution.native_entry_gate import NO_BLOCK
from nifty_scalper_bot.execution.runtime_order_manager import (
    RuntimeOrderManager,
    _strip_exit_identity_kwargs,
)


def test_strip_exit_identity_kwargs_preserves_core_order_kwargs() -> None:
    cleaned, identity = _strip_exit_identity_kwargs(
        {
            "symbol": "NFO:NIFTY2671424250PE",
            "side": "SELL",
            "quantity": 65,
            "intent": "EXIT",
            "linked_entry_order_id": "2074702520085045248",
            "trade_lifecycle_id": "2074702520085045248",
            "bracket_id": "2074702520085045248",
        }
    )

    assert cleaned == {
        "symbol": "NFO:NIFTY2671424250PE",
        "side": "SELL",
        "quantity": 65,
        "intent": "EXIT",
    }
    assert identity == {
        "linked_entry_order_id": "2074702520085045248",
        "trade_lifecycle_id": "2074702520085045248",
        "bracket_id": "2074702520085045248",
    }


def test_runtime_place_order_strips_exit_identity_only_after_native_gate(monkeypatch) -> None:
    manager = object.__new__(RuntimeOrderManager)
    blocked_kwargs: dict[str, Any] = {}
    core_kwargs: dict[str, Any] = {}

    def fake_blocked(self: RuntimeOrderManager, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        blocked_kwargs.update(kwargs)
        return NO_BLOCK

    def fake_core_place_order(self: RuntimeOrderManager, *args: Any, **kwargs: Any) -> str:
        core_kwargs.update(kwargs)
        return "exit_order_1"

    monkeypatch.setattr(RuntimeOrderManager, "_blocked", fake_blocked)
    monkeypatch.setattr(core.OrderManager, "place_order", fake_core_place_order)

    order_id = RuntimeOrderManager.place_order(
        manager,
        symbol="NFO:NIFTY2671424250PE",
        side="SELL",
        quantity=65,
        order_type="MARKET",
        product="MIS",
        tag="exit_sl_20747025",
        check_risk=False,
        intent="EXIT",
        linked_entry_order_id="2074702520085045248",
        trade_lifecycle_id="2074702520085045248",
        bracket_id="2074702520085045248",
    )

    assert order_id == "exit_order_1"
    assert blocked_kwargs["intent"] == "EXIT"
    assert blocked_kwargs["linked_entry_order_id"] == "2074702520085045248"
    assert core_kwargs["intent"] == "EXIT"
    assert "linked_entry_order_id" not in core_kwargs
    assert "trade_lifecycle_id" not in core_kwargs
    assert "bracket_id" not in core_kwargs
    assert manager._last_exit_identity_kwargs["bracket_id"] == "2074702520085045248"
