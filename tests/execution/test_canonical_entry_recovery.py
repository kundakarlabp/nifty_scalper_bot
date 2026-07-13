from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from nifty_scalper_bot.execution.broker_recovery import (
    BrokerFailure,
    RecoveryAction,
    classify_broker_failure,
    decide_recovery,
)
from nifty_scalper_bot.execution.entry_recovery import install_entry_recovery
from nifty_scalper_bot.execution.order_manager import TradePlan, TradePlanSubmitResult


class _Logger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def critical(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Quotes:
    def __init__(self, quote: dict[str, Any]) -> None:
        self.quote = quote
        self.refreshes = 0

    def refresh_quote_now(self, _symbol: str, **_kwargs: Any) -> None:
        self.refreshes += 1

    def get_quote(self, _symbol: str) -> dict[str, Any]:
        return dict(self.quote)


class _Broker:
    def __init__(self) -> None:
        self.orders_payload: list[dict[str, Any]] = []
        self.positions_payload: list[dict[str, Any]] = []
        self.cancel_calls: list[str] = []
        self.status_payload: dict[str, Any] = {}

    def get_orders(self) -> list[dict[str, Any]]:
        return list(self.orders_payload)

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self.positions_payload)

    def cancel_order(self, order_id: str, **_kwargs: Any) -> bool:
        self.cancel_calls.append(order_id)
        return True

    def get_order_status(self, _order_id: str) -> dict[str, Any]:
        return dict(self.status_payload)


class _RecoveryManager:
    responses: list[TradePlanSubmitResult] = []

    def __init__(self, responses: list[TradePlanSubmitResult]) -> None:
        self.responses = list(responses)
        self.plans: list[TradePlan] = []
        self._logger = _Logger()
        self._market_data = _Quotes({"ask": 103.0, "bid": 102.5, "ltp": 102.8})
        self._data_hub = None
        self._broker = _Broker()
        self._last_order_decision: dict[str, Any] = {}
        self._margin_factor = 1.0
        self._margin_buffer = 0.95
        self._instrument_resolver = None
        self.cleared: list[str] = []

    def submit_trade_plan_result(self, plan: TradePlan) -> TradePlanSubmitResult:
        self.plans.append(plan)
        return self.responses.pop(0)

    def _clear_pending_signal(self, signal_id: str) -> None:
        self.cleared.append(signal_id)

    def _lot_size_for_symbol(self, _symbol: str) -> int:
        return 50

    def _resolve_available_margin(self) -> tuple[float, str]:
        return 8_000.0, "fresh"

    def _find_open_order(self, _key: str) -> dict[str, Any] | None:
        return None


install_entry_recovery(_RecoveryManager)


def _plan(quantity: int = 100) -> TradePlan:
    return TradePlan(
        symbol="NFO:NIFTY2662324050PE",
        side="BUY",
        quantity=quantity,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        signal_id="signal-1",
        trace_id="trace-1",
        tag="runner-entry",
    )


def _reject(message: str) -> TradePlanSubmitResult:
    return TradePlanSubmitResult(
        accepted=False,
        reason="broker_rejected",
        details={"broker_message": message},
        broker_attempted=True,
    )


def _accept(order_id: str = "new-order") -> TradePlanSubmitResult:
    return TradePlanSubmitResult(
        accepted=True,
        order_id=order_id,
        reason="accepted",
        broker_attempted=True,
    )


def test_classifier_prefers_trigger_error_and_marks_terminal_auth() -> None:
    trigger = decide_recovery("Trigger price must be lower than limit price")
    assert trigger.failure is BrokerFailure.TRIGGER_INVALID
    assert trigger.action is RecoveryAction.REFRESH_AND_REVALIDATE
    assert trigger.retryable is True

    auth = decide_recovery("Invalid access token; session expired")
    assert auth.failure is BrokerFailure.AUTHENTICATION
    assert auth.retryable is False
    assert (
        classify_broker_failure("HTTP 503 service unavailable")
        is BrokerFailure.TRANSIENT
    )


def test_price_reject_refreshes_and_rebuilds_geometry_once() -> None:
    manager = _RecoveryManager(
        [
            _reject("invalid limit price: price out of range"),
            _accept(),
        ]
    )

    result = manager.submit_trade_plan_result(_plan())

    assert result.accepted is True
    assert len(manager.plans) == 2
    retry = manager.plans[1]
    assert retry.entry_price == 103.0
    assert retry.stop_loss == 93.0
    assert retry.take_profit == 123.0
    assert retry.quantity == 100
    assert manager._market_data.refreshes == 1
    assert manager.cleared == ["signal-1"]
    assert result.details["entry_recovery"]["outcome"] == "retried_once"


def test_margin_reject_resizes_to_affordable_lot_and_reruns_original() -> None:
    manager = _RecoveryManager(
        [
            _reject("insufficient funds; required margin exceeds available margin"),
            _accept("resized-order"),
        ]
    )

    result = manager.submit_trade_plan_result(_plan(quantity=100))

    assert result.accepted is True
    assert result.order_id == "resized-order"
    assert len(manager.plans) == 2
    assert manager.plans[1].quantity == 50
    assert result.details["entry_recovery"]["retry_quantity"] == 50


def test_duplicate_response_reconciles_existing_order_without_resubmit() -> None:
    manager = _RecoveryManager([_reject("duplicate order request already processed")])
    manager._broker.orders_payload = [
        {
            "order_id": "existing-123",
            "symbol": "NIFTY2662324050PE",
            "tag": "runner-entry",
            "status": "OPEN",
        }
    ]

    result = manager.submit_trade_plan_result(_plan())

    assert result.accepted is True
    assert result.order_id == "existing-123"
    assert result.reason == "broker_order_reconciled"
    assert len(manager.plans) == 1


def test_ambiguous_timeout_with_broker_exposure_never_retries() -> None:
    manager = _RecoveryManager([_reject("gateway timeout HTTP 504")])
    manager._broker.positions_payload = [
        {"tradingsymbol": "NIFTY2662324050PE", "quantity": 50}
    ]

    result = manager.submit_trade_plan_result(_plan())

    assert result.accepted is False
    assert result.reason == "entry_exposure_reconciliation_required"
    assert len(manager.plans) == 1


def test_terminal_authentication_failure_is_not_retried() -> None:
    manager = _RecoveryManager([_reject("invalid access token; session expired")])

    result = manager.submit_trade_plan_result(_plan())

    assert result.accepted is False
    assert len(manager.plans) == 1
    assert result.details["entry_recovery"]["outcome"] == "terminal"


@dataclass
class _Order:
    order_id: str = "entry-partial"
    symbol: str = "NFO:NIFTY2662324050PE"
    tag: str = "runner-entry"
    client_order_id: str = "client-partial"
    filled_quantity: int = 50
    fill_price: float = 101.5
    average_price: float = 101.5
    status: Any = None


class _PartialManager:
    def __init__(self) -> None:
        self._logger = _Logger()
        self._broker = _Broker()
        self._broker.status_payload = {
            "status": "CANCELLED",
            "filled_quantity": 50,
            "average_price": 101.5,
        }
        self._last_order_decision: dict[str, Any] = {}
        self.uncertain: list[str] = []
        self.confirmed: list[tuple[str, int, float]] = []
        self._bracket_manager = SimpleNamespace(
            confirm_partial_entry_fill=lambda order_id, qty, price: (
                self.confirmed.append((order_id, qty, price)) or True
            )
        )

    def submit_trade_plan_result(self, _plan: Any) -> TradePlanSubmitResult:
        return _accept()

    def _update_from_response(self, order: _Order, _payload: dict[str, Any]) -> _Order:
        order.status = SimpleNamespace(name="PARTIALLY_FILLED")
        return order

    def _mark_order_uncertain(self, key: str) -> None:
        self.uncertain.append(key)


install_entry_recovery(_PartialManager)


def test_multi_lot_partial_entry_cancels_remainder_and_arms_completed_lot() -> None:
    manager = _PartialManager()
    manager._broker.positions_payload = [
        {"tradingsymbol": "NIFTY2662324050PE", "quantity": 50}
    ]
    order = _Order()
    order.quantity = 100
    order.requested_lots = 2
    order.resolved_lot_size = 50

    updated = manager._update_from_response(
        order,
        {
            "status": "PARTIALLY_FILLED",
            "filled_quantity": 50,
            "pending_quantity": 50,
            "average_price": 101.5,
        },
    )

    assert updated is order
    assert manager._broker.cancel_calls == ["entry-partial"]
    assert manager.confirmed == [("entry-partial", 50, 101.5)]
    assert order.entry_lifecycle_state["broker_filled_lots"] == 1
    assert order.entry_lifecycle_state["protected_lots"] == 1
    assert (
        manager._last_order_decision["block_reason"] == "entry_reconciliation_pending"
    )


def test_one_lot_complete_fill_protects_exact_lot_quantity() -> None:
    manager = _PartialManager()
    manager._broker.positions_payload = [
        {"tradingsymbol": "NIFTY2662324050PE", "quantity": 50}
    ]
    manager._broker.status_payload = {
        "status": "COMPLETE",
        "filled_quantity": 50,
        "pending_quantity": 0,
        "average_price": 101.5,
    }
    manager._bracket_manager = SimpleNamespace(
        confirm_partial_entry_fill=lambda order_id, qty, price: True
    )
    order = _Order(filled_quantity=50)
    order.quantity = 50
    order.requested_lots = 1
    order.resolved_lot_size = 50

    manager._update_from_response(
        order,
        {
            "status": "COMPLETE",
            "filled_quantity": 50,
            "pending_quantity": 0,
            "average_price": 101.5,
        },
    )

    assert order.entry_lifecycle_state["broker_filled_lots"] == 1
    assert order.entry_lifecycle_state["protected_lots"] == 1
    assert order.entry_lifecycle_state["protected_quantity"] == 50
    assert manager._last_order_decision["block_reason"] == "ENTRY_FULL_LOT_PROTECTED"


def test_non_lot_broker_fill_latches_invariant_without_protection() -> None:
    manager = _PartialManager()
    order = _Order(filled_quantity=25)
    order.quantity = 50
    order.requested_lots = 1
    order.resolved_lot_size = 50
    manager._bracket_manager = SimpleNamespace(
        confirm_partial_entry_fill=lambda *_args: (_ for _ in ()).throw(
            AssertionError("must not protect")
        )
    )

    manager._update_from_response(
        order,
        {
            "status": "COMPLETE",
            "filled_quantity": 25,
            "pending_quantity": 25,
            "average_price": 101.5,
        },
    )

    assert order.entry_lifecycle_state["state"] == "ENTRY_RECONCILIATION_UNKNOWN"
    assert (
        manager._last_order_decision["block_reason"]
        == "broker_quantity_lot_invariant_violation"
    )


def test_protection_none_is_not_verified_success() -> None:
    manager = _PartialManager()
    manager._broker.positions_payload = [
        {"tradingsymbol": "NIFTY2662324050PE", "quantity": 50}
    ]
    manager._bracket_manager = SimpleNamespace(
        confirm_partial_entry_fill=lambda *_args: None
    )
    order = _Order(filled_quantity=50)
    order.quantity = 50
    order.requested_lots = 1
    order.resolved_lot_size = 50

    manager._update_from_response(
        order,
        {
            "status": "COMPLETE",
            "filled_quantity": 50,
            "pending_quantity": 0,
            "average_price": 101.5,
        },
    )

    assert order.entry_lifecycle_state["state"] == "ENTRY_RECONCILIATION_UNKNOWN"
    assert manager._last_order_decision["block_reason"] == "entry_protection_failed"
