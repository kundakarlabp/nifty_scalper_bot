"""Legacy two-leg simulation must not confuse broker acceptance with a fill."""

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.order_manager_core import (
    AtomicLeg,
    OrderDetails,
    OrderManager,
    OrderPlacementError,
    OrderStatus,
    OrderType,
)

_LEGS = (
    AtomicLeg(symbol="NFO:NIFTY26NOV24000CE", side="BUY", quantity=65),
    AtomicLeg(symbol="NFO:NIFTY26NOV24000PE", side="BUY", quantity=65),
)


def simulated_manager(statuses):
    pending = list(statuses)
    submitted = []
    reconciled = []

    def submit(**values):
        index = len(submitted)
        status, filled = pending.pop(0)
        result = OrderDetails(
            order_id=f"order-{index}",
            symbol=values["symbol"],
            side=values["side"],
            quantity=values["quantity"],
            order_type=OrderType.MARKET,
            status=status,
            filled_quantity=filled,
        )
        submitted.append(result)
        return result

    owner = SimpleNamespace(
        is_live_mode=lambda: False,
        _normalize_leg=lambda leg: leg,
        _validate_quantity=lambda *args: None,
        _ensure_trading_allowed=lambda **kwargs: True,
        _options_policy=None,
        _place_single_order=submit,
        _leg_failed=OrderManager._leg_failed,
        cancel_and_reconcile=lambda ids: reconciled.append(tuple(ids)),
        _logger=SimpleNamespace(info=lambda *a, **k: None),
    )
    return owner, submitted, reconciled


@pytest.mark.parametrize(
    "status",
    [
        OrderStatus.PENDING,
        OrderStatus.SUBMITTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    ],
)
def test_unfilled_or_partial_first_leg_never_submits_second_leg(status):
    manager, submitted, reconciled = simulated_manager([(status, 0)])
    with pytest.raises(OrderPlacementError, match="Atomic leg"):
        OrderManager.place_atomic_entry(manager, _LEGS)
    assert len(submitted) == 1
    assert reconciled == [("order-0",)]


def test_pending_second_leg_reconciles_both_instead_of_claiming_success():
    manager, submitted, reconciled = simulated_manager(
        [(OrderStatus.FILLED, 65), (OrderStatus.PENDING, 0)]
    )
    with pytest.raises(OrderPlacementError, match="Atomic leg"):
        OrderManager.place_atomic_entry(manager, _LEGS)
    assert len(submitted) == 2
    assert reconciled == [("order-0", "order-1")]


def test_partial_or_impossible_reported_fill_of_filled_leg_is_rejected():
    for filled in (0, 32, 66):
        manager, submitted, reconciled = simulated_manager(
            [(OrderStatus.FILLED, filled)]
        )
        with pytest.raises(OrderPlacementError, match="Atomic leg"):
            OrderManager.place_atomic_entry(manager, _LEGS)
        assert len(submitted) == 1
        assert reconciled == [("order-0",)]


def test_both_exact_broker_reported_fills_only_case_returning_order_ids():
    manager, submitted, reconciled = simulated_manager(
        [(OrderStatus.FILLED, 65), (OrderStatus.FILLED, 65)]
    )
    result = OrderManager.place_atomic_entry(manager, _LEGS)
    assert result == ["order-0", "order-1"]
    assert len(submitted) == 2
    assert reconciled == []


@pytest.mark.parametrize("tolerance", [0.1, 1.0, -0.1, float("nan")])
def test_partial_fill_tolerance_cannot_allow_unconfirmed_leg(tolerance):
    manager, submitted, reconciled = simulated_manager([])
    with pytest.raises(ValueError, match="atomic_partial_fill_tolerance_unsupported"):
        OrderManager.place_atomic_entry(
            manager, _LEGS, partial_fill_tolerance=tolerance
        )
    assert submitted == []
    assert reconciled == []


def test_missing_broker_acknowledgement_is_never_success():
    manager, submitted, reconciled = simulated_manager([])
    manager._place_single_order = lambda **values: None
    with pytest.raises(OrderPlacementError, match="atomic_leg_acknowledgement_missing"):
        OrderManager.place_atomic_entry(manager, _LEGS)
    assert submitted == []
    assert reconciled == []
