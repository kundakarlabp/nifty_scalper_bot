from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.execution.order_manager import (
    OrderDetails,
    OrderStatus,
    OrderType,
    RuntimeOrderManager,
)


ENTRY_ID = "2093222684300599296"
SYMBOL = "NFO:NIFTY2690124100PE"


class _Logger:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class _CanonicalVirtualBrackets:
    def __init__(self, *, initially_present: bool = True) -> None:
        self.bracket = (
            SimpleNamespace(
                bracket_id=ENTRY_ID,
                entry_order_id=ENTRY_ID,
                symbol=SYMBOL,
                is_virtual=True,
                active=True,
                entry_confirmed=True,
            )
            if initially_present
            else None
        )

    def get_bracket(self, order_id: str):
        return self.bracket if order_id == ENTRY_ID else None


def _filled_entry() -> OrderDetails:
    return OrderDetails(
        order_id=ENTRY_ID,
        symbol=SYMBOL,
        side="BUY",
        quantity=65,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        price=65.20,
        average_price=64.75,
        fill_price=64.75,
        filled_quantity=65,
        stop_loss=60.50,
        take_profit=74.75,
        intent="ENTRY",
        product="MIS",
    )


def _manager(*, initially_present: bool = True):
    manager = RuntimeOrderManager.__new__(RuntimeOrderManager)
    manager._logger = _Logger()
    manager._bracket_manager = _CanonicalVirtualBrackets(
        initially_present=initially_present
    )
    manager._bracket_index = {}
    manager._brackets = {}
    manager._indicator_engine = None
    manager._orders = {}
    manager._place_calls = []
    manager._register_bracket_state = lambda state: manager._brackets.update(
        {state.entry_id: state}
    )
    manager._update_entry_children = lambda *_args, **_kwargs: None
    manager._persist_bracket_state = lambda *_args, **_kwargs: None
    manager.attach_dynamic_tp = lambda **_kwargs: None

    def _place_single_order(**kwargs):
        manager._place_calls.append(dict(kwargs))
        return SimpleNamespace(
            order_id=f"legacy-child-{len(manager._place_calls)}",
            price=float(kwargs.get("price") or 0.0),
        )

    manager._place_single_order = _place_single_order
    return manager


def test_canonical_virtual_bracket_never_creates_physical_sl_or_tp_children() -> None:
    """One filled MIS entry must have exactly one exit owner: the virtual bracket."""

    manager = _manager(initially_present=True)

    manager._handle_bracket_update(
        _filled_entry(),
        OrderStatus.SUBMITTED,
        {"status": "COMPLETE"},
    )

    assert manager._place_calls == []
    assert manager._brackets == {}


def test_missing_virtual_bracket_is_recovered_before_legacy_children_are_allowed() -> None:
    """A callback-order race must recover the virtual owner, not fall back to broker BO."""

    manager = _manager(initially_present=False)
    recovery_calls = []

    def _recover(order, *, source: str):
        recovery_calls.append((order.order_id, source))
        manager._bracket_manager.bracket = SimpleNamespace(
            bracket_id=ENTRY_ID,
            entry_order_id=ENTRY_ID,
            symbol=SYMBOL,
            is_virtual=True,
            active=True,
            entry_confirmed=True,
        )

    manager._register_virtual_bracket_for_fill = _recover

    manager._handle_bracket_update(
        _filled_entry(),
        OrderStatus.SUBMITTED,
        {"status": "COMPLETE"},
    )

    assert recovery_calls == [(ENTRY_ID, "bracket_update_single_owner")]
    assert manager._place_calls == []
    assert manager._brackets == {}
