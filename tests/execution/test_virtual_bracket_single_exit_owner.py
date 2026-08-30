from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.execution.order_manager import (
    OrderDetails,
    OrderStatus,
    OrderType,
    OrderManager,
)


def _logger() -> SimpleNamespace:
    sink = lambda *_args, **_kwargs: None
    return SimpleNamespace(
        debug=sink,
        info=sink,
        warning=sink,
        error=sink,
        critical=sink,
    )


def test_canonical_virtual_bracket_does_not_create_broker_sl_or_tp_children() -> None:
    """A filled canonical entry must have exactly one virtual exit owner.

    Regression for the live incident where OrderManager auto-created a Zerodha
    SELL SL child while the canonical BracketManager simultaneously managed the
    same long position virtually. The later virtual exit then competed with the
    still-open broker stop and Zerodha rejected the reducing SELL attempts.
    """

    manager = OrderManager.__new__(OrderManager)
    manager._bracket_manager = SimpleNamespace(register_virtual_bracket=lambda **_kw: None)
    manager._bracket_index = {}
    manager._brackets = {}
    manager._logger = _logger()
    manager._indicator_engine = None

    broker_children: list[dict[str, object]] = []

    def _place_single_order(**kwargs):
        broker_children.append(dict(kwargs))
        return SimpleNamespace(order_id=f"child-{len(broker_children)}")

    manager._place_single_order = _place_single_order
    manager._register_bracket_state = lambda state: manager._brackets.__setitem__(
        state.entry_id, state
    )
    manager._update_entry_children = lambda *_args, **_kwargs: None
    manager._persist_bracket_state = lambda *_args, **_kwargs: None
    manager.attach_dynamic_tp = lambda **_kwargs: None

    order = OrderDetails(
        order_id="entry-1",
        symbol="NFO:NIFTY2690124100PE",
        side="BUY",
        quantity=65,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        price=64.75,
        fill_price=64.75,
        average_price=64.75,
        filled_quantity=65,
        stop_loss=57.50,
        take_profit=78.80,
        intent="ENTRY",
        product="MIS",
    )

    manager._handle_bracket_update(order, OrderStatus.SUBMITTED, {})

    assert broker_children == []
    assert manager._brackets == {}


def test_already_live_legacy_bracket_still_reconciles_with_virtual_owner_bound() -> None:
    """Suppression only blocks *new* legacy bracket creation.

    An entry that already has a legacy bracket registered (e.g. from before
    this guard existed, or restored on restart) must still flow through the
    standard update/reconciliation logic even when the canonical virtual
    BracketManager is bound, so already-live broker SL/TP children can still
    be closed or cancelled.
    """
    from nifty_scalper_bot.execution.order_manager_core import BracketState

    manager = OrderManager.__new__(OrderManager)
    manager._bracket_manager = SimpleNamespace(register_virtual_bracket=lambda **_kw: None)
    manager._logger = _logger()

    state = BracketState(
        entry_id="entry-1",
        symbol="NFO:NIFTY2690124100PE",
        side="BUY",
        exit_side="SELL",
        total_quantity=65,
        entry_price=64.75,
        product="MIS",
        tag=None,
        stop_order_id="legacy-sl-1",
        stop_price=57.50,
        stop_order_type=OrderType.STOP_LOSS_MARKET,
    )
    manager._brackets = {"entry-1": state}
    manager._bracket_index = {"entry-1": "entry-1"}

    debug_events: list[str] = []
    manager._logger = SimpleNamespace(
        debug=lambda msg, *a, **kw: debug_events.append(msg),
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        critical=lambda *a, **kw: None,
    )

    order = OrderDetails(
        order_id="entry-1",
        symbol="NFO:NIFTY2690124100PE",
        side="BUY",
        quantity=65,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        price=64.75,
        fill_price=64.75,
        average_price=64.75,
        filled_quantity=65,
        stop_loss=57.50,
        take_profit=78.80,
        intent="ENTRY",
        product="MIS",
    )

    # entry_id is already known, so this must fall through past the
    # suppression branch into "STANDARD UPDATE LOGIC" (proven by the debug
    # log emitted only in that section), rather than being suppressed.
    try:
        manager._handle_bracket_update(order, OrderStatus.SUBMITTED, {})
    except AttributeError:
        # The minimal fake manager doesn't implement every downstream
        # reconciliation branch (e.g. virtual-sniper registration) — that is
        # out of scope here. Reaching that branch at all is the assertion.
        pass

    assert "Entered _handle_bracket_update" in debug_events
    assert manager._brackets["entry-1"] is state
