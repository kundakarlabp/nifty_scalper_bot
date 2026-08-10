from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from nifty_scalper_bot.execution import bracket_ownership_extension as bracket_owner
from nifty_scalper_bot.execution import position_identity_extension as position_owner
from nifty_scalper_bot.execution import position_manager as position_module
from nifty_scalper_bot.strategies.orchestrator import StrategyAllocation, StrategyOrchestrator


def test_broker_sync_missing_average_price_is_unresolved_not_ltp_cost_basis():
    manager = SimpleNamespace(_positions={})
    prepared, unresolved = position_owner._prepare_broker_positions(
        manager,
        [
            {
                "tradingsymbol": "NIFTY2670724250PE",
                "quantity": 65,
                "average_price": 0,
                "last_price": 88.55,
                "product": "MIS",
            }
        ],
    )

    assert unresolved == {"NFO:NIFTY2670724250PE"}
    assert prepared[0]["symbol"] == "NFO:NIFTY2670724250PE"
    assert prepared[0]["tradingsymbol"] == "NFO:NIFTY2670724250PE"
    assert prepared[0]["average_price"] == 0


def test_broker_sync_uses_existing_local_entry_when_average_price_missing():
    manager = SimpleNamespace(
        _positions={
            "NFO:NIFTY2670724250PE": SimpleNamespace(entry_price=75.0),
        }
    )
    prepared, unresolved = position_owner._prepare_broker_positions(
        manager,
        [
            {
                "tradingsymbol": "NIFTY2670724250PE",
                "quantity": 65,
                "average_price": 0,
                "last_price": 88.55,
                "product": "MIS",
            }
        ],
    )

    assert unresolved == set()
    assert prepared[0]["average_price"] == 75.0


def test_same_side_broker_rebuild_preserves_bot_owned_lifecycle_identity():
    symbol = "NFO:NIFTY2681124500CE"
    opened_at = datetime(2026, 8, 10, 9, 22, 13, tzinfo=timezone.utc)
    owned = SimpleNamespace(
        symbol=symbol,
        side="LONG",
        quantity=65,
        entry_price=133.50,
        current_price=134.00,
        entry_time=opened_at,
        order_id="2086744920689139712",
        stop_loss=122.80,
        take_profit=154.85,
        trailing_stop_distance=2.5,
        state="OPEN_ACTIVE",
    )
    manager = SimpleNamespace(_positions={symbol: owned})
    snapshot = position_owner._snapshot_owned_position_lifecycle(manager)

    rebuilt = SimpleNamespace(
        symbol=symbol,
        side="LONG",
        quantity=50,
        entry_price=134.25,
        current_price=135.00,
        entry_time=datetime(2026, 8, 10, 9, 23, tzinfo=timezone.utc),
        order_id=None,
        stop_loss=None,
        take_profit=None,
        trailing_stop_distance=None,
        state=None,
    )
    manager._positions = {symbol: rebuilt}

    restored = position_owner._restore_owned_position_lifecycle(manager, snapshot)

    assert restored == 1
    assert rebuilt.quantity == 50
    assert rebuilt.entry_price == 134.25
    assert rebuilt.current_price == 135.00
    assert rebuilt.entry_time == opened_at
    assert rebuilt.order_id == "2086744920689139712"
    assert rebuilt.stop_loss == 122.80
    assert rebuilt.take_profit == 154.85
    assert rebuilt.trailing_stop_distance == 2.5
    assert rebuilt.state == "OPEN_ACTIVE"


def test_reversed_broker_position_does_not_inherit_prior_entry_identity():
    symbol = "NFO:NIFTY2681124500CE"
    manager = SimpleNamespace(
        _positions={
            symbol: SimpleNamespace(
                symbol=symbol,
                side="LONG",
                quantity=65,
                order_id="ENTRY-1",
                entry_time=datetime.now(timezone.utc),
                stop_loss=120.0,
                take_profit=150.0,
                trailing_stop_distance=2.0,
                state="OPEN_ACTIVE",
            )
        }
    )
    snapshot = position_owner._snapshot_owned_position_lifecycle(manager)
    reversed_position = SimpleNamespace(
        symbol=symbol,
        side="SHORT",
        quantity=65,
        order_id=None,
        entry_time=datetime.now(timezone.utc),
        stop_loss=None,
        take_profit=None,
        trailing_stop_distance=None,
        state=None,
    )
    manager._positions = {symbol: reversed_position}

    restored = position_owner._restore_owned_position_lifecycle(manager, snapshot)

    assert restored == 0
    assert reversed_position.order_id is None


def _position(*, order_id: str | None) -> position_module.Position:
    return position_module.Position(
        symbol="NFO:NIFTY2681124500CE",
        side="LONG",
        quantity=65,
        entry_price=133.50,
        entry_time=datetime(2026, 8, 10, 9, 22, 13, tzinfo=timezone.utc),
        current_price=133.50,
        order_id=order_id,
    )


def test_position_order_identity_exposes_bot_managed_ownership_to_orchestrator():
    assert _position(order_id="ENTRY-1").strategy_name == "BotManaged"
    assert _position(order_id=None).strategy_name == ""


def test_bot_managed_position_counts_toward_strategy_capital_headroom():
    orchestrator = StrategyOrchestrator(
        risk_manager=SimpleNamespace(current_balance=11381.0),
    )
    allocation = StrategyAllocation(capital_fraction=0.15, tags=())
    managed = _position(order_id="ENTRY-1")
    position_manager = SimpleNamespace(get_all_positions=lambda: [managed])

    assert orchestrator._has_capital_headroom(allocation, position_manager) is False

    managed.order_id = None
    assert orchestrator._has_capital_headroom(allocation, position_manager) is True


def test_existing_nonterminal_bracket_owns_canonical_symbol():
    existing = SimpleNamespace(
        entry_order_id="ENTRY-1",
        symbol="NFO:NIFTY2670724250PE",
        remaining_quantity=65,
        exit_executed=False,
        exit_state="OPEN_ACTIVE",
    )
    manager = SimpleNamespace(_brackets={"ENTRY-1": existing})

    assert bracket_owner._existing_owner(manager, "NIFTY2670724250PE", "ENTRY-2") is existing
    assert bracket_owner._existing_owner(manager, "NFO:NIFTY2670724250PE", "ENTRY-1") is None
