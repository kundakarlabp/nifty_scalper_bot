"""Broker-free fault injection for linked group exits and durable convergence."""

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.bracket_core import BracketManager
from nifty_scalper_bot.execution.position_manager import PositionManager

PARENT = "group-20261101"
ROLES = (
    ("long_put", 23700, "PE", "BUY", 101),
    ("long_call", 24300, "CE", "BUY", 102),
    ("short_put", 23800, "PE", "SELL", 103),
    ("short_call", 24200, "CE", "SELL", 104),
)


class FakeBroker:
    def __init__(self):
        self.orders = []
        self.net = {}
        self.exits_sent = []
        self.exit_status = "COMPLETE"
        self.exit_quantity = None
        self.lost_ack = False

    def get_positions(self):
        return {
            "net": [
                {"tradingsymbol": symbol.split(":", 1)[1], "quantity": qty}
                for symbol, qty in self.net.items()
            ]
        }

    def get_orders(self):
        return list(self.orders)

    def record(
        self, order_id, symbol, side, quantity, *, status="COMPLETE", filled=None
    ):
        amount = quantity if filled is None else filled
        self.orders.append(
            {
                "order_id": order_id,
                "tradingsymbol": symbol.split(":", 1)[1],
                "transaction_type": side,
                "quantity": quantity,
                "filled_quantity": amount,
                "status": status,
            }
        )
        if amount:
            signed = amount if side == "BUY" else -amount
            self.net[symbol] = self.net.get(symbol, 0) + signed

    def exit(self, positions, **kwargs):
        self.exits_sent.append(kwargs)
        order_id = f"exit-{len(self.exits_sent)}"
        filled = self.exit_quantity
        self.record(
            order_id,
            kwargs["symbol"],
            kwargs["side"],
            kwargs["quantity"],
            status=self.exit_status,
            filled=filled,
        )
        positions.add_pending_order(
            order_id=order_id,
            symbol=kwargs["symbol"],
            side=kwargs["side"],
            qty=kwargs["quantity"],
            price=50.0,
            order_type="MARKET",
            intent="REDUCE",
            bracket_id=kwargs["trade_lifecycle_id"],
        )
        if self.lost_ack:
            raise TimeoutError("broker exit occurred but response lost")
        return order_id


def setup(tmp_path):
    pos = PositionManager(state_file=str(tmp_path / "state.json"))
    broker = FakeBroker()
    pos.set_broker_client(broker)
    legs = [
        {
            "role": role,
            "symbol": f"NFO:NIFTY26NOV{strike}{option}",
            "side": side,
            "quantity": 65,
            "lot_size": 65,
            "strike": strike,
            "instrument_token": token,
            "expiry": "2026-11-26",
        }
        for role, strike, option, side, token in ROLES
    ]
    pos.reserve_linked_portfolio(PARENT, "IRON_CONDOR", legs)
    for i, (role, *_rest) in enumerate(ROLES, 1):
        leg = pos.get_linked_portfolio_reservation(PARENT)["legs"][role]
        oid = f"entry-{i}"
        pos.add_pending_order(
            order_id=oid,
            symbol=leg["symbol"],
            side=leg["side"],
            qty=65,
            price=100.0,
            order_type="LIMIT",
            intent="ENTRY",
            bracket_id=PARENT,
        )
        pos.bind_linked_portfolio_order(PARENT, role, oid)
        broker.record(oid, leg["symbol"], leg["side"], 65)
    manager = SimpleNamespace(
        _positions=pos,
        execution_mode="SIMULATION",
        is_live_mode=lambda: False,
        place_order=lambda **kw: broker.exit(pos, **kw),
    )
    bracket = SimpleNamespace(order_manager=manager)
    return pos, broker, bracket


def step(bracket, nonce):
    return BracketManager.execute_linked_group_flatten_step(
        bracket, PARENT, correlation_id=f"exit_{nonce}"
    )


def test_all_four_short_first_exits_then_confirmed_flat(tmp_path):
    pos, broker, bracket = setup(tmp_path)
    assert pos.linked_portfolio_broker_flat_confirmed(PARENT) is False
    for number, role in enumerate(
        ("short_put", "short_call", "long_put", "long_call"), 1
    ):
        result = step(bracket, number)
        if number < 4:
            assert result["state"] == "BROKER_EXIT_FILL_OBSERVED"
            assert result["broker_flat_confirmed"] is False
        else:
            assert result["state"] == "CLOSED_BROKER_CONFIRMED"
            assert result["broker_flat_confirmed"] is True
        assert broker.exits_sent[-1]["symbol"] == (
            pos.get_linked_portfolio_reservation(PARENT)["legs"][role]["symbol"]
        )
    assert [item["side"] for item in broker.exits_sent] == [
        "BUY", "BUY", "SELL", "SELL"
    ]
    assert pos.linked_portfolio_broker_flat_confirmed(PARENT) is True
    assert pos.get_linked_portfolio_reservation(PARENT)["state"] == (
        "CLOSED_BROKER_CONFIRMED"
    )
    observed = BracketManager.linked_portfolio_exit_convergence(bracket, PARENT)
    assert observed["exit_converged"] is True


def test_lost_exit_ack_remains_durable_after_restart_and_no_duplicate(tmp_path):
    pos, broker, bracket = setup(tmp_path)
    broker.lost_ack = True
    with pytest.raises(TimeoutError, match="response lost"):
        step(bracket, "lost")
    saved = pos.get_linked_portfolio_reservation(PARENT)
    assert saved["group_exit_attempts"]["short_put"]["state"] == "UNRESOLVED"
    restored = PositionManager(state_file=str(tmp_path / "state.json"))
    restored.set_broker_client(broker)
    bracket.order_manager._positions = restored
    with pytest.raises(RuntimeError, match="linked_group_exit_submission_ambiguous"):
        step(bracket, "blind-duplicate")
    assert len(broker.exits_sent) == 1


@pytest.mark.parametrize(
    ("status", "filled"),
    [
        ("OPEN", 0),
        ("PARTIALLY FILLED", 25),
        ("REJECTED", 0),
    ],
)
def test_open_partial_and_rejected_exit_never_retried_blindly(
    tmp_path, status, filled
):
    pos, broker, bracket = setup(tmp_path)
    broker.exit_status = status
    broker.exit_quantity = filled
    with pytest.raises(RuntimeError, match="linked_group_exit_broker_fill_unverified"):
        step(bracket, "uncertain")
    assert pos.linked_portfolio_broker_flat_confirmed(PARENT) is False
    with pytest.raises(RuntimeError):
        step(bracket, "another")
    assert len(broker.exits_sent) == 1


def test_live_exits_not_enabled_without_broker_fault_validation(tmp_path):
    pos, broker, bracket = setup(tmp_path)
    bracket.order_manager.is_live_mode = lambda: True
    with pytest.raises(RuntimeError, match="linked_group_live_execution_not_supported"):
        step(bracket, "live")
    assert broker.exits_sent == []


def test_external_order_contamination_blocks_group_dispatch(tmp_path):
    pos, broker, bracket = setup(tmp_path)
    symbol = pos.get_linked_portfolio_reservation(PARENT)["legs"]["short_put"]["symbol"]
    broker.record("manual-extra", symbol, "BUY", 65)
    with pytest.raises(RuntimeError, match="external_order_on_parent_symbol"):
        step(bracket, "manual")
    assert broker.exits_sent == []
