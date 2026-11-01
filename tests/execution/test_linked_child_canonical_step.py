"""Deterministic injected broker fills for one-child-at-a-time canonical steps."""

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.order_manager_core import (
    OrderManager,
    OrderPlacementError,
)
from nifty_scalper_bot.execution.position_manager import PositionManager

PARENT = "parent-20261101"
ROLES = (
    ("long_put", 23700, "PE", "BUY", 101),
    ("long_call", 24300, "CE", "BUY", 102),
    ("short_put", 23800, "PE", "SELL", 103),
    ("short_call", 24200, "CE", "SELL", 104),
)


class Broker:
    def __init__(self):
        self.orderbook = []
        self.positions = {"net": []}
        self.submitted = []
        self.fill_qty = 65
        self.status = "COMPLETE"
        self.timeout_after_submit = False

    def get_orders(self):
        return list(self.orderbook)

    def get_positions(self):
        return {"net": list(self.positions["net"])}

    def place_child(self, positions, **kwargs):
        self.submitted.append(kwargs)
        index = len(self.submitted)
        oid = f"broker-{index}"
        amount = self.fill_qty
        symbol = kwargs["symbol"]
        side = kwargs["side"]
        self.orderbook.append(
            {
                "order_id": oid,
                "tradingsymbol": symbol.split(":", 1)[1],
                "transaction_type": side,
                "quantity": kwargs["quantity"],
                "filled_quantity": amount,
                "status": self.status,
                "tag": "p45_" + f"bot_{kwargs['signal_id'][-12:]}"[-8:],
            }
        )
        if amount:
            self.positions["net"].append(
                {
                    "tradingsymbol": symbol.split(":", 1)[1],
                    "quantity": amount if side == "BUY" else -amount,
                }
            )
        positions.add_pending_order(
            order_id=oid,
            symbol=symbol,
            side=side,
            qty=kwargs["quantity"],
            price=50.0,
            order_type="MARKET",
            intent="ENTRY",
            bracket_id=kwargs["trade_lifecycle_id"],
        )
        if self.timeout_after_submit:
            raise TimeoutError("broker ack lost after execution")
        return oid


def setup(tmp_path):
    positions = PositionManager(state_file=str(tmp_path / "state.json"))
    broker = Broker()
    positions.set_broker_client(broker)
    positions.reserve_linked_portfolio(
        PARENT,
        "IRON_CONDOR",
        [
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
        ],
    )
    manager = SimpleNamespace(
        execution_mode="PAPER",
        is_live_mode=lambda: False,
        _positions=positions,
        screen_linked_portfolio_entry=lambda *a, **k: {
            "risk_screen_passed": True
        },
        place_order=lambda **kw: broker.place_child(positions, **kw),
    )
    return manager, positions, broker


def step(owner, role, nonce):
    return OrderManager.execute_linked_child_step(
        owner,
        PARENT,
        role,
        correlation_id=f"p45_{nonce}",
        product="NRML",
        gross_loss_bound=7000,
        entry_fee_reserve=100,
        exit_fee_reserve=200,
        stressed_slippage_reserve=400,
    )


def test_verified_hedge_first_progression_and_four_full_broker_fills(tmp_path):
    owner, positions, broker = setup(tmp_path)
    for idx, role in enumerate(
        ("long_put", "long_call", "short_put", "short_call"), 1
    ):
        result = step(owner, role, str(idx))
        assert result["broker_order_id"] == f"broker-{idx}"
        assert result["state"] == "BROKER_FILLED_OBSERVED"
        assert result["live_entry_allowed"] is False
    assert len(broker.submitted) == 4
    assert [x["side"] for x in broker.submitted] == [
        "BUY", "BUY", "SELL", "SELL"
    ]
    assert positions.linked_portfolio_broker_state(PARENT)[
        "broker_portfolio_confirmed"
    ] is True


def test_short_before_protective_wings_never_reaches_broker(tmp_path):
    owner, positions, broker = setup(tmp_path)
    with pytest.raises(OrderPlacementError):
        step(owner, "short_put", "short-early")
    assert broker.submitted == []


def test_partial_fill_triggers_abort_and_no_other_child(tmp_path):
    owner, positions, broker = setup(tmp_path)
    broker.fill_qty = 25
    broker.status = "PARTIALLY FILLED"
    with pytest.raises(
        OrderPlacementError, match="linked_child_fill_not_broker_confirmed"
    ):
        step(owner, "long_put", "partial")
    assert positions.get_linked_portfolio_reservation(PARENT)["state"] == (
        "ABORT_RECOVERY_REQUIRED"
    )
    assert len(broker.submitted) == 1
    with pytest.raises(OrderPlacementError):
        step(owner, "long_call", "after-partial")
    assert len(broker.submitted) == 1


def test_timeout_after_broker_fill_survives_restart_without_duplicate(tmp_path):
    owner, positions, broker = setup(tmp_path)
    broker.timeout_after_submit = True
    with pytest.raises(TimeoutError, match="ack lost"):
        step(owner, "long_put", "timedout")
    saved = positions.get_linked_portfolio_reservation(PARENT)
    assert saved["state"] == "ABORT_RECOVERY_REQUIRED"
    assert saved["child_attempts"]["long_put"]["state"] == "UNRESOLVED"
    restored = PositionManager(state_file=str(tmp_path / "state.json"))
    restored.set_broker_client(broker)
    owner._positions = restored
    with pytest.raises(OrderPlacementError):
        step(owner, "long_put", "duplicate")
    assert len(broker.submitted) == 1


def test_acknowledged_order_does_not_prove_full_fill(tmp_path):
    owner, positions, broker = setup(tmp_path)
    broker.fill_qty = 0
    broker.status = "OPEN"
    with pytest.raises(
        OrderPlacementError, match="linked_child_fill_not_broker_confirmed"
    ):
        step(owner, "long_put", "open")
    assert positions.get_linked_portfolio_reservation(PARENT)["state"] == (
        "ABORT_RECOVERY_REQUIRED"
    )


def test_simulation_is_never_live_broker_enabled(tmp_path):
    owner, positions, broker = setup(tmp_path)
    owner.is_live_mode = lambda: True
    with pytest.raises(
        OrderPlacementError, match="linked_child_live_execution_not_supported"
    ):
        step(owner, "long_put", "live")
    assert broker.submitted == []
