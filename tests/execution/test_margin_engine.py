from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.execution.margin_engine import MarginEngine, MarginInputs


class DummyBroker:
    def __init__(self, required: float | dict[int, float] | None = None) -> None:
        self.required = required
        self.quantities: list[int] = []

    def get_required_margin(
        self, *, symbol: str, side: str, quantity: int, product: str
    ) -> dict[str, float]:
        self.quantities.append(quantity)
        if self.required is None:
            raise RuntimeError("not implemented")
        required = (
            self.required.get(quantity, 0.0)
            if isinstance(self.required, dict)
            else self.required
        )
        return {"required": required}


def _inputs(**overrides: float | int | str | None) -> MarginInputs:
    base = {
        "symbol": "NIFTY",
        "side": "BUY",
        "price": 100.0,
        "stop_loss": 90.0,
        "atr": None,
        "requested_qty": 10,
        "product": "NRML",
        "lot_size": 1,
        "balance": 1000.0,
        "per_trade_risk_pct": 100.0,
        "per_trade_cap_pct": 1000.0,
        "margin_factor": 1.0,
        "margin_buffer": 1.0,
        "contract_multiplier": 1.0,
        "ist_now": datetime(2024, 1, 1, 9, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
        "min_lots_per_trade": 1,
        "max_lots_per_trade": 10,
        "atr_multiple": 1.0,
    }
    base.update(overrides)
    return MarginInputs(**base)  # type: ignore[arg-type]


def test_plan_steps_down_fallback_margin_to_affordable_quantity() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    decision = engine.plan(_inputs(balance=500.0))
    assert decision.ok
    assert decision.quantity == 5
    assert decision.est_required == pytest.approx(500.0)


def test_plan_rejects_mis_after_cutoff_without_product_rewrite() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    decision = engine.plan(
        _inputs(
            ist_now=datetime(2024, 1, 1, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
            product="MIS",
            balance=10_000.0,
        )
    )
    assert not decision.ok
    assert decision.order_type == "MIS"
    assert decision.reason == "MIS_WINDOW_CLOSED"


def test_one_lot_is_not_zeroed_only_by_percentage_cap() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    decision = engine.plan(
        _inputs(
            price=200.0,
            stop_loss=150.0,
            requested_qty=5,
            lot_size=5,
            balance=100_000.0,
            per_trade_risk_pct=1.0,
            per_trade_cap_pct=0.5,
        )
    )
    assert decision.ok
    assert decision.quantity == 5


def test_cap_still_limits_additional_lots() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    decision = engine.plan(
        _inputs(
            price=200.0,
            stop_loss=150.0,
            requested_qty=10,
            lot_size=5,
            balance=100_000.0,
            per_trade_risk_pct=10.0,
            per_trade_cap_pct=1.5,
        )
    )
    assert decision.ok
    assert decision.quantity == 5


def test_plan_uses_broker_margin_when_available() -> None:
    engine = MarginEngine(
        broker=DummyBroker(required=250.0),
        data_hub=None,
        lot_size_resolver=None,
        clock=lambda: 0.0,
    )
    decision = engine.plan(_inputs(balance=1_000.0, margin_buffer=0.5))
    assert decision.ok
    assert decision.est_required == pytest.approx(250.0)
    assert decision.available == pytest.approx(1_000.0)


def test_plan_fails_closed_when_concrete_broker_margin_call_fails() -> None:
    decision = MarginEngine(
        broker=DummyBroker(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    ).plan(_inputs())
    assert not decision.ok
    assert decision.reason == "broker_margin_unavailable"
    assert decision.quantity == 0


def test_plan_fails_closed_on_malformed_broker_margin_payload() -> None:
    class MalformedBroker:
        def get_required_margin(self, **_kwargs):
            return {"required": 0}

    decision = MarginEngine(
        broker=MalformedBroker(),
        data_hub=None,
        lot_size_resolver=None,
        clock=lambda: 0.0,
    ).plan(_inputs())
    assert not decision.ok
    assert decision.reason == "broker_margin_unavailable"
    assert decision.quantity == 0


def test_plan_steps_down_by_whole_lots_until_broker_margin_fits() -> None:
    broker = DummyBroker(required={15: 1_500.0, 10: 900.0})
    engine = MarginEngine(
        broker=broker, data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    decision = engine.plan(
        _inputs(
            price=10.0,
            stop_loss=9.0,
            requested_qty=15,
            lot_size=5,
            balance=1_000.0,
            per_trade_risk_pct=100.0,
            per_trade_cap_pct=100.0,
        )
    )
    assert decision.ok
    assert decision.quantity == 10
    assert decision.est_required == pytest.approx(900.0)
    assert broker.quantities == [15, 10]


def test_plan_rejects_positive_non_lot_request() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    decision = engine.plan(_inputs(requested_qty=1, lot_size=25))
    assert not decision.ok
    assert decision.quantity == 0
    assert decision.reason == "invalid_lot_quantity"


def test_plan_rejects_zero_requested_quantity_without_min_lot_promotion() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    decision = engine.plan(
        _inputs(
            requested_qty=0,
            lot_size=75,
            balance=100_000.0,
            per_trade_risk_pct=5.0,
            per_trade_cap_pct=10.0,
            atr=40.0,
            contract_multiplier=1.0,
            min_lots_per_trade=1,
            max_lots_per_trade=1,
        )
    )
    assert not decision.ok
    assert decision.quantity == 0
    assert decision.reason == "invalid_requested_quantity"


@pytest.mark.parametrize("requested_qty", [0, -1, -65])
def test_plan_rejects_non_positive_requested_quantity(requested_qty: int) -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    decision = engine.plan(_inputs(requested_qty=requested_qty, lot_size=65))
    assert not decision.ok
    assert decision.quantity == 0
    assert decision.reason == "invalid_requested_quantity"


def _opt_inputs(**overrides):
    lot = overrides.pop("lot_size", 65)
    base = {
        "symbol": "NFO:NIFTY26AUG25000CE",
        "side": "BUY",
        "price": 100.0,
        "stop_loss": 80.0,
        "atr": None,
        "requested_qty": lot,
        "product": "NRML",
        "lot_size": lot,
        "balance": 200_000.0,
        "per_trade_risk_pct": 1.0,
        "per_trade_cap_pct": 100.0,
        "margin_factor": 1.0,
        "margin_buffer": 1.0,
        "contract_multiplier": float(lot),
        "ist_now": datetime(2024, 1, 1, 9, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
        "min_lots_per_trade": 1,
        "max_lots_per_trade": 10,
        "atr_multiple": 1.5,
    }
    base.update(overrides)
    return MarginInputs(**base)  # type: ignore[arg-type]


def _opt_engine() -> MarginEngine:
    return MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )


def test_option_one_lot_accepted_as_broker_units() -> None:
    decision = _opt_engine().plan(_opt_inputs())
    assert decision.ok
    assert decision.quantity == 65


def test_option_two_lots_accepted() -> None:
    decision = _opt_engine().plan(
        _opt_inputs(requested_qty=130, balance=2_000_000.0)
    )
    assert decision.ok
    assert decision.quantity == 130


def test_option_partial_lot_capacity_is_rejected_not_rounded_up() -> None:
    decision = _opt_engine().plan(_opt_inputs(balance=1_000.0))
    assert not decision.ok
    assert decision.quantity == 0


def test_option_capital_permits_one_of_two_requested_lots() -> None:
    decision = _opt_engine().plan(_opt_inputs(requested_qty=130))
    assert decision.ok
    assert decision.quantity == 65


def test_option_stop_risk_permits_one_of_two_lots() -> None:
    decision = _opt_engine().plan(
        _opt_inputs(
            requested_qty=130, balance=2_000_000.0, per_trade_risk_pct=0.1
        )
    )
    assert decision.ok
    assert decision.quantity == 65


@pytest.mark.parametrize("lot_size", [50, 65, 75])
def test_option_broker_quantity_uses_resolved_lot_size(lot_size: int) -> None:
    decision = _opt_engine().plan(_opt_inputs(lot_size=lot_size))
    assert decision.ok
    assert decision.quantity == lot_size
    assert decision.quantity % lot_size == 0


def test_option_premium_cost_is_not_double_counted_by_contract_size() -> None:
    decision = _opt_engine().plan(_opt_inputs())
    assert decision.quantity == 65
    assert decision.est_required == pytest.approx(6_500.0)
    assert decision.est_required != pytest.approx(422_500.0)


def test_option_affordability_uses_selected_premium_not_underlying() -> None:
    decision = _opt_engine().plan(_opt_inputs(price=120.0, stop_loss=100.0))
    assert decision.ok
    assert decision.quantity == 65
    assert decision.est_required == pytest.approx(7_800.0)
