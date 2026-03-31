from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.execution.margin_engine import MarginEngine, MarginInputs


class DummyBroker:
    def __init__(self, required: float | None = None) -> None:
        self.required = required

    def get_required_margin(
        self, *, symbol: str, side: str, quantity: int, product: str
    ) -> dict[str, float]:
        if self.required is None:
            raise RuntimeError("not implemented")
        return {"required": self.required}


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


def test_plan_blocks_when_needed_gt_available() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    inputs = _inputs(balance=500.0)
    decision = engine.plan(inputs)
    assert not decision.ok
    assert decision.reason is not None and "MARGIN" in decision.reason


def test_plan_downgrades_mis_after_cutoff() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    late_inputs = _inputs(
        ist_now=datetime(2024, 1, 1, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
        product="MIS",
        balance=10_000.0,
    )
    decision = engine.plan(late_inputs)
    assert not decision.ok
    assert decision.order_type == "NRML"
    assert decision.reason == "MIS_WINDOW_CLOSED"


def test_plan_sizes_by_risk_and_caps() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    inputs = _inputs(
        price=200.0,
        stop_loss=150.0,
        requested_qty=100,
        lot_size=5,
        balance=100_000.0,
        per_trade_risk_pct=1.0,
        per_trade_cap_pct=0.5,
    )
    decision = engine.plan(inputs)
    assert decision.ok
    assert decision.quantity == 5


def test_plan_uses_broker_margin_when_available() -> None:
    engine = MarginEngine(
        broker=DummyBroker(required=250.0),
        data_hub=None,
        lot_size_resolver=None,
        clock=lambda: 0.0,
    )
    inputs = _inputs(balance=1_000.0, margin_buffer=0.5)
    decision = engine.plan(inputs)
    assert decision.ok
    assert decision.est_required == pytest.approx(250.0)
    assert decision.available == pytest.approx(1_000.0)


def test_plan_clamps_min_lot_when_margin_permits() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    inputs = _inputs(requested_qty=1, lot_size=25, balance=10_000.0)
    decision = engine.plan(inputs)
    assert decision.ok
    assert decision.quantity == 25
    assert decision.sizing is not None
    assert decision.sizing.reason == "clamped_min_lot"


def test_plan_reports_insufficient_margin_for_min_lot() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    inputs = _inputs(requested_qty=1, lot_size=25, balance=1_000.0, price=120.0)
    decision = engine.plan(inputs)
    assert not decision.ok
    assert decision.reason == "margin_no_qty"
    assert decision.quantity == 0
    assert decision.sizing is not None
    assert decision.sizing.reason == "margin_no_qty"
    assert decision.sizing.needed is not None and decision.sizing.needed > 0


def test_plan_clamps_zero_requested_with_atr_margin() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    inputs = _inputs(
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
    decision = engine.plan(inputs)
    assert decision.ok
    assert decision.quantity == 75
    assert decision.sizing is not None
    assert decision.sizing.reason == "clamped_min_lot"
