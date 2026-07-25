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


# ===================== OPTION LOT-SIZING REGRESSION =====================
# NIFTY option sizing must reason in COMPLETE LOTS and convert to broker
# units exactly once. Regression for the unit/lot mixing that rejected every
# affordable trade with `no_qty_after_risk`, and for the contract-size double
# count that inflated est_required by lot_size.


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
        # Production sets this to the lot size; sizing must NOT count it again.
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
    """Test A: one lot with lot_size=65 becomes broker quantity 65."""
    decision = _opt_engine().plan(_opt_inputs())
    assert decision.ok
    assert decision.quantity == 65


def test_option_two_lots_accepted() -> None:
    """Test B: two affordable lots become broker quantity 130."""
    decision = _opt_engine().plan(
        _opt_inputs(requested_qty=130, balance=2_000_000.0)
    )
    assert decision.ok
    assert decision.quantity == 130


def test_option_partial_lot_capacity_is_rejected_not_rounded_up() -> None:
    """Test C: capacity below one complete lot rejects; never rounds up."""
    decision = _opt_engine().plan(_opt_inputs(balance=1_000.0))
    assert not decision.ok
    assert decision.quantity == 0


def test_option_capital_permits_one_of_two_requested_lots() -> None:
    """Test D: request 2, capital/risk supports 1 -> exactly one lot."""
    decision = _opt_engine().plan(_opt_inputs(requested_qty=130))
    assert decision.ok
    assert decision.quantity == 65


def test_option_stop_risk_permits_one_of_two_lots() -> None:
    """Test E: capital allows two lots, stop-risk budget allows one."""
    decision = _opt_engine().plan(
        _opt_inputs(
            requested_qty=130, balance=2_000_000.0, per_trade_risk_pct=0.1
        )
    )
    assert decision.ok
    assert decision.quantity == 65


@pytest.mark.parametrize("lot_size", [50, 65, 75])
def test_option_broker_quantity_uses_resolved_lot_size(lot_size: int) -> None:
    """Test G: lot size is taken from the contract, never hard-coded."""
    decision = _opt_engine().plan(_opt_inputs(lot_size=lot_size))
    assert decision.ok
    assert decision.quantity == lot_size
    assert decision.quantity % lot_size == 0


def test_option_premium_cost_is_not_double_counted_by_contract_size() -> None:
    """Test H: one lot at premium 100, lot_size 65 costs ~6500, not 422500."""
    decision = _opt_engine().plan(_opt_inputs())
    assert decision.quantity == 65
    assert decision.est_required == pytest.approx(6_500.0)
    assert decision.est_required != pytest.approx(422_500.0)


def test_option_affordability_uses_selected_premium_not_underlying() -> None:
    """Test I: affordability uses the option premium (120), not spot (25000)."""
    decision = _opt_engine().plan(_opt_inputs(price=120.0, stop_loss=100.0))
    assert decision.ok
    assert decision.quantity == 65
    # 120 * 65 = 7800, nowhere near a spot-driven 25000 * 65.
    assert decision.est_required == pytest.approx(7_800.0)
