from __future__ import annotations

from datetime import datetime

import nifty_scalper_bot.config.settings as app_settings
from nifty_scalper_bot.execution.margin_engine import MarginEngine, MarginInputs


class DummyBroker:
    pass


class DummyDataHub:
    pass


class DummyLotResolver:
    pass


def _build_inputs(
    *,
    balance: float,
    requested_qty: int,
    lot_size: int = 50,
    ist_now: datetime | None = None,
) -> MarginInputs:
    session_time = ist_now or datetime(2024, 1, 1, 9, 30, 0)
    return MarginInputs(
        symbol="NFO:NIFTY24OCT20000CE",
        side="BUY",
        price=200.0,
        stop_loss=None,
        atr=None,
        requested_qty=requested_qty,
        product="MIS",
        lot_size=lot_size,
        balance=balance,
        per_trade_risk_pct=100.0,
        per_trade_cap_pct=1_000.0,
        margin_factor=1.0,
        margin_buffer=1.0,
        contract_multiplier=1.0,
        ist_now=session_time,
        min_lots_per_trade=1,
        max_lots_per_trade=5,
        atr_multiple=0.0,
    )


def test_margin_clamp_rejects_zero_request_instead_of_promoting_minimum_lot() -> None:
    """Zero requested quantity must fail closed, not be clamped up.

    Minimum-lot clamping remains valid for a POSITIVE request (covered by
    test_margin_clamp_promotes_minimum_lot_for_positive_request below); it
    must never manufacture live exposure from an empty request.
    """
    engine = MarginEngine(
        broker=DummyBroker(),
        data_hub=DummyDataHub(),
        lot_size_resolver=DummyLotResolver(),
        clock=None,
    )
    inputs = _build_inputs(balance=50_000.0, requested_qty=0, lot_size=25)
    decision = engine.plan(inputs)
    assert decision.ok is False
    assert decision.quantity == 0
    assert decision.reason == "invalid_requested_quantity"


def test_margin_clamp_promotes_minimum_lot_for_positive_request() -> None:
    """Minimum-lot capacity handling is preserved for a positive request."""
    engine = MarginEngine(
        broker=DummyBroker(),
        data_hub=DummyDataHub(),
        lot_size_resolver=DummyLotResolver(),
        clock=None,
    )
    inputs = _build_inputs(balance=50_000.0, requested_qty=25, lot_size=25)
    decision = engine.plan(inputs)
    assert decision.ok is True
    assert decision.quantity == 25


def test_margin_clamp_rejects_with_specific_reason_when_insufficient() -> None:
    engine = MarginEngine(
        broker=DummyBroker(),
        data_hub=DummyDataHub(),
        lot_size_resolver=DummyLotResolver(),
        clock=None,
    )
    inputs = _build_inputs(balance=100.0, requested_qty=25, lot_size=25)
    decision = engine.plan(inputs)
    assert decision.ok is False
    assert decision.reason == "MARGIN no_qty_after_risk"
    assert decision.quantity == 0
    assert decision.sizing is not None
    assert decision.sizing.reason == "insufficient_risk_capacity"
