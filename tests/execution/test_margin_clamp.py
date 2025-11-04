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


def test_margin_clamp_promotes_minimum_lot_when_affordable() -> None:
    engine = MarginEngine(
        broker=DummyBroker(),
        data_hub=DummyDataHub(),
        lot_size_resolver=DummyLotResolver(),
        clock=None,
    )
    inputs = _build_inputs(balance=50_000.0, requested_qty=0, lot_size=25)
    decision = engine.plan(inputs)
    min_qty = max(1, app_settings.MIN_LOTS_PER_TRADE) * inputs.lot_size
    assert decision.ok is True
    assert decision.quantity == min_qty
    assert decision.sizing is not None
    assert decision.sizing.qty == min_qty


def test_margin_clamp_returns_margin_no_qty_when_insufficient() -> None:
    engine = MarginEngine(
        broker=DummyBroker(),
        data_hub=DummyDataHub(),
        lot_size_resolver=DummyLotResolver(),
        clock=None,
    )
    inputs = _build_inputs(balance=500.0, requested_qty=0, lot_size=25)
    decision = engine.plan(inputs)
    assert decision.ok is False
    assert decision.reason == "margin_no_qty"
    assert decision.sizing is not None
    assert decision.sizing.reason == "margin_no_qty"
