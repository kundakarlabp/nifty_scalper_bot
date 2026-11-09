"""Regression guards for strict quote-verified replay; no broker/network use."""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from nifty_scalper_bot.backtesting.execution_evidence import (
    BookLevel, ObservedBook, OrderIntent, classify_barrier,
    execution_readiness, historical_option_fee, replay_market_order,
)

DAY = date(2025, 3, 6)
T0 = datetime(2025, 3, 6, 9, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def book(*, observed_at=None, token="101", sizes=(75, 75, 75, 75, 75)):
    return ObservedBook(
        symbol="NFO:NIFTY25MAR23000CE", token=token,
        expiry=date(2025, 3, 27), lot_size=75,
        observed_at=observed_at or T0 + timedelta(milliseconds=150),
        bid=tuple(BookLevel(100 - i * .05, size) for i, size in enumerate(sizes)),
        ask=tuple(BookLevel(100.20 + i * .05, size) for i, size in enumerate(sizes)),
        source="ARCHIVED_FULL", full_depth_captured=True,
        historical_identity_verified=True, clock_convention_verified=True,
    )


def intent(*, side="BUY", qty=75, latency_ms=100, **overrides):
    row = dict(symbol="NFO:NIFTY25MAR23000CE", token="101",
               expiry=date(2025, 3, 27), lot_size=75, decision_at=T0,
               side=side, quantity=qty, latency_ms=latency_ms,
               short_margin_verified=False)
    row.update(overrides)
    return OrderIntent(**row)


def run(order=None, quotes=None, **kw):
    return replay_market_order(order or intent(), [book()] if quotes is None else quotes,
                               historical_exchange_rate=0.0003503, **kw)


def test_causal_buy_crosses_actual_ask_and_charges_fees():
    result = run(cash_available=10000)
    assert result.status == "MODELED_DEPTH"
    assert result.price == pytest.approx(100.2)
    assert result.fees > 20 and result.filled_quantity == 75
    assert not result.live_eligible


def test_no_depth_never_fills_from_ohlcv_or_ltp():
    result = run(quotes=[])
    assert result.status == "UNRESOLVED"
    assert result.price is None


@pytest.mark.parametrize("defect", [
    {"full_depth_captured": False},
    {"clock_convention_verified": False},
    {"historical_identity_verified": False},
    {"source": "DUMMY"},
    {"bid": (BookLevel(99, 75),)},
    {"ask": tuple(BookLevel(99.0 + i * .05, 75) for i in range(5))},
])
def test_invalid_source_depth_or_crossed_book_blocks(defect):
    assert run(quotes=[replace(book(), **defect)]).reason == "invalid_or_unverified_depth"


def test_order_after_latency_only_cannot_lookahead():
    early = book(observed_at=T0 + timedelta(milliseconds=50))
    late = book(observed_at=T0 + timedelta(milliseconds=200))
    a = run(quotes=[early, late])
    assert a.status == "MODELED_DEPTH"
    assert a.observed_at == late.observed_at
    assert run(quotes=[early]).reason == "no_fresh_quote_after_latency"


def test_quotes_beyond_deadline_do_not_fill():
    assert run(quotes=[book(observed_at=T0+timedelta(seconds=3))]).status == "UNRESOLVED"


def test_duplicate_quote_clock_order_unresolved():
    x=book()
    assert run(quotes=[x, x]).reason == "ambiguous_quote_order"


def test_exact_token_and_contract_identity_required():
    assert run(quotes=[book(token="999")]).reason == "point_in_time_contract_mismatch"


def test_partial_five_level_depth_never_fabricates_remaining_quantity():
    res=run(order=intent(qty=750), quotes=[book(sizes=(75,75,75,75,75))])
    assert res.status == "PARTIAL"
    assert res.filled_quantity == 375 and res.remaining_quantity == 375


def test_missing_short_option_margin_blocked():
    assert run(order=intent(side="SELL")).reason == "short_margin_evidence_missing"
    res=run(order=intent(side="SELL", short_margin_verified=True))
    assert res.status == "MODELED_DEPTH" and res.price == 100.0


def test_tick_rounding_and_adverse_slippage_never_improve_market_price():
    baseline = run()
    stress = run(adverse_slippage_bps=25)
    assert stress.price > baseline.price
    sell0 = run(order=intent(side="SELL",short_margin_verified=True))
    sell1 = run(order=intent(side="SELL",short_margin_verified=True), adverse_slippage_bps=25)
    assert sell1.price < sell0.price


def test_limit_price_and_cash_fail_closed():
    assert run(limit_price=100.15).reason == "no_executable_depth_within_limit"
    assert run(cash_available=10.0).reason == "insufficient_cash_including_fees"


def test_lot_aware_and_expiry_aware():
    assert run(order=intent(qty=1)).status == "UNRESOLVED"
    assert run(order=intent(expiry=date(2025, 2, 1))).status == "UNRESOLVED"


def test_historical_stt_repricing_at_2026_04_01():
    common=dict(side="SELL",price=150.0,quantity=75,exchange_rate=0.0003503)
    old=historical_option_fee(day=date(2026,3,31),**common)
    new=historical_option_fee(day=date(2026,4,1),**common)
    assert new-old == pytest.approx(150*75*.0005)
    with pytest.raises(ValueError,match="historical_fee_inputs_unverified"):
        historical_option_fee(day=DAY,side="BUY",price=100,quantity=75,exchange_rate=0)


def test_unverified_price_on_intrabar_double_touch_is_unresolved():
    assert classify_barrier(opening=100, high=112, low=92, closing=103, stop=95, target=110) == "AMBIGUOUS_UNRESOLVED"
    assert classify_barrier(opening=90,high=112,low=87,closing=100,stop=95,target=110) == "STOP_GAP"
    assert classify_barrier(opening=100,high=110,low=99,closing=101,stop=95,target=110) == "TARGET_TOUCHED_UNPRICED"


def test_evidence_gate_never_auto_promotes():
    a=execution_readiness(point_in_time_contracts=False,verified_clocks=False,
                          five_level_quotes=False,actual_broker_fills=False,
                          exit_ledger_verified=False,chronological_oos=False)
    assert not a["execution_grade_ready"]
    assert len(a["missing_evidence"]) == 6
    b=execution_readiness(point_in_time_contracts=True,verified_clocks=True,
                          five_level_quotes=True,actual_broker_fills=True,
                          exit_ledger_verified=True,chronological_oos=True)
    assert b["execution_grade_ready"] and not b["automatic_live_promotion"]
