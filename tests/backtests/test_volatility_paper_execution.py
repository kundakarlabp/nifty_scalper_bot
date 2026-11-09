"""Broker-free simulated fills only, never evidence of historical profitability."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from nifty_scalper_bot.backtesting.volatility_paper_execution import (
    PaperBasketRehearsal,
    PaperBudget,
    PaperLeg,
    PaperQuote,
)

NOW = datetime(2026, 11, 9, 10, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
EXPIRY = "2026-11-09"


def leg(role, side, strike, right, token):
    return PaperLeg(
        role=role,
        symbol=f"NFO:NIFTY26NOV{strike}{right}",
        token=token,
        side=side,
        quantity=65,
        expiry=EXPIRY,
        strike=strike,
        option_type=right,
    )


def condor_legs():
    return {
        "long_put": leg("long_put", "BUY", 23900, "PE", 11),
        "short_put": leg("short_put", "SELL", 23950, "PE", 12),
        "short_call": leg("short_call", "SELL", 24050, "CE", 13),
        "long_call": leg("long_call", "BUY", 24100, "CE", 14),
    }


def budget():
    return PaperBudget(
        available_margin=100_000,
        required_margin=55_000,
        stress_loss_including_costs=5_000,
        remaining_trade_risk=6_000,
        remaining_daily_risk=7_000,
        observed_at=NOW,
    )


def engine(structure="IRON_CONDOR", legs=None, **kwargs):
    return PaperBasketRehearsal(
        parent_id="paper-test-1",
        structure=structure,
        legs=condor_legs() if legs is None else legs,
        budget=budget(),
        at=NOW,
        **kwargs,
    )


def quote(leg, **kwargs):
    return PaperQuote(
        symbol=leg.symbol,
        token=leg.token,
        bid=98,
        ask=102,
        bid_quantity=65,
        ask_quantity=65,
        observed_at=NOW,
        **kwargs,
    )


def test_sequential_hedge_first_and_short_first_exit():
    b = engine()
    roles = []
    for i in range(4):
        intent = b.next_intent()
        roles.append(intent.role)
        assert (i < 2 and intent.side == "BUY") or (
            i >= 2 and intent.side == "SELL"
        )
        b.apply_paper_fill(
            intent,
            quote(b.legs[intent.role]),
            quantity=65,
            event_id=f"entry-{i}",
            at=NOW,
        )
    assert roles == ["long_call", "long_put", "short_call", "short_put"]
    assert b.next_intent() is None
    assert b.phase == "OPEN"
    b.request_group_exit()
    exits = []
    for i in range(4):
        intent = b.next_intent()
        exits.append(intent.role)
        b.apply_paper_fill(
            intent,
            quote(b.legs[intent.role]),
            quantity=65,
            event_id=f"exit-{i}",
            at=NOW,
        )
    assert exits == ["short_call", "short_put", "long_call", "long_put"]
    assert b.next_intent() is None
    assert b.phase == "FLAT"
    assert len(b.fills) == 8
    assert {fill.source for fill in b.fills} == {"PAPER"}
    assert not b.live_eligible


def test_partial_fill_same_intent_and_idempotent_replay():
    b = engine()
    first = b.next_intent()
    filled = b.apply_paper_fill(
        first, quote(b.legs[first.role]), quantity=25, event_id="p1", at=NOW
    )
    same = b.next_intent()
    assert same.intent_id == first.intent_id
    assert same.quantity == 40
    assert b.apply_paper_fill(
        first, quote(b.legs[first.role]), quantity=25, event_id="p1", at=NOW
    ) == filled
    assert b.entry[first.role] == 25
    b.apply_paper_fill(
        same, quote(b.legs[first.role]), quantity=40, event_id="p2", at=NOW
    )
    assert b.entry[first.role] == 65
    assert b.next_intent().role == "long_put"
    with pytest.raises(ValueError, match="duplicate_event_conflict"):
        b.apply_paper_fill(
            same, quote(b.legs[first.role]), quantity=10, event_id="p2", at=NOW
        )


def test_cancelled_partial_cannot_continue_entry_and_must_exit():
    b = engine()
    first = b.next_intent()
    b.apply_paper_fill(
        first, quote(b.legs[first.role]), quantity=25, event_id="part", at=NOW
    )
    b.cancel_pending()
    assert b.phase == "RECOVERY"
    assert b.next_intent() is None
    b.request_group_exit()
    close = b.next_intent()
    assert close.role == first.role and close.side == "SELL"
    assert close.quantity == 25
    b.apply_paper_fill(
        close, quote(b.legs[close.role]), quantity=25, event_id="flat", at=NOW
    )
    assert b.next_intent() is None and b.phase == "FLAT"


def test_atomic_all_or_none_depth_and_skew():
    b = engine()
    qs = {role: quote(leg) for role, leg in b.legs.items()}
    assert not b.simulate_atomic_snapshot(
        qs, at=NOW, exchange_all_or_none_capability=False
    )
    short_leg = b.legs["short_call"]
    missing_depth = {**qs, "short_call": replace(quote(short_leg), bid_quantity=20)}
    assert not b.simulate_atomic_snapshot(
        missing_depth, at=NOW, exchange_all_or_none_capability=True
    )
    assert sum(b.entry.values()) == 0
    skew = {
        **qs,
        "short_call": replace(
            quote(short_leg), observed_at=NOW - timedelta(seconds=1)
        ),
    }
    assert not b.simulate_atomic_snapshot(
        skew, at=NOW, exchange_all_or_none_capability=True
    )
    assert b.simulate_atomic_snapshot(qs, at=NOW, exchange_all_or_none_capability=True)
    assert b.phase == "OPEN" and len(b.fills) == 4
    b.request_group_exit()
    assert b.simulate_atomic_snapshot(qs, at=NOW, exchange_all_or_none_capability=True)
    assert b.phase == "FLAT" and len(b.fills) == 8


def test_quote_fail_closed_on_stale_or_wrong_symbol():
    b = engine()
    intent = b.next_intent()
    valid = quote(b.legs[intent.role])
    for invalid in (
        replace(valid, observed_at=NOW - timedelta(seconds=3)),
        replace(valid, source="LTP"),
        replace(valid, symbol="NFO:OTHER"),
        replace(valid, bid=103),
        replace(valid, ask_quantity=0),
    ):
        with pytest.raises(
            ValueError, match="paper_quote_unverified|paper_depth_insufficient"
        ):
            b.apply_paper_fill(intent, invalid, quantity=65, event_id="bad", at=NOW)
    assert sum(b.entry.values()) == 0


def test_budget_or_live_mode_fail_closed():
    with pytest.raises(ValueError, match="paper_mode_or_parent_invalid"):
        engine(mode="LIVE")
    for bad in (
        replace(budget(), required_margin=101_000),
        replace(budget(), stress_loss_including_costs=8_000),
        replace(budget(), observed_at=NOW - timedelta(minutes=1)),
        replace(budget(), source="UNVERIFIED"),
    ):
        with pytest.raises(ValueError, match="paper_margin_or_risk_unverified"):
            PaperBasketRehearsal(
                parent_id="p",
                structure="IRON_CONDOR",
                legs=condor_legs(),
                budget=bad,
                at=NOW,
            )


def test_spreads_and_wrong_geometry():
    call_spread = {
        "long_call": leg("long_call", "BUY", 24000, "CE", 1),
        "short_call": leg("short_call", "SELL", 24050, "CE", 2),
    }
    assert engine("BULL_CALL_SPREAD", call_spread).next_intent().role == "long_call"
    put_spread = {
        "short_put": leg("short_put", "SELL", 24000, "PE", 3),
        "long_put": leg("long_put", "BUY", 24050, "PE", 4),
    }
    assert engine("BEAR_PUT_SPREAD", put_spread).next_intent().role == "long_put"
    wrong = {
        **call_spread,
        "short_call": replace(
            call_spread["short_call"],
            strike=23950,
            symbol="NFO:NIFTY26NOV23950CE",
        ),
    }
    with pytest.raises(ValueError, match="paper_strike_geometry_invalid"):
        engine("BULL_CALL_SPREAD", wrong)