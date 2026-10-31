"""Synthetic fixtures only; no claim of observed prospective trading returns."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from nifty_scalper_bot.backtesting.volatility_prospective import (
    ProspectiveLegFill,
    reconcile_prospective_trade,
)
from nifty_scalper_bot.risk.cost_model import estimate_order_cost

_TZ = timezone(timedelta(hours=5, minutes=30))
_DECISION = datetime(2026, 10, 30, 10, 0, tzinfo=_TZ)


def fixture():
    parent = {
        "parent_id": "test-not-real-001",
        "structure": "STRADDLE",
        "legs": {
            "long_call": {
                "symbol": "NFO:NIFTY26NOV24000CE",
                "side": "BUY",
                "quantity": 65,
            },
            "long_put": {
                "symbol": "NFO:NIFTY26NOV24000PE",
                "side": "BUY",
                "quantity": 65,
            },
        },
    }
    entry = {
        role: ProspectiveLegFill(
            order_id=f"entry-{role}",
            symbol=leg["symbol"],
            side="BUY",
            quantity=65,
            price=100,
            filled_at=_DECISION + timedelta(seconds=1),
            source="PAPER",
        )
        for role, leg in parent["legs"].items()
    }
    exit_fills = {
        role: ProspectiveLegFill(
            order_id=f"exit-{role}",
            symbol=leg["symbol"],
            side="SELL",
            quantity=65,
            price=110,
            filled_at=_DECISION + timedelta(minutes=15),
            source="PAPER",
        )
        for role, leg in parent["legs"].items()
    }
    return {
        "parent": parent,
        "decision_at": _DECISION,
        "quote_observed_at": _DECISION - timedelta(milliseconds=250),
        "quote_source": "FULL",
        "executable_entry_quotes": {
            "long_call": 98,
            "long_put": 98,
        },
        "entry_fills": entry,
        "exit_fills": exit_fills,
    }


def test_paper_result_is_cost_adjusted_and_never_live_eligible():
    inputs = fixture()
    result = reconcile_prospective_trade(**inputs)
    fees = 2 * (
        estimate_order_cost(turnover=6500, side="BUY")
        + estimate_order_cost(turnover=7150, side="SELL")
    )
    assert result.source == "PAPER"
    assert result.gross_pnl_rupees == pytest.approx(1300)
    assert result.fees_rupees == pytest.approx(fees)
    assert result.net_pnl_rupees == pytest.approx(1300 - fees)
    assert result.entry_slippage_rupees == pytest.approx(260)
    assert result.live_eligible is False


def test_observed_broker_fills_have_separate_provenance_not_assumed_profit():
    inputs = fixture()
    inputs["entry_fills"] = {
        role: replace(fill, source="BROKER")
        for role, fill in inputs["entry_fills"].items()
    }
    inputs["exit_fills"] = {
        role: replace(fill, source="BROKER")
        for role, fill in inputs["exit_fills"].items()
    }
    result = reconcile_prospective_trade(**inputs)
    assert result.source == "BROKER"
    assert result.live_eligible is False


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (
            lambda i: i["entry_fills"].pop("long_put"),
            "prospective_parent_or_leg_set_invalid",
        ),
        (
            lambda i: i.update(quote_source="ARCHIVED_FULL"),
            "prospective_decision_quote_unverified",
        ),
        (
            lambda i: i.update(quote_observed_at=_DECISION - timedelta(seconds=3)),
            "prospective_decision_quote_unverified",
        ),
        (
            lambda i: i["entry_fills"].update(
                long_put=replace(i["entry_fills"]["long_put"], quantity=25)
            ),
            "prospective_fill_contract_invalid",
        ),
        (
            lambda i: i["exit_fills"].update(
                long_put=replace(i["exit_fills"]["long_put"], source="BROKER")
            ),
            "prospective_fill_sources_mixed",
        ),
        (
            lambda i: i["exit_fills"].update(
                long_put=replace(
                    i["exit_fills"]["long_put"],
                    filled_at=_DECISION - timedelta(minutes=1),
                )
            ),
            "prospective_fill_chronology_invalid",
        ),
        (
            lambda i: i["exit_fills"].update(
                long_put=replace(i["exit_fills"]["long_put"], order_id="exit-long_call")
            ),
            "prospective_fill_contract_invalid",
        ),
        (
            lambda i: i["entry_fills"].update(
                long_call=replace(i["entry_fills"]["long_call"], status="OPEN")
            ),
            "prospective_fill_contract_invalid",
        ),
    ],
)
def test_incomplete_or_unreliable_observation_cannot_be_counted(change, error):
    inputs = fixture()
    change(inputs)
    with pytest.raises(ValueError, match=error):
        reconcile_prospective_trade(**inputs)
