"""Broker-free regression tests for read-only rejected-setup markouts."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.reporting.rejected_setup_forward_markout import (
    evaluate_opportunities,
    observed_markout,
    summarize_groups,
)


def test_spread_aware_markout_uses_prior_ask_and_future_bid() -> None:
    quotes = [
        (99.0, 98.0, 100.0),
        (100.5, 101.0, 103.0),
        (102.0, 104.0, 105.0),
        (104.0, 108.0, 110.0),
    ]

    label, status = observed_markout(quotes, decision_ts=100.0, horizon_s=5)

    assert status == "observed_bid_path_not_broker_fill"
    assert label is not None
    assert label["entry_ask"] == 100.0
    assert label["terminal_bid"] == 108.0
    assert label["gross_spread_adjusted_markout_pct"] == 8.0
    assert label["forward_quote_count"] == 3


def test_future_quote_cannot_be_used_as_entry() -> None:
    label, reason = observed_markout(
        [(101.0, 98.0, 100.0), (102.0, 99.0, 101.0)],
        decision_ts=100.0,
        horizon_s=5,
    )

    assert label is None
    assert reason == "no_point_in_time_entry_ask"


def test_inadequate_or_stale_forward_quotes_remain_unlabelled() -> None:
    quotes = [
        (99.0, 99.0, 100.0),
        (101.0, 105.0, 106.0),
        (102.0, 110.0, 111.0),
        (103.0, 110.0, 111.0),
    ]

    label, status = observed_markout(quotes, decision_ts=100.0, horizon_s=60)

    assert label is None
    assert status == "stale_terminal_bid"


def test_rotated_contract_never_inherits_other_option_markout() -> None:
    opportunity = SimpleNamespace(
        opportunity_id="s1",
        strategy="vwap",
        side="CE",
        symbol="NFO:NIFTY26N0322750CE",
        observed_symbols=("NFO:NIFTY26N0322750CE", "NFO:NIFTY26N0322800CE"),
        first_seen_ts=100.0,
        approved=False,
        final_reason="direction_unavailable",
        blocked_reasons=("direction_unavailable",),
    )
    outcome = evaluate_opportunities(
        [opportunity],
        {"NFO:NIFTY26N0322750CE": [(99.0, 99.0, 100.0)]},
        (60,),
    )

    assert len(outcome) == 1
    assert outcome[0]["markout"] is None
    assert outcome[0]["status"] == "contract_rotation_or_missing_identity"


def test_summary_keeps_unlabelled_independent_setups_in_denominator() -> None:
    rows = [
        {
            "approved": False,
            "reason": "direction_unavailable",
            "horizon_seconds": 60,
            "markout": {"gross_spread_adjusted_markout_pct": 5.0},
        },
        {
            "approved": False,
            "reason": "direction_unavailable",
            "horizon_seconds": 60,
            "markout": None,
        },
    ]
    groups = summarize_groups(rows, min_labelled=2)

    assert len(groups) == 1
    assert groups[0]["independent_setups"] == 2
    assert groups[0]["labelled"] == 1
    assert groups[0]["unlabelled"] == 1
    assert groups[0]["label_coverage_fraction"] == 0.5
    assert groups[0]["mean_gross_markout_pct"] is None
    assert groups[0]["causal_or_profitability_inference_ready"] is False
