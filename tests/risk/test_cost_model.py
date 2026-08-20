"""Tests for the round-trip transaction cost model."""

from __future__ import annotations

import pytest

from nifty_scalper_bot.risk.cost_model import (
    estimate_order_cost,
    estimate_round_trip_cost,
    evaluate_net_reward_risk,
    minimum_net_reward_risk,
    passes_cost_edge_gate,
)


def test_one_way_execution_fees_reconcile_to_round_trip_model() -> None:
    entry = 120.0
    exit_price = 135.0
    quantity = 65
    one_way = estimate_order_cost(
        turnover=entry * quantity, side="BUY"
    ) + estimate_order_cost(turnover=exit_price * quantity, side="SELL")
    round_trip = estimate_round_trip_cost(
        entry_price=entry,
        exit_price=exit_price,
        quantity=quantity,
    )

    assert one_way == pytest.approx(round_trip.total)


def test_cost_breakdown_components_positive() -> None:
    c = estimate_round_trip_cost(
        entry_price=120, exit_price=135, quantity=65, half_spread=0.5
    )
    assert c.brokerage == 40.0
    assert c.stt > 0 and c.exchange_txn > 0 and c.gst > 0
    assert c.total > 60.0
    assert abs(c.cost_per_unit - c.total / 65) < 1e-9


def test_round_trip_cost_counts_each_partial_exit_order() -> None:
    two_orders = estimate_round_trip_cost(
        entry_price=100.0,
        exit_price=110.0,
        quantity=130,
        executed_orders=2,
    )
    three_orders = estimate_round_trip_cost(
        entry_price=100.0,
        exit_price=110.0,
        quantity=130,
        executed_orders=3,
    )
    assert three_orders.brokerage - two_orders.brokerage == 20.0
    assert three_orders.total > two_orders.total


def test_gate_blocks_thin_target() -> None:
    ok, edge, _ = passes_cost_edge_gate(
        entry_price=120, target_price=122, quantity=65, half_spread=0.5
    )
    assert not ok and edge < 2.0


def test_gate_allows_wide_target() -> None:
    ok, edge, _ = passes_cost_edge_gate(
        entry_price=120, target_price=140, quantity=65, half_spread=0.5
    )
    assert ok and edge >= 2.0


def test_env_override_min_edge(monkeypatch) -> None:
    monkeypatch.setenv("MIN_EDGE_MULTIPLE", "1.0")
    ok, _, _ = passes_cost_edge_gate(
        entry_price=120, target_price=124, quantity=65, half_spread=0.3
    )
    assert ok


def test_cost_adjusted_rr_counts_both_win_and_stop_outcomes(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.5")
    legacy_ok, legacy_edge, _ = passes_cost_edge_gate(
        entry_price=100.0,
        target_price=103.6,
        quantity=65,
        half_spread=0.1,
    )

    result = evaluate_net_reward_risk(
        entry_price=100.0,
        stop_price=98.0,
        target_price=103.6,
        quantity=65,
        half_spread=0.1,
    )

    assert legacy_ok is True and legacy_edge > 2.0
    assert result.allowed is False
    assert result.net_rr < 1.5
    assert result.net_reward < result.gross_reward
    assert result.net_risk > result.gross_risk


def test_cost_adjusted_rr_and_threshold_share_one_owner(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.25")

    result = evaluate_net_reward_risk(
        entry_price=100.0,
        stop_price=90.0,
        target_price=120.0,
        quantity=65,
        half_spread=0.1,
    )

    assert minimum_net_reward_risk() == 1.25
    assert result.minimum == 1.25
    assert result.allowed is (result.net_rr >= 1.25)
