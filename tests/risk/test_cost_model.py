"""Tests for the round-trip transaction cost model."""

from __future__ import annotations

from nifty_scalper_bot.risk.cost_model import (
    estimate_round_trip_cost,
    passes_cost_edge_gate,
)


def test_cost_breakdown_components_positive() -> None:
    c = estimate_round_trip_cost(
        entry_price=120, exit_price=135, quantity=65, half_spread=0.5
    )
    assert c.brokerage == 40.0
    assert c.stt > 0 and c.exchange_txn > 0 and c.gst > 0
    assert c.total > 60.0
    assert abs(c.cost_per_unit - c.total / 65) < 1e-9


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
