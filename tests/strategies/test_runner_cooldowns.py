from __future__ import annotations

from pathlib import Path

import pytest

from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_on_tick_error_phase_updates_for_premium_squeeze() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(
        encoding="utf-8"
    )
    assert 'phase = "phase8_premium_squeeze"' in source


def test_active_option_selection_uses_atm_strike_key() -> None:
    source = Path("src/nifty_scalper_bot/core/app.py").read_text(encoding="utf-8")
    assert 'atm=basket.get("atm_strike")' in source


def test_cooldown_first_trade_no_epoch_age() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(
        encoding="utf-8"
    )
    assert "first_trade_for_key" in source
    assert "age_seconds=None" not in source or "age_seconds=%s" in source


def test_rejected_signal_cooldown_prevents_spam() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(
        encoding="utf-8"
    )
    assert "SIGNAL_REJECT_COOLDOWN_ACTIVE" in source
    assert "_signal_reject_cooldown_ts" in source


def test_candidate_enrichment_raises_rr_and_option_score() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(
        encoding="utf-8"
    )
    assert 'metadata["rr_score"] = max(' in source
    assert 'metadata["option_score"] = max(' in source


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("entry_rr_below_floor", "entry_rr_below_floor"),
        ("MARGIN no_qty_after_risk", "risk_capacity_unavailable"),
        ("insufficient_margin", "risk_capacity_unavailable"),
        ("available_balance_unavailable", "risk_capacity_unavailable"),
        ("single_position_gate", "position_conflict"),
        ("open_position_exists", "position_conflict"),
        ("quote_stale", None),
    ],
)
def test_deterministic_execution_reject_reason_families(reason, expected) -> None:
    assert StrategyRunner._deterministic_execution_reject_reason(reason) == expected


def test_deterministic_reject_cooldown_requires_no_broker_attempt() -> None:
    runner = object.__new__(StrategyRunner)
    runner._execution_reject_cooldown_ts = {}

    marked = runner._mark_deterministic_execution_reject_cooldown(
        symbol="NFO:NIFTY26AUG25000CE",
        reason_key="orderflow",
        reason="MARGIN no_qty_after_risk",
        now_epoch=100.0,
        broker_attempted=True,
    )

    assert marked is None
    assert runner._execution_reject_cooldown_ts == {}


def test_deterministic_local_reject_stamps_existing_cooldown_cache() -> None:
    runner = object.__new__(StrategyRunner)
    runner._execution_reject_cooldown_ts = {}

    marked = runner._mark_deterministic_execution_reject_cooldown(
        symbol="NFO:NIFTY26AUG25000CE",
        reason_key="orderflow",
        reason="MARGIN no_qty_after_risk",
        now_epoch=100.0,
        broker_attempted=False,
    )

    assert marked == "risk_capacity_unavailable"
    assert runner._execution_reject_cooldown_ts == {
        "NFO:NIFTY26AUG25000CE:orderflow:risk_capacity_unavailable": 100.0
    }
