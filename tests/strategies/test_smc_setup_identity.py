from __future__ import annotations

from datetime import timedelta

import pytest

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    SMCStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy

from .test_smc_sweep_history_recovery import (
    CE,
    FUTURES,
    SPOT,
    _bar,
    _base_rows,
    _Engine,
    _indicators,
)


def _confirmed_signal(
    *, shift=timedelta(), source=FUTURES, symbol=CE, premium_swing=99.0, restart=True
):
    rows = _base_rows()
    sweep = _bar(30, open_=23988, high=23991, low=23974, close=23984, volume=2500)
    confirm = _bar(31, open_=23984, high=24000, low=23982, close=23998, volume=2200)
    rows.extend([sweep, confirm])
    for row in rows:
        row["timestamp"] += shift
    engine = _Engine(rows)
    strategy = SMCStrategy(SMCStrategyConfig(min_confidence=0.0), engine)

    def context(timestamp):
        payload = _indicators(timestamp)
        payload.update(
            futures_symbol=source if source == FUTURES else "",
            spot_symbol=SPOT,
            prior_swing_low=premium_swing,
        )
        return payload

    if not restart:
        rows.pop()
        assert (
            strategy.generate_signal(symbol, context(sweep["timestamp"]), 103.0) is None
        )
        rows.append(confirm)
    signal = strategy.generate_signal(symbol, context(confirm["timestamp"]), 103.0)
    assert signal is not None
    return signal


@pytest.mark.parametrize("shift", [timedelta(minutes=10), timedelta(days=1)])
def test_distinct_sweeps_at_same_level_have_distinct_identity(monkeypatch, shift):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    first = _confirmed_signal()
    later = _confirmed_signal(shift=shift)
    assert first.metadata["sweep_level"] == later.metadata["sweep_level"]
    assert first.deterministic_id != later.deterministic_id


def test_structure_source_is_part_of_smc_identity(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    futures = _confirmed_signal()
    spot = _confirmed_signal(source=SPOT)
    assert futures.deterministic_id != spot.deterministic_id


def test_option_rotation_cannot_change_underlying_sweep_identity(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    first = _confirmed_signal(premium_swing=99.0)
    rotated = _confirmed_signal(symbol="NFO:NIFTY2690823700CE", premium_swing=79.0)
    assert first.metadata["setup_id"] == rotated.metadata["setup_id"]
    assert first.deterministic_id == rotated.deterministic_id


def test_recovered_sweep_has_same_identity_as_uninterrupted_evaluation(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    uninterrupted = _confirmed_signal(restart=False)
    recovered = _confirmed_signal(restart=True)
    assert recovered.metadata["sweep_recovered_from_history"] is True
    assert uninterrupted.metadata["sweep_recovered_from_history"] is False
    assert uninterrupted.deterministic_id == recovered.deterministic_id
