from __future__ import annotations

from datetime import datetime

import pytest

from nifty_scalper_bot.core.option_universe import (
    OptionUniverseConfig,
    OptionUniverseManager,
)


@pytest.fixture(autouse=True)
def legacy_option_universe_enabled(monkeypatch):
    monkeypatch.setenv("OPTION_UNIVERSE_LEGACY_FALLBACK_ENABLED", "true")
    monkeypatch.delenv("EXECUTION_MODE", raising=False)
    monkeypatch.delenv("MODE", raising=False)


def test_option_universe_builds_atm_range() -> None:
    manager = OptionUniverseManager(
        OptionUniverseConfig(strike_step=50, strikes_around_atm=1),
        now_fn=lambda: datetime(2026, 2, 1, 10, 0, 0),
    )

    manager.update_underlying(18638.0)
    symbols = manager.get_current_universe()

    assert len(symbols) == 6
    assert any(symbol.endswith('18650CE') for symbol in symbols)
    assert any(symbol.endswith('18600PE') for symbol in symbols)
    assert any(symbol.endswith('18700PE') for symbol in symbols)


def test_option_universe_rolls_expiry_near_threshold() -> None:
    manager = OptionUniverseManager(
        OptionUniverseConfig(expiry_roll_hours=2.0),
        now_fn=lambda: datetime(2026, 2, 3, 11, 0, 0),
    )

    manager.update_underlying(18650.0)
    first_universe = manager.get_current_universe()

    manager_late = OptionUniverseManager(
        OptionUniverseConfig(expiry_roll_hours=2.0),
        now_fn=lambda: datetime(2026, 2, 3, 15, 10, 0),
    )
    manager_late.update_underlying(18650.0)
    rolled_universe = manager_late.get_current_universe()

    assert first_universe
    assert rolled_universe
    assert first_universe != rolled_universe


def test_primary_symbols_matches_current_universe() -> None:
    manager = OptionUniverseManager(
        OptionUniverseConfig(strike_step=100, strikes_around_atm=2),
        now_fn=lambda: datetime(2026, 3, 10, 11, 0, 0),
    )

    manager.update_underlying(22540.0)

    assert manager.get_primary_symbols() == manager.get_current_universe()


def test_filtered_universe_uses_nearest_weekly_and_caps_size() -> None:
    manager = OptionUniverseManager(
        OptionUniverseConfig(strike_step=50, strikes_around_atm=1),
        now_fn=lambda: datetime(2026, 2, 2, 10, 0, 0),
    )

    symbols = manager.get_filtered_universe(18630.0)

    assert symbols
    assert len(symbols) <= 8
    assert any(symbol.endswith('18650CE') for symbol in symbols)


def test_filtered_universe_reuses_spot_within_threshold() -> None:
    manager = OptionUniverseManager(
        OptionUniverseConfig(
            strike_step=50,
            strikes_around_atm=1,
            spot_recalc_threshold_points=1000.0,
        ),
        now_fn=lambda: datetime(2026, 2, 2, 10, 0, 0),
    )

    first = manager.get_filtered_universe(18600.0)
    second = manager.get_filtered_universe(18640.0)

    assert first == second


def test_live_option_universe_manual_generation_raises(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    manager = OptionUniverseManager(
        OptionUniverseConfig(strike_step=50, strikes_around_atm=1),
        now_fn=lambda: datetime(2026, 2, 1, 10, 0, 0),
    )
    with pytest.raises(RuntimeError, match="InstrumentManager ActiveContractBasket"):
        manager.update_underlying(18638.0)


def test_option_universe_legacy_requires_env(monkeypatch) -> None:
    monkeypatch.delenv("OPTION_UNIVERSE_LEGACY_FALLBACK_ENABLED", raising=False)
    manager = OptionUniverseManager(
        OptionUniverseConfig(strike_step=50, strikes_around_atm=1),
        now_fn=lambda: datetime(2026, 2, 1, 10, 0, 0),
    )
    with pytest.raises(RuntimeError, match="manual generation disabled"):
        manager.update_underlying(18638.0)
