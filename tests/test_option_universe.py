from __future__ import annotations

from datetime import datetime

from nifty_scalper_bot.core.option_universe import (
    OptionUniverseConfig,
    OptionUniverseManager,
)


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
