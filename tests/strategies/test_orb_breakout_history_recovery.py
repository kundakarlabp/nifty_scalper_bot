from __future__ import annotations

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    ORBProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy

from .test_orb_pro_v2 import (
    CE,
    FUTURE,
    _bar,
    _base_indicators,
    _IndicatorEngine,
    _opening_rows,
)


def _new_strategy(rows):
    return ORBProStrategy(
        ORBProStrategyConfig(min_confidence=0.0, orb_minutes=15),
        _IndicatorEngine({FUTURE: rows}),
    )


def test_restart_recovers_unconfirmed_breakout_for_current_retest(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ORB_MOMENTUM_BRANCH_ENABLED", "false")
    rows = _opening_rows()
    breakout = _bar(
        15,
        open_=24_014.0,
        high=24_029.0,
        low=24_012.0,
        close=24_025.0,
        volume=1_100.0,
    )
    retest = _bar(
        16,
        open_=24_024.0,
        high=24_030.0,
        low=24_019.0,
        close=24_027.0,
        volume=1_300.0,
    )
    rows.extend([breakout, retest])

    signal = _new_strategy(rows).generate_signal(
        CE,
        _base_indicators("CE", retest["timestamp"]),
        50.0,
    )

    assert signal is not None
    assert signal.metadata["entry_branch"] == "retest"
    assert signal.metadata["breakout_recovered_from_history"] is True
    assert signal.metadata["breakout_timestamp"] == breakout["timestamp"].timestamp()


def test_recovery_does_not_late_enter_after_prior_retest(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ORB_MOMENTUM_BRANCH_ENABLED", "false")
    rows = _opening_rows()
    rows.extend(
        [
            _bar(
                15,
                open_=24_014.0,
                high=24_029.0,
                low=24_012.0,
                close=24_025.0,
                volume=1_100.0,
            ),
            _bar(
                16,
                open_=24_024.0,
                high=24_030.0,
                low=24_019.0,
                close=24_027.0,
                volume=1_300.0,
            ),
            _bar(
                17,
                open_=24_027.0,
                high=24_032.0,
                low=24_022.0,
                close=24_029.0,
                volume=1_200.0,
            ),
        ]
    )

    strategy = _new_strategy(rows)
    assert (
        strategy.generate_signal(
            CE,
            _base_indicators("CE", rows[-1]["timestamp"]),
            50.0,
        )
        is None
    )
    assert strategy.last_no_vote_reason == "no_fresh_underlying_breakout"


def test_recovery_never_replays_a_momentum_breakout(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ORB_MOMENTUM_BRANCH_ENABLED", "true")
    rows = _opening_rows()
    rows.extend(
        [
            _bar(
                15,
                open_=24_010.0,
                high=24_050.0,
                low=24_008.0,
                close=24_046.0,
                volume=4_000.0,
            ),
            _bar(
                16,
                open_=24_046.0,
                high=24_052.0,
                low=24_040.0,
                close=24_048.0,
                volume=1_500.0,
            ),
        ]
    )

    strategy = _new_strategy(rows)
    assert (
        strategy.generate_signal(
            CE,
            _base_indicators("CE", rows[-1]["timestamp"]),
            50.0,
        )
        is None
    )
    assert strategy.last_no_vote_reason == "no_fresh_underlying_breakout"
