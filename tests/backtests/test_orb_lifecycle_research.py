"""ORB lifecycle research mirrors production exit settings without look-ahead."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from scripts.research_orb_iterative import BASE_OVERRIDES, candidates

from nifty_scalper_bot.backtesting.strategy_research import (
    _apply_bar_lifecycle_proxy,
    _bar_lifecycle_time_stop_due,
)


IST = ZoneInfo("Asia/Kolkata")


def _position() -> dict:
    opened = datetime(2026, 1, 5, 10, 0, tzinfo=IST)
    return {
        "entry_time": opened,
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "initial_stop_loss": 95.0,
        "take_profit": 112.5,
        "quantity": 75,
        "high_water": 100.0,
        "trail_updates": 0,
    }


def test_lifecycle_proxy_ratchets_stop_from_completed_bar_only():
    position = _position()
    bar = {
        "timestamp": position["entry_time"] + timedelta(minutes=5),
        "open": 104.0,
        "high": 110.0,
        "low": 103.0,
        "close": 109.0,
        "volume": 100.0,
    }
    changed = _apply_bar_lifecycle_proxy(position, bar, prior_atr=2.0)
    assert changed is True
    assert position["stop_loss"] > 100.0
    assert position["stop_loss"] < bar["high"]
    assert position["trail_updates"] == 1


def test_lifecycle_time_stop_matches_12_minute_half_r_progress_rule():
    position = _position()
    position["high_water"] = 102.0  # 0.4R
    assert _bar_lifecycle_time_stop_due(
        position, position["entry_time"] + timedelta(minutes=11)
    ) is False
    assert _bar_lifecycle_time_stop_due(
        position, position["entry_time"] + timedelta(minutes=12)
    ) is True
    position["high_water"] = 102.5  # exactly 0.5R
    assert _bar_lifecycle_time_stop_due(
        position, position["entry_time"] + timedelta(minutes=12)
    ) is False


def test_five_runs_change_exactly_one_orb_setting_from_lifecycle_reference():
    rows = candidates()
    assert [row["name"] for row in rows[:2]] == [
        "static_reference_075_25",
        "lifecycle_reference_075_25",
    ]
    experiments = rows[2:]
    assert len(experiments) == 5
    assert all(row["lifecycle_proxy"] is True for row in experiments)
    for row in experiments:
        extra = {
            key: value
            for key, value in row["overrides"].items()
            if BASE_OVERRIDES.get(key) != value
        }
        assert len(extra) == 1
