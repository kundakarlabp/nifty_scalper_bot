"""Tests for the expiry-day theta gate."""

from __future__ import annotations

from datetime import datetime

from nifty_scalper_bot.risk.expiry_gate import IST, expiry_theta_block


def _ist(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=IST)


def test_blocks_expiry_tuesday_afternoon() -> None:
    # 2026-06-09 is a Tuesday
    blocked, reason = expiry_theta_block(_ist(2026, 6, 9, 14, 0))
    assert blocked and "expiry_day_after" in reason


def test_allows_expiry_tuesday_morning() -> None:
    blocked, reason = expiry_theta_block(_ist(2026, 6, 9, 10, 30))
    assert not blocked and reason == "before_cutoff"


def test_allows_non_expiry_day_afternoon() -> None:
    # 2026-06-10 is a Wednesday
    blocked, reason = expiry_theta_block(_ist(2026, 6, 10, 14, 30))
    assert not blocked and reason == "not_expiry_day"


def test_disabled_via_env(monkeypatch) -> None:
    monkeypatch.setenv("EXPIRY_THETA_GATE_ENABLED", "false")
    blocked, reason = expiry_theta_block(_ist(2026, 6, 9, 15, 0))
    assert not blocked and reason == "gate_disabled"


def test_custom_cutoff(monkeypatch) -> None:
    monkeypatch.setenv("EXPIRY_ENTRY_CUTOFF_IST", "15:00")
    blocked, _ = expiry_theta_block(_ist(2026, 6, 9, 14, 30))
    assert not blocked


def test_bad_cutoff_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("EXPIRY_ENTRY_CUTOFF_IST", "garbage")
    blocked, _ = expiry_theta_block(_ist(2026, 6, 9, 14, 0))
    assert blocked  # falls back to 13:30


def test_midday_pause_blocks_within_window() -> None:
    from nifty_scalper_bot.risk.expiry_gate import midday_pause_block
    blocked, reason = midday_pause_block(_ist(2026, 6, 10, 12, 0))
    assert blocked and "midday_pause" in reason


def test_midday_pause_allows_outside_window() -> None:
    from nifty_scalper_bot.risk.expiry_gate import midday_pause_block
    blocked, _ = midday_pause_block(_ist(2026, 6, 10, 9, 45))
    assert not blocked
    blocked, _ = midday_pause_block(_ist(2026, 6, 10, 14, 0))
    assert not blocked


def test_midday_pause_disabled_via_env(monkeypatch) -> None:
    from nifty_scalper_bot.risk.expiry_gate import midday_pause_block
    monkeypatch.setenv("MIDDAY_PAUSE_ENABLED", "false")
    blocked, reason = midday_pause_block(_ist(2026, 6, 10, 12, 0))
    assert not blocked and reason == "pause_disabled"


def test_midday_pause_default_enabled_and_window(monkeypatch) -> None:
    from nifty_scalper_bot.risk.expiry_gate import midday_pause_block

    monkeypatch.delenv("MIDDAY_PAUSE_ENABLED", raising=False)
    monkeypatch.delenv("MIDDAY_PAUSE_START", raising=False)
    monkeypatch.delenv("MIDDAY_PAUSE_END", raising=False)

    blocked, reason = midday_pause_block(_ist(2026, 6, 10, 12, 0))

    assert blocked
    assert reason == "midday_pause_11:30-13:15_ist"


def test_midday_pause_false_allows_inside_window(monkeypatch) -> None:
    from nifty_scalper_bot.risk.expiry_gate import midday_pause_block

    monkeypatch.setenv("MIDDAY_PAUSE_ENABLED", "false")

    blocked, reason = midday_pause_block(_ist(2026, 6, 10, 12, 0))

    assert not blocked
    assert reason == "pause_disabled"


def test_midday_pause_truthy_values_enable_pause(monkeypatch) -> None:
    from nifty_scalper_bot.risk.expiry_gate import midday_pause_block

    for value in ("true", "yes", "1", "on"):
        monkeypatch.setenv("MIDDAY_PAUSE_ENABLED", value)
        blocked, reason = midday_pause_block(_ist(2026, 6, 10, 12, 0))
        assert blocked, value
        assert "midday_pause" in reason


def test_midday_pause_falsey_values_disable_pause(monkeypatch) -> None:
    from nifty_scalper_bot.risk.expiry_gate import midday_pause_block

    for value in ("false", "no", "0", "off"):
        monkeypatch.setenv("MIDDAY_PAUSE_ENABLED", value)
        blocked, reason = midday_pause_block(_ist(2026, 6, 10, 12, 0))
        assert not blocked, value
        assert reason == "pause_disabled"


def test_midday_pause_invalid_times_fall_back_to_default_window(monkeypatch) -> None:
    from nifty_scalper_bot.risk.expiry_gate import midday_pause_block

    monkeypatch.setenv("MIDDAY_PAUSE_ENABLED", "true")
    monkeypatch.setenv("MIDDAY_PAUSE_START", "25:00")
    monkeypatch.setenv("MIDDAY_PAUSE_END", "garbage")

    blocked, reason = midday_pause_block(_ist(2026, 6, 10, 12, 0))

    assert blocked
    assert reason == "midday_pause_11:30-13:15_ist"


def test_midday_pause_custom_window_blocks_and_allows(monkeypatch) -> None:
    from nifty_scalper_bot.risk.expiry_gate import midday_pause_block

    monkeypatch.setenv("MIDDAY_PAUSE_ENABLED", "true")
    monkeypatch.setenv("MIDDAY_PAUSE_START", "12:00")
    monkeypatch.setenv("MIDDAY_PAUSE_END", "12:30")

    blocked, reason = midday_pause_block(_ist(2026, 6, 10, 12, 15))
    assert blocked
    assert reason == "midday_pause_12:00-12:30_ist"

    blocked, reason = midday_pause_block(_ist(2026, 6, 10, 12, 45))
    assert not blocked
    assert reason == "outside_pause"
