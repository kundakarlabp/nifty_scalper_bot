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
