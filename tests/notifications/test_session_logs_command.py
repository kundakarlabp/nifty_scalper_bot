"""/sessionlogs must return the whole trading session, not a truncated tail.

The Telegram log ring was fixed at 2000 lines. At live tick-log rates that
holds roughly 15-90 minutes, so /dumplogs could never return a full trading
session and post-session diagnosis worked from a truncated window -- observed
directly: uploaded 31 Jul logs covered only 13:00-14:42 of a 09:00-15:30 day,
showing 3 closed trades out of a reported 36.
"""

from __future__ import annotations

from nifty_scalper_bot.notifications.operator_telegram import (
    OPERATOR_COMMANDS,
    _filter_session_lines,
    _session_window_ist,
)
from nifty_scalper_bot.notifications.telegram_controller import _Ring, _ring_capacity


def test_command_is_registered() -> None:
    names = [c.name for c in OPERATOR_COMMANDS]
    assert "sessionlogs" in names
    # Existing commands are preserved.
    assert "logs" in names and "dumplogs" in names


def test_default_window_is_the_nse_session() -> None:
    assert _session_window_ist() == ("09:00", "15:30")


def test_window_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("SESSION_LOG_START_IST", "09:15")
    monkeypatch.setenv("SESSION_LOG_END_IST", "15:29")
    assert _session_window_ist() == ("09:15", "15:29")


def test_pre_open_and_post_close_lines_are_excluded() -> None:
    lines = [
        "08:59:59 [INFO] pre-open",
        "09:00:00 [INFO] open",
        "12:30:00 [INFO] midday",
        "15:30:00 [INFO] close",
        "15:31:00 [INFO] post-close",
    ]
    kept = _filter_session_lines(lines, "09:00", "15:30")
    assert "08:59:59 [INFO] pre-open" not in kept
    assert "15:31:00 [INFO] post-close" not in kept
    assert len(kept) == 3


def test_multiline_records_are_not_broken_apart() -> None:
    """Tracebacks have no leading timestamp and must follow their record."""
    lines = [
        "12:00:00 [ERROR] boom",
        "Traceback (most recent call last):",
        "  File 'x.py', line 1",
        "15:31:00 [INFO] post-close",
        "  orphaned continuation after the window",
    ]
    kept = _filter_session_lines(lines, "09:00", "15:30")
    assert "Traceback (most recent call last):" in kept
    assert "  File 'x.py', line 1" in kept
    # A continuation belonging to an excluded record stays excluded.
    assert "  orphaned continuation after the window" not in kept


def test_ring_capacity_holds_a_full_session_by_default() -> None:
    """2000 lines could not span 09:00-15:30 at live log rates."""
    assert _ring_capacity() == 50_000
    assert _Ring().buf.maxlen == 50_000


def test_ring_capacity_is_bounded_and_configurable(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_LOG_RING_LINES", "80000")
    assert _ring_capacity() == 80_000
    # Floor protects the existing /logs and /dumplogs behaviour.
    monkeypatch.setenv("TELEGRAM_LOG_RING_LINES", "10")
    assert _ring_capacity() == 2_000
    # Ceiling bounds memory (~200 bytes/line).
    monkeypatch.setenv("TELEGRAM_LOG_RING_LINES", "999999")
    assert _ring_capacity() == 200_000
    # Malformed config must not shrink the buffer.
    monkeypatch.setenv("TELEGRAM_LOG_RING_LINES", "abc")
    assert _ring_capacity() == 50_000
