"""Tests for the central market-session helper and stale-threshold function."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.utils import market_hours


IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture(autouse=True)
def _no_offhours_override(monkeypatch):
    """Disable any off-hours testing override that may be set in CI."""
    monkeypatch.delenv("ALLOW_OFFHOURS_TESTING", raising=False)
    monkeypatch.delenv("SESSION_ALLOW_OUT_OF_HOURS", raising=False)
    monkeypatch.delenv("EXECUTION_MODE", raising=False)
    monkeypatch.delenv("ENABLE_LIVE", raising=False)


def test_midnight_ist_returns_closed():
    """Args: none. Returns: None. Raises: AssertionError."""
    # Mon Apr 27 2026 00:00 IST -> closed
    now = datetime(2026, 4, 27, 0, 0, tzinfo=IST)
    assert market_hours.get_market_session_state(now) == "closed"


def test_weekday_open_returns_open():
    """Args: none. Returns: None. Raises: AssertionError."""
    # Mon Apr 27 2026 10:00 IST -> open
    now = datetime(2026, 4, 27, 10, 0, tzinfo=IST)
    assert market_hours.get_market_session_state(now) == "open"


def test_weekend_returns_closed():
    """Args: none. Returns: None. Raises: AssertionError."""
    # Sun Apr 26 2026 11:00 IST -> closed
    now = datetime(2026, 4, 26, 11, 0, tzinfo=IST)
    assert market_hours.get_market_session_state(now) == "closed"


def test_preopen_returns_preopen():
    """Args: none. Returns: None. Raises: AssertionError."""
    # Mon Apr 27 2026 09:05 IST -> preopen
    now = datetime(2026, 4, 27, 9, 5, tzinfo=IST)
    assert market_hours.get_market_session_state(now) == "preopen"


def test_after_close_returns_closed():
    """Args: none. Returns: None. Raises: AssertionError."""
    # Mon Apr 27 2026 16:00 IST -> closed
    now = datetime(2026, 4, 27, 16, 0, tzinfo=IST)
    assert market_hours.get_market_session_state(now) == "closed"


def test_is_market_open_now_uses_session_state(monkeypatch):
    """Args: monkeypatch. Returns: None. Raises: AssertionError."""
    monkeypatch.setattr(market_hours, "get_market_session_state", lambda: "open")
    assert market_hours.is_market_open_now() is True

    monkeypatch.setattr(market_hours, "get_market_session_state", lambda: "closed")
    assert market_hours.is_market_open_now() is False


def test_nse_nifty_open_threshold_is_120():
    """Args: none. Returns: None. Raises: AssertionError."""
    assert market_hours.stale_threshold_for_symbol("NSE:NIFTY", market_open=True) == 120.0


def test_nse_nifty_closed_threshold_is_3600():
    """Args: none. Returns: None. Raises: AssertionError."""
    assert market_hours.stale_threshold_for_symbol("NSE:NIFTY", market_open=False) == 3600.0


def test_nfo_option_open_threshold_is_900():
    """Args: none. Returns: None. Raises: AssertionError."""
    sym = "NFO:NIFTY24DEC25000CE"
    assert market_hours.stale_threshold_for_symbol(sym, market_open=True) == 900.0


def test_nfo_option_closed_threshold_is_3600():
    """Args: none. Returns: None. Raises: AssertionError."""
    sym = "NFO:NIFTY24DEC25000CE"
    assert market_hours.stale_threshold_for_symbol(sym, market_open=False) == 3600.0


def test_nifty_future_open_threshold_is_120():
    """Args: none. Returns: None. Raises: AssertionError."""
    assert market_hours.stale_threshold_for_symbol("NIFTY24DECFUT", market_open=True) == 120.0


def test_nifty_future_closed_threshold_is_3600():
    """Args: none. Returns: None. Raises: AssertionError."""
    assert (
        market_hours.stale_threshold_for_symbol("NIFTY24DECFUT", market_open=False)
        == 3600.0
    )


def test_thresholds_respect_env_overrides(monkeypatch):
    """Args: monkeypatch. Returns: None. Raises: AssertionError."""
    monkeypatch.setenv("MDM_INDEX_LTP_STALE_SECONDS", "200")
    monkeypatch.setenv("MDM_OFFMARKET_INDEX_LTP_STALE_SECONDS", "5400")
    monkeypatch.setenv("MDM_OPTION_LTP_STALE_SECONDS", "300")
    monkeypatch.setenv("MDM_OFFMARKET_OPTION_LTP_STALE_SECONDS", "7200")

    assert market_hours.stale_threshold_for_symbol("NSE:NIFTY", market_open=True) == 200.0
    assert market_hours.stale_threshold_for_symbol("NSE:NIFTY", market_open=False) == 5400.0
    assert (
        market_hours.stale_threshold_for_symbol("NFO:NIFTY24DEC25000CE", market_open=True)
        == 300.0
    )
    assert (
        market_hours.stale_threshold_for_symbol("NFO:NIFTY24DEC25000CE", market_open=False)
        == 7200.0
    )


def test_no_code_path_can_log_threshold_10_for_nifty():
    """Args: none. Returns: None. Raises: AssertionError."""
    # Both paths must yield ≥ 120s, never 10s.
    assert market_hours.stale_threshold_for_symbol("NSE:NIFTY", market_open=True) >= 120.0
    assert market_hours.stale_threshold_for_symbol("NSE:NIFTY", market_open=False) >= 120.0
