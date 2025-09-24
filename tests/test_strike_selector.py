# tests/test_strike_selector.py
"""
Tests for the strike selector and market hours utilities.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from freezegun import freeze_time

from src.utils.strike_selector import is_market_open, get_instrument_tokens
import src.utils.strike_selector as strike_selector
from src.config import settings


# Test cases for is_market_open
# Note: These dates are chosen as examples.

# A trading day (Monday) during hours.
@freeze_time("2024-08-05 11:00:00+05:30")
def test_market_open_during_hours():
    assert is_market_open() is True

# A trading day (Monday) after hours.
@freeze_time("2024-08-05 16:00:00+05:30")
def test_market_closed_after_hours():
    assert is_market_open() is False

# A weekend (Saturday).
@freeze_time("2024-08-10 11:00:00+05:30")
def test_market_closed_on_weekend():
    assert is_market_open() is False

# A non-trading day (Sunday) to simulate a market holiday.
@freeze_time("2024-08-18 11:00:00+05:30")
def test_market_closed_on_holiday():
    assert is_market_open() is False


@pytest.fixture
def mock_kite():
    """Provides a mock KiteConnect instance."""
    kite = MagicMock()

    # Mock for get_instrument_tokens
    kite.ltp.return_value = {
        "NSE:NIFTY 50": {"instrument_token": 256265, "last_price": 19525.50}
    }

    # Mock for get_next_expiry_date
    kite.instruments.return_value = [
        {
            "name": "NIFTY",
            "segment": "NFO-OPT",
            "expiry": datetime(2024, 8, 6).date(),
            "instrument_type": "CE",
            "strike": 19500,
            "tradingsymbol": "NIFTY2480819500CE",
            "instrument_token": 1,
            "lot_size": 75,
        },
        {
            "name": "NIFTY",
            "segment": "NFO-OPT",
            "expiry": datetime(2024, 8, 6).date(),
            "instrument_type": "PE",
            "strike": 19500,
            "tradingsymbol": "NIFTY2480819500PE",
            "instrument_token": 2,
            "lot_size": 75,
        },
        {
            "name": "NIFTY",
            "segment": "NFO-OPT",
            "expiry": datetime(2024, 8, 6).date(),
            "instrument_type": "CE",
            "strike": 19550,
            "tradingsymbol": "NIFTY2480819550CE",
            "instrument_token": 3,
            "lot_size": 75,
        },
        {
            "name": "NIFTY",
            "segment": "NFO-OPT",
            "expiry": datetime(2024, 8, 6).date(),
            "instrument_type": "PE",
            "strike": 19550,
            "tradingsymbol": "NIFTY2480819550PE",
            "instrument_token": 4,
            "lot_size": 75,
        },
    ]
    return kite


@freeze_time("2024-08-05 10:00:00+05:30")  # A Monday before weekly expiry
def test_get_instrument_tokens_weekly_expiry(mock_kite):
    """Tests instrument selection for a standard weekly expiry."""
    tokens = get_instrument_tokens(
        kite_instance=mock_kite,
    )
    assert tokens is not None
    assert tokens["expiry"] == "2024-08-06"
    assert tokens["lot_size"] == 75
    assert "contracts" in tokens
    ce_contract = tokens["contracts"].get("ce")
    pe_contract = tokens["contracts"].get("pe")
    assert ce_contract is not None and pe_contract is not None
    assert ce_contract["lot_size"] == 75
    assert ce_contract["token"] == tokens["tokens"]["ce"]
    assert ce_contract["segment"] == "NFO-OPT"
    assert ce_contract["expiry"] == "2024-08-06"
    prewarm = tokens.get("prewarm_contracts", {})
    assert {19500, 19550}.issubset(prewarm.get("ce", {}).keys())
    assert {19500, 19550}.issubset(prewarm.get("pe", {}).keys())
    assert tokens.get("prewarm_tokens", {}).get("ce")
    assert tokens.get("prewarm_tokens", {}).get("pe")


@freeze_time("2024-08-26 10:00:00+05:30")  # A Monday before monthly expiry
def test_get_instrument_tokens_monthly_expiry(mock_kite):
    """Tests instrument selection for a monthly expiry."""
    mock_kite.instruments.return_value = [
        {
            "name": "NIFTY",
            "segment": "NFO-OPT",
            "expiry": datetime(2024, 8, 27).date(),
            "instrument_type": "CE",
            "strike": 19500,
            "tradingsymbol": "NIFTY24AUG19500CE",
            "instrument_token": 5,
            "lot_size": 75,
        },
        {
            "name": "NIFTY",
            "segment": "NFO-OPT",
            "expiry": datetime(2024, 8, 27).date(),
            "instrument_type": "PE",
            "strike": 19500,
            "tradingsymbol": "NIFTY24AUG19500PE",
            "instrument_token": 6,
            "lot_size": 75,
        },
    ]
    tokens = get_instrument_tokens(
        kite_instance=mock_kite,
    )
    assert tokens is not None
    assert tokens["expiry"] == "2024-08-27"
    assert tokens["lot_size"] == 75
    assert tokens.get("prewarm_tokens", {}).get("ce")


@freeze_time("2024-08-05 10:00:00+05:30")
def test_get_instrument_tokens_missing_trade_symbol(mock_kite, monkeypatch):
    """Returns None when trade symbol not found in instruments."""
    monkeypatch.setattr(settings.instruments, "trade_symbol", "FAKE", raising=False)
    tokens = get_instrument_tokens(kite_instance=mock_kite)
    assert tokens is None


@freeze_time("2024-08-05 10:00:00+05:30")
def test_get_instrument_tokens_reports_missing_option_tokens(mock_kite):
    mock_kite.instruments.return_value = [
        {
            "name": "NIFTY",
            "segment": "NFO-OPT",
            "expiry": datetime(2024, 8, 6).date(),
            "instrument_type": "CE",
            "strike": 19500,
            "tradingsymbol": "NIFTY2480819500CE",
            "instrument_token": 1,
        },
    ]
    tokens = get_instrument_tokens(kite_instance=mock_kite)
    assert tokens is not None
    assert tokens.get("error") == "no_option_token"


@freeze_time("2024-08-05 10:00:00+05:30")
def test_get_instrument_tokens_refreshes_dump(mock_kite, monkeypatch):
    monkeypatch.setattr(settings.instruments, "strike_range", 0, raising=False)
    dump1 = [
        {
            "name": "NIFTY",
            "segment": "NFO-OPT",
            "expiry": datetime(2024, 8, 6).date(),
            "instrument_type": "CE",
            "strike": 19550,
            "instrument_token": 10,
        }
    ]
    dump2 = dump1 + [
        {
            "name": "NIFTY",
            "segment": "NFO-OPT",
            "expiry": datetime(2024, 8, 6).date(),
            "instrument_type": "PE",
            "strike": 19550,
            "instrument_token": 11,
        }
    ]
    fetch = MagicMock(side_effect=[dump1, dump2])
    monkeypatch.setattr(strike_selector, "_fetch_instruments_nfo", fetch)
    tokens = get_instrument_tokens(kite_instance=mock_kite)
    assert tokens["tokens"] == {"ce": 10, "pe": 11}
    assert "error" not in tokens
    assert fetch.call_count == 2


@freeze_time("2024-08-05 10:00:00+05:30")
def test_today_mode_falls_back_to_next_expiry(mock_kite, monkeypatch):
    """When option_expiry_mode='today' on a non-expiry day, picks next weekly."""
    monkeypatch.setattr(settings.strategy, "option_expiry_mode", "today", raising=False)
    tokens = get_instrument_tokens(kite_instance=mock_kite)
    assert tokens is not None
    assert tokens["expiry"] == "2024-08-06"
