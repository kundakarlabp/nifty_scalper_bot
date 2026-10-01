"""Offline archive orchestration uses the existing read-only broker seam."""

import datetime as dt
import json

import pytest
from scripts.archive_kite_history import archive_history


class Broker:
    def __init__(self):
        self.calls = []

    def instruments(self, exchange):
        return [
            {
                "exchange": "NFO",
                "tradingsymbol": "NIFTY26OCT25000CE",
                "name": "NIFTY",
                "instrument_type": "CE",
                "instrument_token": "123",
                "expiry": "2026-10-27",
                "strike": "25000",
                "lot_size": "65",
            }
        ]

    def historical_data(self, token, start, end, interval, continuous=False, oi=False):
        self.calls.append((token, start, end, interval, continuous, oi))
        return [[start.isoformat(), 100, 105, 95, 102, 130, 650]]


def test_kite_archive_chunks_resumes_and_preserves_raw_oi_and_identity(tmp_path):
    broker = Broker()
    args = (
        broker,
        ["NFO:NIFTY26OCT25000CE"],
        dt.date(2026, 8, 1),
        dt.date(2026, 9, 30),
        tmp_path,
    )
    first = archive_history(*args)
    assert first["saved_requests"] == 3
    assert len(broker.calls) == 3
    for token, start, end, interval, continuous, oi in broker.calls:
        assert token == 123
        assert (end.date() - start.date()).days < 30
        assert str(start.tzinfo) == "Asia/Kolkata"
        assert interval == "minute" and oi and not continuous
    files = list(tmp_path.glob("candles/*.json"))
    assert len(files) == 3
    data = json.loads(files[0].read_text())
    assert data["symbol"] == "NFO:NIFTY26OCT25000CE"
    assert data["instrument"]["lot_size"] == "65"
    assert data["candles"][0][-1] == 650
    assert data["timestamp_convention"] == "bar_start"
    second = archive_history(*args)
    assert second["cached_requests"] == 3
    assert len(broker.calls) == 3


def test_unavailable_contract_is_rejected_before_fetching(tmp_path):
    broker = Broker()
    with pytest.raises(ValueError, match="current instrument master"):
        archive_history(
            broker,
            ["NFO:NIFTY22OCT25000CE"],
            dt.date(2026, 9, 1),
            dt.date(2026, 9, 2),
            tmp_path,
        )
    assert not broker.calls


def test_empty_and_failed_requests_remain_explicit_and_retryable(tmp_path):
    class MissingBroker(Broker):
        def historical_data(self, *args, **kwargs):
            self.calls.append(args)
            if len(self.calls) == 1:
                return []
            raise RuntimeError("sensitive upstream request detail")

    broker = MissingBroker()
    args = (
        broker,
        ["NFO:NIFTY26OCT25000CE"],
        dt.date(2026, 8, 1),
        dt.date(2026, 9, 1),
        tmp_path,
    )
    report = archive_history(*args)
    assert len(report["empty_requests"]) == 1
    assert len(report["failed_requests"]) == 1
    assert report["saved_requests"] == 0
    assert not list(tmp_path.glob("candles/*.json"))
    assert "sensitive" not in (tmp_path / "coverage.json").read_text()
    archive_history(*args)
    assert len(broker.calls) == 4


def test_corrupt_cache_is_refetched(tmp_path):
    broker = Broker()
    args = (
        broker,
        ["NFO:NIFTY26OCT25000CE"],
        dt.date(2026, 9, 1),
        dt.date(2026, 9, 2),
        tmp_path,
    )
    archive_history(*args)
    path = next(tmp_path.glob("candles/*.json"))
    path.write_text("truncated json")
    report = archive_history(*args)
    assert report["saved_requests"] == 1
    assert len(broker.calls) == 2
