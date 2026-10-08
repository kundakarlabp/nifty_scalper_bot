"""Offline archive orchestration uses the existing read-only broker seam."""

import datetime as dt
import json
import os

import pytest
from scripts import archive_kite_history as collector
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
    assert len(broker.calls) == 10  # bounded retries; failures remain retryable


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


@pytest.mark.parametrize("existing_token", [None, "existing-session"])
def test_cli_loads_external_env_without_overwriting_existing_session(
    tmp_path, monkeypatch, capsys, existing_token
):
    env_file = tmp_path / "operator.env"
    env_file.write_text("BROKER_ACCESS_TOKEN=file-session\n")
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.delenv("BROKER_ACCESS_TOKEN", raising=False)
    if existing_token is not None:
        monkeypatch.setenv("BROKER_ACCESS_TOKEN", existing_token)
    seen = []

    class OfflineBroker(Broker):
        def __init__(self):
            super().__init__()
            seen.append(os.environ.get("BROKER_ACCESS_TOKEN"))

        def close(self):
            seen.append("closed")

    monkeypatch.setattr(collector, "ZerodhaKiteClient", OfflineBroker)
    assert (
        collector.main(
            [
                "--env-file",
                str(env_file),
                "--symbols",
                "NFO:NIFTY26OCT25000CE",
                "--start",
                "2026-09-01",
                "--end",
                "2026-09-02",
                "--outdir",
                str(tmp_path / "archive"),
            ]
        )
        == 0
    )
    assert seen == [existing_token or "file-session", "closed"]
    assert (
        json.loads((tmp_path / "archive/coverage.json").read_text())["saved_requests"]
        == 1
    )
    output = capsys.readouterr().out
    assert "file-session" not in output and "existing-session" not in output


def test_cli_missing_external_env_fails_before_broker_access(tmp_path, monkeypatch):
    broker_calls = []
    monkeypatch.setattr(
        collector, "ZerodhaKiteClient", lambda: broker_calls.append(True)
    )
    with pytest.raises(FileNotFoundError, match="Environment file does not exist"):
        collector.main(
            [
                "--env-file",
                str(tmp_path / "absent.env"),
                "--symbols",
                "NFO:NIFTY26OCT25000CE",
                "--start",
                "2026-09-01",
                "--end",
                "2026-09-02",
            ]
        )
    assert broker_calls == []


def test_shared_history_cache_reuses_valid_bytes_between_jobs(tmp_path):
    broker = Broker()
    common = (
        broker,
        ["NFO:NIFTY26OCT25000CE"],
        dt.date(2026, 9, 1),
        dt.date(2026, 9, 2),
    )
    archive_history(*common, tmp_path / "one", cache_dir=tmp_path / "shared")
    second = archive_history(*common, tmp_path / "two", cache_dir=tmp_path / "shared")
    assert second["cached_requests"] == 1
    assert len(broker.calls) == 1
    assert (
        next((tmp_path / "one/candles").glob("*.json")).read_bytes()
        == next((tmp_path / "two/candles").glob("*.json")).read_bytes()
    )


def test_history_universe_matches_canonical_active_expiry_and_option_cap(monkeypatch):
    from scripts.archive_kite_history import history_universe

    monkeypatch.setenv("MAX_ACTIVE_OPTION_SYMBOLS", "8")
    rows = []
    token = 1000
    for expiry, expiry_code in (("2026-10-13", "13"), ("2026-10-20", "20")):
        for strike in range(22000, 23001, 50):
            for side in ("CE", "PE"):
                token += 1
                rows.append(
                    {
                        "exchange": "NFO",
                        "tradingsymbol": f"NIFTY26O{expiry_code}{strike}{side}",
                        "name": "NIFTY",
                        "instrument_type": side,
                        "instrument_token": token,
                        "expiry": expiry,
                        "strike": strike,
                        "lot_size": 65,
                    }
                )

    selected = {
        "ce": "NFO:NIFTY26O1322500CE",
        "pe": "NFO:NIFTY26O1322500PE",
    }
    universe = history_universe(rows, selected, "NFO:NIFTY26OCTFUT")

    option_symbols = [
        symbol for symbol in universe if symbol.startswith("NFO:NIFTY26O")
    ]
    assert universe[:2] == ["NSE:NIFTY 50", "NFO:NIFTY26OCTFUT"]
    assert selected["ce"] in option_symbols
    assert selected["pe"] in option_symbols
    assert len(option_symbols) == 8
    assert all("NIFTY26O13" in symbol for symbol in option_symbols)
    assert not any("NIFTY26O20" in symbol for symbol in option_symbols)
