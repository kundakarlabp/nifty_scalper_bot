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

    option_symbols = [symbol for symbol in universe if symbol.endswith(("CE", "PE"))]
    assert universe[:2] == ["NSE:NIFTY 50", "NFO:NIFTY26OCTFUT"]
    assert selected["ce"] in option_symbols
    assert selected["pe"] in option_symbols
    assert len(option_symbols) == 8
    assert all("NIFTY26O13" in symbol for symbol in option_symbols)
    assert not any("NIFTY26O20" in symbol for symbol in option_symbols)


def test_checksum_valid_but_out_of_range_cached_candles_are_not_trusted(tmp_path):
    broker = Broker()
    args = (
        broker,
        ["NFO:NIFTY26OCT25000CE"],
        dt.date(2026, 9, 1),
        dt.date(2026, 9, 2),
        tmp_path,
    )
    archive_history(*args)
    path = next((tmp_path / "candles").glob("*.json"))
    payload = json.loads(path.read_text())
    payload["candles"][0][0] = "2026-09-10T09:15:00+05:30"
    payload["candles_sha256"] = collector._candle_digest(payload["candles"])
    path.write_text(json.dumps(payload))

    result = archive_history(*args)

    assert result["saved_requests"] == 1
    assert result["cached_requests"] == 0
    assert len(broker.calls) == 2


def test_retained_cache_inventory_counts_synchronized_historical_contracts(tmp_path):
    directory = tmp_path / "candles"
    directory.mkdir()
    stamp = "2026-10-12T09:30:00+05:30"

    def keep(symbol, kind, token, *, strike=None, valid=True):
        instrument = {
            "exchange": "NFO" if kind != "EQ" else "NSE",
            "tradingsymbol": symbol.split(":", 1)[1],
            "instrument_type": kind,
            "name": "NIFTY",
            "instrument_token": token,
            "expiry": "2026-10-13",
            "strike": strike,
            "lot_size": 65,
        }
        candles = [[stamp, 100, 105, 95, 102, 200, 600]]
        payload = {
            "symbol": symbol,
            "instrument_token": token,
            "interval": "minute",
            "continuous": False,
            "oi": True,
            "timestamp_convention": "bar_start",
            "from": "2026-10-12T00:00:00+05:30",
            "to": "2026-10-12T23:59:59+05:30",
            "captured_at": "2026-10-12T16:00:00+05:30",
            "instrument": instrument,
            "candles": candles,
            "candles_sha256": collector._candle_digest(candles),
        }
        if not valid:
            payload["candles_sha256"] = "forged"
        (directory / f"{token}.json").write_text(json.dumps(payload))

    keep("NSE:NIFTY 50", "EQ", 101)
    keep("NFO:NIFTY26OCTFUT", "FUT", 102)
    keep("NFO:NIFTY26O1322500CE", "CE", 103, strike=22500)
    keep("NFO:NIFTY26O1322500PE", "PE", 104, strike=22500)
    keep("NFO:NIFTY26O1322600CE", "CE", 105, strike=22600, valid=False)
    result = collector.historical_cache_inventory(
        tmp_path, dt.date(2026, 10, 12), dt.date(2026, 10, 12)
    )

    assert result["corrupt_or_untrusted_files"] == 1
    assert result["synchronized_contract_minutes"] == 1
    assert result["sessions"][0]["date"] == "2026-10-12"
    assert result["sessions"][0]["synchronized_contract_minutes"] == 1
    assert result["sessions"][0]["valid_paired_strikes"] == 1
    assert result["promotion_eligible"] is False


def test_retained_cache_inventory_rejects_post_expiry_candles(tmp_path):
    directory = tmp_path / "candles"
    directory.mkdir()
    payload = {
        "symbol": "NFO:NIFTY26O0622500CE",
        "instrument_token": 123,
        "interval": "minute",
        "continuous": False,
        "oi": True,
        "timestamp_convention": "bar_start",
        "from": "2026-10-07T00:00:00+05:30",
        "to": "2026-10-07T23:59:59+05:30",
        "captured_at": "2026-10-05T10:00:00+05:30",
        "instrument": {
            "exchange": "NFO",
            "tradingsymbol": "NIFTY26O0622500CE",
            "instrument_type": "CE",
            "name": "NIFTY",
            "instrument_token": 123,
            "expiry": "2026-10-06",
            "strike": 22500,
            "lot_size": 65,
        },
        "candles": [["2026-10-07T09:30:00+05:30", 100, 105, 95, 102, 200]],
    }
    payload["candles_sha256"] = collector._candle_digest(payload["candles"])
    (directory / "invalid.json").write_text(json.dumps(payload))
    result = collector.historical_cache_inventory(
        tmp_path, dt.date(2026, 10, 7), dt.date(2026, 10, 7)
    )
    assert result["corrupt_or_untrusted_files"] == 1
    assert result["synchronized_contract_minutes"] == 0


def _retained_session_payload(
    symbol,
    kind,
    token,
    *,
    strike=None,
    expiry="2026-10-13",
    first_minute=30,
):
    timestamps = [
        dt.datetime(2026, 10, 12, 9, minute, tzinfo=collector.IST)
        for minute in range(first_minute, 34)
    ]
    candles = [
        [timestamp.isoformat(), 22525, 22535, 22520, 22530, 1000, 12000]
        for timestamp in timestamps
    ]
    instrument = {
        "exchange": symbol.split(":", 1)[0],
        "tradingsymbol": symbol.split(":", 1)[1],
        "name": "NIFTY",
        "instrument_type": kind,
        "instrument_token": token,
        "expiry": expiry,
        "strike": strike,
        "lot_size": 65,
    }
    return {
        "symbol": symbol,
        "instrument_token": token,
        "interval": "minute",
        "oi": True,
        "continuous": False,
        "from": "2026-10-12T00:00:00+05:30",
        "to": "2026-10-12T23:59:59+05:30",
        "captured_at": "2026-10-12T15:36:00+05:30",
        "timestamp_convention": "bar_start",
        "instrument": instrument,
        "candles": candles,
        "candles_sha256": collector._candle_digest(candles),
    }


def test_retained_session_stages_only_asof_synchronized_observed_contracts(tmp_path):
    from nifty_scalper_bot.backtesting.strategy_research import load_archive

    candles = tmp_path / "cache/candles"
    candles.mkdir(parents=True)
    payloads = [
        _retained_session_payload("NSE:NIFTY 50", "EQ", 100),
        _retained_session_payload("NFO:NIFTY26OCTFUT", "FUT", 101, expiry="2026-10-27"),
        _retained_session_payload("NFO:NIFTY26O1322500CE", "CE", 102, strike=22500),
        _retained_session_payload("NFO:NIFTY26O1322500PE", "PE", 103, strike=22500),
        # Later quote discovery cannot change the decision-time universe.
        _retained_session_payload(
            "NFO:NIFTY26O2022500CE",
            "CE",
            104,
            strike=22500,
            expiry="2026-10-20",
            first_minute=31,
        ),
        _retained_session_payload(
            "NFO:NIFTY26O2022500PE",
            "PE",
            105,
            strike=22500,
            expiry="2026-10-20",
            first_minute=31,
        ),
    ]
    for index, payload in enumerate(payloads):
        (candles / f"{index}.json").write_text(json.dumps(payload))
    staging = tmp_path / "staged"
    report = collector.historical_cache_inventory(
        tmp_path / "cache",
        dt.date(2026, 10, 12),
        dt.date(2026, 10, 12),
        stage_dir=staging,
        min_synchronized_minutes=3,
    )
    assert report["staged_session_dates"] == ["2026-10-12"]
    assert report["promotion_eligible"] is False
    histories, instruments, _ = load_archive(staging / "2026-10-12")
    assert set(instruments) == {
        "NSE:NIFTY 50",
        "NFO:NIFTY26OCTFUT",
        "NFO:NIFTY26O1322500CE",
        "NFO:NIFTY26O1322500PE",
    }
    assert all(
        timestamp.date() == dt.date(2026, 10, 12)
        for history in histories.values()
        for timestamp in history
    )


def test_retained_session_ambiguous_futures_fail_closed(tmp_path):
    candles = tmp_path / "cache/candles"
    candles.mkdir(parents=True)
    rows = [
        _retained_session_payload("NSE:NIFTY 50", "EQ", 100),
        _retained_session_payload("NFO:NIFTY26OCTFUT", "FUT", 101, expiry="2026-10-27"),
        _retained_session_payload("NFO:NIFTY26NOVFUT", "FUT", 106, expiry="2026-11-24"),
        _retained_session_payload("NFO:NIFTY26O1322500CE", "CE", 102, strike=22500),
        _retained_session_payload("NFO:NIFTY26O1322500PE", "PE", 103, strike=22500),
    ]
    for index, payload in enumerate(rows):
        (candles / f"{index}.json").write_text(json.dumps(payload))
    staging = tmp_path / "staged"
    report = collector.historical_cache_inventory(
        tmp_path / "cache",
        dt.date(2026, 10, 12),
        dt.date(2026, 10, 12),
        stage_dir=staging,
        min_synchronized_minutes=3,
    )
    assert report["staged_session_dates"] == []
    assert report["staging_rejections"]["2026-10-12"] == (
        "historical_future_selection_ambiguous"
    )
    assert not (staging / "2026-10-12").exists()


def test_retained_session_corrupt_cache_cannot_stage(tmp_path):
    candles = tmp_path / "cache/candles"
    candles.mkdir(parents=True)
    rows = [
        _retained_session_payload("NSE:NIFTY 50", "EQ", 100),
        _retained_session_payload("NFO:NIFTY26OCTFUT", "FUT", 101, expiry="2026-10-27"),
        _retained_session_payload("NFO:NIFTY26O1322500CE", "CE", 102, strike=22500),
        _retained_session_payload("NFO:NIFTY26O1322500PE", "PE", 103, strike=22500),
    ]
    rows[-1]["candles_sha256"] = "not-a-valid-digest"
    for index, payload in enumerate(rows):
        (candles / f"{index}.json").write_text(json.dumps(payload))
    staging = tmp_path / "staged"
    report = collector.historical_cache_inventory(
        tmp_path / "cache",
        dt.date(2026, 10, 12),
        dt.date(2026, 10, 12),
        stage_dir=staging,
        min_synchronized_minutes=3,
    )
    assert report["corrupt_or_untrusted_files"] == 1
    assert report["staged_session_dates"] == []

@pytest.mark.parametrize(
    "failure_type,expected_reason",
    [
        ("same_side_duplicate", "historical_option_selection_ambiguous"),
        ("token_collision", "historical_contract_token_collision"),
    ],
)
def test_retained_session_conflicting_contract_identity_is_not_staged(
    tmp_path, failure_type, expected_reason
):
    candles = tmp_path / "cache/candles"
    candles.mkdir(parents=True)
    rows = [
        _retained_session_payload("NSE:NIFTY 50", "EQ", 100),
        _retained_session_payload(
            "NFO:NIFTY26OCTFUT", "FUT", 101, expiry="2026-10-27"
        ),
        _retained_session_payload(
            "NFO:NIFTY26O1322500CE", "CE", 102, strike=22500
        ),
        _retained_session_payload(
            "NFO:NIFTY26O1322500PE", "PE", 103, strike=22500
        ),
    ]
    if failure_type == "same_side_duplicate":
        rows.append(
            _retained_session_payload(
                "NFO:NIFTY26O13X22500CE", "CE", 105, strike=22500
            )
        )
    else:
        rows[-1]["instrument_token"] = 101
        rows[-1]["instrument"]["instrument_token"] = 101
    for index, payload in enumerate(rows):
        (candles / f"{index}.json").write_text(json.dumps(payload))
    report = collector.historical_cache_inventory(
        tmp_path / "cache",
        dt.date(2026, 10, 12),
        dt.date(2026, 10, 12),
        stage_dir=tmp_path / "staged",
        min_synchronized_minutes=3,
    )
    assert report["staged_session_dates"] == []
    assert report["staging_rejections"]["2026-10-12"] == expected_reason
