import json
from datetime import datetime, timezone

from nifty_scalper_bot.storage.replay_archive import ReplayArchive, load_session


def test_archive_preserves_depth_basket_history_and_sanitizes_config(tmp_path):
    archive = ReplayArchive(tmp_path)
    now = datetime(2026, 10, 1, 4, 0, tzinfo=timezone.utc)
    archive.record(
        "snapshot",
        {
            "basket": {"selected_ce_token": 123},
            "history": {},
            "configuration": {"ORB_TARGET_RR": "1.8"},
        },
        now,
    )
    archive.record(
        "tick",
        {
            "symbol": "NFO:NIFTY26O0125000CE",
            "ltp": 100,
            "depth": {"buy": [{"price": 99, "quantity": 65}]},
            "instrument_token": 123,
        },
        now,
    )
    archive.close()
    events = load_session(tmp_path / "2026-10-01.jsonl")
    assert [row["kind"] for row in events] == ["snapshot", "tick"]
    assert events[1]["payload"]["depth"]["buy"][0]["quantity"] == 65
    assert events[0]["sequence"] < events[1]["sequence"]


def test_archive_rejects_corrupted_or_truncated_event(tmp_path):
    import pytest

    path = tmp_path / "2026-10-01.jsonl"
    path.write_text('{"sequence": 1}\n')
    with pytest.raises(ValueError, match="invalid_replay_event"):
        load_session(path)


def test_capture_environment_excludes_credentials(monkeypatch):
    from nifty_scalper_bot.storage.replay_archive import capture_environment

    monkeypatch.setenv("ORB_TARGET_RR", "1.8")
    monkeypatch.setenv("BROKER_API_KEY", "SECRET")
    monkeypatch.setenv("RISK_API_SECRET", "SECRET")
    assert capture_environment()["ORB_TARGET_RR"] == "1.8"
    assert "SECRET" not in json.dumps(capture_environment())


def test_archive_rejects_missing_tail_even_without_sequence_gap(tmp_path):
    import pytest

    archive = ReplayArchive(tmp_path)
    now = datetime(2026, 10, 1, 4, 0, tzinfo=timezone.utc)
    archive.record("snapshot", {}, now)
    archive.record("tick", {"symbol": "NSE:NIFTY 50", "ltp": 25000}, now)
    archive.close()
    path = tmp_path / "2026-10-01.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="replay_capture_incomplete"):
        load_session(path)


def test_writer_size_cap_invalidates_evidence(tmp_path):
    import pytest

    archive = ReplayArchive(tmp_path, max_session_bytes=300)
    now = datetime(2026, 10, 1, 4, 0, tzinfo=timezone.utc)
    archive.record("snapshot", {}, now)
    archive.record("tick", {"large": "x" * 1000}, now)
    archive.close()
    assert archive.stats()["failed"] == 1
    with pytest.raises(ValueError, match="replay_capture_incomplete"):
        load_session(tmp_path / "2026-10-01.jsonl")


def test_recorder_serializes_only_on_background_writer(tmp_path, monkeypatch):
    """FULL quote conversion must not run on the WebSocket/tick callback."""
    import threading

    import nifty_scalper_bot.storage.replay_archive as archive_module

    observed_threads = []
    original = archive_module.to_json_safe

    def inspect_conversion(payload):
        observed_threads.append(threading.current_thread().name)
        return original(payload)

    monkeypatch.setattr(archive_module, "to_json_safe", inspect_conversion)
    archive = ReplayArchive(tmp_path)
    t = datetime(2026, 11, 3, 4, 0, tzinfo=timezone.utc)
    archive.record("snapshot", {"basket": {"selected_ce": "NFO:CE"}}, t)
    for index in range(200):
        archive.record(
            "tick",
            {
                "symbol": "NFO:NIFTY26NOV24000CE",
                "ltp": 101.0 + index / 10,
                "source": "ws",
                "depth": {
                    "buy": [{"price": 100.0, "quantity": 75}],
                    "sell": [{"price": 101.0, "quantity": 75}],
                },
            },
            t,
        )
    archive.close()
    assert observed_threads
    assert set(observed_threads) == {"replay-archive"}
    events = load_session(tmp_path / "2026-11-03.jsonl")
    assert len(events) == 201
    assert events[-1]["sequence"] == 201
    stats = archive.stats()
    assert stats["written"] == 201
    assert stats["failed"] == 0
    assert stats["dropped"] == 0
    assert stats["queue_capacity"] >= 201


def test_recorder_upstream_coalescing_invalidates_replay(tmp_path):
    import pytest

    upstream = {
        "raw_ws_receipts": 10,
        "submitted": 9,
        "processed": 9,
        "dropped": 0,
        "coalesced": 0,
        "legacy_queue_dropped": 0,
    }
    archive = ReplayArchive(tmp_path, upstream_provider=lambda: dict(upstream))
    now = datetime(2026, 11, 3, 4, 0, tzinfo=timezone.utc)
    archive.record("snapshot", {"basket": {}}, now)
    archive.record("tick", {"symbol": "NSE:NIFTY", "ltp": 100}, now)
    upstream["raw_ws_receipts"] += 2
    upstream["submitted"] += 2
    upstream["processed"] += 1
    upstream["coalesced"] += 1
    archive.close()
    sidecar = json.loads((tmp_path / f"{archive.run_id}.integrity.json").read_text())
    assert sidecar["upstream_start"]["coalesced"] == 0
    assert sidecar["upstream_latest"]["coalesced"] == 1
    with pytest.raises(ValueError, match="replay_capture_upstream_loss"):
        load_session(tmp_path / "2026-11-03.jsonl")


def test_recorder_reports_bounded_queue_and_writer_lag(tmp_path):
    archive = ReplayArchive(tmp_path, capacity=3)
    now = datetime(2026, 11, 3, 4, 0, tzinfo=timezone.utc)
    archive.record("snapshot", {}, now)
    archive.close()
    sidecar = json.loads((tmp_path / f"{archive.run_id}.integrity.json").read_text())
    assert sidecar["max_pending"] >= 1
    assert sidecar["max_pending"] <= 3
    assert sidecar["queue_capacity"] == 3
    assert sidecar["max_writer_lag_ms"] >= 0.0
    assert sidecar["written_by_day"]["2026-11-03"] == 1


def test_archive_freezes_nested_depth_before_writer_runs(tmp_path):
    archive = ReplayArchive(tmp_path)
    instant = datetime(2026, 11, 3, 4, 0, tzinfo=timezone.utc)
    archive.record("snapshot", {"basket": {}}, instant)
    raw = {
        "symbol": "NFO:NIFTY26NOV24000PE",
        "ltp": 90.0,
        "depth": {
            "buy": [{"price": 89.95, "quantity": 25}],
            "sell": [{"price": 90.05, "quantity": 50}],
        },
        "ohlc": {"open": 90.0},
    }
    archive.record("tick", raw, instant)
    raw["depth"]["buy"][0]["price"] = 999.0
    raw["ohlc"]["open"] = 999.0
    archive.close()
    observed = load_session(tmp_path / "2026-11-03.jsonl")[1]["payload"]
    assert observed["depth"]["buy"][0]["price"] == 89.95
    assert observed["ohlc"]["open"] == 90.0
