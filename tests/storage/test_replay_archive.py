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
