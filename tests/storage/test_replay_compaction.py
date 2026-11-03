"""Completed replay compression cannot change or fabricate archived market data."""

from datetime import datetime, timedelta, timezone

import pytest

from nifty_scalper_bot.storage.replay_archive import ReplayArchive, load_session
from nifty_scalper_bot.storage.replay_compaction import (
    compact_completed_session,
    compact_prior_sessions,
)

SESSION = datetime(2026, 11, 3, 4, 0, tzinfo=timezone.utc)
AFTER = datetime(2026, 11, 4, 0, 30, tzinfo=timezone.utc)


def _archive(directory):
    recorder = ReplayArchive(directory)
    recorder.record(
        "snapshot",
        {"basket": {"selected_ce": "NFO:NIFTY26N0323750CE"}, "history": {}},
        SESSION,
    )
    recorder.record(
        "tick",
        {
            "symbol": "NFO:NIFTY26N0323750CE",
            "source": "ws",
            "bid": 28.25,
            "ask": 28.35,
            "depth_available": True,
            "tradable_quote": True,
        },
        SESSION + timedelta(seconds=1),
    )
    recorder.close()
    return directory / "2026-11-03.jsonl"


def test_completed_capture_compacts_losslessly_and_prunes_raw(tmp_path):
    original = _archive(tmp_path)
    expected = load_session(original)
    result = compact_completed_session(original, now=AFTER)
    compressed = tmp_path / "2026-11-03.jsonl.gz"
    assert result["complete_integrity_verified"] is True
    assert result["events"] == 2
    assert compressed.exists()
    assert not original.exists()
    assert load_session(compressed) == expected


def test_live_day_cannot_be_compacted(tmp_path):
    original = _archive(tmp_path)
    with pytest.raises(ValueError, match="live_session_forbidden"):
        compact_completed_session(original, now=SESSION)
    assert original.exists()


def test_corrupt_capture_never_deletes_original(tmp_path):
    original = _archive(tmp_path)
    original.write_text(original.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="replay_capture_incomplete"):
        compact_completed_session(original, now=AFTER)
    assert original.exists()
    assert not (tmp_path / "2026-11-03.jsonl.gz").exists()


def test_compaction_skips_market_and_preopen_hours(tmp_path):
    original = _archive(tmp_path)
    assert (
        compact_prior_sessions(
            tmp_path, now=datetime(2026, 11, 4, 9, 30, tzinfo=timezone.utc)
        )
        == []
    )
    assert original.exists()


def test_compaction_rejects_corrupted_existing_gzip(tmp_path):
    original = _archive(tmp_path)
    target = tmp_path / "2026-11-03.jsonl.gz"
    target.write_bytes(b"not gzip")
    with pytest.raises((ValueError, OSError)):
        compact_completed_session(original, now=AFTER)
    assert original.exists()
