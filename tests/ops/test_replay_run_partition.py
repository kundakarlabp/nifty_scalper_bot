"""Replaying multiple captured process runs must preserve each run's state."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from nifty_scalper_bot.backtesting.runtime_research import validate_replay_epochs
from nifty_scalper_bot.ops.research_jobs import partition_replay_runs
from nifty_scalper_bot.storage.replay_archive import ReplayArchive, archive_fingerprint


def _snapshot() -> dict:
    return {
        "instruments": [{"instrument_token": 1}],
        "basket": {"selected_ce": "NFO:NIFTY26OCT22600CE"},
        "effective_settings": {"risk": {"enabled": True}},
        "runner_config": {"strategy": "test"},
        "initial_balance": 15000.0,
        "initial_positions": [],
    }


def _record_two_runs(tmp_path):
    now = datetime.fromisoformat("2026-10-21T10:00:00+05:30")
    for offset in (0, 2):
        archive = ReplayArchive(tmp_path / "archive")
        archive.record("snapshot", _snapshot(), now + timedelta(minutes=offset))
        archive.record(
            "tick",
            {"symbol": "NFO:NIFTY26OCT22600CE", "ltp": 100 + offset},
            now + timedelta(minutes=offset + 1),
        )
        archive.close()
    return tmp_path / "archive" / "2026-10-21.jsonl"


def test_same_day_restart_replays_as_two_isolated_epochs(tmp_path):
    path = _record_two_runs(tmp_path)
    before = archive_fingerprint(path)
    with pytest.raises(ValueError, match="replay_multiple_runs_unrestored_state"):
        validate_replay_epochs(path)

    runs = partition_replay_runs(path, tmp_path / "epochs")
    assert len(runs) == 2
    assert runs[0][0] != runs[1][0]
    for _run_id, epoch_path in runs:
        event_count, first, snapshots = validate_replay_epochs(epoch_path)
        assert (event_count, snapshots) == (2, 1)
        assert first["initial_positions"] == []
        assert epoch_path.stem == "2026-10-21"
    assert archive_fingerprint(path) == before


def test_single_epoch_reuses_original_verified_archive(tmp_path):
    now = datetime.fromisoformat("2026-10-21T10:00:00+05:30")
    archive = ReplayArchive(tmp_path / "archive")
    archive.record("snapshot", _snapshot(), now)
    archive.close()
    path = tmp_path / "archive" / "2026-10-21.jsonl"

    runs = partition_replay_runs(path, tmp_path / "epochs")
    assert runs == [(archive.run_id, path)]
    assert not (tmp_path / "epochs").exists()


def test_corrupt_archive_is_not_partitioned(tmp_path):
    path = tmp_path / "2026-10-21.jsonl"
    path.write_text('{"bad": true}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        partition_replay_runs(path, tmp_path / "epochs")
    assert not (tmp_path / "epochs").exists()


def test_research_worker_runs_each_epoch_without_live_credentials(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from nifty_scalper_bot.ops.research_jobs import run_recorded_replays

    path = _record_two_runs(tmp_path)
    before = archive_fingerprint(path)
    archive_dir = tmp_path / "data" / "replay_archive"
    archive_dir.mkdir(parents=True)
    for item in path.parent.iterdir():
        (archive_dir / item.name).write_bytes(item.read_bytes())
    monkeypatch.setattr(
        "nifty_scalper_bot.config.paths.get_data_dir", lambda: tmp_path / "data"
    )
    sessions = []

    def worker(command, **kwargs):
        session = command[command.index("--session") + 1]
        output = command[command.index("--output") + 1]
        sessions.append(session)
        assert kwargs["env"]["ENABLE_LIVE"] == "false"
        assert kwargs["env"]["ALLOW_REAL_BROKER"] == "false"
        assert "KITE_API_KEY" not in kwargs["env"]
        assert "--compare-adaptive" in command
        assert command[command.index("--stress-bps") + 1 :] == ["10", "25", "50"]
        (Path(output) / "paired-comparison.json").write_text(
            '{"state":"MODELED_COMPARISON_ONLY","stress":[]}',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("nifty_scalper_bot.ops.research_jobs.subprocess.run", worker)
    report = run_recorded_replays(
        tmp_path,
        {
            "id": "epoch-fixture",
            "start": "2026-10-21",
            "end": "2026-10-21",
        },
    )
    assert report["completed_sessions"] == 2
    assert report["failed_sessions"] == 0
    assert report["state"] == "completed"
    assert report["live_equivalent"] is False
    assert len(set(sessions)) == 2
    scratch = tmp_path / "data/research/epoch-fixture/runtime/2026-10-21/_epochs"
    assert not scratch.exists()
    assert archive_fingerprint(path) == before


def test_partition_rejects_reentry_of_a_previous_process_run(tmp_path):
    path = _record_two_runs(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    repeated = dict(rows[1])
    repeated["sequence"] += 1
    repeated["available_at"] = "2026-10-21T10:04:00+05:30"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(repeated) + "\n")
    identity = rows[0]["run_id"]
    sidecar = path.parent / f"{identity}.integrity.json"
    counters = json.loads(sidecar.read_text())
    counters["written_by_day"]["2026-10-21"] += 1
    sidecar.write_text(json.dumps(counters))
    destination = tmp_path / "epochs"
    with pytest.raises(ValueError, match="replay_epoch_noncontiguous"):
        partition_replay_runs(path, destination)
    assert not destination.exists()


def test_partition_preserves_unrestored_open_position_as_independent_blocker(
    tmp_path,
):
    path = _record_two_runs(tmp_path)
    events = [json.loads(line) for line in path.read_text().splitlines()]
    second_run = events[-1]["run_id"]
    for event in events:
        if event["run_id"] == second_run and event["kind"] == "snapshot":
            event["payload"]["initial_positions"] = [{"quantity": 65}]
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    epochs = partition_replay_runs(path, tmp_path / "epochs")
    assert len(epochs) == 2
    assert validate_replay_epochs(epochs[0][1])[0] == 2
    with pytest.raises(ValueError, match="replay_initial_positions_unsupported"):
        validate_replay_epochs(epochs[1][1])
