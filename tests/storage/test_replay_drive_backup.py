"""Prove off-market Google Drive backups preserve replay evidence and fail closed."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.storage.replay_archive import ReplayArchive, load_session
from nifty_scalper_bot.storage.replay_compaction import compact_completed_session
from nifty_scalper_bot.storage.replay_drive_backup import backup_prior_sessions

IST = ZoneInfo("Asia/Kolkata")
SESSION = datetime(2026, 11, 4, 9, 15, tzinfo=IST)
AFTER = datetime(2026, 11, 5, 1, 0, tzinfo=IST)


def _source(tmp_path: Path) -> Path:
    directory = tmp_path / "replay_archive"
    archive = ReplayArchive(directory)
    archive.record(
        "snapshot",
        {
            "basket": {"selected_ce": "NFO:NIFTY26N0923850CE"},
            "initial_balance": 100000,
        },
        SESSION,
    )
    archive.record(
        "tick",
        {"symbol": "NFO:NIFTY26N0923850CE", "ltp": 100.5},
        SESSION,
    )
    archive.close()
    original = directory / "2026-11-04.jsonl"
    assert len(load_session(original)) == 2
    result = compact_completed_session(original, now=AFTER)
    assert result["complete_integrity_verified"] is True
    return directory / "2026-11-04.jsonl.gz"


def _fake_rclone(remote_data: dict[str, bytes]):
    def execute(args, **_kwargs):
        operation = args[1]
        if operation == "copyto":
            remote_data[args[3]] = Path(args[2]).read_bytes()
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if operation == "md5sum":
            data = remote_data[args[2]]
            md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
            return subprocess.CompletedProcess(
                args, 0, stdout=f"{md5}  file\n", stderr=""
            )
        raise AssertionError(operation)

    return execute


def test_uploads_verified_tape_and_integrity_with_no_live_io(tmp_path, monkeypatch):
    source = _source(tmp_path)
    (tmp_path / "research" / "fidelity").mkdir(parents=True)
    (tmp_path / "research" / "fidelity" / "2026-11-04.json").write_text(
        json.dumps({"archive_integrity_verified": True})
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.storage.replay_drive_backup.shutil.which",
        lambda _name: "/usr/bin/rclone",
    )
    remote = {}
    fake = _fake_rclone(remote)
    before_open = backup_prior_sessions(
        tmp_path,
        remote="niftybackup:",
        now=SESSION,
        runner=fake,
    )
    assert before_open == []
    assert not remote

    reports = backup_prior_sessions(
        tmp_path, remote="niftybackup:", now=AFTER, runner=fake
    )
    assert reports[0]["status"] == "uploaded_verified"
    assert reports[0]["files"] == 4
    prefix = "niftybackup:2026/2026-11-04/"
    assert remote[prefix + "2026-11-04.jsonl.gz"] == source.read_bytes()
    assert prefix + "capture-fidelity.json" in remote
    assert prefix + "2026-11-04.manifest.json" in remote
    assert any(key.endswith(".integrity.json") for key in remote)
    receipt = json.loads(
        (tmp_path / "replay_drive_backups" / "2026-11-04.receipt.json").read_text()
    )
    assert receipt["remote_md5_verified"] is True
    assert receipt["session_date_ist"] == "2026-11-04"
    # No retransfers once an immutable, verified receipt is present.
    assert (
        backup_prior_sessions(tmp_path, remote="niftybackup:", now=AFTER, runner=fake)
        == []
    )


def test_checksum_mismatch_never_marks_uploaded_or_removes_source(
    tmp_path, monkeypatch
):
    source = _source(tmp_path)
    monkeypatch.setattr(
        "nifty_scalper_bot.storage.replay_drive_backup.shutil.which",
        lambda _name: "/usr/bin/rclone",
    )

    def corrupt_remote(args, **_kwargs):
        if args[1] == "copyto":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args, 0, stdout=f"{'0' * 32}  file\n", stderr=""
        )

    with pytest.raises(ValueError, match="remote_checksum_mismatch"):
        backup_prior_sessions(
            tmp_path,
            remote="niftybackup:",
            now=AFTER,
            runner=corrupt_remote,
        )
    assert source.exists()
    assert not (tmp_path / "replay_drive_backups" / "2026-11-04.receipt.json").exists()


def test_unconfigured_and_invalid_remotes_are_safe(tmp_path, monkeypatch):
    _source(tmp_path)
    assert backup_prior_sessions(tmp_path, remote="", now=AFTER) == []
    with pytest.raises(ValueError, match="remote_name_invalid"):
        backup_prior_sessions(tmp_path, remote="nifty:; rm -rf /", now=AFTER)
    monkeypatch.setattr(
        "nifty_scalper_bot.storage.replay_drive_backup.shutil.which",
        lambda _name: None,
    )
    with pytest.raises(RuntimeError, match="rclone_not_installed"):
        backup_prior_sessions(tmp_path, remote="niftybackup:", now=AFTER)
