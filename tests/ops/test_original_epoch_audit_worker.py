"""Bounded, source-pinned host-local epoch audit through the existing worker."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from nifty_scalper_bot.ops.research_jobs import (
    _same_request_definition,
    require_epoch_audit_offmarket,
    validate_request,
)

_DAY = "2026-11-11"
_SHA = "048bb7d03bd2ac021fe204c06c34bb8542fd678d4ee5968a41376979eb1e5922"


def _payload() -> dict:
    return {
        "id": "original-epoch-test",
        "days": 1,
        "mode": "epoch_audit",
        "session_date": _DAY,
        "expected_uncompressed_sha256": _SHA,
    }


def test_epoch_request_requires_sha_day_and_offmarket_clock():
    after_close = datetime.fromisoformat("2026-11-13T18:00:00+05:30")
    request = validate_request(_payload(), now=after_close)
    assert request["session_date"] == _DAY
    assert request["expected_uncompressed_sha256"] == _SHA
    assert request["mode"] == "epoch_audit"

    for invalid in (
        {**_payload(), "days": 2},
        {**_payload(), "expected_uncompressed_sha256": None},
        {**_payload(), "expected_uncompressed_sha256": "a" * 63},
        {**_payload(), "session_date": None},
        {**_payload(), "profile": "captured_baseline"},
        {**_payload(), "extra_untrusted_path": "/tmp/tape"},
    ):
        with pytest.raises(ValueError):
            validate_request(invalid, now=after_close)

    for blocked in (
        "2026-11-13T05:00:00+05:30",
        "2026-11-13T09:30:00+05:30",
        "2026-11-13T16:59:59+05:30",
    ):
        with pytest.raises(ValueError, match="epoch_audit_off_market_only"):
            validate_request(_payload(), now=datetime.fromisoformat(blocked))
    with pytest.raises(
        ValueError, match="epoch_audit_insufficient_offmarket_budget"
    ):
        validate_request(
            _payload(), now=datetime.fromisoformat("2026-11-13T03:59:00+05:30")
        )
    for permitted in ("2026-11-13T02:59:00+05:30", "2026-11-13T17:00:00+05:30"):
        require_epoch_audit_offmarket(datetime.fromisoformat(permitted))


def test_epoch_request_rejects_digest_substitution_and_other_modes():
    baseline = _payload()
    changed = {**baseline, "expected_uncompressed_sha256": "f" * 64}
    assert not _same_request_definition(baseline, changed)
    with pytest.raises(ValueError, match="source_digest_only_valid"):
        validate_request(
            {**_payload(), "mode": "runtime"},
            now=datetime.fromisoformat("2026-11-13T18:00:00+05:30"),
        )


def test_fixed_manifest_pins_original_source():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "deploy/research_request.json").read_text())
    assert manifest["mode"] == "epoch_audit"
    assert manifest["session_date"] == _DAY
    assert manifest["expected_uncompressed_sha256"] == _SHA


def test_worker_reads_original_local_tape_without_broker_or_archive_fetch(
    tmp_path, monkeypatch
):
    from scripts import run_research_job as worker

    data_dir = tmp_path / "data"
    folder = data_dir / "replay_archive"
    folder.mkdir(parents=True)
    tape = folder / f"{_DAY}.jsonl.gz"
    tape.write_bytes(b"original-unmodified-tape")
    source_bytes = tape.read_bytes()
    journal = data_dir / "trades.db"
    journal.write_bytes(b"canonical-unmodified-journal")
    captured = []

    def audit(session, trade_db, *, expected_uncompressed_sha256):
        captured.append((session, trade_db, expected_uncompressed_sha256))
        return {
            "session_date_ist": _DAY,
            "source_uncompressed_sha256": _SHA,
            "original_epoch_count": 2,
            "original_writer_integrity_verified": True,
            "journal_same_day_decision_rows": 40,
            "journal_assigned_same_epoch_decision_rows": 39,
            "independent_same_epoch_setups": 30,
            "grouped_outcomes": [
                {
                    "strategy": "smc",
                    "horizon_seconds": 60,
                    "strict_full_depth_valid": 4,
                    "strict_full_depth_missing": 26,
                }
            ],
            "epochs": [
                {
                    "event_count": 100,
                    "snapshot_count": 2,
                    "writer_integrity_verified": True,
                    "full_book_token_verified_ticks": 22,
                    "full_book_tick_reasons": {"recorded_full_book_token_verified": 22},
                },
                {
                    "event_count": 200,
                    "snapshot_count": 2,
                    "writer_integrity_verified": True,
                    "full_book_token_verified_ticks": 50,
                    "full_book_tick_reasons": {"recorded_full_book_token_verified": 50},
                },
            ],
            "outcomes": [{"symbol": "NFO:PRIVATE", "markout": {"entry_ask": 10}}],
            "promotion_eligible": False,
            "actual_broker_fills_verified": False,
        }

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr("nifty_scalper_bot.config.paths.get_data_dir", lambda: data_dir)
    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.require_epoch_audit_offmarket",
        lambda moment: None,
    )
    monkeypatch.setattr(
        "scripts.validate_epoch_structural_markouts.audit_original_session",
        audit,
    )
    result = worker.run_worker(_payload(), tmp_path / "no-credentials.env")
    assert captured == [(tape, journal, _SHA)]
    assert result["state"] == "completed"
    assert result["requested_work_completed"] is True
    assert result["epoch_audit_completed"] is True
    assert result["backtest_completed"] is False
    assert result["promotion_eligible"] is False
    assert result["epoch_audit"]["original_epoch_count"] == 2
    assert result["epoch_audit"]["grouped_outcomes"][0]["strict_full_depth_valid"] == 4
    assert "outcomes" not in result["epoch_audit"]
    private = json.loads(
        (data_dir / "research/original-epoch-test/epoch_audit.json").read_text()
    )
    assert private["outcomes"][0]["symbol"] == "NFO:PRIVATE"
    assert tape.read_bytes() == source_bytes
    assert journal.read_bytes() == b"canonical-unmodified-journal"


def test_worker_fails_closed_on_live_mode_and_missing_original(tmp_path, monkeypatch):
    from scripts import run_research_job as worker

    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(ValueError, match="epoch_audit_requires_shadow"):
        worker.run_worker(_payload(), tmp_path / "unused.env")

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setattr(
        "nifty_scalper_bot.ops.research_jobs.require_epoch_audit_offmarket",
        lambda moment: None,
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.config.paths.get_data_dir", lambda: tmp_path / "data"
    )
    with pytest.raises(ValueError, match="epoch_audit_original_tape_missing"):
        worker.run_worker(_payload(), tmp_path / "unused.env")
