"""Read-only captured-epoch setup and quote-alignment regression fixtures."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from scripts import validate_epoch_structural_markouts as audit

from nifty_scalper_bot.storage.replay_archive import ReplayArchive

_CE1 = "NFO:NIFTY26N0323700CE"
_CE2 = "NFO:NIFTY26N0323750CE"
_PE = "NFO:NIFTY26N0323700PE"
_FUT = "NFO:NIFTY26NOVFUT"
_SPOT = "NSE:NIFTY 50"


def _snapshot(ce: str) -> dict:
    return {
        "instruments": [{"instrument_token": 1}],
        "basket": {
            "selected_ce": ce,
            "selected_pe": _PE,
            "futures_symbol": _FUT,
        },
        "effective_settings": {"risk": {"enabled": True}},
        "runner_config": {"strategy": "test"},
        "initial_balance": 15000.0,
        "initial_positions": [],
    }


def _tick(symbol: str) -> dict:
    depth = {
        "buy": [{"price": 99 - i, "quantity": 65} for i in range(5)],
        "sell": [{"price": 101 + i, "quantity": 65} for i in range(5)],
    }
    return {
        "symbol": symbol,
        "source": "ws_full",
        "depth_available": True,
        "tradable_quote": True,
        "bid": 99.0,
        "ask": 101.0,
        "ltp": 100.0,
        "depth": depth,
    }


def _two_run_source(tmp_path: Path):
    folder = tmp_path / "replay_archive"
    day = datetime.fromisoformat("2026-11-11T09:20:00+05:30")
    runs = []
    decision_times = []
    for index, ce in enumerate((_CE1, _CE2)):
        start = day + timedelta(hours=index)
        archive = ReplayArchive(folder)
        archive.record("snapshot", _snapshot(ce), start)
        for symbol in (ce, _PE, _FUT, _SPOT):
            archive.record("tick", _tick(symbol), start + timedelta(seconds=5))
        for sec in range(9, 80, 5):
            archive.record("tick", _tick(ce), start + timedelta(seconds=sec))
        archive.close()
        decision_times.append((start + timedelta(seconds=10)).timestamp())
        runs.append(archive.run_id)
    return folder / "2026-11-11.jsonl", runs, decision_times


def _row(at: float, symbol: str, setup: str = "shared") -> dict:
    return {
        "timestamp": at,
        "event_name": "candidate.approved",
        "symbol": symbol,
        "reason_code": "order_submitted",
        "meta": {
            "research_context": {
                "setup_id": setup,
                "strategy": "smc",
                "contract_side": "CE",
                "lot_size": 65,
            }
        },
    }


def test_original_two_epochs_never_mix_decisions_or_quotes(tmp_path, monkeypatch):
    session, _runs, times = _two_run_source(tmp_path)
    raw_before = hashlib.sha256(session.read_bytes()).hexdigest()
    db = tmp_path / "trades.db"
    db.touch()
    decisions = [
        _row(times[0] - 86400, _CE1, "yesterday"),
        _row(times[0], _CE1),
        _row(times[0] + 15 * 60, _CE1, "between_epochs"),
        _row(times[1], _CE2),
    ]
    monkeypatch.setattr(audit, "load_candidate_decision_rows", lambda _db: decisions)
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    result = audit.audit_original_session(
        session, db, expected_uncompressed_sha256=raw_before
    )
    assert result["original_writer_integrity_verified"] is True
    assert result["original_epoch_count"] == 2
    assert result["independent_same_epoch_setups"] == 2
    assert result["journal_same_day_decision_rows"] == 3
    assert result["journal_out_of_day_decision_rows"] == 1
    assert result["journal_unassigned_same_day_decision_rows"] == {
        "decision_outside_valid_recording_epochs": 1
    }
    grouped = {item["horizon_seconds"]: item for item in result["grouped_outcomes"]}
    assert grouped[60]["quote_valid"] == 2, {
        "coverage": grouped[60],
        "epochs": result["epochs"],
        "outcomes": [o for o in result["outcomes"] if o["horizon_seconds"] == 60],
    }
    assert grouped[60]["cost_modelled"] == 2
    assert grouped[900]["quote_valid"] == 0
    assert grouped[900]["status_counts"] == {"horizon_crosses_epoch_end": 2}
    assert all(item["writer_integrity_verified"] for item in result["epochs"])
    assert result["comparison"]["profitability_inference_allowed"] is False
    assert hashlib.sha256(session.read_bytes()).hexdigest() == raw_before


def test_contract_selection_mismatch_blocks_label(tmp_path, monkeypatch):
    session, _runs, times = _two_run_source(tmp_path)
    db = tmp_path / "trades.db"
    db.touch()
    monkeypatch.setattr(
        audit, "load_candidate_decision_rows", lambda _db: [_row(times[0], _CE2)]
    )
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    result = audit.audit_original_session(session, db)
    item = next(row for row in result["outcomes"] if row["horizon_seconds"] == 60)
    assert item["status"] == "contract_not_selected_at_decision"
    assert item["markout"] is None
    assert item["cost_scenarios"] is None


def test_dropped_capture_and_wrong_sha_fail_closed(tmp_path, monkeypatch):
    session, runs, _times = _two_run_source(tmp_path)
    db = tmp_path / "trades.db"
    db.touch()
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    with pytest.raises(ValueError, match="pinned_uncompressed_sha256_mismatch"):
        audit.audit_original_session(session, db, expected_uncompressed_sha256="0" * 64)
    sidecar = session.parent / f"{runs[0]}.integrity.json"
    metadata = json.loads(sidecar.read_text())
    metadata["dropped"] = 1
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="replay_capture_incomplete"):
        audit.audit_original_session(session, db)


def test_research_only_without_local_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(ValueError, match="epoch_audit_requires_shadow"):
        audit.audit_original_session(
            tmp_path / "missing.jsonl", tmp_path / "missing.db"
        )
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    with pytest.raises(ValueError, match="epoch_audit_original_tape_missing"):
        audit.audit_original_session(
            tmp_path / "missing.jsonl", tmp_path / "missing.db"
        )
