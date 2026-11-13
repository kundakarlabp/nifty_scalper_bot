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
_TOKENS = {_CE1: 101, _CE2: 102, _PE: 103, _FUT: 104, _SPOT: 105}


def _snapshot(ce: str) -> dict:
    return {
        "instruments": [
            {
                "instrument_token": token,
                "tradingsymbol": symbol.split(":")[-1],
                "exchange": symbol.split(":")[0],
            }
            for symbol, token in _TOKENS.items()
        ],
        "basket": {
            "selected_ce": ce,
            "selected_pe": _PE,
            "futures_symbol": _FUT,
            "token_by_symbol": dict(_TOKENS),
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
        "instrument_token": _TOKENS[symbol],
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
    assert grouped[60]["quote_valid"] == 2, [
        o["status"] for o in result["outcomes"] if o["horizon_seconds"] == 60
    ]
    assert grouped[60]["cost_modelled"] == 2
    assert grouped[60]["strict_full_depth_valid"] == 2
    assert grouped[60]["strict_full_depth_missing"] == 0
    assert grouped[900]["strict_full_depth_valid"] == 0
    assert all(
        epoch["full_book_token_verified_ticks"] > 0 for epoch in result["epochs"]
    )
    assert all(
        sum(epoch["full_book_tick_reasons"].values())
        == epoch["selected_option_full_book_candidate_ticks"]
        for epoch in result["epochs"]
    )
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


def test_epoch_decision_funnel_accounts_for_missing_setup_ownership(
    tmp_path, monkeypatch
):
    session, _runs, times = _two_run_source(tmp_path)
    db = tmp_path / "trades.db"
    db.touch()
    first = _row(times[0], _CE1)
    repeated = _row(times[0] + 1, _CE1)
    missing_setup = _row(times[0], _CE1)
    missing_setup["meta"]["research_context"].pop("setup_id")
    missing_strategy = _row(times[0], _CE1)
    missing_strategy["meta"]["research_context"].pop("strategy")
    invalid_side = _row(times[0], _CE1)
    invalid_side["meta"]["research_context"]["contract_side"] = "BUY"
    not_candidate = _row(times[0], _CE1)
    not_candidate["event_name"] = "decision.other"
    decisions = [
        _row(times[0] - 86400, _CE1, "yesterday"),
        first,
        repeated,
        missing_setup,
        missing_strategy,
        invalid_side,
        not_candidate,
        _row(times[0] + 15 * 60, _CE1, "between_epochs"),
        _row(times[1], _CE2),
    ]
    monkeypatch.setattr(audit, "load_candidate_decision_rows", lambda _db: decisions)
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    result = audit.audit_original_session(session, db)
    assert result["journal_same_day_decision_rows"] == 8
    assert result["journal_assigned_same_epoch_decision_rows"] == 7
    assert result["journal_canonicalized_decision_rows"] == 3
    assert result["journal_excluded_before_canonicalization"] == 4
    assert result["journal_unassigned_same_day_decision_rows"] == {
        "decision_outside_valid_recording_epochs": 1
    }
    first_epoch = result["epochs"][0]["canonical_inclusion_funnel"]
    assert first_epoch["canonicalized_decision_rows"] == 2
    assert first_epoch["independent_setup_count"] == 1
    assert first_epoch["repeated_decisions_collapsed"] == 1
    assert first_epoch["exclusion_reasons"] == {
        "invalid_option_side": 1,
        "missing_owned_setup_id": 1,
        "missing_owned_strategy": 1,
        "not_candidate_decision_event": 1,
    }
    assert result["independent_same_epoch_setups"] == 2
    assert result["promotion_eligible"] is False


def test_missing_futures_quote_context_remains_explicitly_unlabelled(
    tmp_path, monkeypatch
):
    folder = tmp_path / "replay_archive"
    now = datetime.fromisoformat("2026-11-11T09:20:00+05:30")
    archive = ReplayArchive(folder)
    archive.record("snapshot", _snapshot(_CE1), now)
    for symbol in (_CE1, _PE, _SPOT):
        archive.record("tick", _tick(symbol), now + timedelta(seconds=5))
    archive.close()
    db = tmp_path / "trades.db"
    db.touch()
    monkeypatch.setattr(
        audit,
        "load_candidate_decision_rows",
        lambda _db: [_row((now + timedelta(seconds=5)).timestamp(), _CE1)],
    )
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    report = audit.audit_original_session(folder / "2026-11-11.jsonl", db)
    assert report["original_writer_integrity_verified"] is True
    first = report["epochs"][0]
    assert first["selected_quote_context_available"] is False
    assert first["selected_quote_context_missing_by_requirement"] == {
        "FUT_no_ticks": 1,
        "FUT_no_websocket_ticks": 1,
    }
    assert report["independent_same_epoch_setups"] == 1
    assert all(
        row["status"] == "epoch_selected_quote_context_incomplete"
        and row["markout"] is None
        and row["cost_scenarios"] is None
        for row in report["outcomes"]
    )


def test_recorded_full_book_requires_actual_token_and_all_five_levels():
    full = _tick(_CE1)
    reason, book = audit._recorded_full_book_quote(full, _TOKENS[_CE1])
    assert reason == "recorded_full_book_token_verified"
    assert book == (99.0, 101.0)

    wrong_token = dict(full, instrument_token=998)
    assert audit._recorded_full_book_quote(wrong_token, _TOKENS[_CE1])[0] == (
        "tick_instrument_token_mismatch"
    )
    assert audit._recorded_full_book_quote(full, None)[0] == (
        "selected_catalog_token_unverified"
    )
    missing_token = {k: v for k, v in full.items() if k != "instrument_token"}
    assert audit._recorded_full_book_quote(missing_token, _TOKENS[_CE1])[0] == (
        "tick_instrument_token_missing"
    )

    incomplete = dict(
        full,
        depth={
            "buy": full["depth"]["buy"][:4],
            "sell": full["depth"]["sell"],
        },
    )
    assert audit._recorded_full_book_quote(incomplete, _TOKENS[_CE1])[0] == (
        "five_level_depth_missing"
    )
    reversed_bid = dict(
        full,
        depth={
            "buy": list(reversed(full["depth"]["buy"])),
            "sell": full["depth"]["sell"],
        },
    )
    assert audit._recorded_full_book_quote(reversed_bid, _TOKENS[_CE1])[0] == (
        "five_level_order_invalid"
    )
    duplicate_bid_prices = dict(
        full,
        depth={
            "buy": [dict(full["depth"]["buy"][0]) for _ in range(5)],
            "sell": full["depth"]["sell"],
        },
    )
    assert (
        audit._recorded_full_book_quote(duplicate_bid_prices, _TOKENS[_CE1])[0]
        == "five_level_order_invalid"
    )
    duplicate_ask_prices = dict(
        full,
        depth={
            "buy": full["depth"]["buy"],
            "sell": [dict(full["depth"]["sell"][0]) for _ in range(5)],
        },
    )
    assert (
        audit._recorded_full_book_quote(duplicate_ask_prices, _TOKENS[_CE1])[0]
        == "five_level_order_invalid"
    )

    conflicting_top = dict(full, bid=98.0)
    assert audit._recorded_full_book_quote(conflicting_top, _TOKENS[_CE1])[0] == (
        "top_of_book_depth_conflict"
    )
    assert (
        audit._recorded_full_book_quote(dict(full, source="rest"), _TOKENS[_CE1])[0]
        == "non_full_ws_source"
    )


def test_catalog_token_requires_continuous_snapshot_identity():
    snapshot = _snapshot(_CE1)
    catalog = audit._recorded_option_tokens(snapshot)
    assert catalog[_CE1] == _TOKENS[_CE1]
    assert audit._selected_snapshot_token(_CE1, snapshot, catalog) == (_TOKENS[_CE1])
    snapshot["basket"]["token_by_symbol"][_CE1] = 909
    assert audit._selected_snapshot_token(_CE1, snapshot, catalog) is None


def test_full_book_evidence_is_not_an_executable_broker_fill(tmp_path, monkeypatch):
    session, _runs, times = _two_run_source(tmp_path)
    db = tmp_path / "trades.db"
    db.touch()
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setattr(
        audit,
        "load_candidate_decision_rows",
        lambda _db: [_row(times[0], _CE1)],
    )
    result = audit.audit_original_session(session, db)
    record = next(x for x in result["outcomes"] if x["horizon_seconds"] == 60)
    assert record["strict_full_depth_markout"] is not None
    assert record["strict_full_depth_status"] == "observed_bid_path_not_broker_fill"
    assert result["actual_broker_fills_verified"] is False
    assert result["promotion_eligible"] is False


def test_mid_epoch_token_identity_gap_cannot_bridge_full_book_markout(
    tmp_path, monkeypatch
):
    directory = tmp_path / "replay_archive"
    start = datetime.fromisoformat("2026-11-11T09:20:00+05:30")
    recorder = ReplayArchive(directory)
    recorder.record("snapshot", _snapshot(_CE1), start)
    for symbol in (_CE1, _PE, _FUT, _SPOT):
        recorder.record("tick", _tick(symbol), start + timedelta(seconds=3))
    recorder.record("tick", _tick(_CE1), start + timedelta(seconds=9))
    invalid = _snapshot(_CE1)
    invalid["basket"]["token_by_symbol"][_CE1] = 999
    recorder.record("snapshot", invalid, start + timedelta(seconds=10))
    recorder.record("tick", _tick(_CE1), start + timedelta(seconds=11))
    recorder.record("snapshot", _snapshot(_CE1), start + timedelta(seconds=13))
    recorder.record("tick", _tick(_CE1), start + timedelta(seconds=14))
    for offset in range(20, 86, 5):
        recorder.record("tick", _tick(_CE1), start + timedelta(seconds=offset))
    recorder.close()

    db = tmp_path / "trades.db"
    db.touch()
    decisions = [
        _row((start + timedelta(seconds=5)).timestamp(), _CE1, "before_gap"),
        _row((start + timedelta(seconds=11)).timestamp(), _CE1, "during_gap"),
        _row((start + timedelta(seconds=15)).timestamp(), _CE1, "restored"),
    ]
    monkeypatch.setattr(audit, "load_candidate_decision_rows", lambda _db: decisions)
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    result = audit.audit_original_session(directory / "2026-11-11.jsonl", db)
    assert result["independent_same_epoch_setups"] == 3
    sixty = [r for r in result["outcomes"] if r["horizon_seconds"] == 60]
    by_time = {int(r["decision_timestamp"]): r for r in sixty}
    for offset in (5, 11):
        item = by_time[int((start + timedelta(seconds=offset)).timestamp())]
        assert item["strict_full_depth_markout"] is None
        assert item["strict_full_depth_status"] == (
            "selected_quote_identity_phase_changed_during_horizon"
        )
    restored = by_time[int((start + timedelta(seconds=15)).timestamp())]
    assert restored["strict_full_depth_markout"] is not None
    assert restored["strict_full_depth_status"] == "observed_bid_path_not_broker_fill"
    epoch = result["epochs"][0]
    assert epoch["full_book_tick_reasons"]["selected_catalog_token_unverified"] == 1
    assert epoch["full_book_token_verified_ticks"] > 0
    assert sum(epoch["full_book_tick_reasons"].values()) == (
        epoch["selected_option_full_book_candidate_ticks"]
    )
