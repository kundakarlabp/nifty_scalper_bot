"""Breeze bridge tests: causal identities, SHA checks and no fabricated fills."""

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from scripts.stage_breeze_sqlite_research import run


def _sources(tmp_path: Path, *, previous: bool = True, gaps: tuple[int, ...] = ()):
    eod_path = tmp_path / "eod.sqlite"
    with sqlite3.connect(eod_path) as db:
        db.execute("CREATE TABLE sessions (day TEXT, available_not_before TEXT)")
        db.execute(
            "CREATE TABLE daily_index (day TEXT, close REAL, available_not_before TEXT)"
        )
        db.execute(
            "CREATE TABLE contract_eod (day TEXT, instrument_name TEXT, expiry TEXT, "
            "strike REAL, lot_size INT, kind TEXT)"
        )
        if previous:
            db.execute("INSERT INTO sessions VALUES ('2025-02-03', '2025-02-04')")
            db.execute(
                "INSERT INTO daily_index VALUES ('2025-02-03', 23500, '2025-02-04')"
            )
            db.executemany(
                "INSERT INTO contract_eod VALUES (?,?,?,?,?,?)",
                [
                    ("2025-02-03", "NIFTY25FEBFUT", "2025-02-27", None, 75, "FUT"),
                    (
                        "2025-02-03",
                        "NIFTY25FEB0623500CE",
                        "2025-02-06",
                        23500,
                        75,
                        "CE",
                    ),
                    (
                        "2025-02-03",
                        "NIFTY25FEB0623500PE",
                        "2025-02-06",
                        23500,
                        75,
                        "PE",
                    ),
                ],
            )
        # Same-day data cannot repair lack of previous EOD master.
        db.execute("INSERT INTO sessions VALUES ('2025-02-04', '2025-02-05')")
        db.execute("INSERT INTO daily_index VALUES ('2025-02-04', 23450, '2025-02-05')")
        db.commit()
    breeze_path = tmp_path / "breeze.sqlite"
    with sqlite3.connect(breeze_path) as db:
        db.execute(
            "CREATE TABLE series (id INT, trade_day TEXT, product_type TEXT, "
            "expiry TEXT, strike INT, side TEXT, observed_minutes INT, "
            "missing_minutes INT, status TEXT)"
        )
        db.execute(
            "CREATE TABLE candles (series_id INT, stamp TEXT, open REAL, high REAL, "
            "low REAL, close REAL, volume INT, oi INT)"
        )
        start = datetime.fromisoformat("2025-02-04T09:15:00+05:30")
        families = [
            (1, "cash", None, None, "INDEX"),
            (2, "futures", "2025-02-27", None, "FUT"),
            (3, "options", "2025-02-06", 23500, "CE"),
            (4, "options", "2025-02-06", 23500, "PE"),
        ]
        for sid, kind, expiry, strike, side in families:
            count = 90 - (len(gaps) if side == "CE" else 0)
            db.execute(
                "INSERT INTO series VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    "2025-02-04",
                    kind,
                    expiry,
                    strike,
                    side,
                    count,
                    375 - count,
                    "PARTIAL",
                ),
            )
            for minute in range(90):
                if side == "CE" and minute in gaps:
                    continue
                price = (100 if kind == "options" else 23500) + minute * 0.1
                db.execute(
                    "INSERT INTO candles VALUES (?,?,?,?,?,?,?,?)",
                    (
                        sid,
                        (start + timedelta(minutes=minute)).isoformat(),
                        price,
                        price + 1,
                        price - 1,
                        price,
                        200,
                        1000,
                    ),
                )
        db.commit()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "verified_windows": 1,
                "bars_1m": 360 - len(gaps),
                "dataset_sha256": hashlib.sha256(breeze_path.read_bytes()).hexdigest(),
                "research_only": True,
                "promotion_eligible": False,
                "execution_grade_ready": False,
            }
        )
    )
    return breeze_path, eod_path, summary_path


def test_stage_prior_session_only_and_load_through_existing_runner(tmp_path):
    breeze, eod, summary = _sources(tmp_path)
    dest = tmp_path / "stage"
    report = run(
        breeze,
        eod,
        dest,
        summary_path=summary,
        expected_windows=1,
        day="2025-02-04",
        permit_partial=True,
    )
    assert report["staged_days"] == 1
    assert report["research_executions_completed"] == 0
    staged = dest / "2025-02-04"
    manifest = json.loads((staged / "manifest.json").read_text())
    assert manifest["continuous_minutes_staged"] == 90
    assert manifest["missing_or_unsynchronized_minutes"] == 285
    assert not manifest["execution_grade_ready"]
    assert not manifest["historical_token_verified"]
    assert manifest["selected"]["CE"]["symbol"].endswith("CE")
    from nifty_scalper_bot.backtesting.strategy_research import load_archive

    histories, instruments, digest = load_archive(staged)
    assert len(histories) == 4
    assert len(instruments) == 4
    assert len(digest) == 64


def test_partial_session_not_silently_accepted_by_default(tmp_path):
    breeze, eod, summary = _sources(tmp_path)
    result = run(
        breeze,
        eod,
        tmp_path / "staged",
        summary_path=summary,
        expected_windows=1,
        day="2025-02-04",
    )
    assert result["staged_days"] == 0
    assert result["rejections"] == {"synchronized_minute_coverage_incomplete": 1}


def test_same_day_master_never_backfilled(tmp_path):
    breeze, eod, summary = _sources(tmp_path, previous=False)
    result = run(
        breeze,
        eod,
        tmp_path / "staged",
        summary_path=summary,
        expected_windows=1,
        day="2025-02-04",
        permit_partial=True,
    )
    assert result["staged_days"] == 0
    assert result["rejections"] == {
        "previous_eod_contract_master_or_spot_unavailable": 1
    }


def test_provenance_digest_mismatch_prevents_dataset_staging(tmp_path):
    breeze, eod, summary = _sources(tmp_path)
    with sqlite3.connect(breeze) as db:
        db.execute(
            "DELETE FROM candles WHERE rowid = "
            "(SELECT rowid FROM candles WHERE series_id=1 LIMIT 1)"
        )
        db.commit()
    with pytest.raises(ValueError, match="breeze_database_digest_mismatch"):
        run(
            breeze,
            eod,
            tmp_path / "stage",
            summary_path=summary,
            expected_windows=1,
            day="2025-02-04",
            permit_partial=True,
        )
    assert not (tmp_path / "stage").exists()


def test_partial_gap_keeps_contiguous_cohort_only(tmp_path):
    breeze, eod, summary = _sources(tmp_path, gaps=(80,))
    result = run(
        breeze,
        eod,
        tmp_path / "stage",
        summary_path=summary,
        expected_windows=1,
        day="2025-02-04",
        permit_partial=True,
    )
    assert result["staged_days"] == 1
    report = json.loads((tmp_path / "stage/2025-02-04/manifest.json").read_text())
    assert report["continuous_minutes_staged"] == 80
    assert report["discontinuous_minutes_excluded"] == 9
    assert report["missing_or_unsynchronized_minutes"] == 286


def test_shadows_existing_research_engine_without_broker(tmp_path, monkeypatch):
    breeze, eod, summary = _sources(tmp_path)
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    called = []

    def stub(path):
        called.append(path.name)
        return {"evidence_label": "RESEARCH_CANDIDATE", "live_equivalent": False}

    monkeypatch.setattr(
        "nifty_scalper_bot.backtesting.strategy_research.run_archived_research",
        stub,
    )
    report = run(
        breeze,
        eod,
        tmp_path / "staged",
        summary_path=summary,
        expected_windows=1,
        day="2025-02-04",
        permit_partial=True,
        execute=True,
    )
    assert report["research_executions_completed"] == 1
    assert called == ["2025-02-04"]
    assert (
        json.loads((tmp_path / "staged/2025-02-04/strategy_research.json").read_text())[
            "live_equivalent"
        ]
        is False
    )
