"""Private Breeze research import: signed-off hashes, gaps and fail-closed exports."""
import csv
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from scripts.prepare_breeze_research_dataset import (
    build,
    digest_json,
    export_series,
    verify_window,
)


def _payload(*, special=False, skip=(), changed=0, expiry="2025-02-27"):
    day = date(2025, 10, 21) if special else date(2025, 2, 3)
    start = datetime.fromisoformat(
        f"{day}T{'13:45' if special else '09:15'}:00+05:30"
    )
    bars = [
        [
            (start + timedelta(minutes=i)).isoformat(),
            100.0 + i + changed,
            102.0 + i + changed,
            99.0 + i + changed,
            101.0 + i + changed,
            100,
            1000,
        ]
        for i in range(60 if special else 15)
        if i not in skip
    ]
    if special:
        bars.append(
            [
                (start + timedelta(minutes=60)).isoformat(),
                161.0, 162.0, 160.0, 161.5, 100, 1000,
            ]
        )
        expiry = "2025-10-28"
    raw = []
    p = {
        "schema_version": 1,
        "day": day.isoformat(),
        "expiry": expiry,
        "strike": 22500,
        "side": "CE",
        "source": "icici_breeze_historical_v2",
        "product_type": "options",
        "normalized_timestamp_convention": "bar_start",
        "promotion_eligible": False,
        "source_timezone_verified": False,
        "source_timestamp_convention_verified": False,
        "provider_rows": raw,
        "provider_sha256": digest_json(raw),
        "candles": bars,
        "candles_sha256": digest_json(bars),
    }
    return p


def _window(root: Path, name: str, payload: dict):
    base = root / name
    src = base / "breeze_history" / payload["day"]
    src.mkdir(parents=True)
    path = src / "NIFTY_option.json"
    path.write_text(json.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (base / "SHA256SUMS").write_text(
        f"{digest}  breeze_history/{payload['day']}/NIFTY_option.json\n"
    )
    return path


def test_build_and_export_keep_gap_and_research_only(tmp_path):
    source = tmp_path / "windows"
    payload = _payload(skip=(2,))
    _window(source, "part-a", payload)
    _window(source, "part-b", payload)
    output = tmp_path / "private" / "candles.sqlite"
    summary = tmp_path / "private" / "summary.json"
    result = build(source, output, summary, expected_windows=2)
    assert result["verified_windows"] == 2
    assert result["series"] == 1
    assert result["identical_duplicates"] == 1
    assert result["bars_1m"] == 14
    assert result["gap_minutes"] == 361
    assert result["execution_grade_ready"] is False
    assert not result["historical_quote_depth_present"]
    assert output.stat().st_mode & 0o077 == 0
    path = tmp_path / "private" / "CE-5m.csv"
    exported = export_series(
        output, path, day="2025-02-03", product="options",
        expiry="2025-02-27", strike=22500, side="CE", interval=5
    )
    assert exported["bars"] == 2
    assert exported["incomplete_windows_excluded"] == 73
    assert not exported["execution_eligible"]
    with path.open() as f:
        data = list(csv.DictReader(f))
    assert len(data) == 2
    assert data[0]["start_ist"] == "2025-02-03T09:20:00+05:30"
    assert data[0]["available_at_ist"] == "2025-02-03T09:25:00+05:30"
    assert data[0]["execution_eligible"] == "False"
    assert payload["candles"][0][0].startswith("2025")
    assert json.loads(summary.read_text())["bars_1m"] == 14
    with pytest.raises(ValueError, match="private_output_exists"):
        build(source, output, summary, expected_windows=2)


def test_hash_mismatch_blocks_any_build(tmp_path):
    source = tmp_path / "src"
    file = _window(source, "one", _payload())
    file.write_text(file.read_text() + " ")
    with pytest.raises(ValueError, match="window_member_sha256_mismatch"):
        verify_window(source / "one")
    with pytest.raises(ValueError, match="window_member_sha256_mismatch"):
        build(
            source, tmp_path / "cache.sqlite",
            tmp_path / "summary.json", expected_windows=1
        )
    assert not (tmp_path / "cache.sqlite").exists()


def test_conflicting_overlapping_windows_are_not_merged(tmp_path):
    source = tmp_path / "src"
    _window(source, "one", _payload())
    _window(source, "two", _payload(changed=1))
    with pytest.raises(ValueError, match="overlap_contract_history_conflict"):
        build(source, tmp_path / "cache.sqlite", tmp_path / "summary.json",
              expected_windows=2)
    assert not (tmp_path / "cache.sqlite").exists()


def test_special_boundary_observation_is_flagged_not_filled(tmp_path):
    source = tmp_path / "src"
    _window(source, "one", _payload(special=True))
    output = tmp_path / "research.sqlite"
    result = build(
        source, output, tmp_path / "summary.json", expected_windows=1
    )
    assert result["bars_1m"] == 60
    assert result["special_boundary_exclusions"] == 1
    exported = export_series(
        output, tmp_path / "15m.csv", day="2025-10-21", product="options",
        expiry="2025-10-28", strike=22500, side="CE", interval=15
    )
    assert exported["bars"] == 4
    assert exported["incomplete_windows_excluded"] == 0


def test_export_rejects_incomplete_contract_identity(tmp_path):
    with pytest.raises(ValueError, match="research_identity_incomplete"):
        export_series(
            tmp_path / "none.sqlite", tmp_path / "out.csv",
            day="2025-02-03", product="options",
            expiry="2025-02-27", strike=None, side="CE", interval=5
        )


def test_muhurat_spot_legacy_clock_is_quarantined(tmp_path):
    source = tmp_path / "src"
    payload = _payload(special=True)
    payload["product_type"] = "cash"
    payload["expiry"] = None
    payload["strike"] = None
    payload["side"] = "INDEX"
    payload["candles"] = [
        [
            (datetime.fromisoformat("2025-10-21T09:15:00+05:30")
             + timedelta(minutes=i)).isoformat(),
            100.0, 102.0, 99.0, 101.0, 100, None,
        ]
        for i in range(370)
    ]
    payload["candles_sha256"] = digest_json(payload["candles"])
    _window(source, "one", payload)
    database = tmp_path / "private.sqlite"
    report = build(source, database, tmp_path / "summary.json",
                   expected_windows=1)
    assert report["quarantined_special_index_series"] == 1
    assert report["quarantined_source_minutes"] == 370
    assert report["bars_1m"] == 0
    with pytest.raises(ValueError, match="research_contract_day_not_available"):
        export_series(
            database, tmp_path / "invalid.csv", day="2025-10-21",
            product="cash", expiry=None, strike=None, side="INDEX", interval=1
        )
