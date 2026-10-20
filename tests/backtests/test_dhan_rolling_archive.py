"""Regression tests for the research-only Dhan rolling-options adapter."""

import gzip
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.backtesting.dhan_rolling_archive import (
    _canonical_sha,
    load_dhan_rolling_day,
)


def _bytes(obj):
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _fixture(root: Path, *, switch=False, empty_future=False):
    date_text = "2025-01-02"
    base_epoch = int(
        datetime(2025, 1, 2, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata")).timestamp()
    )
    epochs = [base_epoch + i * 60 for i in range(61)]
    shard = "dhan-half-01"
    directory = root / "extracted" / shard
    readiness = root / "readiness"
    readiness.mkdir(parents=True)
    for dirname in ("receipts", "normalized"):
        (directory / dirname).mkdir(parents=True)
    tasks = []
    segments = []
    pairs = (
        ("spot", None),
        ("future", None),
        ("option", "CALL"),
        ("option", "PUT"),
    )
    for idx, (kind, side) in enumerate(pairs):
        task = {
            "index": idx,
            "kind": kind,
            "label": kind + (side or ""),
            "first": date_text,
            "stop": "2025-01-03",
            "expiry_code": 1 if kind == "option" else None,
            "relative_strike": "ATM" if kind == "option" else None,
            "option_side": side,
        }
        current_epochs = [] if empty_future and kind == "future" else epochs
        size = len(current_epochs)
        series = {key: [100.0] * size for key in ("open", "high", "low", "close")}
        series.update(timestamp=current_epochs, volume=[20] * size)
        if kind == "option":
            series.update(
                strike=[23000 if i < 30 or not switch else 23050 for i in range(61)]
            )
        normalized = {
            "task": task,
            "quality": {"rows": size},
            "series": series,
        }
        rel = f"normalized/{idx:05d}.json.gz"
        with gzip.open(directory / rel, "wb") as handle:
            handle.write(_bytes(normalized))
        receipt = {
            "task": task,
            "normalized_path": rel,
            "normalized_sha256": _canonical_sha(normalized),
        }
        (directory / "receipts" / f"{idx:05d}.json").write_bytes(_bytes(receipt))
        tasks.append(
            {
                "task": task,
                "shard": shard,
                "normalized_path": rel,
                "backtest_eligible": bool(size),
                "documented_provider_scope": True,
                "empty": not size,
                "rows": size,
            }
        )
        bounds = (
            []
            if not size
            else [(0, 29), (30, 60)] if kind == "option" and switch else [(0, 60)]
        )
        for n, (start, end) in enumerate(bounds):
            segments.append(
                {
                    "shard": shard,
                    "task_index": idx,
                    "trading_date": date_text,
                    "segment_id": f"{kind}{side or ''}-{n}",
                    "backtest_eligible": True,
                    "first_provider_epoch": epochs[start],
                    "last_provider_epoch": epochs[end],
                }
            )
    (readiness / "backtest-quality-report.json").write_bytes(
        _bytes(
            {
                "source": "DhanHQ v2",
                "readiness": {
                    "rolling_intraday_component_research": True,
                    "immutable_contract_backtest": False,
                    "historical_execution_replay": False,
                    "live_equivalent": False,
                },
            }
        )
    )
    (readiness / "backtest-catalog.json").write_bytes(
        _bytes(
            {
                "profile": "dhan_rolling_intraday_v1",
                "availability_policy": "provider_epoch_plus_60_seconds",
                "cross_segment_positions_allowed": False,
                "tasks": tasks,
            }
        )
    )
    with gzip.open(readiness / "backtest-segments.jsonl.gz", "wt") as handle:
        for segment in segments:
            handle.write(json.dumps(segment) + "\n")
    (readiness / "source-archive-sha256.txt").write_text("synthetic fixture only\n")
    lines = []
    for path in sorted(readiness.iterdir()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (readiness / "backtest-readiness.sha256").write_text("\n".join(lines) + "\n")
    return root / "extracted", readiness


def test_research_session_is_causal_not_executable(tmp_path):
    shards, readiness = _fixture(tmp_path)
    report = load_dhan_rolling_day(shards, readiness, date(2025, 1, 2))
    assert report["component_research_eligible"] is True
    assert report["aligned_minutes"] == 61
    assert report["longest_contiguous_unswitched_aligned_minutes"] == 61
    assert report["immutable_contract_identity_verified"] is False
    assert report["historical_execution_replay_supported"] is False
    assert report["live_equivalent"] is False
    first = report["streams"]["CALL"][0]
    assert first["available_epoch"] == first["provider_epoch"] + 60
    assert "tradingsymbol" not in first and "expiry" not in first


def test_strike_switch_prevents_60_min_continuous_context(tmp_path):
    shards, readiness = _fixture(tmp_path, switch=True)
    report = load_dhan_rolling_day(shards, readiness, date(2025, 1, 2))
    assert report["aligned_minutes"] == 61
    assert report["longest_contiguous_unswitched_aligned_minutes"] == 31
    assert report["component_research_eligible"] is False


def test_tampered_catalog_is_rejected(tmp_path):
    shards, readiness = _fixture(tmp_path)
    (readiness / "backtest-catalog.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum_mismatch"):
        load_dhan_rolling_day(shards, readiness, date(2025, 1, 2))


def test_tampered_normalized_rows_are_rejected(tmp_path):
    shards, readiness = _fixture(tmp_path)
    path = shards / "dhan-half-01" / "normalized" / "00002.json.gz"
    with gzip.open(path, "rt") as handle:
        obj = json.load(handle)
    obj["series"]["close"][0] = 500.0
    with gzip.open(path, "wb") as handle:
        handle.write(_bytes(obj))
    with pytest.raises(ValueError, match="normalized_checksum_mismatch"):
        load_dhan_rolling_day(shards, readiness, date(2025, 1, 2))


def test_missing_segment_is_rejected(tmp_path):
    shards, readiness = _fixture(tmp_path)
    segments_path = readiness / "backtest-segments.jsonl.gz"
    with gzip.open(segments_path, "rt") as handle:
        segments = [json.loads(line) for line in handle]
    with gzip.open(segments_path, "wt") as handle:
        for segment in segments[:-1]:
            handle.write(json.dumps(segment) + "\n")
    sha_path = readiness / "backtest-readiness.sha256"
    entries = sha_path.read_text().splitlines()
    checksum = hashlib.sha256(segments_path.read_bytes()).hexdigest()
    sha_path.write_text(
        "\n".join(
            (
                f"{checksum}  backtest-segments.jsonl.gz"
                if "backtest-segments.jsonl.gz" in line
                else line
            )
            for line in entries
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="row_segment_missing"):
        load_dhan_rolling_day(shards, readiness, date(2025, 1, 2))

def test_absent_future_does_not_invent_data_or_block_spot_option_research(tmp_path):
    shards, readiness = _fixture(tmp_path, empty_future=True)
    report = load_dhan_rolling_day(shards, readiness, date(2025, 1, 2))
    assert report["missing_streams"] == ["future"]
    assert report["streams"]["future"] == []
    assert report["aligned_minutes"] == 0
    assert report["component_research_eligible"] is False
    assert report["spot_option_aligned_minutes"] == 61
    assert report["spot_option_longest_contiguous_minutes"] == 61
    assert report["spot_option_research_eligible"] is True
    assert report["live_equivalent"] is False


def test_missing_future_with_strike_switch_stays_ineligible(tmp_path):
    shards, readiness = _fixture(tmp_path, switch=True, empty_future=True)
    report = load_dhan_rolling_day(shards, readiness, date(2025, 1, 2))
    assert report["missing_streams"] == ["future"]
    assert report["spot_option_longest_contiguous_minutes"] == 31
    assert report["spot_option_research_eligible"] is False
    assert report["component_research_eligible"] is False
