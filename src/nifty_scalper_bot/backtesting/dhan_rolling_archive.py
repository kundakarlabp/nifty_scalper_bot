"""Offline Dhan rolling-options reader for non-executable component research.

A rolling ATM series is not an identified historical option contract. Never
feed the resulting rows to an execution or contract-specific P&L simulator.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
_ASSETS = (
    "backtest-quality-report.json",
    "backtest-catalog.json",
    "backtest-segments.jsonl.gz",
    "source-archive-sha256.txt",
)


def _canonical_sha(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _verified_readiness(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checksums = (directory / "backtest-readiness.sha256").read_text().splitlines()
    found: dict[str, str] = {}
    for line in checksums:
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2 or pieces[1].lstrip("* ") not in _ASSETS:
            raise ValueError("dhan_readiness_checksum_manifest_invalid")
        name = pieces[1].lstrip("* ")
        if name in found:
            raise ValueError("dhan_readiness_checksum_duplicate")
        found[name] = pieces[0]
    if set(found) != set(_ASSETS):
        raise ValueError("dhan_readiness_checksum_manifest_incomplete")
    for name, expected in found.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError("dhan_readiness_checksum_mismatch")
    quality = _read_json(directory / _ASSETS[0])
    catalog = _read_json(directory / _ASSETS[1])
    readiness = quality.get("readiness", {})
    if (
        catalog.get("profile") != "dhan_rolling_intraday_v1"
        or catalog.get("availability_policy") != "provider_epoch_plus_60_seconds"
        or catalog.get("cross_segment_positions_allowed") is not False
        or readiness.get("rolling_intraday_component_research") is not True
        or readiness.get("immutable_contract_backtest") is not False
        or readiness.get("historical_execution_replay") is not False
        or readiness.get("live_equivalent") is not False
    ):
        raise ValueError("dhan_rolling_research_not_ready")
    return quality, catalog


def load_dhan_rolling_day(
    archive_root: Path, readiness_dir: Path, trading_date: date
) -> dict[str, Any]:
    """Load verified spot/future/near-ATM CE+PE minute rows for one session.

    Caller must supply extracted private release shards and the private release's
    validated readiness assets. Rows retain segment and next-minute availability;
    no executable contract symbol, expiry, fills or historical lot sizes exist.
    """
    quality, catalog = _verified_readiness(readiness_dir)
    segments: dict[tuple[str, int], list[dict[str, Any]]] = {}
    with gzip.open(readiness_dir / "backtest-segments.jsonl.gz", "rt") as handle:
        for line in handle:
            segment = json.loads(line)
            if segment.get("trading_date") == trading_date.isoformat():
                key = (segment["shard"], segment["task_index"])
                segments.setdefault(key, []).append(segment)

    streams: dict[str, list[dict[str, Any]]] = {
        "spot": [],
        "future": [],
        "CALL": [],
        "PUT": [],
    }
    for entry in catalog["tasks"]:
        task = entry["task"]
        kind = task["kind"]
        if kind == "option":
            if task.get("expiry_code") != 1 or task.get("relative_strike") != "ATM":
                continue
            stream = task["option_side"]
        elif kind in {"spot", "future"}:
            stream = kind
        else:
            continue
        if not (task["first"] <= trading_date.isoformat() < task["stop"]):
            continue
        if stream not in streams:
            raise ValueError("dhan_selected_stream_unknown")
        if not entry["backtest_eligible"]:
            if (
                entry.get("documented_provider_scope") is True
                and entry.get("empty") is True
                and entry.get("rows") == 0
            ):
                continue
            raise ValueError("dhan_selected_stream_quarantined")
        shard = entry["shard"]
        index = task["index"]
        if not re.fullmatch(r"dhan-half-[0-9]{2}", shard) or type(index) is not int:
            raise ValueError("dhan_shard_identity_invalid")
        relative = f"normalized/{index:05d}.json.gz"
        if entry["normalized_path"] != relative:
            raise ValueError("dhan_normalized_path_invalid")
        directory = archive_root / shard
        receipt = _read_json(directory / "receipts" / f"{index:05d}.json")
        if receipt["task"] != task or receipt["normalized_path"] != relative:
            raise ValueError("dhan_receipt_identity_mismatch")
        with gzip.open(directory / relative, "rt") as handle:
            payload = json.load(handle)
        if _canonical_sha(payload) != receipt["normalized_sha256"]:
            raise ValueError("dhan_normalized_checksum_mismatch")
        if payload["task"] != task or payload["quality"]["rows"] != entry["rows"]:
            raise ValueError("dhan_normalized_catalog_mismatch")
        series = payload["series"]
        epochs = series["timestamp"]
        for i, epoch in enumerate(epochs):
            timestamp = datetime.fromtimestamp(epoch, IST)
            if timestamp.date() != trading_date:
                continue
            matching = [
                item
                for item in segments.get((shard, index), [])
                if item["backtest_eligible"]
                and item["first_provider_epoch"] <= epoch <= item["last_provider_epoch"]
            ]
            if len(matching) != 1:
                raise ValueError("dhan_row_segment_missing_or_ambiguous")
            row = {
                name: values[i]
                for name, values in series.items()
                if name != "timestamp"
            }
            row.update(
                {
                    "provider_epoch": epoch,
                    "available_epoch": epoch + 60,
                    "segment_id": matching[0]["segment_id"],
                }
            )
            streams[stream].append(row)

    for stream, rows in streams.items():
        rows.sort(key=lambda row: row["provider_epoch"])
        if len({row["provider_epoch"] for row in rows}) != len(rows):
            raise ValueError(f"dhan_duplicate_observation:{stream}")

    by_stream = {
        key: {row["provider_epoch"]: row for row in rows}
        for key, rows in streams.items()
    }
    aligned = sorted(set.intersection(*(set(rows) for rows in by_stream.values())))
    longest = run = 0
    previous: int | None = None
    last_segments: tuple[str, str] | None = None
    for epoch in aligned:
        pair = (
            by_stream["CALL"][epoch]["segment_id"],
            by_stream["PUT"][epoch]["segment_id"],
        )
        run = (
            run + 1
            if previous is not None and epoch - previous == 60 and pair == last_segments
            else 1
        )
        longest = max(longest, run)
        previous, last_segments = epoch, pair
    return {
        "trading_date": trading_date.isoformat(),
        "source": quality["source"],
        "evidence_label": "RESEARCH_CANDIDATE",
        "immutable_contract_identity_verified": False,
        "historical_execution_replay_supported": False,
        "live_equivalent": False,
        "aligned_minutes": len(aligned),
        "longest_contiguous_unswitched_aligned_minutes": longest,
        "component_research_eligible": longest >= 60,
        "streams": streams,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = load_dhan_rolling_day(args.shards, args.readiness, args.date)
    args.output.write_text(json.dumps(result, sort_keys=True))
    print(json.dumps({key: value for key, value in result.items() if key != "streams"}))


if __name__ == "__main__":
    main()
