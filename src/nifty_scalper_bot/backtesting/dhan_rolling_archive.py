"""Offline Dhan rolling-options reader for non-executable component research.

A rolling ATM series is not an identified historical option contract. Never
feed the resulting rows to an execution or contract-specific P&L simulator.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
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


def _identified_future_day(
    path: Path, trading_date: date
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the private identified-contract release without inventing quotes.

    A sidecar verifies bytes, not independent exchange authenticity. Contract
    selection remains the caller's explicit choice, not a reconstructed basket.
    """
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    manifest = path.with_name(path.name + ".sha256").read_text().splitlines()
    if len(manifest) != 1 or manifest[0].split() != [checksum, path.name]:
        raise ValueError("dhan_future_checksum_mismatch")
    payload = json.loads(gzip.decompress(raw))
    identity = payload.get("identity", {})
    security = identity.get("security_id", "")
    symbol = identity.get("symbol", "")
    expiry = date.fromisoformat(identity.get("expiry", ""))
    collected = date.fromisoformat(payload.get("collected_through", ""))
    if (
        payload.get("profile") != "nifty_identified_future_v1"
        or payload.get("historical_contract_identity_verified") is not True
        or payload.get("historical_execution_quotes_available") is not False
        or payload.get("timestamp_convention")
        != "provider_epoch_interpreted_as_completed_minute_open_time"
        or not isinstance(security, str)
        or not security.isdigit()
        or int(security) <= 0
        or security == "13"
        or not isinstance(symbol, str)
        or not re.fullmatch(r"NIFTY(?=\d|-).*FUT", symbol, re.IGNORECASE)
        or path.name != f"nifty-future-{expiry}.json.gz"
        or not trading_date <= collected <= expiry
    ):
        raise ValueError("dhan_future_identity_invalid")
    candles = payload.get("candles")
    if (
        not isinstance(candles, list)
        or not candles
        or payload.get("rows") != len(candles)
    ):
        raise ValueError("dhan_future_rows_invalid")
    result = []
    close_boundary_rows = 0
    previous = None
    for row in candles:
        epoch = row.get("timestamp")
        if type(epoch) is not int or (previous is not None and epoch <= previous):
            raise ValueError("dhan_future_timestamp_invalid")
        timestamp = datetime.fromtimestamp(epoch, IST)
        if (
            timestamp.second != 0
            or not "09:15" <= timestamp.strftime("%H:%M") <= "15:30"
            or timestamp.date() > min(collected, expiry)
        ):
            raise ValueError("dhan_future_session_invalid")
        values = [row.get(key) for key in ("open", "high", "low", "close", "volume")]
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("dhan_future_numeric_invalid")
        op, high, low, close, volume = values
        oi = row.get("open_interest", row.get("oi"))
        if (
            min(op, high, low, close) <= 0
            or not low <= min(op, close) <= max(op, close) <= high
            or volume < 0
            or (
                oi is not None
                and (type(oi) not in (int, float) or not math.isfinite(oi) or oi < 0)
            )
        ):
            raise ValueError("dhan_future_bar_invalid")
        previous = epoch
        if timestamp.date() == trading_date and timestamp.strftime("%H:%M") == "15:30":
            close_boundary_rows += 1
        elif timestamp.date() == trading_date:
            result.append(
                {
                    **{
                        key: row[key]
                        for key in ("open", "high", "low", "close", "volume")
                    },
                    "open_interest": oi,
                    "provider_epoch": epoch,
                    "available_epoch": epoch + 60,
                    "segment_id": (
                        f"identified-future:{security}:{expiry}:{trading_date}"
                    ),
                }
            )
    return result, {
        "source": payload.get("source"),
        "sha256": checksum,
        "identity": identity,
        "selected_session_rows": len(result),
        "excluded_close_boundary_rows": close_boundary_rows,
        "historical_basket_selection_verified": False,
        "contains_bid_ask_depth": False,
    }


def load_dhan_rolling_day(
    archive_root: Path,
    readiness_dir: Path,
    trading_date: date,
    *,
    futures_archive: Path | None = None,
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

    future_provenance = None
    if futures_archive is not None:
        if streams["future"]:
            raise ValueError("dhan_future_sources_conflict")
        streams["future"], future_provenance = _identified_future_day(
            futures_archive, trading_date
        )

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
    spot_option_epochs = sorted(
        set(by_stream["spot"]) & set(by_stream["CALL"]) & set(by_stream["PUT"])
    )
    spot_option_longest = option_run = 0
    last_option_epoch: int | None = None
    last_option_segments: tuple[str, str, str] | None = None
    for epoch in spot_option_epochs:
        option_segments = (
            by_stream["spot"][epoch]["segment_id"],
            by_stream["CALL"][epoch]["segment_id"],
            by_stream["PUT"][epoch]["segment_id"],
        )
        option_run = (
            option_run + 1
            if last_option_epoch is not None
            and epoch - last_option_epoch == 60
            and option_segments == last_option_segments
            else 1
        )
        spot_option_longest = max(spot_option_longest, option_run)
        last_option_epoch = epoch
        last_option_segments = option_segments
    missing_streams = sorted(name for name, rows in streams.items() if not rows)
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
        "spot_option_aligned_minutes": len(spot_option_epochs),
        "spot_option_longest_contiguous_minutes": spot_option_longest,
        "spot_option_research_eligible": spot_option_longest >= 60,
        "missing_streams": missing_streams,
        "futures_supplement": future_provenance,
        "futures_context_identity_verified": future_provenance is not None
        and bool(streams["future"]),
        "streams": streams,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--futures-archive", type=Path)
    args = parser.parse_args()
    result = load_dhan_rolling_day(
        args.shards, args.readiness, args.date, futures_archive=args.futures_archive
    )
    args.output.write_text(json.dumps(result, sort_keys=True))
    print(json.dumps({key: value for key, value in result.items() if key != "streams"}))


if __name__ == "__main__":
    main()
