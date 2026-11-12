#!/usr/bin/env python3
"""Build a private, research-only SQLite NIFTY candle catalogue.

Convert checksum-verified Breeze windows without modifying originals.

Ownership: isolated offline research import and read-only export of named contracts.
Safety: never decrypts, fetches, trades, or asserts executable quotes; rejects checksum
conflicts and invalid candles and never silently patches gaps or modifies raw archives.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from nifty_scalper_bot.backtesting.breeze_bar_research import aggregate_breeze_session

SCHEMA = """
CREATE TABLE series (
  id INTEGER PRIMARY KEY,
  trade_day TEXT NOT NULL,
  contract_key TEXT NOT NULL,
  product_type TEXT NOT NULL,
  expiry TEXT,
  strike INTEGER,
  side TEXT NOT NULL,
  status TEXT NOT NULL,
  observed_minutes INTEGER NOT NULL,
  missing_minutes INTEGER NOT NULL,
  flagged_boundary_minutes INTEGER NOT NULL,
  quarantined_source_minutes INTEGER NOT NULL,
  source_digest TEXT NOT NULL,
  timestamp_verified INTEGER NOT NULL DEFAULT 0,
  execution_eligible INTEGER NOT NULL DEFAULT 0,
  UNIQUE(trade_day, contract_key)
);
CREATE TABLE candles (
  series_id INTEGER NOT NULL REFERENCES series(id),
  stamp TEXT NOT NULL,
  available_at TEXT NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  volume INTEGER NOT NULL,
  oi INTEGER,
  PRIMARY KEY(series_id, stamp)
) WITHOUT ROWID;
CREATE INDEX series_by_day ON series(trade_day, product_type, expiry, strike, side);
"""
SHA_LINE = re.compile(r"^([a-fA-F0-9]{64})  (.+)$")
SHA_VALUE = re.compile(r"^[a-fA-F0-9]{64}$")
IST_SUFFIX = timedelta(hours=5, minutes=30)


def digest_json(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_window(window: Path) -> tuple[list[Path], int]:
    """Verify every named member and prohibit symlinks/unsafe manifest paths."""
    manifest = window / "SHA256SUMS"
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("window_sha256_manifest_missing")
    verified: list[Path] = []
    seen: set[str] = set()
    for line in manifest.read_text().splitlines():
        match = SHA_LINE.fullmatch(line)
        if match is None:
            raise ValueError("window_sha256_manifest_invalid")
        sha, relative = match.groups()
        item_path = Path(relative)
        if item_path.is_absolute() or ".." in item_path.parts or relative in seen:
            raise ValueError("window_manifest_path_invalid")
        seen.add(relative)
        file_path = window / item_path
        if not file_path.is_file() or file_path.is_symlink():
            raise ValueError("window_member_missing_or_symlink")
        h = hashlib.sha256()
        with file_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != sha.lower():
            raise ValueError("window_member_sha256_mismatch")
        verified.append(file_path)
    if not verified:
        raise ValueError("window_sha256_manifest_empty")
    observations = [
        p for p in verified if "breeze_history" in p.relative_to(window).parts
        and p.suffix == ".json"
    ]
    if not observations:
        raise ValueError("window_has_no_contract_archives")
    return observations, len(verified)


def contract_identity(
    p: dict[str, Any],
) -> tuple[str, str, str | None, int | None, str]:
    product = p.get("product_type", "options")
    side = p.get("side")
    day = p.get("day")
    expiry = p.get("expiry")
    strike = p.get("strike")
    if not isinstance(day, str):
        raise ValueError("archive_day_invalid")
    try:
        session = date.fromisoformat(day)
        expiry_day = date.fromisoformat(expiry) if expiry is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("archive_date_invalid") from exc
    if not date(2024, 11, 6) <= session <= date(2026, 11, 5):
        raise ValueError("archive_day_outside_study")
    if product == "cash" and (side, expiry, strike) == ("INDEX", None, None):
        key = "NIFTY_INDEX"
    elif product == "futures" and side == "FUT" and strike is None and (
        expiry_day is not None and expiry_day >= session
    ):
        key = f"NIFTY_{expiry}_FUT"
    elif product == "options" and side in ("CE", "PE") and (
        type(strike) is int and strike > 0 and strike % 50 == 0
        and expiry_day is not None and expiry_day >= session
    ):
        key = f"NIFTY_{expiry}_{strike}_{side}"
    else:
        raise ValueError("archive_contract_identity_invalid")
    return key, product, expiry, strike, side


def session_hours(day: date) -> tuple[time, time]:
    if day == date(2025, 10, 21):
        return time(13, 45), time(14, 45)
    return time(9, 15), time(15, 30)


def validated_series(payload: dict[str, Any]) -> tuple[dict[str, Any], list[list[Any]]]:
    if payload.get("schema_version") != 1 or (
        payload.get("normalized_timestamp_convention") != "bar_start"
        or payload.get("source") != "icici_breeze_historical_v2"
        or payload.get("promotion_eligible") is not False
    ):
        raise ValueError("breeze_source_provenance_invalid")
    for field, value in (
        ("candles", payload.get("candles")),
        ("provider_rows", payload.get("provider_rows")),
    ):
        if not isinstance(value, list):
            raise ValueError("breeze_source_array_missing")
        if (
            not isinstance(payload.get(f"{field}_sha256"), str)
            or not SHA_VALUE.fullmatch(payload[f"{field}_sha256"])
            or digest_json(value) != payload[f"{field}_sha256"]
        ):
            raise ValueError("breeze_source_digest_mismatch")
    key, product, expiry, strike, side = contract_identity(payload)
    session = date.fromisoformat(payload["day"])
    start, stop = session_hours(session)
    window_start = datetime.combine(session, start)
    window_end = datetime.combine(session, stop)
    normal_rows: list[list[Any]] = []
    boundaries = 0
    quarantined = 0
    for row in payload["candles"]:
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError("breeze_candle_schema_invalid")
        try:
            stamp = datetime.fromisoformat(str(row[0]))
        except (TypeError, ValueError) as exc:
            raise ValueError("breeze_candle_timestamp_invalid") from exc
        if (
            stamp.utcoffset() != IST_SUFFIX or stamp.second or stamp.microsecond
            or stamp.tzinfo is None or stamp.date() != session
        ):
            raise ValueError("breeze_candle_timestamp_invalid")
        naive = stamp.replace(tzinfo=None)
        if window_start <= naive < window_end:
            normal_rows.append(row)
        elif session == date(2025, 10, 21) and naive == window_end:
            boundaries += 1
        elif session == date(2025, 10, 21) and product == "cash":
            # Archived Muhurat spot source contains regular-session timestamps.
            # Preserve explicit quarantine metadata but never expose such bars.
            quarantined += 1
        else:
            raise ValueError("breeze_candle_outside_verified_session")
    if boundaries > 1:
        raise ValueError("breeze_duplicate_special_boundary")
    if quarantined:
        if product != "cash" or session != date(2025, 10, 21):
            raise ValueError("unexpected_source_quarantine")
        quarantined += len(normal_rows)
        normal_rows = []
    audit = aggregate_breeze_session(
        normal_rows, day=session, interval_minutes=1,
        session_start=start, session_end=stop
    )
    if len(audit.bars) != len(normal_rows):
        raise ValueError("breeze_normalized_series_conflict")
    state = "QUARANTINED_SPECIAL_INDEX_CLOCK" if quarantined else (
        "EMPTY" if not normal_rows else (
            "SPECIAL_BOUNDARY_FLAGGED" if boundaries else (
                "PARTIAL" if audit.missing_windows else "COMPLETE"
            )
        )
    )
    summary = {
        "trade_day": payload["day"],
        "contract_key": key,
        "product_type": product,
        "expiry": expiry,
        "strike": strike,
        "side": side,
        "status": state,
        "observed_minutes": len(normal_rows),
        "missing_minutes": len(audit.missing_windows),
        "flagged_boundary_minutes": boundaries,
        "quarantined_source_minutes": quarantined,
        "source_digest": payload["candles_sha256"],
    }
    return summary, normal_rows


def build(
    windows_root: Path,
    database: Path,
    summary_file: Path,
    *,
    expected_windows: int = 25,
) -> dict[str, Any]:
    """Build an atomic, private SQLite research cache; immutable source is untouched."""
    root = windows_root.resolve()
    if not root.is_dir() or not 1 <= expected_windows <= 100:
        raise ValueError("extracted_windows_or_count_invalid")
    for output in (database, summary_file):
        if output.exists() or output.resolve().is_relative_to(root):
            raise ValueError("private_output_exists_or_overlaps_source")
    manifests = sorted(root.rglob("SHA256SUMS"))
    if len(manifests) != expected_windows:
        raise ValueError("archive_window_count_mismatch")
    sources: list[Path] = []
    verified_files = 0
    for manifest in manifests:
        observations, n = verify_window(manifest.parent)
        sources.extend(observations)
        verified_files += n
    database.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = database.with_name(database.name + ".tmp")
    if tmp.exists():
        raise ValueError("incomplete_private_dataset_exists")
    counters = {
        "verified_windows": len(manifests),
        "verified_files": verified_files,
        "input_contract_files": len(sources),
        "identical_duplicates": 0,
        "series": 0,
        "bars_1m": 0,
        "gap_minutes": 0,
        "special_boundary_exclusions": 0,
        "quarantined_special_index_series": 0,
        "quarantined_source_minutes": 0,
    }
    try:
        tmp.touch(mode=0o600, exist_ok=False)
        os.chmod(tmp, 0o600)
        with sqlite3.connect(tmp) as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(SCHEMA)
            known: dict[tuple[str, str], dict[str, Any]] = {}
            for file in sorted(sources):
                payload = json.loads(file.read_text())
                info, candles = validated_series(payload)
                identity = (info["trade_day"], info["contract_key"])
                if identity in known:
                    if known[identity] != info:
                        raise ValueError("overlap_contract_history_conflict")
                    counters["identical_duplicates"] += 1
                    continue
                known[identity] = info
                cursor = db.execute(
                    "INSERT INTO series (trade_day,contract_key,product_type,expiry,"
                    "strike,side,status,observed_minutes,missing_minutes,"
                    "flagged_boundary_minutes,quarantined_source_minutes,"
                    "source_digest) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    tuple(info[k] for k in (
                        "trade_day","contract_key","product_type","expiry","strike",
                        "side","status","observed_minutes","missing_minutes",
                        "flagged_boundary_minutes","quarantined_source_minutes",
                        "source_digest"
                    )),
                )
                n = cursor.lastrowid
                values = []
                for row in candles:
                    ts = datetime.fromisoformat(row[0])
                    available = (ts + timedelta(minutes=1)).isoformat()
                    values.append((n, row[0], available, *row[1:]))
                db.executemany(
                    "INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?)", values
                )
                counters["series"] += 1
                counters["bars_1m"] += len(candles)
                counters["gap_minutes"] += info["missing_minutes"]
                counters["special_boundary_exclusions"] += info[
                    "flagged_boundary_minutes"
                ]
                counters["quarantined_source_minutes"] += info[
                    "quarantined_source_minutes"
                ]
                if info["status"] == "QUARANTINED_SPECIAL_INDEX_CLOCK":
                    counters["quarantined_special_index_series"] += 1
            db.commit()
            audit_result = db.execute("PRAGMA integrity_check").fetchone()[0]
            if audit_result != "ok":
                raise ValueError("private_sqlite_integrity_failed")
        os.replace(tmp, database)
        report = {
            **counters,
            "study": "NIFTY_BREEZE_IDENTIFIED_OHLCV",
            "research_only": True,
            "historical_quote_depth_present": False,
            "source_clock_convention_verified": False,
            "all_historical_strikes_collected": False,
            "execution_grade_ready": False,
            "promotion_eligible": False,
            "special_budget_2025_02_01_intraday_verified": False,
            "dataset_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
            "trading_days_not_in_archive_are_not_assumed_closed": True,
        }
        temporary_summary = summary_file.with_name(summary_file.name + ".tmp")
        with temporary_summary.open("x") as dest:
            json.dump(report, dest, indent=2, sort_keys=True)
            dest.write("\n")
        os.chmod(temporary_summary, 0o600)
        os.replace(temporary_summary, summary_file)
        return report
    finally:
        tmp.unlink(missing_ok=True)


def export_series(
    database: Path, output: Path, *, day: str, product: str,
    expiry: str | None, strike: int | None, side: str | None, interval: int
) -> dict[str, Any]:
    if output.exists() or not database.is_file() or interval not in (1, 5, 15):
        raise ValueError("research_export_input_invalid")
    if product not in ("cash", "futures", "options"):
        raise ValueError("research_product_invalid")
    if (product, side, expiry, strike) not in (
        ("cash", "INDEX", None, None),
    ) and not (
        product == "futures" and side == "FUT" and expiry is not None and strike is None
    ) and not (
        product == "options" and side in ("CE", "PE")
        and expiry is not None and type(strike) is int
    ):
        raise ValueError("research_identity_incomplete")
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as conn:
        candidate = conn.execute(
            "SELECT id,status FROM series WHERE trade_day=? AND product_type=? "
            "AND expiry IS ? AND strike IS ? AND side=?",
            (day, product, expiry, strike, side),
        ).fetchall()
        if len(candidate) != 1 or candidate[0][1] in (
            "EMPTY", "QUARANTINED_SPECIAL_INDEX_CLOCK"
        ):
            raise ValueError("research_contract_day_not_available")
        source = conn.execute(
            "SELECT stamp,open,high,low,close,volume,oi FROM candles "
            "WHERE series_id=? ORDER BY stamp", (candidate[0][0],)
        ).fetchall()
    start, stop = session_hours(date.fromisoformat(day))
    audit = aggregate_breeze_session(
        source, day=date.fromisoformat(day), interval_minutes=interval,
        session_start=start, session_end=stop
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        raise ValueError("incomplete_research_export_exists")
    try:
        with tmp.open("x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("start_ist","available_at_ist","open","high","low",
                             "close","volume","oi","source_minutes","execution_eligible"))
            writer.writerows(
                (bar.start.isoformat(),bar.available_at.isoformat(),bar.open,
                 bar.high,bar.low,bar.close,bar.volume,bar.open_interest,
                 bar.source_minutes,False) for bar in audit.bars
            )
        os.chmod(tmp, 0o600)
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "bars": len(audit.bars),
        "incomplete_windows_excluded": len(audit.missing_windows),
        "interval_minutes": interval,
        "research_only": True,
        "execution_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_subparsers(dest="command", required=True)
    b = action.add_parser("build")
    b.add_argument("--extracted-windows", type=Path, required=True)
    b.add_argument("--database", type=Path, required=True)
    b.add_argument("--summary", type=Path, required=True)
    b.add_argument("--expected-windows", type=int, default=25)
    e = action.add_parser("export")
    e.add_argument("--database", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--day", required=True)
    e.add_argument("--product", choices=("cash","futures","options"), required=True)
    e.add_argument("--expiry")
    e.add_argument("--strike", type=int)
    e.add_argument("--side", choices=("INDEX","FUT","CE","PE"), required=True)
    e.add_argument("--interval", type=int, choices=(1,5,15), default=5)
    args = parser.parse_args()
    if args.command == "build":
        result = build(
            args.extracted_windows,args.database,args.summary,
            expected_windows=args.expected_windows
        )
    else:
        result = export_series(
            args.database,args.output,day=args.day,product=args.product,
            expiry=args.expiry,strike=args.strike,side=args.side,
            interval=args.interval
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
