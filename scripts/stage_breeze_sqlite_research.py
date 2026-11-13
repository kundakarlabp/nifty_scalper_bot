"""Stage verified Breeze SQLite candles for offline strategy research.

Responsibilities: Point-in-time EOD contract referencing, gap-aware session gating,
optional existing strategy-research replay and inspectable exclusion counts.
Ownership: Operator-invoked offline research only; not the production research worker.
Safety: No broker API, quote reconstruction, synthetic token, same-day EOD lookahead,
Live promotion, or mutation of either the historical database or raw archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
MIN_BARS = 60


def _readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ValueError("historical_research_database_missing")
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _previous_master(conn: sqlite3.Connection, day: str) -> str | None:
    row = conn.execute(
        "SELECT MAX(day) FROM sessions WHERE day < ? AND available_not_before <= ?",
        (day, day),
    ).fetchone()
    if not row or not row[0]:
        return None
    previous = date.fromisoformat(row[0])
    if (date.fromisoformat(day) - previous).days > 7:
        return None
    return row[0]


def _previous_spot(conn: sqlite3.Connection, day: str) -> float | None:
    row = conn.execute(
        "SELECT day,close FROM daily_index WHERE day < ? "
        "AND available_not_before <= ? ORDER BY day DESC LIMIT 1",
        (day, day),
    ).fetchone()
    if not row or (date.fromisoformat(day) - date.fromisoformat(row[0])).days > 7:
        return None
    return float(row[1])


def _series(
    conn: sqlite3.Connection, day: str
) -> dict[tuple[str, str | None, int | None, str], dict[str, Any]]:
    result: dict[tuple[str, str | None, int | None, str], dict[str, Any]] = {}
    rows = conn.execute(
        "SELECT id,product_type,expiry,strike,side,observed_minutes,"
        "missing_minutes,status FROM series WHERE trade_day=?",
        (day,),
    )
    for sid, kind, expiry, strike, side, count, missing, state in rows:
        if state in {"EMPTY", "QUARANTINED_SPECIAL_INDEX_CLOCK"} or count < MIN_BARS:
            continue
        result[(kind, expiry, strike, side)] = {
            "id": sid,
            "observed_minutes": count,
            "missing_minutes": missing,
            "status": state,
        }
    return result


def _available_contracts(
    eod: sqlite3.Connection, previous: str, kind: str
) -> list[dict[str, Any]]:
    rows = eod.execute(
        "SELECT instrument_name,expiry,strike,lot_size FROM contract_eod "
        "WHERE day=? AND kind=? AND lot_size>0 ORDER BY expiry,strike",
        (previous, kind),
    ).fetchall()
    return [
        {"name": name, "expiry": expiry, "strike": strike, "lot": int(lot)}
        for name, expiry, strike, lot in rows
        if name and name.startswith("NIFTY") and name.endswith(kind)
    ]


def _find_pair(
    futures: sqlite3.Connection, eod: sqlite3.Connection, day: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    avail = _series(futures, day)
    index = avail.get(("cash", None, None, "INDEX"))
    if not index:
        raise ValueError("spot_minute_history_not_available")
    previous = _previous_master(eod, day)
    anchor = _previous_spot(eod, day)
    if previous is None or anchor is None:
        raise ValueError("previous_eod_contract_master_or_spot_unavailable")
    future = next(
        (
            {**c, **avail[("futures", c["expiry"], None, "FUT")]}
            for c in _available_contracts(eod, previous, "FUT")
            if c["expiry"] >= day
            and ("futures", c["expiry"], None, "FUT") in avail
        ),
        None,
    )
    if future is None:
        raise ValueError("previous_day_futures_identity_not_available")
    calls = {
        (c["expiry"], int(c["strike"])): c
        for c in _available_contracts(eod, previous, "CE")
        if c["strike"] is not None and c["expiry"] >= day
    }
    puts = {
        (c["expiry"], int(c["strike"])): c
        for c in _available_contracts(eod, previous, "PE")
        if c["strike"] is not None and c["expiry"] >= day
    }
    possible = sorted(
        (
            (expiry, abs(strike - anchor), strike)
            for expiry, strike in calls.keys() & puts.keys()
            if ("options", expiry, strike, "CE") in avail
            and ("options", expiry, strike, "PE") in avail
        )
    )
    if not possible:
        raise ValueError("previous_day_option_pair_not_available")
    expiry, _, strike = possible[0]
    ce = {**calls[(expiry, strike)], **avail[("options", expiry, strike, "CE")]}
    pe = {**puts[(expiry, strike)], **avail[("options", expiry, strike, "PE")]}
    # Lot mismatches need independent resolution, never silently choose one.
    if ce["lot"] != pe["lot"] or ce["lot"] != future["lot"]:
        raise ValueError("historical_lot_size_mismatch")
    return index, future, ce, pe


def _candles(db: sqlite3.Connection, sid: int) -> list[tuple[Any, ...]]:
    return db.execute(
        "SELECT stamp,open,high,low,close,volume,oi "
        "FROM candles WHERE series_id=? ORDER BY stamp",
        (sid,),
    ).fetchall()


def _session_grid(day: str) -> tuple[datetime, int]:
    if day == "2025-10-21":
        return datetime.combine(date.fromisoformat(day), time(13, 45), IST), 60
    return datetime.combine(date.fromisoformat(day), time(9, 15), IST), 375


def _prepare_day(
    breeze: sqlite3.Connection,
    eod: sqlite3.Connection,
    day: str,
    outdir: Path,
    *,
    permit_partial: bool,
) -> dict[str, Any]:
    index, fut, ce, pe = _find_pair(breeze, eod, day)
    selected = [("INDEX", index), ("FUT", fut), ("CE", ce), ("PE", pe)]
    originals = {kind: _candles(breeze, item["id"]) for kind, item in selected}
    first, minutes = _session_grid(day)
    expected = {
        (first + timedelta(minutes=i)).isoformat() for i in range(minutes)
    }
    present = {kind: {row[0] for row in rows} for kind, rows in originals.items()}
    synchronized = set.intersection(*(stamps for stamps in present.values()))
    synchronized &= expected
    missing = expected - synchronized
    if not permit_partial and missing:
        raise ValueError("synchronized_minute_coverage_incomplete")
    if len(synchronized) < MIN_BARS:
        raise ValueError("synchronized_minute_coverage_too_short")
    # A partial session is split to retain only its longest contiguous segment;
    # never stitch discontinuous candles into a false regular minute history.
    sorted_times = sorted(synchronized)
    blocks: list[list[str]] = []
    for stamp in sorted_times:
        contiguous = bool(blocks) and (
            datetime.fromisoformat(stamp) - datetime.fromisoformat(blocks[-1][-1])
            == timedelta(minutes=1)
        )
        if not contiguous:
            blocks.append([stamp])
        else:
            blocks[-1].append(stamp)
    longest = max(blocks, key=len)
    if len(longest) < MIN_BARS:
        raise ValueError("continuous_minute_cohort_too_short")
    kept = set(longest)
    outdir.mkdir(mode=0o700, parents=True, exist_ok=False)
    candle_dir = outdir / "candles"
    candle_dir.mkdir(mode=0o700)
    outputs: dict[str, dict[str, Any]] = {}
    for kind, contract in selected:
        rows = [row for row in originals[kind] if row[0] in kept]
        if kind == "INDEX":
            symbol = "NSE:NIFTY 50"
            name, lot, expiry, strike = "NIFTY 50", 1, None, None
        else:
            name, lot, expiry = contract["name"], contract["lot"], contract["expiry"]
            strike = contract["strike"]
            symbol = f"NFO:{name}"
        instrument: dict[str, Any] = {
            "exchange": "NSE" if kind == "INDEX" else "NFO",
            "tradingsymbol": name,
            "name": "NIFTY",
            "instrument_type": "EQ" if kind == "INDEX" else kind,
            "lot_size": lot,
        }
        if expiry is not None:
            instrument["expiry"] = expiry
        if strike is not None:
            instrument["strike"] = strike
        payload = {
            "symbol": symbol,
            "timestamp_convention": "bar_start",
            "instrument": instrument,
            "candles": [list(r) for r in rows],
            "evidence": "RESEARCH_CANDIDATE_BAR_ONLY",
            "historical_instrument_token_verified": False,
        }
        path = candle_dir / f"{kind}.json"
        path.write_text(json.dumps(payload, sort_keys=True))
        path.chmod(0o600)
        outputs[kind] = {"symbol": symbol, "bars": len(rows)}
    report = {
        "trade_day": day,
        "status": "staged",
        "source": "verified_Breeze_SQLite_and_prior_session_EOD",
        "source_minutes_per_contract": {k: len(v) for k, v in originals.items()},
        "complete_synchronized_minutes": len(synchronized),
        "missing_or_unsynchronized_minutes": len(missing),
        "continuous_minutes_staged": len(longest),
        "orb_opening_range_complete": all(
            (first + timedelta(minutes=i)).isoformat() in kept for i in range(15)
        ),
        "discontinuous_minutes_excluded": len(synchronized) - len(longest),
        "selected": outputs,
        "historical_token_verified": False,
        "historical_quote_depth_available": False,
        "historical_broker_fills_available": False,
        "source_clock_independently_verified": False,
        "research_only": True,
        "execution_grade_ready": False,
        "live_promotion_eligible": False,
    }
    (outdir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (outdir / "manifest.json").chmod(0o600)
    return report


def _verify_research_source(
    path: Path, summary_path: Path, expected_windows: int
) -> dict[str, Any]:
    if not summary_path.is_file():
        raise ValueError("breeze_validation_summary_missing")
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("verified_windows") != expected_windows
        or summary.get("research_only") is not True
        or summary.get("promotion_eligible") is not False
        or summary.get("execution_grade_ready") is not False
    ):
        raise ValueError("breeze_validation_summary_untrusted")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != summary.get("dataset_sha256"):
        raise ValueError("breeze_database_digest_mismatch")
    with closing(_readonly(path)) as db:
        row = db.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    if row != summary.get("bars_1m"):
        raise ValueError("breeze_database_row_count_mismatch")
    return summary


def run(
    breeze_db: Path,
    eod_db: Path,
    output_root: Path,
    *,
    summary_path: Path,
    expected_windows: int = 25,
    day: str | None = None,
    permit_partial: bool = False,
    execute: bool = False,
    limit: int = 500,
) -> dict[str, Any]:
    """Create offline strategy input and optionally use the existing research engine."""
    if output_root.exists() or not 1 <= limit <= 500:
        raise ValueError("research_output_exists_or_limit_invalid")
    if execute and (
        os.getenv("EXECUTION_MODE", "SHADOW").upper() != "SHADOW"
        or os.getenv("ENABLE_LIVE", "false").lower() == "true"
        or os.getenv("ALLOW_REAL_BROKER", "false").lower() == "true"
    ):
        raise ValueError("research_requires_shadow_process")
    _verify_research_source(breeze_db, summary_path, expected_windows)
    with closing(_readonly(breeze_db)) as breeze, closing(_readonly(eod_db)) as eod:
        if day is not None:
            date.fromisoformat(day)
            days = [day]
        else:
            days = [
                x[0]
                for x in breeze.execute(
                    "SELECT DISTINCT trade_day FROM series ORDER BY trade_day"
                ).fetchall()
            ][:limit]
        # No empty directory or success artifact when dataset itself is invalid.
        if not days:
            raise ValueError("intraday_breeze_sqlite_has_no_sessions")
        output_root.mkdir(mode=0o700, parents=True)
        accepted: list[dict[str, Any]] = []
        rejected: Counter[str] = Counter()
        for session in days:
            try:
                item = _prepare_day(
                    breeze, eod, session, output_root / session,
                    permit_partial=permit_partial,
                )
            except (ValueError, sqlite3.Error) as exc:
                rejected[str(exc)] += 1
                continue
            if execute and not item["orb_opening_range_complete"]:
                item["strategy_research_status"] = "skipped"
                item["strategy_research_reason"] = "orb_opening_range_incomplete"
            elif execute:
                from nifty_scalper_bot.backtesting.strategy_research import (
                    run_archived_research,
                )
                try:
                    result = run_archived_research(output_root / session)
                except Exception as exc:
                    # Research failure is audited; no retry into broker runtime.
                    item["strategy_research_status"] = "failed"
                    item["strategy_research_error_type"] = type(exc).__name__
                else:
                    item["strategy_research_status"] = "completed"
                    report_file = output_root / session / "strategy_research.json"
                    report_file.write_text(json.dumps(result, default=str))
                    report_file.chmod(0o600)
            accepted.append(item)
    report = {
        "input_days": len(days),
        "staged_days": len(accepted),
        "rejected_days": sum(rejected.values()),
        "rejections": dict(rejected),
        "research_executions_skipped": sum(
            x.get("strategy_research_status") == "skipped" for x in accepted
        ),
        "research_executions_completed": sum(
            x.get("strategy_research_status") == "completed" for x in accepted
        ),
        "research_executions_failed": sum(
            x.get("strategy_research_status") == "failed" for x in accepted
        ),
        "research_only": True,
        "execution_grade_ready": False,
        "live_promotion_eligible": False,
    }
    path = output_root / "research_summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    path.chmod(0o600)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breeze-db", type=Path, required=True)
    parser.add_argument("--eod-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-windows", type=int, default=25)
    parser.add_argument("--day")
    parser.add_argument("--permit-partial", action="store_true")
    parser.add_argument("--run-strategies", action="store_true")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    result = run(
        args.breeze_db, args.eod_db, args.output_root,
        summary_path=args.summary, expected_windows=args.expected_windows, day=args.day,
        permit_partial=args.permit_partial, execute=args.run_strategies,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()