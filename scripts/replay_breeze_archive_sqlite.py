#!/usr/bin/env python3
"""Stage Breeze intraday candles for the existing offline strategy research engine.

Responsibilities: Select pre-listed NIFTY contracts from preceding EOD facts,
join to exact Breeze 1m identities, and optionally run SHADOW component replay.
Ownership: Offline experimental replay; not the production runner or order path.
Safety: Never invent quotes/tokens, use same-day EOD or promote modeled fills.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROXY_LABEL = "RESEARCH_CANDIDATE_BAR_PROXY_ONLY"


def _reader(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ValueError("offline_dataset_missing")
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    return db


def _prior_context(db: sqlite3.Connection, day: str) -> tuple[str, float] | None:
    """A complete prior EOD record, published strictly before the target day."""
    row = db.execute(
        "SELECT s.day, i.close FROM sessions s "
        "JOIN daily_index i ON s.day=i.day "
        "WHERE s.day<? AND s.available_not_before<=? "
        "AND i.available_not_before<=? "
        "ORDER BY s.day DESC LIMIT 1",
        (day, day, day),
    ).fetchone()
    if row is None or not float(row[1]) > 0:
        return None
    return str(row[0]), float(row[1])


def _prior_listed_contracts(
    db: sqlite3.Connection, prior_day: str, close: float, target_day: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    future = db.execute(
        "SELECT expiry, lot_size, instrument_name FROM contract_eod "
        "WHERE day=? AND kind='FUT' AND expiry>=? AND lot_size>0 "
        "ORDER BY expiry LIMIT 1",
        (prior_day, target_day),
    ).fetchone()
    if future is None:
        return None
    option_expiry = db.execute(
        "SELECT ce.expiry FROM contract_eod ce JOIN contract_eod pe "
        "ON pe.day=ce.day AND pe.expiry=ce.expiry "
        "AND pe.strike=ce.strike AND pe.kind='PE' "
        "WHERE ce.day=? AND ce.kind='CE' AND ce.expiry>=? "
        "AND ce.strike>0 AND ce.lot_size>0 AND pe.lot_size>0 "
        "ORDER BY ce.expiry LIMIT 1",
        (prior_day, target_day),
    ).fetchone()
    if option_expiry is None:
        return None
    expiry = str(option_expiry[0])
    pair = db.execute(
        "SELECT ce.strike,ce.lot_size,pe.lot_size,"
        "ce.instrument_name,pe.instrument_name FROM contract_eod ce "
        "JOIN contract_eod pe ON pe.day=ce.day AND pe.expiry=ce.expiry "
        "AND pe.strike=ce.strike AND pe.kind='PE' "
        "WHERE ce.day=? AND ce.expiry=? AND ce.kind='CE' "
        "AND ce.strike>0 AND ce.lot_size>0 AND pe.lot_size>0 "
        "ORDER BY ABS(ce.strike-?),ce.strike LIMIT 1",
        (prior_day, expiry, close),
    ).fetchone()
    if pair is None or int(pair[1]) != int(pair[2]):
        return None
    strike = float(pair[0])
    if not strike.is_integer():
        return None
    futures = {
        "key": f"NIFTY_{future[0]}_FUT",
        "expiry": str(future[0]),
        "lot_size": int(future[1]),
        "name": str(future[2]),
        "side": "FUT",
    }
    ce = {
        "key": f"NIFTY_{expiry}_{int(strike)}_CE",
        "expiry": expiry,
        "lot_size": int(pair[1]),
        "name": str(pair[3]),
        "side": "CE",
        "strike": int(strike),
    }
    pe = {
        "key": f"NIFTY_{expiry}_{int(strike)}_PE",
        "expiry": expiry,
        "lot_size": int(pair[2]),
        "name": str(pair[4]),
        "side": "PE",
        "strike": int(strike),
    }
    return futures, ce, pe


def _series_rows(
    db: sqlite3.Connection, day: str, key: str
) -> tuple[list[list[Any]], dict[str, Any]] | None:
    record = db.execute(
        "SELECT id,status,observed_minutes,missing_minutes,"
        "quarantined_source_minutes FROM series "
        "WHERE trade_day=? AND contract_key=?",
        (day, key),
    ).fetchone()
    if record is None or record[1] in ("EMPTY", "QUARANTINED_SPECIAL_INDEX_CLOCK"):
        return None
    candles = db.execute(
        "SELECT stamp,open,high,low,close,volume,oi "
        "FROM candles WHERE series_id=? ORDER BY stamp",
        (record[0],),
    ).fetchall()
    if not candles or len(candles) != record[2]:
        raise ValueError("breeze_contract_row_count_mismatch")
    return (
        [list(row) for row in candles],
        {
            "status": record[1],
            "observed_minutes": record[2],
            "missing_minutes": record[3],
            "quarantined_source_minutes": record[4],
        },
    )


def prepare_causal_day(
    minute_db: Path,
    eod_db: Path,
    day: str,
    output: Path,
) -> dict[str, Any]:
    """Write exactly four causal source series; abstain on any missing identity."""
    session = date.fromisoformat(day)
    if not date(2024, 11, 6) <= session <= date(2026, 11, 5):
        raise ValueError("research_day_outside_verified_study")
    if output.exists():
        raise ValueError("research_day_output_exists")
    with _reader(minute_db) as minute, _reader(eod_db) as eod:
        prior = _prior_context(eod, day)
        if prior is None:
            return {"status": "ABSTAIN", "reason": "prior_eod_unavailable", "day": day}
        prior_day, prior_close = prior
        listed = _prior_listed_contracts(eod, prior_day, prior_close, day)
        if listed is None:
            return {
                "status": "ABSTAIN", "reason": "previously_listed_contract_pair_missing",
                "day": day, "prior_eod_day": prior_day,
            }
        fut, ce, pe = listed
        selected: list[tuple[str, dict[str, Any]]] = [
            ("INDEX", {"key": "NIFTY_INDEX", "side": "INDEX"}),
            ("FUT", fut),
            ("CE", ce),
            ("PE", pe),
        ]
        bundles: list[tuple[str, dict[str, Any], list[list[Any]], dict[str, Any]]] = []
        for kind, ident in selected:
            matched = _series_rows(minute, day, ident["key"])
            if matched is None:
                return {
                    "status": "ABSTAIN",
                    "reason": f"selected_{kind.lower()}_intraday_unavailable",
                    "day": day,
                    "prior_eod_day": prior_day,
                    "contract_key": ident["key"],
                }
            rows, coverage = matched
            bundles.append((kind, ident, rows, coverage))
    # Explicitly never ask same-day EOD which strike was successful.
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    candle_dir = output / "candles"
    candle_dir.mkdir(mode=0o700)
    coverage_by_role: dict[str, dict[str, Any]] = {}
    for kind, ident, rows, coverage in bundles:
        symbol = (
            "NSE:NIFTY 50"
            if kind == "INDEX"
            else (
                f"NFO:NIFTY{ident['expiry'].replace('-', '')}"
                + (f"_{ident['strike']}" if "strike" in ident else "")
                + kind
            )
        )
        instrument = {
            "name": "NIFTY",
            "instrument_type": kind if kind != "INDEX" else "EQ",
            "lot_size": ident.get("lot_size", 1),
            "expiry": ident.get("expiry"),
            "strike": ident.get("strike"),
        }
        payload = {
            "symbol": symbol,
            "instrument": instrument,
            "timestamp_convention": "bar_start",
            "candles": rows,
            "research_only": True,
            "historical_broker_token_verified": False,
            "historical_quote_depth_available": False,
        }
        artifact = candle_dir / f"{kind}.json"
        with artifact.open("x") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.chmod(artifact, 0o600)
        coverage_by_role[kind] = {**coverage, "contract_key": ident["key"]}
    report = {
        "day": day,
        "status": "RESEARCH_BAR_PROXY_READY",
        "evidence_label": PROXY_LABEL,
        "prior_eod_day": prior_day,
        "prior_eod_close_used_for_selection": prior_close,
        "selection_method": "prior_day_earliest_expiry_atm_exact_identity",
        "selected_contracts": coverage_by_role,
        "historical_broker_token_verified": False,
        "historical_bid_ask_available": False,
        "execution_grade_ready": False,
        "promotion_eligible": False,
        "source_timestamp_convention_independently_verified": False,
    }
    with (output / "selection_report.json").open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    os.chmod(output / "selection_report.json", 0o600)
    return report


def execute(
    minute_db: Path,
    eod_db: Path,
    day: str,
    output: Path,
    *,
    prepare_only: bool = False,
) -> dict[str, Any]:
    if os.getenv("EXECUTION_MODE", "SHADOW").upper() != "SHADOW":
        raise ValueError("breeze_replay_requires_shadow_mode")
    if os.getenv("ENABLE_LIVE", "false").strip().lower() in (
        "true", "1", "yes", "on"
    ):
        raise ValueError("breeze_replay_refuses_live_environment")
    result = prepare_causal_day(minute_db, eod_db, day, output)
    if result["status"] != "RESEARCH_BAR_PROXY_READY" or prepare_only:
        return result
    # Calls the existing modelled OHLC strategy research engine, never the
    # production order manager or broker. No live promotion is possible here.
    from nifty_scalper_bot.backtesting.strategy_research import (
        run_archived_research,
    )

    report = run_archived_research(output, slippage_bps=10.0)
    report["breeze_evidence_label"] = PROXY_LABEL
    report["prior_day_instrument_selection"] = result
    report["execution_grade_ready"] = False
    report["promotion_eligible"] = False
    filename = output / "research_scenarios.json"
    with filename.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
    os.chmod(filename, 0o600)
    return {**result, "status": "RESEARCH_BAR_PROXY_REPLAYED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-db", required=True, type=Path)
    parser.add_argument("--eod-db", required=True, type=Path)
    parser.add_argument("--day", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    result = execute(
        args.minute_db, args.eod_db, args.day, args.output,
        prepare_only=args.prepare_only,
    )
    print(
        json.dumps(
            {
                "day": result["day"],
                "status": result["status"],
                "reason": result.get("reason"),
                "evidence_label": result.get("evidence_label"),
                "execution_grade_ready": False,
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] != "ABSTAIN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
