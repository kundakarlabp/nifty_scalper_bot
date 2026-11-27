#!/usr/bin/env python3
"""Replay checksum-verified Breeze baskets with the existing SHADOW strategy engine.

Responsibilities: Resume session-level offline research safely, authenticate
SQLite/EOD provenance, aggregate resolved cost-stress P&L and no-vote reasons.
Ownership: Explicitly invoked research process, never the production runner.
Safety: Reads private market data locally; cannot call brokers or promote LIVE.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.stage_breeze_sqlite_research import _verify_research_source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shadow_only() -> None:
    if (
        os.getenv("EXECUTION_MODE", "").upper() != "SHADOW"
        or os.getenv("ENABLE_LIVE", "false").lower() in {"true", "1", "yes"}
        or os.getenv("ENABLE_LIVE_TRADING", "false").lower()
        in {"true", "1", "yes"}
        or os.getenv("ALLOW_REAL_BROKER", "false").lower()
        in {"true", "1", "yes"}
    ):
        raise ValueError("breeze_offline_batch_requires_explicit_shadow")


def _private_json(path: Path, payload: dict[str, Any]) -> None:
    target = path.with_suffix(path.suffix + ".tmp")
    if target.exists():
        raise ValueError("research_temporary_output_already_exists")
    with target.open("x") as stream:
        json.dump(payload, stream, sort_keys=True, default=str)
        stream.write("\n")
    target.chmod(0o600)
    target.replace(path)


def run_batch(
    *,
    breeze_db: Path,
    eod_db: Path,
    summary_path: Path,
    staged_dir: Path,
    output_dir: Path,
    selected_days: tuple[str, ...] = (),
    include_partial: bool = False,
    expected_windows: int = 25,
    limit: int = 1000,
) -> dict[str, Any]:
    """Reuse canonical bar research; never fabricate options or executable quotes."""
    _shadow_only()
    if not 1 <= limit <= 1000 or not staged_dir.is_dir():
        raise ValueError("historical_stage_or_limit_invalid")
    if output_dir.resolve() == staged_dir.resolve():
        raise ValueError("research_output_must_be_separate_from_source")
    validated = _verify_research_source(breeze_db, summary_path, expected_windows)
    database_digest = str(validated["dataset_sha256"])
    eod_digest = _sha256(eod_db)
    from nifty_scalper_bot.backtesting.strategy_research import (
        load_archive,
        run_archived_research,
        summarize,
    )
    from nifty_scalper_bot.config.settings import get_settings

    experiment_settings = {
        "elite": asdict(get_settings().elite),
        "cost_controls": {
            key: value for key, value in os.environ.items()
            if key.startswith(("COST_", "MIN_NET_", "RISK_MIN_"))
        },
    }
    configuration_digest = hashlib.sha256(
        json.dumps(experiment_settings, sort_keys=True, default=str).encode()
    ).hexdigest()
    # Hash every Python source in the installed bot package. Changes to any
    # strategy, regime, cost model or replay dependency invalidate resume.
    package_root = Path(inspect.getfile(load_archive)).resolve().parents[1]
    code_hash = hashlib.sha256()
    for source in sorted(package_root.rglob("*.py")):
        code_hash.update(str(source.relative_to(package_root)).encode())
        code_hash.update(source.read_bytes())
    engine_digest = code_hash.hexdigest()
    all_days = sorted(path.name for path in staged_dir.glob("20??-??-??") if path.is_dir())
    if selected_days:
        wanted = set(selected_days)
        if len(wanted) != len(selected_days) or not wanted.issubset(all_days):
            raise ValueError("selected_research_session_not_staged")
        days = [day for day in all_days if day in wanted]
    else:
        if len(all_days) > limit:
            raise ValueError("historical_session_limit_would_truncate_data")
        days = all_days
    if not days:
        raise ValueError("historical_session_selection_empty")
    # All input checks must complete before writing any outputs.
    eligible: list[tuple[str, str]] = []
    skipped: Counter[str] = Counter()
    for day in days:
        path = staged_dir / day
        manifest = json.loads((path / "manifest.json").read_text())
        if (manifest.get("trade_day") != day or manifest.get("research_only") is not True
                or manifest.get("live_promotion_eligible") is not False
                or manifest.get("execution_grade_ready") is not False):
            raise ValueError("staged_session_not_research_verified")
        if manifest.get("source_dataset_sha256") != database_digest:
            raise ValueError("staged_session_dataset_digest_mismatch")
        if manifest.get("source_eod_sha256") != eod_digest:
            raise ValueError("staged_session_eod_digest_mismatch")
        if not manifest.get("orb_opening_range_complete"):
            skipped["opening_range_incomplete"] += 1
            continue
        if not include_partial and manifest.get("missing_or_unsynchronized_minutes"):
            skipped["partial_synchronized_session"] += 1
            continue
        _, instruments, digest = load_archive(path)
        if len(instruments) != 4:
            raise ValueError("historical_session_instrument_count_invalid")
        eligible.append((day, digest))
    if output_dir.exists() and not (output_dir / "runs").is_dir():
        raise ValueError("research_existing_output_missing_run_directory")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(mode=0o700, exist_ok=True)
    trades: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    funnels: dict[str, Counter[str]] = defaultdict(Counter)
    completed = 0
    reused = 0
    for day, data_digest in eligible:
        artifact = runs_dir / f"{day}.json"
        if artifact.exists():
            item = json.loads(artifact.read_text())
            if (
                item.get("day") != day
                or item.get("source_sha256") != data_digest
                or item.get("engine_sha256") != engine_digest
                or item.get("configuration_sha256") != configuration_digest
                or item.get("evidence") != "RESEARCH_CANDIDATE"
            ):
                raise ValueError("research_resume_stale_or_mismatched_artifact")
            reused += 1
        else:
            report = run_archived_research(staged_dir / day, slippage_bps=10.0)
            if (
                report.get("evidence_label") != "RESEARCH_CANDIDATE"
                or report.get("live_equivalent") is not False
                or report.get("data_sha256") != data_digest
            ):
                raise ValueError("canonical_research_result_provenance_mismatch")
            # Option basket is only known after the first underlying minute.
            earliest = datetime.fromisoformat(
                json.loads((staged_dir / day / "manifest.json").read_text())[
                    "contract_selection_not_before"
                ]
            )
            for scenario in report["scenarios"]:
                for body in scenario["strategies"].values():
                    for trade in body.get("trades") or []:
                        if datetime.fromisoformat(trade["entry_time"]) < earliest:
                            raise ValueError("research_trade_precedes_contract_selection")
            item = {
                "day": day,
                "source_sha256": data_digest,
                "engine_sha256": engine_digest,
                "configuration_sha256": configuration_digest,
                "evidence": report["evidence_label"],
                "report": report,
            }
            _private_json(artifact, item)
            completed += 1
        for scenario in item["report"]["scenarios"]:
            bps = float(scenario["slippage_bps_per_side"])
            for strategy, body in scenario["strategies"].items():
                trades[(bps, strategy)].extend(body.get("trades") or [])
                if bps == 10.0:
                    reasons[strategy].update(body.get("no_vote_reasons") or {})
                    funnels[strategy].update(body.get("entry_funnel") or {})
    result_rows = []
    for bps, strategy in sorted(trades):
        metrics = summarize(trades[(bps, strategy)])
        result_rows.append({
            "strategy": strategy,
            "slippage_bps_per_side": bps,
            "trade_count": metrics["trade_count"],
            "net_pnl": metrics["net_pnl"],
            "fees": metrics["fees"],
            "profit_factor": metrics["profit_factor"],
            "max_realized_drawdown": metrics["max_realized_drawdown"],
        })
    report = {
        "scope": "Breeze_identified_OHLCV_strategy_component_research",
        "dataset_sha256": database_digest,
        "eod_sha256": eod_digest,
        "engine_sha256": engine_digest,
        "configuration_sha256": configuration_digest,
        "requested_sessions": len(days),
        "eligible_sessions": len(eligible),
        "skipped": dict(skipped),
        "newly_completed": completed,
        "verified_resume_count": reused,
        "results": result_rows,
        "no_vote_reasons_10bps": {key: dict(val.most_common(12)) for key, val in reasons.items()},
        "entry_funnel_10bps": {key: dict(val) for key, val in funnels.items()},
        "source_timestamp_semantics_independently_verified": False,
        "historical_bid_ask_or_broker_fills_present": False,
        "research_only": True,
        "execution_grade_ready": False,
        "live_promotion_eligible": False,
    }
    _private_json(output_dir / "aggregate.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breeze-db", required=True, type=Path)
    parser.add_argument("--eod-db", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--staged-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--day", action="append", default=[])
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--expected-windows", type=int, default=25)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    result = run_batch(
        breeze_db=args.breeze_db, eod_db=args.eod_db,
        summary_path=args.summary, staged_dir=args.staged_dir,
        output_dir=args.output_dir, selected_days=tuple(args.day),
        include_partial=args.include_partial,
        expected_windows=args.expected_windows, limit=args.limit,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
