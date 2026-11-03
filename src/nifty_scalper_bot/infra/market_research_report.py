"""Post-session, broker-free analysis of authoritative replay plus journal samples.

Responsibilities: validate existing archived tick/quote provenance, separate
structural setup decisions from minute-sampled strategy rejections, and produce
60/180/300-second exact-contract ask-to-bid research labels and parity evidence.
Ownership: SQLite journal and the replay archive remain sole evidence owners;
this reader persists an atomic local report and bounded Supabase journal lines.
Safety: No broker orders or network calls, missing evidence fails closed, and
observational quotes cannot authorize changing any live entry or risk policy.
"""

from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.symbols import normalize_symbol

LOGGER = get_logger(__name__)
_IST = ZoneInfo("Asia/Kolkata")
_HORIZONS = (60, 180, 300)
_MIN_LABELLED = 10
_STRATEGIES = {"SMC", "VWAPPro", "ORBPro"}


def capture_storage_status(
    session_date: date, *, directory: Path | None = None
) -> dict[str, Any]:
    """Cheap, read-only replay size, writer-drop and queue health evidence."""
    from nifty_scalper_bot.config.paths import get_data_dir

    folder = directory if directory is not None else get_data_dir() / "replay_archive"
    tape = folder / f"{session_date.isoformat()}.jsonl"
    try:
        configured = int(os.getenv("REPLAY_CAPTURE_MAX_SESSION_MB", "512"))
    except (ValueError, TypeError):
        configured = 512
    ceiling = max(128, min(2048, configured)) * 1024 * 1024
    try:
        size = tape.stat().st_size
    except OSError:
        size = 0
    runs = 0
    dropped = 0
    failed = 0
    pending = 0
    recorded = 0
    for sidecar in folder.glob("*.integrity.json"):
        try:
            obj = json.loads(sidecar.read_text(encoding="utf-8"))
            counted = int(
                obj.get("written_by_day", {}).get(session_date.isoformat(), 0)
            )
            if counted <= 0:
                continue
            runs += 1
            recorded += counted
            dropped += max(0, int(obj.get("dropped") or 0))
            failed += max(0, int(obj.get("failed") or 0))
            pending += max(0, int(obj.get("pending") or 0))
        except (OSError, TypeError, ValueError, KeyError):
            continue
    enabled = os.getenv("REPLAY_CAPTURE_ENABLED", "true").lower() == "true"
    used_fraction = round(size / ceiling, 4)
    warnings = []
    if enabled and not tape.is_file():
        warnings.append("session_tape_not_recorded")
    if enabled and size and runs == 0:
        warnings.append("writer_integrity_sidecars_missing")
    if dropped or failed:
        warnings.append("writer_integrity_degraded")
    if used_fraction >= 0.80:
        warnings.append("replay_session_capacity_near_limit")
    if pending:
        warnings.append("writer_queue_pending")
    return {
        "v": 1,
        "date": session_date.isoformat(),
        "capture_enabled": enabled,
        "tape_present": tape.is_file(),
        "bytes": size,
        "capacity_bytes": ceiling,
        "capacity_used_fraction": used_fraction,
        "writer_runs": runs,
        "recorded_events": recorded,
        "dropped_events": dropped,
        "failed_events": failed,
        "pending_events": pending,
        "warnings": warnings,
        "complete_capture_verified": False,
    }


def _parse_records(
    journal: str, marker: str, session_date: date
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in journal.splitlines():
        if marker not in line:
            continue
        _, _, raw = line.partition(marker + " ")
        if not raw:
            continue
        try:
            payload = json.loads(raw.strip())
            if not isinstance(payload, dict) or payload.get("v") != 1:
                continue
            instant = datetime.fromisoformat(
                str(payload.get("decision_at") or payload.get("checked_at"))
            )
            if (
                instant.tzinfo is None
                or instant.astimezone(_IST).date() != session_date
            ):
                continue
            records.append(payload)
        except (ValueError, TypeError, KeyError):
            continue
    return records


def _parity_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    # A restarted process can re-observe one completed minute. Last evidence
    # for the exact contract/minute wins; missing is never converted to match.
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        symbol = normalize_symbol(str(row.get("symbol") or ""))
        minute = str(row.get("indicator_minute") or row.get("checked_at") or "")
        if symbol.startswith("NFO:") and minute:
            unique[(symbol, minute)] = row
    statuses: dict[str, int] = defaultdict(int)
    by_side: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (symbol, _minute), row in unique.items():
        status = str(row.get("status") or "missing")
        statuses[status] += 1
        by_side[symbol[-2:]][status] += 1
    matches = statuses.get("same_minute_volume_match", 0)
    return {
        "unit": "distinct_selected_contract_completed_minute",
        "observations": len(unique),
        "positive_volume_matches": matches,
        "verified_fraction_of_observed": (
            round(matches / len(unique), 4) if unique else 0.0
        ),
        "statuses": dict(sorted(statuses.items())),
        "by_side": {
            key: dict(sorted(val.items())) for key, val in sorted(by_side.items())
        },
        "complete_day_coverage_proven": False,
    }


def _sampled_markouts(
    records: list[dict[str, Any]],
    quotes: Mapping[str, list[tuple[float, float, float]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from scripts.reporting.rejected_setup_forward_markout import observed_markout

    # One timestamped no-vote per strategy/contract/UTC minute. This is NOT
    # deduplication by structural setup ID and cannot justify changing a gate.
    samples: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in records:
        strategy = str(row.get("strategy") or "")
        contract = normalize_symbol(str(row.get("symbol") or ""))
        minute = str(row.get("evaluation_minute") or "")
        if (
            strategy not in _STRATEGIES
            or not contract.startswith("NFO:")
            or not contract.endswith(("CE", "PE"))
            or not minute
        ):
            continue
        samples.setdefault((strategy, contract, minute), row)

    outcomes: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for (strategy, contract, minute), row in sorted(samples.items()):
        try:
            instant = datetime.fromisoformat(str(row["decision_at"]))
            if instant.tzinfo is None:
                continue
            decision_ts = instant.timestamp()
        except (ValueError, TypeError, KeyError):
            continue
        reason = str(row.get("reason") or "none")[:100]
        for horizon in _HORIZONS:
            observation, status = observed_markout(
                quotes.get(contract, ()),
                decision_ts=decision_ts,
                horizon_s=horizon,
            )
            result = {
                "strategy": strategy,
                "contract": contract,
                "side": contract[-2:],
                "minute": minute,
                "decision_at": instant.isoformat(),
                "reason": reason,
                "horizon_seconds": horizon,
                "markout": observation,
                "label_status": status,
                "sampling": "first_no_vote_per_contract_strategy_utc_minute",
                "independent_structural_setup": False,
            }
            outcomes.append(result)
            groups[(strategy, reason, horizon)].append(result)

    strata: list[dict[str, Any]] = []
    for (strategy, reason, horizon), items in sorted(groups.items()):
        labelled = [
            float(item["markout"]["gross_spread_adjusted_markout_pct"])
            for item in items
            if isinstance(item["markout"], dict)
        ]
        enough = len(labelled) >= _MIN_LABELLED
        strata.append(
            {
                "strategy": strategy,
                "reason": reason,
                "horizon_seconds": horizon,
                "sampled_minutes": len(items),
                "labelled": len(labelled),
                "unlabelled": len(items) - len(labelled),
                "coverage_fraction": round(len(labelled) / len(items), 4),
                "mean_ask_to_bid_pct": (
                    round(statistics.mean(labelled), 5) if enough else None
                ),
                "median_ask_to_bid_pct": (
                    round(statistics.median(labelled), 5) if enough else None
                ),
                "positive_fraction": (
                    round(sum(value > 0 for value in labelled) / len(labelled), 4)
                    if enough
                    else None
                ),
                "minimum_labelled": _MIN_LABELLED,
                "independent_structural_setup": False,
                "eligible_for_filter_change": False,
            }
        )
    return outcomes, strata


def build_session_research_report(
    *,
    session_date: date,
    journal: str,
    replay_path: Path,
    trade_db: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build an integrity-checked report; missing inputs fail closed.

    Only file-system and CPU work after trading hours. No market/broker/network
    access; atomic persisted output remains inspectable across process restarts.
    """
    from scripts.reporting.rejected_setup_forward_markout import (
        evaluate_opportunities,
        load_archive_quotes,
        summarize_groups,
    )

    from nifty_scalper_bot.backtesting.setup_opportunity_analysis import (
        canonicalize_setup_opportunities,
    )
    from nifty_scalper_bot.journal.trade_ledger import load_candidate_decision_rows

    parity = _parity_summary(
        _parse_records(journal, "OPTION_COMPLETED_VOLUME_PARITY", session_date)
    )
    rejection_records = _parse_records(
        journal, "RESEARCH_STRATEGY_REJECTION", session_date
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "date_ist": session_date.isoformat(),
        "horizons_seconds": list(_HORIZONS),
        "analysis": "observed_exact_contract_ask_to_bid_not_broker_fills",
        "volume_parity": parity,
        "sampled_observations": [],
        "sampled_strata": [],
        "structural_outcomes": [],
        "structural_strata": [],
        "independent_setup_count": 0,
        "replay_sha256": None,
        "eligible_for_entry_filter_change": False,
        "status": "unavailable",
        "blocker": None,
    }
    storage = capture_storage_status(session_date, directory=replay_path.parent)
    report["capture_storage"] = storage
    try:
        # Pending observations may not yet have reached the tape.
        # Once settled, the existing strict reader validates drops,
        # failures, sequence continuity and all per-run sidecars.
        if storage["pending_events"]:
            raise ValueError("replay_capture_writer_not_settled")
        quotes, sha, quote_count = load_archive_quotes(replay_path)
        report["writer_integrity_verified"] = True
        report["replay_sha256"] = sha
        report["tradable_ws_quote_ticks"] = quote_count
        if not quote_count:
            report["blocker"] = "no_tradable_depth_ws_quotes"
        else:
            sampled, strata = _sampled_markouts(rejection_records, quotes)
            report["sampled_observations"] = sampled
            report["sampled_strata"] = strata
            if trade_db.is_file():
                structural = canonicalize_setup_opportunities(
                    load_candidate_decision_rows(trade_db)
                )
                report["independent_setup_count"] = len(structural)
                events = evaluate_opportunities(structural, quotes, _HORIZONS)
                report["structural_outcomes"] = events
                report["structural_strata"] = summarize_groups(
                    events, min_labelled=_MIN_LABELLED
                )
            else:
                report["structural_data_blocker"] = "canonical_trade_db_missing"
            report["status"] = "analysed"
    except (OSError, ValueError, TypeError) as exc:
        report["blocker"] = (
            str(exc)[:110]
            if isinstance(exc, ValueError)
            else f"research_source_{type(exc).__name__}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(output_path)
    return report


def research_report_log_lines(report: Mapping[str, Any]) -> list[str]:
    """Serialize bounded, non-secret results for the existing archive text."""
    base = {
        "v": 1,
        "date": report["date_ist"],
        "status": report["status"],
        "blocker": report.get("blocker"),
        "replay_sha256": report.get("replay_sha256"),
        "tradable_ws_quote_ticks": report.get("tradable_ws_quote_ticks"),
        "independent_setup_count": report["independent_setup_count"],
        "sampled_observations": len(report["sampled_observations"]),
        "volume_parity": report["volume_parity"],
    }

    def encoded(row: Mapping[str, Any]) -> str:
        return json.dumps(row, separators=(",", ":"), allow_nan=False)

    lines = ["RESEARCH_SESSION_EVIDENCE " + encoded(base)]
    for kind, strata in (
        ("minute_sample", report["sampled_strata"]),
        ("structural_setup", report["structural_strata"]),
    ):
        for row in strata:
            lines.append(
                "RESEARCH_MARKOUT_STRATUM " + encoded({"v": 1, "kind": kind, **row})
            )
    return lines


def log_research_report(report: Mapping[str, Any]) -> None:
    """Local operator telemetry; archiver appends these records to its snapshot."""
    for line in research_report_log_lines(report):
        LOGGER.info("%s", line)


def run_postmarket_research(session_date: date, journal: str) -> dict[str, Any]:
    """Run once after session close using only existing host-local evidence."""
    from nifty_scalper_bot.config.paths import get_data_dir

    data_dir = get_data_dir()
    report = build_session_research_report(
        session_date=session_date,
        journal=journal,
        replay_path=data_dir / "replay_archive" / f"{session_date}.jsonl",
        trade_db=data_dir / "trades.db",
        output_path=data_dir / "research" / "markouts" / f"{session_date}.json",
    )
    log_research_report(report)
    return report
