#!/usr/bin/env python3
"""Purpose: Audit original captured replay epochs and same-epoch quote outcomes.

Responsibilities: Verify writer sidecars and isolate original process epochs,
scope canonical decisions to the IST session, and measure exact-contract quote
coverage without crossing a restart or selected-basket change.
Boundaries: Offline SHADOW-only, no broker or network calls and no live changes.
Safety: Missing quote paths, incomplete segment fidelity and absent fills never
become synthesized outcomes or a strategy-promotion recommendation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from scripts.reporting.rejected_setup_forward_markout import (
    load_archive_quotes,
    observed_markout,
)

from nifty_scalper_bot.backtesting.replay_fidelity import _selected, audit_session
from nifty_scalper_bot.backtesting.runtime_research import validate_replay_epochs
from nifty_scalper_bot.backtesting.setup_opportunity_analysis import (
    _research_context,
    canonicalize_setup_opportunities,
)
from nifty_scalper_bot.infra.market_research_report import (
    _cost_model,
    _quote_path_dense,
    _session_decision_rows,
)
from nifty_scalper_bot.journal.trade_ledger import load_candidate_decision_rows
from nifty_scalper_bot.ops.research_jobs import partition_replay_runs
from nifty_scalper_bot.storage.replay_archive import archive_fingerprint, iter_session
from nifty_scalper_bot.utils.symbols import normalize_symbol

_IST = ZoneInfo("Asia/Kolkata")
_HORIZONS = (60, 180, 300, 600, 900)
_STRESS_BPS = (10.0, 25.0, 50.0)
_DEFAULT_LOT = 65  # Research assumption, not an exchange or broker fill.


def _semantic_sha256(path: Path) -> str:
    """Pin uncompressed JSONL bytes across validated gzip compaction."""
    digest = hashlib.sha256()
    reader = gzip.open if path.name.endswith(".gz") else Path.open
    with reader(path, "rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def _day_from_filename(path: Path) -> date:
    name = path.name
    if name.endswith(".jsonl.gz"):
        day_text = name.removesuffix(".jsonl.gz")
    elif name.endswith(".jsonl"):
        day_text = name.removesuffix(".jsonl")
    else:
        raise ValueError("epoch_audit_invalid_archive_name")
    return date.fromisoformat(day_text)


def _epoch_context(path: Path, run_id: str) -> dict[str, Any]:
    """Take all evidence from existing validated original epoch and sidecar."""
    event_count, first, snapshots = validate_replay_epochs(path)
    fidelity = audit_session(path)
    if fidelity.get("archive_integrity_verified") is not True:
        raise ValueError("epoch_audit_fidelity_not_verified")
    first_ts: float | None = None
    last_ts: float | None = None
    selections: list[tuple[float, str, str]] = []
    for event in iter_session(path):
        instant = datetime.fromisoformat(event["available_at"])
        timestamp = instant.timestamp()
        if instant.astimezone(_IST).date() != _day_from_filename(path):
            raise ValueError("epoch_audit_cross_session_event")
        if first_ts is None:
            first_ts = timestamp
        last_ts = timestamp
        if event["kind"] == "snapshot":
            chosen = _selected(event["payload"])
            if not chosen:
                raise ValueError("epoch_audit_selected_contracts_unproven")
            selections.append(
                (
                    timestamp,
                    normalize_symbol(chosen["CE"]),
                    normalize_symbol(chosen["PE"]),
                )
            )
    if first_ts is None or last_ts is None or not selections:
        raise ValueError("epoch_audit_empty_epoch")
    quotes, quote_sha, quote_count = load_archive_quotes(path)
    return {
        "run_id": run_id,
        "start": first_ts,
        "end": last_ts,
        "event_count": event_count,
        "snapshot_count": snapshots,
        "archive_sha256": quote_sha,
        "quote_count": quote_count,
        "fidelity": fidelity,
        "selections": selections,
        "quotes": quotes,
        "captured_revision": first.get("code_revision"),
    }


def _exact_selected(
    context: Mapping[str, Any],
    *,
    symbol: str,
    side: str,
    at: float,
    horizon: int,
) -> str | None:
    """Require matching original selection at decision and through exit horizon."""
    if side not in {"CE", "PE"}:
        return "unsupported_option_side"
    selections = context["selections"]
    active: tuple[float, str, str] | None = None
    for snapshot in selections:
        if snapshot[0] <= at:
            active = snapshot
        else:
            break
    if active is None:
        return "selected_contract_before_first_snapshot_unknown"
    selected = active[1] if side == "CE" else active[2]
    if selected != symbol:
        return "contract_not_selected_at_decision"
    for snapshot in selections:
        if at < snapshot[0] <= at + horizon:
            current = snapshot[1] if side == "CE" else snapshot[2]
            if current != symbol:
                return "selected_contract_rotated_during_horizon"
    return None


def _epoch_for_decision(
    contexts: list[dict[str, Any]], timestamp: float
) -> dict[str, Any] | None:
    matches = [c for c in contexts if c["start"] <= timestamp <= c["end"]]
    if len(matches) > 1:
        raise ValueError("epoch_audit_ambiguous_epoch_timestamp")
    return matches[0] if matches else None


def _label(opportunity: Any, context: dict[str, Any], horizon: int) -> dict[str, Any]:
    ts = float(opportunity.decision_ts)
    symbol = normalize_symbol(str(opportunity.symbol))
    entry: dict[str, Any] = {
        "run_id": context["run_id"],
        "opportunity_id": opportunity.opportunity_id,
        "strategy": opportunity.strategy,
        "approved": bool(opportunity.approved),
        "reason": opportunity.final_reason,
        "symbol": symbol,
        "decision_timestamp": ts,
        "horizon_seconds": horizon,
        "status": "unavailable",
        "markout": None,
        "cost_scenarios": None,
        "cost_quantity_provenance": None,
        "independent_setup": True,
    }
    if not symbol.startswith("NFO:") or not symbol.endswith(("CE", "PE")):
        entry["status"] = "missing_exact_option_contract"
        return entry
    recorded = {normalize_symbol(x) for x in opportunity.observed_symbols}
    if recorded != {symbol}:
        entry["status"] = "contract_rotation_or_missing_identity"
        return entry
    if ts + horizon > context["end"]:
        entry["status"] = "horizon_crosses_epoch_end"
        return entry
    selection_failure = _exact_selected(
        context, symbol=symbol, side=symbol[-2:], at=ts, horizon=horizon
    )
    if selection_failure:
        entry["status"] = selection_failure
        return entry
    quotes = context["quotes"].get(symbol, ())
    timestamps = [row[0] for row in quotes]
    markout, status = observed_markout(
        quotes, decision_ts=ts, horizon_s=horizon, timestamps=timestamps
    )
    entry["status"] = status
    entry["markout"] = markout
    if markout is None:
        return entry
    if not _quote_path_dense(timestamps, decision_ts=ts, horizon_s=horizon):
        entry["cost_label_status"] = "sparse_or_gapped_quote_path"
        return entry
    lot = opportunity.research_context.get("lot_size")
    verified_lot = type(lot) is int and 0 < lot < 10000
    quantity = lot if verified_lot else _DEFAULT_LOT
    entry["cost_scenarios"] = {
        str(int(bps)): _cost_model(
            markout, quantity=quantity, adverse_slippage_bps=bps
        )["terminal"]
        for bps in _STRESS_BPS
    }
    entry["cost_label_status"] = "modeled_not_broker_fill"
    entry["cost_quantity_provenance"] = (
        "recorded_decision_lot_size" if verified_lot else "unverified_default_lot"
    )
    return entry


def _candidate_inclusion_funnel(
    rows: list[dict[str, Any]], opportunities: Any
) -> dict[str, Any]:
    """Account for journal decisions excluded before unique-setup attribution."""
    excluded: Counter[str] = Counter()
    eligible = 0
    for row in rows:
        if row.get("event_name") not in {"candidate.approved", "candidate.blocked"}:
            excluded["not_candidate_decision_event"] += 1
            continue
        context = _research_context(row)
        setup_id = str(
            context.get("setup_id") or context.get("setup_structure_id") or ""
        ).strip()
        if not setup_id:
            excluded["missing_owned_setup_id"] += 1
            continue
        if not str(context.get("strategy") or "").strip():
            excluded["missing_owned_strategy"] += 1
            continue
        side = str(
            context.get("contract_side")
            or context.get("trade_side")
            or row.get("direction")
            or ""
        ).strip().upper()
        if side not in {"CE", "PE"}:
            excluded["invalid_option_side"] += 1
            continue
        eligible += 1

    included = sum(int(item.decision_count) for item in opportunities)
    if included != eligible or included + sum(excluded.values()) != len(rows):
        raise ValueError("epoch_audit_canonical_candidate_count_conflict")
    return {
        "source_decision_rows": len(rows),
        "canonicalized_decision_rows": included,
        "excluded_before_canonicalization": sum(excluded.values()),
        "exclusion_reasons": dict(sorted(excluded.items())),
        "independent_setup_count": len(opportunities),
        "repeated_decisions_collapsed": included - len(opportunities),
    }


def _report_grouped(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["strategy"], row["horizon_seconds"])].append(row)
    report = []
    for (strategy, horizon), observed in sorted(groups.items()):
        statuses = Counter(row["status"] for row in observed)
        quote_valid = sum(row["markout"] is not None for row in observed)
        cost_valid = sum(row["cost_scenarios"] is not None for row in observed)
        report.append(
            {
                "strategy": strategy,
                "horizon_seconds": horizon,
                "independent_setup_outcomes": len(observed),
                "quote_valid": quote_valid,
                "quote_missing": len(observed) - quote_valid,
                "cost_modelled": cost_valid,
                "cost_unavailable": len(observed) - cost_valid,
                "status_counts": dict(sorted(statuses.items())),
                "complete_trade_outcome_count": 0,
                "comparison_eligible": False,
            }
        )
    return report


def audit_original_session(
    session: Path,
    trade_db: Path,
    *,
    expected_uncompressed_sha256: str | None = None,
) -> dict[str, Any]:
    """Strict original-epoch research; preserve every input file unchanged."""
    if os.getenv("EXECUTION_MODE", "").strip().upper() != "SHADOW":
        raise ValueError("epoch_audit_requires_shadow")
    if not session.is_file():
        raise ValueError("epoch_audit_original_tape_missing")
    if not trade_db.is_file():
        raise ValueError("epoch_audit_canonical_trade_db_missing")
    day = _day_from_filename(session)
    physical_sha = archive_fingerprint(session)
    semantic_sha = _semantic_sha256(session)
    if expected_uncompressed_sha256 and semantic_sha != expected_uncompressed_sha256:
        raise ValueError("epoch_audit_pinned_uncompressed_sha256_mismatch")
    raw_decisions = load_candidate_decision_rows(trade_db)
    day_decisions = _session_decision_rows(raw_decisions, day)
    with tempfile.TemporaryDirectory(prefix="nifty-epoch-audit-") as work:
        epochs = partition_replay_runs(session, Path(work) / "epochs")
        contexts = [_epoch_context(path, run_id) for run_id, path in epochs]
        if not contexts:
            raise ValueError("epoch_audit_no_valid_epochs")
        if any(
            contexts[i]["end"] >= contexts[i + 1]["start"]
            for i in range(len(contexts) - 1)
        ):
            raise ValueError("epoch_audit_overlapping_epoch_windows")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        unassigned: Counter[str] = Counter()
        for row in day_decisions:
            try:
                raw = row["timestamp"]
                if raw is None or isinstance(raw, bool):
                    raise ValueError("invalid_timestamp")
                timestamp = float(raw)
                if not math.isfinite(timestamp):
                    raise ValueError("invalid_timestamp")
                epoch = _epoch_for_decision(contexts, timestamp)
            except (ValueError, TypeError, KeyError, OverflowError):
                unassigned["invalid_decision_timestamp"] += 1
                continue
            if epoch is None:
                unassigned["decision_outside_valid_recording_epochs"] += 1
            else:
                grouped[epoch["run_id"]].append(row)
        outcomes: list[dict[str, Any]] = []
        epoch_meta: list[dict[str, Any]] = []
        setup_total = 0
        for c in contexts:
            decisions = grouped[c["run_id"]]
            opportunities = canonicalize_setup_opportunities(decisions)
            funnel = _candidate_inclusion_funnel(decisions, opportunities)
            setup_total += len(opportunities)
            for opportunity in opportunities:
                if not c["fidelity"].get("observed_selected_context_available"):
                    for horizon in _HORIZONS:
                        outcomes.append(
                            {
                                "run_id": c["run_id"],
                                "opportunity_id": opportunity.opportunity_id,
                                "strategy": opportunity.strategy,
                                "approved": bool(opportunity.approved),
                                "reason": opportunity.final_reason,
                                "symbol": normalize_symbol(opportunity.symbol),
                                "decision_timestamp": opportunity.decision_ts,
                                "horizon_seconds": horizon,
                                "status": "epoch_selected_quote_context_incomplete",
                                "markout": None,
                                "cost_scenarios": None,
                                "independent_setup": True,
                            }
                        )
                else:
                    outcomes.extend(
                        _label(opportunity, c, horizon) for horizon in _HORIZONS
                    )
            epoch_meta.append(
                {
                    "run_id": c["run_id"],
                    "start_ist": datetime.fromtimestamp(c["start"], _IST).isoformat(),
                    "end_ist": datetime.fromtimestamp(c["end"], _IST).isoformat(),
                    "event_count": c["event_count"],
                    "snapshot_count": c["snapshot_count"],
                    "quote_count": c["quote_count"],
                    "archive_sha256": c["archive_sha256"],
                    "captured_revision": c["captured_revision"],
                    "basket_segments": c["fidelity"]["basket_segments"],
                    "verified_basket_segments": c["fidelity"][
                        "basket_segments_with_quote_context"
                    ],
                    "writer_integrity_verified": True,
                    "selected_quote_context_available": c["fidelity"][
                        "observed_selected_context_available"
                    ],
                    "canonical_decision_rows": len(decisions),
                    "canonical_inclusion_funnel": funnel,
                    "selected_quote_context_missing_by_requirement": c["fidelity"][
                        "basket_segments_missing_by_requirement"
                    ],
                    "independent_setups": len(opportunities),
                }
            )
    if archive_fingerprint(session) != physical_sha:
        raise ValueError("epoch_audit_source_mutated_during_research")
    counts = Counter(
        row["strategy"] for row in outcomes if row["horizon_seconds"] == 900
    )
    assigned = sum(
        item["canonical_inclusion_funnel"]["source_decision_rows"]
        for item in epoch_meta
    )
    if assigned + sum(unassigned.values()) != len(day_decisions):
        raise ValueError("epoch_audit_session_decision_denominator_conflict")
    return {
        "scope": "original_captured_epochs_exact_contract_observational_markout",
        "session_date_ist": day.isoformat(),
        "status": "analysed_research_only",
        "source_bytes_sha256": physical_sha,
        "source_uncompressed_sha256": semantic_sha,
        "source_hash_pinned": expected_uncompressed_sha256 is not None,
        "original_writer_integrity_verified": True,
        "original_epoch_count": len(epoch_meta),
        "epochs": epoch_meta,
        "journal_total_decision_rows": len(raw_decisions),
        "journal_same_day_decision_rows": len(day_decisions),
        "journal_out_of_day_decision_rows": len(raw_decisions) - len(day_decisions),
        "journal_unassigned_same_day_decision_rows": dict(unassigned),
        "journal_assigned_same_epoch_decision_rows": sum(
            item["canonical_inclusion_funnel"]["source_decision_rows"]
            for item in epoch_meta
        ),
        "journal_excluded_before_canonicalization": sum(
            item["canonical_inclusion_funnel"]["excluded_before_canonicalization"]
            for item in epoch_meta
        ),
        "journal_canonicalized_decision_rows": sum(
            item["canonical_inclusion_funnel"]["canonicalized_decision_rows"]
            for item in epoch_meta
        ),
        "independent_same_epoch_setups": setup_total,
        "independent_setup_counts_by_strategy": dict(sorted(counts.items())),
        "grouped_outcomes": _report_grouped(outcomes),
        "outcomes": outcomes,
        "comparison": {
            "status": "BLOCKED_NO_MATCHED_FAILED_BREAKOUT_TRADE_EVIDENCE",
            "reason": (
                "VWAP/ORB/SMC canonical setup decisions and observed quote "
                "markouts cannot be compared causally with an unregistered "
                "failed-breakout hypothesis. Align underlying events and "
                "obtain executable quotes, fees, stops and fill evidence."
            ),
            "profitability_inference_allowed": False,
        },
        "live_equivalent": False,
        "promotion_eligible": False,
        "actual_broker_fills_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--trade-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-uncompressed-sha256")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite any previous report")
    result = audit_original_session(
        args.session,
        args.trade_db,
        expected_uncompressed_sha256=args.expected_uncompressed_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
