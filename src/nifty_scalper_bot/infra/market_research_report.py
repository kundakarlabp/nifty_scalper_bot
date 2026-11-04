"""Post-session, broker-free analysis of authoritative replay plus journal samples.

Responsibilities: validate existing archived tick/quote provenance, separate
structural setup decisions from minute-sampled strategy rejections, and produce
60/180/300/600/900-second exact-contract ask-to-bid research labels, modeled
execution costs, adverse excursion and volume parity evidence.
Ownership: SQLite journal and the replay archive remain sole evidence owners;
this reader persists an atomic local report and bounded Supabase journal lines.
Safety: No broker orders or network calls, missing evidence fails closed, and
observational quotes cannot authorize changing any live entry or risk policy.
"""

from __future__ import annotations

import bisect
import json
import os
import statistics
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from nifty_scalper_bot.risk.cost_model import estimate_round_trip_cost
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.symbols import normalize_symbol

LOGGER = get_logger(__name__)
_IST = ZoneInfo("Asia/Kolkata")
_HORIZONS = (60, 180, 300, 600, 900)
_MODELED_QUANTITY_DEFAULT = 65  # Research assumption; verify instrument lot.
_MODELED_ADVERSE_SLIPPAGE_BPS = 5.0  # Per-side model, not observed broker fill.
_MIN_LABELLED = 10
_STRATEGIES = {"SMC", "VWAPPro", "ORBPro"}


def capture_storage_status(
    session_date: date, *, directory: Path | None = None
) -> dict[str, Any]:
    """Cheap, read-only replay size, writer-drop and queue health evidence."""
    from nifty_scalper_bot.config.paths import get_data_dir

    folder = directory if directory is not None else get_data_dir() / "replay_archive"
    plain = folder / f"{session_date.isoformat()}.jsonl"
    compressed = plain.with_name(plain.name + ".gz")
    tape = plain if plain.is_file() else compressed
    is_compressed = tape == compressed
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
    upstream_missing = 0
    upstream_losses = 0
    max_queue = 0
    max_writer_lag_ms = 0.0
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
            max_queue = max(max_queue, int(obj.get("max_pending") or 0))
            max_writer_lag_ms = max(
                max_writer_lag_ms, float(obj.get("max_writer_lag_ms") or 0)
            )
            initial = obj.get("upstream_start")
            latest = obj.get("upstream_latest")
            if isinstance(initial, dict) and isinstance(latest, dict):
                for key in ("dropped", "coalesced", "legacy_queue_dropped"):
                    if key in initial and key in latest:
                        upstream_losses += max(0, int(latest[key]) - int(initial[key]))
                    else:
                        upstream_missing += 1
                        break
            else:
                upstream_missing += 1
            dropped += max(0, int(obj.get("dropped") or 0))
            failed += max(0, int(obj.get("failed") or 0))
            pending += max(0, int(obj.get("pending") or 0))
        except (OSError, TypeError, ValueError, KeyError):
            continue
    enabled = os.getenv("REPLAY_CAPTURE_ENABLED", "true").lower() == "true"
    used_fraction = None if is_compressed else round(size / ceiling, 4)
    warnings = []
    if enabled and not tape.is_file():
        warnings.append("session_tape_not_recorded")
    if enabled and size and runs == 0:
        warnings.append("writer_integrity_sidecars_missing")
    if dropped or failed:
        warnings.append("writer_integrity_degraded")
    if used_fraction is not None and used_fraction >= 0.80:
        warnings.append("replay_session_capacity_near_limit")
    if pending:
        warnings.append("writer_queue_pending")
    if upstream_losses:
        warnings.append("upstream_drop_or_coalescing")
    if upstream_missing:
        warnings.append("upstream_accounting_unverified")
    return {
        "v": 1,
        "date": session_date.isoformat(),
        "capture_enabled": enabled,
        "tape_present": tape.is_file(),
        "tape_compressed": is_compressed,
        "bytes": size,
        "capacity_bytes": ceiling,
        "capacity_used_fraction": used_fraction,
        "writer_runs": runs,
        "recorded_events": recorded,
        "dropped_events": dropped,
        "failed_events": failed,
        "pending_events": pending,
        "max_writer_queue_pending": max_queue,
        "max_writer_lag_ms": round(max_writer_lag_ms, 3),
        "upstream_lost_or_coalesced": upstream_losses,
        "upstream_accounting_unavailable_runs": upstream_missing,
        "observed_loss_free": bool(
            runs
            and not (dropped or failed or pending or upstream_losses)
            and not upstream_missing
        ),
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


def _cost_model(
    markout: Mapping[str, Any],
    *,
    quantity: int,
    adverse_slippage_bps: float,
) -> dict[str, Any]:
    """Apply the canonical fee model to crossed ask/bid, never double-pay spread."""
    ask = float(markout["entry_ask"])
    slip = adverse_slippage_bps / 10_000
    paid = ask * (1 + slip)

    def exit_value(bid: float) -> dict[str, float]:
        sold = bid * (1 - slip)
        cost = estimate_round_trip_cost(
            entry_price=paid,
            exit_price=sold,
            quantity=quantity,
            half_spread=0.0,
        )
        gross = (sold - paid) * quantity
        return {
            "gross_rupees": round(gross, 2),
            "fees_rupees": round(cost.total, 2),
            "net_rupees": round(gross - cost.total, 2),
        }

    low = ask * (1 + float(markout["mae_markout_pct"]) / 100)
    high = ask * (1 + float(markout["mfe_markout_pct"]) / 100)
    return {
        "terminal": exit_value(float(markout["terminal_bid"])),
        "observed_mae": exit_value(low),
        "observed_mfe": exit_value(high),
        "quantity_units_assumed": quantity,
        "adverse_slippage_bps_per_side": adverse_slippage_bps,
        "spread_crossed_using_actual_ask_bid": True,
        "depth_quantity_verified": False,
        "actual_stop_executed": None,
    }


def _quote_path_dense(
    timestamps: Sequence[float],
    *,
    decision_ts: float,
    horizon_s: int,
    max_gap_s: float = 10.0,
) -> bool:
    """Withhold cost/MAE labels if the interior exact-contract tape has gaps."""
    if not timestamps:
        return False
    prior = bisect.bisect_right(timestamps, decision_ts) - 1
    end = bisect.bisect_right(timestamps, decision_ts + horizon_s)
    if prior < 0 or end - prior < 4:
        return False
    return all(
        0 <= right - left <= max_gap_s
        for left, right in zip(timestamps[prior : end - 1], timestamps[prior + 1 : end])
    )


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

    quote_times = {
        contract: [item[0] for item in books] for contract, books in quotes.items()
    }
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
                timestamps=quote_times.get(contract, ()),
            )
            raw_context = row.get("context")
            context: dict[str, Any] = (
                raw_context if isinstance(raw_context, dict) else {}
            )
            recorded_lot_size = context.get("lot_size")
            verified_lot_size = (
                isinstance(recorded_lot_size, int)
                and not isinstance(recorded_lot_size, bool)
                and 0 < recorded_lot_size < 10000
            )
            quantity = (
                recorded_lot_size
                if isinstance(recorded_lot_size, int) and verified_lot_size
                else _MODELED_QUANTITY_DEFAULT
            )
            selected_side_key = (
                "selected_ce" if contract.endswith("CE") else "selected_pe"
            )
            selected = str(context.get(selected_side_key) or "")
            if selected and normalize_symbol(selected) != contract:
                observation, status = None, "contract_not_selected_at_decision"
            continuous = bool(
                observation is not None
                and _quote_path_dense(
                    quote_times.get(contract, ()),
                    decision_ts=decision_ts,
                    horizon_s=horizon,
                )
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
                "cost_adjusted": (
                    _cost_model(
                        observation,
                        quantity=quantity,
                        adverse_slippage_bps=_MODELED_ADVERSE_SLIPPAGE_BPS,
                    )
                    if observation is not None and continuous
                    else None
                ),
                "cost_label_status": (
                    "modelled_not_filled"
                    if continuous
                    else "missing_or_gapped_quote_path"
                ),
                "cost_quantity_provenance": (
                    "recorded_decision_lot_size"
                    if verified_lot_size
                    else "unverified_default_nifty_lot_assumption"
                ),
                "decision_context": context,
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
        net_labelled = [
            float(item["cost_adjusted"]["terminal"]["net_rupees"])
            for item in items
            if isinstance(item.get("cost_adjusted"), dict)
        ]
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
                "cost_adjusted_labelled": len(net_labelled),
                "mean_modelled_net_rupees": (
                    round(statistics.mean(net_labelled), 2)
                    if len(net_labelled) >= _MIN_LABELLED
                    else None
                ),
                "actual_stop_outcomes_verified": False,
                "minimum_labelled": _MIN_LABELLED,
                "independent_structural_setup": False,
                "eligible_for_filter_change": False,
            }
        )
    return outcomes, strata


def _attach_structural_costs(
    outcomes: list[dict[str, Any]],
    quotes: Mapping[str, list[tuple[float, float, float]]],
) -> None:
    """Cost-label original setup decisions only if their full book path is dense."""
    quote_times = {
        contract: [item[0] for item in books] for contract, books in quotes.items()
    }
    for outcome in outcomes:
        markout = outcome.get("markout")
        at = outcome.get("decision_timestamp")
        try:
            if not isinstance(at, (int, float)) or isinstance(at, bool):
                raise ValueError("invalid_original_decision_timestamp")
            decision_ts = float(at)
            horizon = int(outcome["horizon_seconds"])
            contract = str(outcome["contract"])
        except (ValueError, TypeError, KeyError):
            outcome["cost_adjusted"] = None
            outcome["cost_label_status"] = "invalid_original_decision"
            continue
        supported = isinstance(markout, Mapping) and _quote_path_dense(
            quote_times.get(contract, ()),
            decision_ts=decision_ts,
            horizon_s=horizon,
        )
        if isinstance(markout, Mapping) and supported:
            outcome["cost_adjusted"] = _cost_model(
                markout,
                quantity=_MODELED_QUANTITY_DEFAULT,
                adverse_slippage_bps=_MODELED_ADVERSE_SLIPPAGE_BPS,
            )
        else:
            outcome["cost_adjusted"] = None
        outcome["cost_label_status"] = (
            "modelled_not_filled" if supported else "missing_or_gapped_quote_path"
        )
        outcome["cost_quantity_provenance"] = "unverified_default_nifty_lot_assumption"


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
    if not replay_path.is_file() and replay_path.name.endswith(".jsonl"):
        compressed_path = replay_path.with_name(replay_path.name + ".gz")
        if compressed_path.is_file():
            replay_path = compressed_path
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
        "cost_model_provenance": {
            "one_lot_default_units": _MODELED_QUANTITY_DEFAULT,
            "default_lot_size_is_assumption": True,
            "adverse_slippage_bps_per_side": _MODELED_ADVERSE_SLIPPAGE_BPS,
            "implementation": "risk.cost_model.estimate_round_trip_cost",
            "fills_and_protective_stops_verified": False,
        },
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
                _attach_structural_costs(events, quotes)
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
    if isinstance(report.get("capture_fidelity"), Mapping):
        lines.append("REPLAY_SESSION_FIDELITY " + encoded(report["capture_fidelity"]))
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
    """Run post-close markouts and strict capture-fidelity preflight offline."""
    from nifty_scalper_bot.backtesting.replay_fidelity import audit_session
    from nifty_scalper_bot.config.paths import get_data_dir

    data_dir = get_data_dir()
    path = data_dir / "replay_archive" / f"{session_date}.jsonl"
    if not path.is_file():
        compressed = path.with_name(path.name + ".gz")
        if compressed.is_file():
            path = compressed
    output = data_dir / "research" / "markouts" / f"{session_date}.json"
    report = build_session_research_report(
        session_date=session_date,
        journal=journal,
        replay_path=path,
        trade_db=data_dir / "trades.db",
        output_path=output,
    )
    quality_path = data_dir / "research" / "fidelity" / f"{session_date}.json"
    try:
        fidelity = audit_session(path, gap_seconds=3)
        role_coverage = {
            role: {
                "observed_ticks": fields["observed_ticks"],
                "distinct_seconds_with_updates": fields[
                    "distinct_seconds_with_updates"
                ],
                "missing_market_minutes": fields["missing_market_minutes"],
                "missing_market_minute_ranges": fields[
                    "missing_market_minute_ranges"
                ],
                "first_market_tick_ist": fields["first_market_tick_ist"],
                "last_market_tick_ist": fields["last_market_tick_ist"],
                "gaps_over_threshold": fields["gaps_over_threshold"],
            }
            for role, fields in fidelity["roles"].items()
        }
        summary = {
            "status": "audited",
            "archive_integrity_verified": True,
            "observed_selected_context_available": fidelity[
                "observed_selected_context_available"
            ],
            "selected_context_quote_update_coverage": role_coverage,
            "full_session_selected_coverage_proven": False,
            "actual_exchange_stream_completeness_proven": False,
            "replay_sha256": fidelity["capture_sha256"],
        }
    except (OSError, ValueError, TypeError) as exc:
        fidelity = {
            "status": "unavailable",
            "blocker": (
                str(exc)[:110] if isinstance(exc, ValueError) else type(exc).__name__
            ),
            "archive_integrity_verified": False,
            "full_session_selected_coverage_proven": False,
            "actual_exchange_stream_completeness_proven": False,
        }
        summary = dict(fidelity)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = quality_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(fidelity, allow_nan=False) + "\n")
    temporary.replace(quality_path)
    report["capture_fidelity"] = summary
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, allow_nan=False) + "\n")
    temporary.replace(output)
    log_research_report(report)
    return report
