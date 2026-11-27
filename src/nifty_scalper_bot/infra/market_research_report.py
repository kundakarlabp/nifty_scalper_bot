"""Post-session, broker-free analysis of authoritative replay plus journal samples.

Responsibilities: validate archived quote provenance, report canonical
first-observed RangeVWAP futures excursions with 5/10/15-minute futures and
full-lot option bid outcomes stressed at the same default slippage as live,
with frozen local/session target touch and prior-session clock-matched ATR
research, while preserving the existing structural decisions,
regime-stratified minute-sampled rejections, cost/MAE labels and option volume
parity evidence. Range admission cohorts distinguish observed markouts from
risk-verified outcomes and are mirrored in bounded archive telemetry.
Independent rolling-VWAP reclaims are separate from session-VWAP excursions.
Ownership: SQLite journal and the replay archive remain sole evidence owners;
this reader persists an atomic local report and bounded Supabase journal lines.
Safety: No broker orders or network calls, missing evidence fails closed, and
observational quotes cannot authorize changing any live entry or risk policy.
Date-scoped journal decisions must match the replay session's IST calendar date.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from nifty_scalper_bot.config.regime_ontology import regime_label
from nifty_scalper_bot.risk.cost_model import estimate_round_trip_cost
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    RangeVWAPReclaimStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.symbols import normalize_symbol

LOGGER = get_logger(__name__)
_IST = ZoneInfo("Asia/Kolkata")
_HORIZONS = (60, 180, 300, 600, 900)
_MODELED_QUANTITY_DEFAULT = 65  # Research assumption; verify instrument lot.
_MODELED_ADVERSE_SLIPPAGE_BPS = 5.0  # Generic sampling, not broker fill.
_RANGE_VWAP_SLIPPAGE_BPS = RangeVWAPReclaimStrategyConfig().slippage_stress_bps
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
        configured = int(os.getenv("REPLAY_CAPTURE_MAX_SESSION_MB", "1024"))
    except (ValueError, TypeError):
        configured = 1024
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
    effective_capacities: list[int] = []
    capacity_guards: set[str] = set()
    for sidecar in folder.glob("*.integrity.json"):
        try:
            obj = json.loads(sidecar.read_text(encoding="utf-8"))
            counted = int(
                obj.get("written_by_day", {}).get(session_date.isoformat(), 0)
            )
            if counted <= 0:
                continue
            runs += 1
            observed_capacity = obj.get("max_session_bytes")
            if type(observed_capacity) is int and observed_capacity > 0:
                effective_capacities.append(observed_capacity)
            guard = obj.get("capacity_guard")
            if isinstance(guard, str) and guard:
                capacity_guards.add(guard)
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
    # Report the actual narrowest writer ceiling, not the larger requested
    # limit when disk headroom forced a safe fallback.
    requested_ceiling = ceiling
    if effective_capacities:
        ceiling = min(effective_capacities)
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
    if any(guard != "requested" for guard in capacity_guards):
        warnings.append("replay_capacity_disk_guard_fallback")
    return {
        "v": 1,
        "date": session_date.isoformat(),
        "capture_enabled": enabled,
        "tape_present": tape.is_file(),
        "tape_compressed": is_compressed,
        "bytes": size,
        "capacity_bytes": ceiling,
        "requested_capacity_bytes": requested_ceiling,
        "capacity_guard": sorted(capacity_guards),
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


def _local_regime_minutes(
    records: list[dict[str, Any]], *, session_date: date, report_directory: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """First observation per completed future/minute, with prior-day ATR reference."""
    from nifty_scalper_bot.instruments.active_contracts import (
        canonical_nifty_future_symbol,
    )

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    first_seen: dict[tuple[str, str], datetime] = {}
    for row in records:
        try:
            stamp = datetime.fromisoformat(str(row["bar_timestamp"]))
            decision = datetime.fromisoformat(str(row["decision_at"]))
            future = canonical_nifty_future_symbol(row.get("futures_symbol"))
            atr_bps = row.get("futures_atr_bps")
            if (
                future is None
                or stamp.tzinfo is None
                or decision.tzinfo is None
                or stamp.astimezone(_IST).date() != session_date
                or decision.astimezone(_IST).date() != session_date
                or stamp.second != 0
                or stamp.microsecond != 0
                or not 60 <= (decision - stamp).total_seconds() <= 150
                or row.get("source") != "contiguous_session_futures_1m"
                or not isinstance(atr_bps, (int, float))
                or isinstance(atr_bps, bool)
                or not math.isfinite(atr_bps)
                or atr_bps <= 0
            ):
                continue
            stamp = stamp.astimezone(_IST)
            key = (future, stamp.isoformat())
            if key in first_seen and first_seen[key] <= decision:
                continue
            first_seen[key] = decision
            unique[key] = {**row, "clock_minute_ist": stamp.strftime("%H:%M")}
        except (ValueError, TypeError, KeyError, OverflowError):
            continue

    # Existing dated reports are the only historical source. No future day,
    # same-day report, missing minute, or synthetic baseline is eligible.
    references: dict[str, list[float]] = defaultdict(list)
    prior_sessions = 0
    for path in sorted(report_directory.glob("????-??-??.json"), reverse=True):
        if prior_sessions >= 10:
            break
        try:
            prior_date = date.fromisoformat(path.stem)
            if prior_date >= session_date:
                continue
            prior = json.loads(path.read_text(encoding="utf-8"))
            if prior.get("date_ist") != prior_date.isoformat():
                continue
            daily: dict[str, float] = {}
            for row in prior.get("local_regime_minutes", []):
                stamp = datetime.fromisoformat(str(row.get("bar_timestamp")))
                value = row.get("futures_atr_bps")
                if (
                    stamp.tzinfo is not None
                    and stamp.astimezone(_IST).date() == prior_date
                    and row.get("source") == "contiguous_session_futures_1m"
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value > 0
                ):
                    daily.setdefault(stamp.astimezone(_IST).strftime("%H:%M"), value)
            if daily:
                prior_sessions += 1
                for minute, value in daily.items():
                    references[minute].append(value)
        except (OSError, ValueError, TypeError, AttributeError):
            continue

    rows = [unique[key] for key in sorted(unique, key=lambda key: key[1])]
    state_counts: dict[str, int] = defaultdict(int)
    shadow_transition_counts: dict[str, int] = defaultdict(int)
    unknown_local_range = 0
    denied_observations = 0
    confirmed_in_trend = 0
    independent_local_reclaims = 0
    for row in rows:
        matched = references.get(row["clock_minute_ist"], [])
        baseline = statistics.median(matched) if len(matched) >= 5 else None
        row["clock_matched_prior_sessions"] = len(matched)
        row["clock_matched_atr_bps_median"] = baseline
        row["atr_vs_clock_matched_median"] = (
            round(row["futures_atr_bps"] / baseline, 4) if baseline else None
        )
        state = str(row.get("local_range_state") or "UNPROVEN")
        state_counts[state] += 1
        candidate = row.get("shadow_transition_candidate")
        if candidate in ("TREND", "RANGE"):
            shadow_transition_counts[str(candidate)] += 1
        denied_observations += int(row.get("regime_gate_allowed") is False)
        unknown_local_range += int(
            state == "RANGE" and str(row.get("regime_raw") or "").upper() == "UNKNOWN"
        )
        confirmed_in_trend += int(
            state == "RANGE"
            and str(row.get("regime") or "").upper() == "TREND"
            and str(row.get("regime_raw") or "").upper() == "TREND"
            and row.get("regime_gate_allowed") is True
            and row.get("regime_transition_pending") is not True
        )
        independent_local_reclaims += int(
            row.get("local_reclaim_side") in {"CE", "PE"}
            and row.get("session_vwap_setup_qualified") is not True
        )
    return rows, {
        "unit": "distinct_completed_futures_minute_not_trade",
        "observed_minutes": len(rows),
        "state_counts": dict(state_counts),
        "shadow_transition_counts": dict(shadow_transition_counts),
        "canonical_denied_observed_minutes": denied_observations,
        "unknown_with_confirmed_local_range_minutes": unknown_local_range,
        "clock_matched_calibration_status": (
            "reference_available_research_only"
            if any(row["atr_vs_clock_matched_median"] is not None for row in rows)
            else "insufficient_clock_matched_evidence"
        ),
        "confirmed_local_range_in_trend_minutes": confirmed_in_trend,
        "local_reclaim_without_session_setup_minutes": independent_local_reclaims,
        "clock_matched_reference_sessions": prior_sessions,
        "clock_matched_minimum_sessions": 5,
        "clock_matched_labelled_minutes": sum(
            row["atr_vs_clock_matched_median"] is not None for row in rows
        ),
        "reference": "prior_dated_reports_same_ist_minute_atr_bps_median",
        "live_policy_changed": False,
    }


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
    latest_unaligned = sum(
        row.get("latest_sources_aligned") is False for row in unique.values()
    )
    latest_aligned_matches = sum(
        row.get("status") == "same_minute_volume_match"
        and row.get("latest_sources_aligned") is True
        for row in unique.values()
    )
    return {
        "unit": "distinct_selected_contract_completed_minute",
        "observations": len(unique),
        "positive_volume_matches": matches,
        "latest_source_unaligned_observations": latest_unaligned,
        "latest_aligned_positive_matches": latest_aligned_matches,
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
    groups: dict[tuple[str, str, int, str, str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
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
            regime = (
                regime_label(context["regime"])
                if context.get("regime") is not None
                else "UNRECORDED"
            )
            raw_regime = (
                regime_label(context["regime_raw"])
                if context.get("regime_raw") is not None
                else "UNRECORDED"
            )
            gate = context.get("regime_gate_allowed")
            gate_state = (
                "allowed"
                if gate is True
                else "blocked" if gate is False else "unrecorded"
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
                "regime": regime,
                "regime_raw": raw_regime,
                "regime_gate_state": gate_state,
                "sampling": "first_no_vote_per_contract_strategy_utc_minute",
                "independent_structural_setup": False,
            }
            outcomes.append(result)
            groups[(strategy, reason, horizon, regime, raw_regime, gate_state)].append(
                result
            )

    strata: list[dict[str, Any]] = []
    for (strategy, reason, horizon, regime, raw_regime, gate_state), items in sorted(
        groups.items()
    ):
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
                "regime": regime,
                "regime_raw": raw_regime,
                "regime_gate_state": gate_state,
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


def _session_decision_rows(
    rows: Sequence[Mapping[str, Any]], session_date: date
) -> list[Mapping[str, Any]]:
    """Scope canonical decision rows to one IST date before quote-time matching.

    The canonical trade journal loader is not date-filtered. Mixing its old
    decisions with one day's immutable option tape produces false missing-quote
    outcomes even when that day's own quote stream has valid coverage.
    """
    start = datetime.combine(session_date, time.min, tzinfo=_IST).timestamp()
    end = datetime.combine(
        session_date + timedelta(days=1), time.min, tzinfo=_IST
    ).timestamp()
    scoped = []
    for row in rows:
        raw = row.get("timestamp")
        if raw is None or isinstance(raw, bool):
            continue
        try:
            observed = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(observed) and start <= observed < end:
            scoped.append(row)
    return scoped


def _range_excursion_markouts(
    excursions: Sequence[Mapping[str, Any]],
    futures: Mapping[str, Sequence[tuple[float, float, str]]],
    books: Mapping[str, Sequence[tuple[float, float, float, int, int, str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Price the same *first-seen* setup; keep thin/unverified paths unlabelled."""
    from scripts.reporting.rejected_setup_forward_markout import (
        observed_futures_markout,
        observed_markout,
    )

    outcomes: list[dict[str, Any]] = []
    for row in excursions:
        symbol = normalize_symbol(str(row["symbol"]))
        future = str(row["futures_symbol"])
        when = float(row["decision_timestamp"])
        quote = row.get("quote")
        evidence = quote if isinstance(quote, Mapping) else {}
        raw_quantity = evidence.get("lot_size")
        quantity: int | None = (
            raw_quantity
            if isinstance(raw_quantity, int)
            and not isinstance(raw_quantity, bool)
            and 0 < raw_quantity < 10000
            else None
        )
        lot_known = quantity is not None
        original_future = futures.get(future, ())
        future_times = [tick[0] for tick in original_future]
        previous = bisect.bisect_right(future_times, when) - 1
        # A restart or selected-contract rotation is not one continuous tape.
        phase = (
            original_future[previous][2]
            if previous >= 0 and 0 <= when - future_times[previous] <= 3
            else None
        )
        scoped_future = [
            (timestamp, price)
            for timestamp, price, scope in original_future
            if phase is not None and scope == phase
        ]
        full_lot = (
            [
                (timestamp, bid, ask)
                for timestamp, bid, ask, bid_qty, ask_qty, scope in books.get(
                    symbol, ()
                )
                if scope == phase and bid_qty >= quantity and ask_qty >= quantity
            ]
            if phase is not None and quantity is not None
            else []
        )
        times = [item[0] for item in full_lot]
        stop_risk = row.get("modeled_stop_risk")
        net_rr = row.get("modeled_net_rr")
        risk_verified = (
            isinstance(stop_risk, (float, int))
            and not isinstance(stop_risk, bool)
            and isinstance(net_rr, (float, int))
            and not isinstance(net_rr, bool)
            and math.isfinite(stop_risk)
            and math.isfinite(net_rr)
            and 0 < stop_risk <= 773
            and net_rr >= 1.5
        )
        first_rejection = str(row.get("first_rejection_reason") or "")
        shadow_unknown_reclaim = (
            str(row.get("regime_raw") or "").upper() == "UNKNOWN"
            and row.get("regime_gate_allowed") is False
            and row.get("shadow_transition_candidate") == "RANGE"
            and row.get("local_range_state") == "RANGE"
            and row.get("local_reclaim_side") == row.get("side")
            and first_rejection == "canonical_regime_gate_block"
        )
        for horizon in (300, 600, 900):
            fut, fut_status = observed_futures_markout(
                scoped_future,
                decision_ts=when,
                horizon_s=horizon,
                side=str(row["side"]),
            )
            option, option_status = (
                observed_markout(
                    full_lot,
                    decision_ts=when,
                    horizon_s=horizon,
                    timestamps=times,
                )
                if full_lot and not row.get("contract_identity_conflicted")
                else (
                    None,
                    "rotated_contract_or_missing_full_lot_verified_book",
                )
            )
            dense = option is not None and _quote_path_dense(
                times, decision_ts=when, horizon_s=horizon
            )
            if not dense:
                option = None
                if option_status == "observed_bid_path_not_broker_fill":
                    option_status = "option_full_lot_book_path_gapped"
            costed = (
                _cost_model(
                    option,
                    quantity=quantity,
                    adverse_slippage_bps=_RANGE_VWAP_SLIPPAGE_BPS,
                )
                if option is not None and quantity is not None
                else None
            )
            # Compare frozen VWAP targets against the same observed futures
            # path. A touch is not an option fill or a stop-adjusted outcome.
            target_reach: dict[str, Any] = {}
            direction = 1 if row["side"] == "CE" else -1
            initial = float(fut["entry_futures_price"]) if fut is not None else None
            for name, center in (
                ("session_vwap", row.get("session_vwap")),
                ("local_vwap", row.get("rolling_center")),
            ):
                distance: float | None = None
                touch: float | None = None
                if (
                    initial is not None
                    and isinstance(center, (int, float))
                    and not isinstance(center, bool)
                    and math.isfinite(center)
                    and center > 0
                ):
                    target = float(center)
                    distance = direction * (target - initial)
                    if distance > 0:
                        touch = next(
                            (
                                timestamp
                                for timestamp, price in scoped_future
                                if when < timestamp <= when + horizon
                                and direction * (price - target) >= 0
                            ),
                            None,
                        )
                target_reach[name] = {
                    "forward_distance_points": (
                        round(distance, 4) if distance is not None else None
                    ),
                    "touched": (
                        touch is not None
                        if distance is not None and distance > 0
                        else None
                    ),
                    "first_touch_seconds": (
                        round(touch - when, 3) if touch is not None else None
                    ),
                    "option_fill_verified": False,
                }
            outcomes.append(
                {
                    "excursion_id": row["excursion_id"],
                    "excursion_basis": row.get("excursion_basis", "session_vwap"),
                    "frozen_vwap_target_reach": target_reach,
                    "first_seen_ts": when,
                    "futures_contract": future,
                    "option_contract": symbol,
                    "side": row["side"],
                    "replay_epoch_phase_verified": phase is not None,
                    "regime": row.get("regime"),
                    "center_gap_atr": row.get("center_gap_atr"),
                    "volatility_ratio": row.get("volatility_ratio"),
                    "first_rejection_reason": first_rejection,
                    "all_reasons": row.get("observed_decision_reasons", []),
                    "baseline_signal_emitted_at_first_decision": row[
                        "same_decision_signal_emitted"
                    ],
                    "counterfactual_center_gate_candidate": (
                        first_rejection == "session_rolling_center_disagree"
                    ),
                    "counterfactual_range_width_candidate": (
                        first_rejection == "range_width_unproven"
                    ),
                    "shadow_local_range_in_trend_candidate": (
                        row.get("local_range_session_vwap_setup") is True
                        and first_rejection == "strategy_regime_ineligible"
                    ),
                    "shadow_confirmed_local_range_in_trend_candidate": (
                        row.get("local_range_confirmed_in_trend") is True
                        and first_rejection == "strategy_regime_ineligible"
                    ),
                    "shadow_rolling_vwap_only_candidate": (
                        row.get("excursion_basis") == "rolling_vwap"
                    ),
                    "shadow_unknown_local_range_candidate": shadow_unknown_reclaim,
                    "shadow_unknown_session_reclaim_candidate": (
                        shadow_unknown_reclaim
                        and row.get("structural_qualified") is True
                    ),
                    "quantity_units_from_observed_contract": (
                        quantity if lot_known else None
                    ),
                    "modeled_cash_stop_risk": stop_risk,
                    "modeled_net_rr": net_rr,
                    "same_decision_risk_773_and_net_rr_verified": risk_verified,
                    "horizon_seconds": horizon,
                    "futures_markout": fut,
                    "futures_status": fut_status,
                    "option_full_lot_markout": option,
                    "option_status": option_status,
                    "option_cost_adjusted": costed,
                    "adverse_slippage_bps_per_side": _RANGE_VWAP_SLIPPAGE_BPS,
                    "observed_book_and_futures_complete": (
                        fut is not None and costed is not None
                    ),
                    "quote_model_not_broker_fill": True,
                }
            )

    comparisons: list[dict[str, Any]] = []
    for policy, predicate in (
        ("current_admission", "baseline_signal_emitted_at_first_decision"),
        ("ignore_center_disagreement_only", "counterfactual_center_gate_candidate"),
        ("ignore_range_width_only", "counterfactual_range_width_candidate"),
        ("shadow_local_range_in_trend", "shadow_local_range_in_trend_candidate"),
        (
            "shadow_confirmed_local_range_in_trend",
            "shadow_confirmed_local_range_in_trend_candidate",
        ),
        ("shadow_unknown_local_range", "shadow_unknown_local_range_candidate"),
        ("shadow_rolling_vwap_only", "shadow_rolling_vwap_only_candidate"),
        (
            "shadow_unknown_session_reclaim",
            "shadow_unknown_session_reclaim_candidate",
        ),
    ):
        for horizon in (300, 600, 900):
            group = [
                row
                for row in outcomes
                if row["horizon_seconds"] == horizon and row[predicate]
            ]
            feasible = [
                row
                for row in group
                if row["same_decision_risk_773_and_net_rr_verified"]
                and row["observed_book_and_futures_complete"]
            ]
            observed_net = [
                float(row["option_cost_adjusted"]["terminal"]["net_rupees"])
                for row in group
                if row["observed_book_and_futures_complete"]
            ]
            comparisons.append(
                {
                    "policy": policy,
                    "horizon_seconds": horizon,
                    "adverse_slippage_bps_per_side": _RANGE_VWAP_SLIPPAGE_BPS,
                    "unique_first_seen_excursions": len(group),
                    "full_lot_bid_futures_labelled": sum(
                        row["observed_book_and_futures_complete"] for row in group
                    ),
                    "same_decision_cash_risk_unverified": sum(
                        not row["same_decision_risk_773_and_net_rr_verified"]
                        for row in group
                    ),
                    "verified_risk_and_observed_horizon": len(feasible),
                    "observed_full_lot_net_rupees_mean": (
                        round(statistics.mean(observed_net), 2)
                        if len(observed_net) >= _MIN_LABELLED
                        else None
                    ),
                    "observed_full_lot_net_positive_fraction": (
                        round(
                            sum(value > 0 for value in observed_net)
                            / len(observed_net),
                            4,
                        )
                        if len(observed_net) >= _MIN_LABELLED
                        else None
                    ),
                    "observed_markout_not_risk_verified": True,
                    "futures_target_touch_counts": {
                        name: {
                            "forward_target_observed": sum(
                                item["frozen_vwap_target_reach"][name]["touched"]
                                is not None
                                for item in group
                            ),
                            "touched": sum(
                                item["frozen_vwap_target_reach"][name]["touched"]
                                is True
                                for item in group
                            ),
                        }
                        for name in ("session_vwap", "local_vwap")
                    },
                    "modeled_net_rupees_mean": (
                        round(
                            statistics.mean(
                                float(
                                    row["option_cost_adjusted"]["terminal"][
                                        "net_rupees"
                                    ]
                                )
                                for row in feasible
                            ),
                            2,
                        )
                        if len(feasible) >= _MIN_LABELLED
                        else None
                    ),
                    "broker_fill_and_stop_simulated": False,
                    "comparison_not_causal_or_promotable": True,
                }
            )
    return outcomes, comparisons


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
        load_excursion_quote_paths,
        summarize_groups,
    )

    from nifty_scalper_bot.backtesting.setup_opportunity_analysis import (
        canonicalize_range_excursions,
        canonicalize_setup_opportunities,
    )
    from nifty_scalper_bot.journal.trade_ledger import load_candidate_decision_rows

    parity = _parity_summary(
        _parse_records(journal, "OPTION_COMPLETED_VOLUME_PARITY", session_date)
    )
    rejection_records = _parse_records(
        journal, "RESEARCH_STRATEGY_REJECTION", session_date
    )
    range_excursion_observed = _parse_records(
        journal, "RANGE_VWAP_EXCURSION_OBSERVED", session_date
    )
    range_excursion_decisions = _parse_records(
        journal, "RANGE_VWAP_EXCURSION_DECISION", session_date
    )
    local_minutes, local_summary = _local_regime_minutes(
        _parse_records(journal, "RANGE_VWAP_LOCAL_REGIME", session_date),
        session_date=session_date,
        report_directory=output_path.parent,
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
        "range_excursion_opportunities": [],
        "local_regime_minutes": local_minutes,
        "local_regime_summary": local_summary,
        "range_excursion_horizons": [],
        "range_admission_comparisons": [],
        "range_excursion_observed_events": len(range_excursion_observed),
        "range_excursion_duplicate_observations": 0,
        "replay_sha256": None,
        "eligible_for_entry_filter_change": False,
        "cost_model_provenance": {
            "one_lot_default_units": _MODELED_QUANTITY_DEFAULT,
            "default_lot_size_is_assumption": True,
            "adverse_slippage_bps_per_side": _MODELED_ADVERSE_SLIPPAGE_BPS,
            "range_vwap_adverse_slippage_bps_per_side": _RANGE_VWAP_SLIPPAGE_BPS,
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
        futures_tape, full_lot_books = load_excursion_quote_paths(replay_path)
        opportunities = canonicalize_range_excursions(
            range_excursion_observed, range_excursion_decisions
        )
        range_outcomes, comparisons = _range_excursion_markouts(
            opportunities, futures_tape, full_lot_books
        )
        report["range_excursion_opportunities"] = opportunities
        report["range_excursion_horizons"] = range_outcomes
        report["range_admission_comparisons"] = comparisons
        report["range_excursion_duplicate_observations"] = sum(
            row["duplicate_restart_observations"] for row in opportunities
        )
        if not quote_count:
            report["blocker"] = "no_tradable_depth_ws_quotes"
        else:
            sampled, strata = _sampled_markouts(rejection_records, quotes)
            report["sampled_observations"] = sampled
            report["sampled_strata"] = strata
            if trade_db.is_file():
                raw_decisions = load_candidate_decision_rows(trade_db)
                day_decisions = _session_decision_rows(raw_decisions, session_date)
                report["structural_decision_rows_scoped"] = len(day_decisions)
                report["structural_decision_rows_excluded_non_session"] = len(
                    raw_decisions
                ) - len(day_decisions)
                structural = canonicalize_setup_opportunities(day_decisions)
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
        "range_excursions": len(report["range_excursion_opportunities"]),
        "range_excursion_duplicate_observations": report[
            "range_excursion_duplicate_observations"
        ],
        "range_option_futures_horizons_fully_labelled": sum(
            row["observed_book_and_futures_complete"]
            for row in report["range_excursion_horizons"]
        ),
        "structural_decision_rows_scoped": report.get(
            "structural_decision_rows_scoped"
        ),
        "structural_decision_rows_excluded_non_session": report.get(
            "structural_decision_rows_excluded_non_session"
        ),
        "sampled_observations": len(report["sampled_observations"]),
        "volume_parity": report["volume_parity"],
        "local_regime_summary": report.get("local_regime_summary"),
    }

    def encoded(row: Mapping[str, Any]) -> str:
        return json.dumps(row, separators=(",", ":"), allow_nan=False)

    lines = ["RESEARCH_SESSION_EVIDENCE " + encoded(base)]
    if isinstance(report.get("capture_fidelity"), Mapping):
        lines.append("REPLAY_SESSION_FIDELITY " + encoded(report["capture_fidelity"]))
    for kind, strata in (
        ("minute_sample", report["sampled_strata"]),
        ("structural_setup", report["structural_strata"]),
        (
            "range_admission",
            [
                row
                for row in report.get("range_admission_comparisons", [])
                if row["unique_first_seen_excursions"] > 0
            ],
        ),
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
        # Both analyses must observe identical immutable tape bytes. The
        # recorder may still be appending after the nominal settlement window;
        # never certify quote markouts from different archive revisions.
        if (
            report.get("replay_sha256") is not None
            and report["replay_sha256"] != fidelity["capture_sha256"]
        ):
            raise ValueError("replay_archive_changed_between_passes")
        role_coverage = {
            role: {
                "observed_ticks": fields["observed_ticks"],
                "distinct_seconds_with_updates": fields[
                    "distinct_seconds_with_updates"
                ],
                "missing_market_minutes": fields["missing_market_minutes"],
                "missing_market_minute_ranges": fields["missing_market_minute_ranges"],
                "observed_ws_two_sided_quote_minutes": fields.get(
                    "observed_ws_two_sided_quote_minutes"
                ),
                "missing_ws_two_sided_quote_minutes": fields.get(
                    "missing_ws_two_sided_quote_minutes"
                ),
                "missing_ws_two_sided_quote_minute_ranges": fields.get(
                    "missing_ws_two_sided_quote_minute_ranges"
                ),
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
        if isinstance(exc, ValueError) and str(exc) == (
            "replay_archive_changed_between_passes"
        ):
            report["status"] = "unavailable"
            report["blocker"] = str(exc)
            report["sampled_observations"] = []
            report["sampled_strata"] = []
            report["structural_outcomes"] = []
            report["structural_strata"] = []
            report["range_excursion_horizons"] = []
            report["range_admission_comparisons"] = []
            report["eligible_for_entry_filter_change"] = False
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
