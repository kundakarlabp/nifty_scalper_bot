#!/usr/bin/env python3
"""Purpose: Run bounded, broker-free historical comparisons of VWAP, ORB and reversal.

Responsibilities: Use pre-existing, checksum-prepared PUBLIC monthly-option
sessions; select a small date-stratified archive cohort independent of outcomes;
replay the existing strategies, retain frozen matched-event IDs, inferred-contract
labels, source provenance, rejection evidence and unresolved exit statuses.
Boundary: SHADOW-only; monthly OHLC and inferred contract labels are not
historically verified instruments, direction provenance or executable quotes.
Safety: Never submits orders, changes live configuration or promotes parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from scripts.audit_directional_strategies import run_directional_research_bundle

from nifty_scalper_bot.backtesting.strategy_research import summarize

PHASES = {
    "2017": "development",
    "2018": "development",
    "2019": "validation",
    "2020": "retrospective_check",
}
REQUIRED_KINDS = ("EQ", "FUT", "CE", "PE")
SLIPPAGE_BPS = (10.0, 25.0, 50.0)


def select_sessions(manifest: Mapping[str, Any], per_quarter: int = 2) -> list[str]:
    """Preselect centrally spaced eligible dates in each calendar quarter.

    This is a bounded *feasibility pilot*, not a statistically representative
    sample or a new untouched OOS. Missing archived data are never imputed.
    """
    if not 1 <= per_quarter <= 5:
        raise ValueError("directional_pilot_per_quarter_invalid")
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    seen: set[str] = set()
    for row in manifest.get("sessions", ()):
        day = str(row.get("day", ""))
        if day in seen:
            raise ValueError("directional_pilot_duplicate_session")
        seen.add(day)
        if len(day) != 10 or day[:4] not in PHASES:
            continue
        bars = row.get("bars") or {}
        if not all(int(bars.get(key, 0)) >= 45 for key in REQUIRED_KINDS):
            continue
        try:
            quarter = (int(day[5:7]) - 1) // 3 + 1
        except ValueError as exc:
            raise ValueError("directional_pilot_date_invalid") from exc
        if not 1 <= quarter <= 4:
            raise ValueError("directional_pilot_date_invalid")
        groups[(day[:4], quarter)].append(day)
    selected: list[str] = []
    for key, days in sorted(groups.items()):
        days = sorted(days)
        count = min(per_quarter, len(days))
        indices = sorted(
            {((2 * i + 1) * len(days)) // (2 * count) for i in range(count)}
        )
        selected.extend(days[i] for i in indices)
    return selected


def _compact_candidates(study: Mapping[str, Any], kind: str, phase: str, day: str):
    rows = []
    for candidate in study.get("candidates", ()):
        for scenario in candidate.get("scenarios", ()):
            bps = float(scenario["slippage_bps_per_side"])
            if bps not in SLIPPAGE_BPS:
                raise ValueError("directional_pilot_slippage_drift")
            trades = list(scenario.get("trades", ()))
            unresolved = list(scenario.get("unresolved_trades", ()))
            measured = summarize(trades)
            rows.append(
                {
                    "study": kind,
                    "phase": phase,
                    "day": day,
                    "candidate": candidate["name"],
                    "slippage_bps": bps,
                    "resolved": len(trades),
                    "unresolved": len(unresolved),
                    "net_pnl": measured["net_pnl"],
                    "fees": measured["fees"],
                    "no_vote_reasons": scenario.get("no_vote_reasons", {}),
                    "entry_funnel": scenario.get("entry_funnel", {}),
                }
            )
    return rows


def _compact_reversal(study: Mapping[str, Any], phase: str, day: str):
    rows = []
    for scenario in study.get("scenarios", ()):
        bps = float(scenario["slippage_bps_per_side"])
        events = scenario["events"]
        for candidate in ("SMC", "FailedBreakoutResearch"):
            outcomes = [row["outcomes"][candidate] for row in events]
            resolved = [row for row in outcomes if row["resolved_fill"]]
            unresolved = sum(bool(row["unresolved_exit"]) for row in outcomes)
            if any(row["net_pnl"] is None for row in resolved):
                raise ValueError("directional_pilot_resolved_without_pnl")
            rows.append(
                {
                    "study": "same_event_reversal",
                    "phase": phase,
                    "day": day,
                    "candidate": candidate,
                    "slippage_bps": bps,
                    "resolved": len(resolved),
                    "unresolved": unresolved,
                    "net_pnl": sum(float(row["net_pnl"]) for row in resolved),
                    "fees": sum(float(row["costs"]) for row in resolved),
                    "matched_option_event_count": len(events),
                    "same_minute_both_filled": scenario[
                        "both_resolved_same_entry_count"
                    ],
                    "reversal_only_positive_count": scenario[
                        "reversal_only_positive_resolved_fill_count"
                    ],
                }
            )
    return rows


def summarize_pilot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    for item in rows:
        key = (
            item["study"],
            item["phase"],
            item["candidate"],
            float(item["slippage_bps"]),
        )
        entry = grouped.setdefault(
            key,
            {
                "study": key[0],
                "phase": key[1],
                "candidate": key[2],
                "slippage_bps_per_side": key[3],
                "sessions": set(),
                "resolved": 0,
                "unresolved": 0,
                "net_pnl": 0.0,
                "fees": 0.0,
                "matched_option_event_count": 0,
                "entry_funnel": {},
                "no_vote_reasons": {},
            },
        )
        entry["sessions"].add(item["day"])
        for field in ("resolved", "unresolved", "matched_option_event_count"):
            entry[field] += item.get(field, 0)
        for field in ("net_pnl", "fees"):
            entry[field] += float(item[field])
        for field in ("entry_funnel", "no_vote_reasons"):
            counters = item.get(field) or {}
            if not isinstance(counters, Mapping):
                raise ValueError("directional_pilot_diagnostics_invalid")
            for reason, count in counters.items():
                if (
                    not isinstance(reason, str)
                    or not reason
                    or not isinstance(count, int)
                    or isinstance(count, bool)
                    or count < 0
                ):
                    raise ValueError("directional_pilot_diagnostics_invalid")
                entry[field][reason] = entry[field].get(reason, 0) + count
    result = []
    for value in grouped.values():
        value["session_count"] = len(value.pop("sessions"))
        value["mean_net_per_resolved_fill"] = (
            value["net_pnl"] / value["resolved"] if value["resolved"] else None
        )
        # This is a diagnostic stage, not a causal explanation of profitability.
        funnel = value["entry_funnel"]
        if value["resolved"] > 0:
            value["zero_fill_status"] = "RESOLVED_FILLS_PRESENT"
        elif value["unresolved"] > 0:
            value["zero_fill_status"] = "UNRESOLVED_EXIT_PRESENT"
        elif value["study"] == "same_event_reversal":
            value["zero_fill_status"] = (
                "NO_MATCHED_EVENTS"
                if value["matched_option_event_count"] == 0
                else "MATCHED_EVENTS_WITHOUT_RESOLVED_FILL"
            )
        elif funnel.get("evaluations", 0) == 0:
            value["zero_fill_status"] = "NO_RECORDED_EVALUATIONS"
        elif funnel.get("signals_emitted", 0) == 0:
            value["zero_fill_status"] = "EVALUATIONS_WITHOUT_SIGNALS"
        else:
            value["zero_fill_status"] = "SIGNALS_WITHOUT_RESOLVED_FILL"
        value["attribution_valid"] = False
        value["promotion_eligible"] = False
        result.append(value)
    return sorted(
        result,
        key=lambda row: (
            row["study"],
            row["phase"],
            row["candidate"],
            row["slippage_bps_per_side"],
        ),
    )


def _matched_reversal_event_roster(
    study: Mapping[str, Any], *, session_path: Path, phase: str
) -> list[dict[str, Any]]:
    """Retain the original matched cohort, never retroactively choose a strike.

    The public archive's expiry, lot and continuous-future labels are inferred,
    not contemporaneous exchange-master or quote-depth proof.
    """
    scenarios = study["failed_breakout_vs_smc"]["scenarios"]
    if not scenarios:
        raise ValueError("directional_pilot_reversal_scenarios_missing")
    events = scenarios[0]["events"]
    index = {}
    for kind in ("CE", "PE"):
        payload = json.loads((session_path / "candles" / f"{kind}.json").read_text())
        index[payload["symbol"]] = dict(payload["instrument"])
    retained = []
    seen = set()
    for row in events:
        symbol, side = row["symbol"], row["side"]
        instrument = index.get(symbol)
        if instrument is None or instrument.get("instrument_type") != side:
            raise ValueError("directional_pilot_reversal_instrument_mismatch")
        key = "|".join(
            (
                row["event_bar_start"],
                row["earliest_entry"],
                row["future"],
                symbol,
                side,
            )
        )
        identity = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if identity in seen:
            raise ValueError("directional_pilot_reversal_event_duplicate")
        seen.add(identity)
        retained.append(
            {
                "event_id": identity,
                "phase": phase,
                "event_bar_start": row["event_bar_start"],
                "earliest_entry": row["earliest_entry"],
                "future": row["future"],
                "symbol": symbol,
                "side": side,
                "archive_instrument": {
                    "expiry": instrument.get("expiry"),
                    "strike": instrument.get("strike"),
                    "lot_size": instrument.get("lot_size"),
                    "token": instrument.get("instrument_token"),
                },
                "archive_instrument_basis": "INFERRED_FROM_MONTHLY_CANDLES",
                "historical_master_verified": False,
                "source_clock_verified": False,
                "historical_futures_roll_verified": False,
                "observed_depth_available": False,
                "live_promotion_eligible": False,
            }
        )
    return retained


def run(root: Path, *, per_quarter: int = 2) -> dict[str, Any]:
    if os.getenv("EXECUTION_MODE", "").upper() != "SHADOW":
        raise ValueError("directional_pilot_requires_shadow")
    manifest_file = root / "data_manifest.json"
    raw = manifest_file.read_bytes()
    manifest = json.loads(raw)
    selected = select_sessions(manifest, per_quarter)
    if not selected:
        raise ValueError("directional_pilot_no_eligible_sessions")
    observations = []
    matched_reversal_events = []
    failures = []
    digest_by_day = {}
    for index, day in enumerate(selected, start=1):
        try:
            session_path = root / "sessions" / day
            if not all(
                (session_path / "candles" / f"{kind}.json").is_file()
                for kind in REQUIRED_KINDS
            ):
                raise ValueError("directional_pilot_missing_session_candles")
            study = run_directional_research_bundle(session_path)
            if study["promotion_eligible"] or study["live_equivalent"]:
                raise ValueError("directional_pilot_live_boundary_violation")
            digest_by_day[day] = study["data_sha256"]
            phase = PHASES[day[:4]]
            observations.extend(
                _compact_candidates(study[name], name, phase, day)
                for name in ("vwap", "orb")
            )
            observations.append(
                _compact_reversal(study["failed_breakout_vs_smc"], phase, day)
            )
            matched_reversal_events.extend(
                _matched_reversal_event_roster(
                    study, session_path=session_path, phase=phase
                )
            )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failures.append(
                {"day": day, "type": type(exc).__name__, "reason": str(exc)[:160]}
            )
        print(
            f"Directional archive pilot: {index}/{len(selected)} sessions", flush=True
        )
    flat = [row for sublist in observations for row in sublist]
    if not flat:
        raise ValueError("directional_pilot_all_sessions_unusable")
    return {
        "scope": "public_monthly_options_bounded_directional_pilot",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "session_archive_digests": digest_by_day,
        "preselected_sessions": selected,
        "successful_sessions": sorted(digest_by_day),
        "failures": failures,
        "per_quarter": per_quarter,
        "results": summarize_pilot(flat),
        "matched_reversal_events": matched_reversal_events,
        "historical_point_in_time_identity_verified": False,
        "observed_historical_quote_books_present": False,
        "promotion_eligible": False,
        "live_equivalent": False,
        "outperformance_inference_allowed": False,
        "limitations": [
            "Small date-stratified feasibility sample; not an untouched holdout",
            "Monthly fixed-opening ATM option baskets; "
            "not current weekly or rotated contracts",
            "Public pre-2018 bar timestamp conventions and "
            "continuous future identity unverified",
            "OHLC simulated execution lacks historical bid/ask/depth and partial fills",
            "SMC/reversal compared at same next-minute fill; "
            "earlier SMC entries not attributed",
            "Per-session candidate competition is not reconstructed "
            "across a real live portfolio",
            "Failed archive sessions are reported and never silently imputed",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-quarter", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite historical research result")
    result = run(args.study_dir, per_quarter=args.per_quarter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(args.output)


if __name__ == "__main__":
    main()
