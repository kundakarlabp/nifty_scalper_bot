"""Research-only comparison of SMC confirmation evidence in archived journald.

Responsibilities: deduplicate both SMC lifecycle and trigger-vote setup IDs, compare
strict and conservative neutral-tolerant logged confirmation sequences,
reconstruct the original smcv2 structural sweep timestamp at manager approval
and broker acceptance (using explicit timezone-aware event provenance), and
report missing or invalid evidence without fabricating executable opportunities.

Safety boundary: audits *logged* evidence, not the real-time strategy state
machine. Journald may omit ticks, candles, fills, bid/ask and book updates.
Neither an observed pair nor its absence proves a missed profitable entry.
This module cannot alter live strategy policy or place an order.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

IST = timezone(timedelta(hours=5, minutes=30))
TIME_PATTERN = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) IST\]")
FIELD_PATTERN = re.compile(r"(?:^|\s)([a-z][a-z0-9_]*)=([^\s]+)")
# Reset confirmation on direction disagreement, holdoff, incomplete or adverse
# book support. Only absent independent/depth persistence is a neutral proxy.
NEUTRAL_REASONS = {"no_independent_flow_confirmation", "depth_persistence_unconfirmed"}


@dataclass(frozen=True, slots=True)
class Evidence:
    at: datetime
    symbol: str
    quote_version: int
    reason: str
    classification: str


def _fields(line: str) -> dict[str, str]:
    return dict(FIELD_PATTERN.findall(line))


def _when(line: str) -> datetime | None:
    match = TIME_PATTERN.match(line)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)


def _smc_setup_anchor_time(setup_id: str) -> datetime | None:
    """Parse only the original timezone-aware structural anchor from smcv2 IDs."""
    prefix = "smcv2:NFO:"
    if not setup_id.startswith(prefix):
        return None
    pieces = setup_id.split(":", 4)
    if len(pieces) != 5 or pieces[3] not in {"CE", "PE"}:
        return None
    try:
        when = datetime.fromisoformat(pieces[4].replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo is not None and when.utcoffset() is not None else None


def _reference_sweep_age(
    setup_id: str, observed_at: datetime | None
) -> float | None:
    """Retrospective proxy only; does not replay final live signal metadata."""
    sweep_at = _smc_setup_anchor_time(setup_id)
    if sweep_at is None or observed_at is None:
        return None
    return round((observed_at - sweep_at).total_seconds(), 3)


def _float(fields: dict[str, str], name: str, default: float = -1.0) -> float:
    try:
        return float(fields[name])
    except (KeyError, TypeError, ValueError, OverflowError):
        return default


def _evidence(line: str, at: datetime) -> Evidence | None:
    fields = _fields(line)
    symbol = fields.get("symbol", "")
    if (
        not symbol.startswith("NFO:")
        or not symbol.endswith(("CE", "PE"))
        or fields.get("side") != symbol[-2:]
    ):
        return None
    reason = fields.get("block_reason", "unknown")
    version_float = _float(fields, "quote_version", 0)
    version = int(version_float) if math.isfinite(version_float) else 0
    qualifies = (
        fields.get("eligible") == "True"
        and fields.get("aligned") == "True"
        and fields.get("conflict") == "False"
        and fields.get("source") == "temporal_ofi"
        and fields.get("ofi_ready") == "True"
        and fields.get("ofi_sustained") == "True"
        and fields.get("ofi_reversal_holdoff") == "False"
        and reason == "None"
        and _float(fields, "age_s") >= 0
        and _float(fields, "age_s") <= 3
        and _float(fields, "ofi_updates_3s") >= 3
        and _float(fields, "ofi_3s_span") >= 1.5
        and version > 0
    )
    hard = (
        fields.get("eligible") != "True"
        or not 0 <= _float(fields, "age_s") <= 3
        or fields.get("conflict") == "True"
        or fields.get("ofi_reversal_holdoff") == "True"
        or (not qualifies and reason not in NEUTRAL_REASONS)
    )
    classification = "hard_reset" if hard else "qualifying" if qualifies else "neutral"
    return Evidence(at, symbol, version, reason, classification)


def _has_pair(events: Sequence[Evidence], *, allow_one_neutral: bool) -> bool:
    """Compare logged proxies; never pretend to reproduce live manager calls."""
    first: Evidence | None = None
    neutrals = 0
    for item in events:
        if item.classification == "hard_reset":
            first = None
            neutrals = 0
        elif item.classification == "neutral":
            neutrals += 1
            if not allow_one_neutral or neutrals > 1:
                first = None
                neutrals = 0
        else:
            if first is not None:
                age = (item.at - first.at).total_seconds()
                if item.quote_version > first.quote_version and 4 <= age <= 12:
                    return True
            first = item
            neutrals = 0
    return False


def analyze(log_text: str, *, window_seconds: int = 120) -> dict[str, Any]:
    """Deduplicate structural setup IDs; compare strictly logged flow proxies."""
    if window_seconds < 1:
        raise ValueError("window_seconds must be positive")
    setups: dict[str, dict[str, Any]] = {}
    votes: list[tuple[str, str, datetime, int]] = []
    flow: list[Evidence] = []

    def observe_setup(
        setup_id: str, symbol: str, side: str, at: datetime, *, stage: bool
    ) -> None:
        """Count a unique vote-only setup without inventing its missing stage."""
        if not setup_id:
            return
        existing = setups.get(setup_id)
        if existing is not None:
            if (
                existing["side"] != side
                or existing["symbol"] != symbol
                or not symbol.endswith(side)
            ):
                existing["identity_conflict"] = True
            existing["lifecycle_stage_observed"] |= stage
            if at < existing["first_at"]:
                existing["first_at"] = at
                existing["first_trigger_at_ist"] = at.isoformat()
            return
        if side not in {"CE", "PE"} or not symbol.endswith(side):
            return
        setups[setup_id] = {
            "setup_id": setup_id,
            "symbol": symbol,
            "side": side,
            "first_trigger_at_ist": at.isoformat(),
            "first_at": at,
            "lifecycle_stage_observed": stage,
            "identity_conflict": False,
        }

    # Read-only lifecycle markers; qualification must precede broker acceptance
    # and must not be mislabeled a fresh structural setup.
    admission_events: dict[str, dict[str, list[tuple[datetime, str, str]]]] = {}
    restarts = 0
    submission_lines = 0
    for line in log_text.splitlines():
        if "Started server process [" in line:
            restarts += 1
        if "ORDER_SUBMIT" in line or "order.submit_attempt" in line:
            submission_lines += 1
        if not any(
            tag in line
            for tag in (
                "SETUP_LIFECYCLE stage=TRIGGER_QUALIFIED",
                "SETUP_LIFECYCLE stage=MANAGER_QUALIFIED",
                "SETUP_LIFECYCLE stage=BROKER_ACCEPTED",
                "STRATEGY_TRIGGER_VOTE strategy=SMC",
                "ORDERFLOW_CONTEXT_EVIDENCE",
            )
        ):
            continue
        at = _when(line)
        if at is None:
            continue
        fields = _fields(line)
        if "SETUP_LIFECYCLE stage=TRIGGER_QUALIFIED" in line:
            if fields.get("strategy") != "SMC":
                continue
            observe_setup(
                fields.get("setup_id", ""),
                fields.get("symbol", ""),
                fields.get("side", ""),
                at,
                stage=True,
            )
        elif (
            "SETUP_LIFECYCLE stage=MANAGER_QUALIFIED" in line
            or "SETUP_LIFECYCLE stage=BROKER_ACCEPTED" in line
        ):
            if fields.get("strategy") != "SMC":
                continue
            setup_id = fields.get("setup_id", "")
            symbol = fields.get("symbol", "")
            side = fields.get("side", "")
            if not setup_id or side not in {"CE", "PE"} or not symbol.endswith(side):
                continue
            stage_key = (
                "manager"
                if "stage=MANAGER_QUALIFIED" in line
                else "broker"
            )
            admission_events.setdefault(setup_id, {"manager": [], "broker": []})[
                stage_key
            ].append((at, symbol, side))
        elif "STRATEGY_TRIGGER_VOTE strategy=SMC" in line:
            setup_id = fields.get("setup_id", "")
            symbol = fields.get("symbol", "")
            side = fields.get("side", "")
            if setup_id and symbol and side in {"CE", "PE"} and symbol.endswith(side):
                observe_setup(setup_id, symbol, side, at, stage=False)
                version_float = _float(fields, "quote_update_version", 0)
                version = int(version_float) if math.isfinite(version_float) else 0
                votes.append((setup_id, symbol, at, version))
        elif "ORDERFLOW_CONTEXT_EVIDENCE" in line:
            item = _evidence(line, at)
            if item is not None:
                flow.append(item)

    report: list[dict[str, Any]] = []
    for item in sorted(setups.values(), key=lambda entry: entry["first_at"]):
        at = item["first_at"]
        symbol = item["symbol"]
        setup_id = item["setup_id"]
        selected = [
            (vote_time, symbol, vote_version)
            for vote_setup, vote_symbol, vote_time, vote_version in votes
            if vote_setup == setup_id
            and vote_symbol == symbol
            and at <= vote_time <= at + timedelta(seconds=window_seconds)
        ]
        # Only attach book observations close to an explicit SMC trigger vote.
        # Absence of an observed vote in a throttled journal is NOT proof that
        # the engine did not evaluate a setup at that instant.
        used: set[int] = set()
        matched: list[Evidence] = []
        unique_votes = sorted(set(selected))
        for vote_at, _, vote_version in unique_votes:
            matches = [
                (j, entry)
                for j, entry in enumerate(flow)
                if j not in used
                and entry.symbol == symbol
                and entry.quote_version == vote_version
                and vote_version > 0
                and abs((entry.at - vote_at).total_seconds()) <= 1
            ]
            if not matches:
                continue
            j, entry = min(
                matches,
                key=lambda pair: (abs((pair[1].at - vote_at).total_seconds()), pair[0]),
            )
            used.add(j)
            matched.append(entry)
        matched.sort(key=lambda evidence: (evidence.at, evidence.quote_version))
        counts = Counter(event.classification for event in matched)
        recorded = admission_events.get(setup_id, {"manager": [], "broker": []})
        manager_times = sorted(
            timestamp
            for timestamp, observed_symbol, observed_side in recorded["manager"]
            if observed_symbol == symbol and observed_side == item["side"]
        )
        broker_times = sorted(
            timestamp
            for timestamp, observed_symbol, observed_side in recorded["broker"]
            if observed_symbol == symbol and observed_side == item["side"]
        )
        manager_at = manager_times[0] if manager_times else None
        broker_at = broker_times[0] if broker_times else None
        manager_sweep_age = _reference_sweep_age(setup_id, manager_at)
        broker_sweep_age = _reference_sweep_age(setup_id, broker_at)
        # Missing/malformed historical provenance is indeterminate, not "pass".
        five_minute_reference = (
            "no_manager_qualification_logged"
            if manager_at is None
            else "unknown_original_sweep"
            if manager_sweep_age is None
            else "invalid_future_anchor"
            if manager_sweep_age < 0
            else "would_reject_stale_sweep"
            if manager_sweep_age > 300
            else "within_five_minute_limit"
        )
        result = {
            "manager_qualified_at_ist": (
                manager_at.isoformat() if manager_at else None
            ),
            "broker_accepted_at_ist": broker_at.isoformat() if broker_at else None,
            "original_sweep_at_ist": (
                _smc_setup_anchor_time(setup_id).astimezone(IST).isoformat()
                if _smc_setup_anchor_time(setup_id) is not None
                else None
            ),
            "sweep_age_at_manager_qualification_seconds": manager_sweep_age,
            "sweep_age_at_broker_acceptance_seconds": broker_sweep_age,
            "five_minute_reference": five_minute_reference,
            "setup_id": setup_id,
            "first_trigger_at_ist": item["first_trigger_at_ist"],
            "symbol": symbol,
            "side": item["side"],
            "identity_conflict": item["identity_conflict"],
            "lifecycle_stage_observed": item["lifecycle_stage_observed"],
            "trigger_vote_lines": len(selected),
            "distinct_vote_snapshots": len(unique_votes),
            "matched_book_observations": len(matched),
            "unmatched_votes": len(unique_votes) - len(matched),
            "qualifying_book_observations": counts["qualifying"],
            "neutral_observations": counts["neutral"],
            "hard_reset_observations": counts["hard_reset"],
            "strict_logged_pair": None,
            "neutral_tolerant_logged_pair": None,
        }
        if (
            item["lifecycle_stage_observed"]
            and not result["identity_conflict"]
            and result["unmatched_votes"] == 0
            and matched
        ):
            result["strict_logged_pair"] = _has_pair(matched, allow_one_neutral=False)
            result["neutral_tolerant_logged_pair"] = _has_pair(
                matched, allow_one_neutral=True
            )
        report.append(result)
    return {
        "schema_version": 1,
        "evidence_grade": "throttled_journald_observational_proxy",
        "execution_or_pnl_replay_valid": False,
        "reason": (
            "Missing complete quote/candle/bid-ask/stop/fill provenance; "
            "runtime calls may be throttled; vote-only setups lack stages; "
            "historical setup-ID sweep ages are not executable P&L replay"
        ),
        "setup_count": len(report),
        "manager_qualified_setup_count": sum(
            entry["manager_qualified_at_ist"] is not None for entry in report
        ),
        "manager_qualified_5m_reference_rejections": sum(
            entry["five_minute_reference"] == "would_reject_stale_sweep"
            for entry in report
        ),
        "lifecycle_stage_setup_count": sum(
            bool(item["lifecycle_stage_observed"]) for item in report
        ),
        "vote_only_setup_count": sum(
            not item["lifecycle_stage_observed"] for item in report
        ),
        "server_start_lines": restarts,
        "order_submission_marker_lines": submission_lines,
        "comparison": {
            "baseline": (
                "two logged qualifying observations 4-12 seconds apart, "
                "no intervening nonqualifying observation"
            ),
            "research_variant": (
                "one explicitly neutral missing-confirmation observation tolerated; "
                "direction mismatch, OFI misalignment, adverse flow, missing 3s "
                "coverage and holdoff always reset"
            ),
        },
        "setups": report,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Locally exported journald text")
    parser.add_argument("--window-seconds", type=int, default=120)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args(argv)
    report = analyze(
        args.log_path.read_text(encoding="utf-8"),
        window_seconds=args.window_seconds,
    )
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
