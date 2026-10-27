"""Research-only comparison of SMC confirmation evidence in archived journald.

Responsibilities: deduplicate strategy-owned SMC setup markers, compare
strict and conservative neutral-tolerant logged confirmation sequences,
and report incompleteness without fabricating executable opportunities.

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
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=IST
    )


def _float(fields: dict[str, str], name: str, default: float = -1.0) -> float:
    try:
        return float(fields[name])
    except (KeyError, TypeError, ValueError, OverflowError):
        return default


def _evidence(line: str, at: datetime) -> Evidence | None:
    fields = _fields(line)
    symbol = fields.get("symbol", "")
    if not symbol.startswith("NFO:") or not symbol.endswith(("CE", "PE")):
        return None
    reason = fields.get("block_reason", "unknown")
    version_float = _float(fields, "quote_version", 0)
    version = int(version_float) if math.isfinite(version_float) else 0
    qualifies = (
        fields.get("aligned") == "True"
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
        fields.get("conflict") == "True"
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
    votes: list[tuple[str, str, datetime]] = []
    flow: list[Evidence] = []
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
            setup_id, symbol, side = (
                fields.get("setup_id", ""),
                fields.get("symbol", ""),
                fields.get("side", ""),
            )
            if not setup_id:
                continue
            if setup_id in setups:
                current = setups[setup_id]
                if current["side"] != side or current["symbol"] != symbol:
                    current["identity_conflict"] = True
                continue
            if side not in {"CE", "PE"} or not symbol.endswith(side):
                continue
            setups[setup_id] = {
                "setup_id": setup_id,
                "symbol": symbol,
                "side": side,
                "first_trigger_at_ist": at.isoformat(),
                "first_at": at,
                "identity_conflict": False,
            }
        elif "STRATEGY_TRIGGER_VOTE strategy=SMC" in line:
            if fields.get("setup_id") and fields.get("symbol"):
                votes.append((fields["setup_id"], fields["symbol"], at))
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
            (vote_time, symbol)
            for vote_setup, vote_symbol, vote_time in votes
            if vote_setup == setup_id
            and vote_symbol == symbol
            and at <= vote_time <= at + timedelta(seconds=window_seconds)
        ]
        # Only attach book observations close to an explicit SMC trigger vote.
        # Absence of an observed vote in a throttled journal is NOT proof that
        # the engine did not evaluate a setup at that instant.
        used: set[int] = set()
        matched: list[Evidence] = []
        for vote_at, _ in sorted(set(selected)):
            matches = [
                (j, entry)
                for j, entry in enumerate(flow)
                if j not in used
                and entry.symbol == symbol
                and abs((entry.at - vote_at).total_seconds()) <= 1
            ]
            if not matches:
                continue
            j, entry = min(
                matches,
                key=lambda pair: (
                    abs((pair[1].at - vote_at).total_seconds()), pair[0]
                ),
            )
            used.add(j)
            matched.append(entry)
        matched.sort(key=lambda evidence: (evidence.at, evidence.quote_version))
        counts = Counter(event.classification for event in matched)
        result = {
            "setup_id": setup_id,
            "first_trigger_at_ist": item["first_trigger_at_ist"],
            "symbol": symbol,
            "side": item["side"],
            "identity_conflict": item["identity_conflict"],
            "trigger_vote_lines": len(selected),
            "matched_book_observations": len(matched),
            "unmatched_votes": len(selected) - len(matched),
            "qualifying_book_observations": counts["qualifying"],
            "neutral_observations": counts["neutral"],
            "hard_reset_observations": counts["hard_reset"],
            "strict_logged_pair": None,
            "neutral_tolerant_logged_pair": None,
        }
        if (
            not result["identity_conflict"]
            and result["unmatched_votes"] == 0
            and matched
        ):
            result["strict_logged_pair"] = _has_pair(
                matched, allow_one_neutral=False
            )
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
            "runtime calls may be throttled"
        ),
        "setup_count": len(report),
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
