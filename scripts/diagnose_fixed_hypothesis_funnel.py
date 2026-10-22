#!/usr/bin/env python3
"""Read-only diagnostics of fixed monthly-option hypothesis artifacts.

No strategy configuration, trade execution, or broker integration is imported.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED_PHASES = {"development": 461, "validation": 229, "final": 215}
EXPECTED_STRATEGIES = {
    "ORBNetGateBaseline", "ORBRangeScaledResearch",
    "PullbackResearch", "CompressionResearch", "MomentumResearch",
}
EXPECTED_SLIPPAGE = {10.0, 25.0, 50.0}


def diagnose(report: dict) -> list[dict]:
    rows = report["results"]
    if len(rows) != 45:
        raise ValueError(f"expected 45 phase/strategy/slippage cells; got {len(rows)}")
    observed = set()
    output = []
    for row in rows:
        phase = row["phase"]
        strategy = row["candidate"]
        slip = float(row["slippage_bps_per_side"])
        key = (phase, strategy, slip)
        if key in observed:
            raise ValueError(f"duplicate result: {key}")
        observed.add(key)
        if (\n            phase not in EXPECTED_PHASES\n            or strategy not in EXPECTED_STRATEGIES\n            or slip not in EXPECTED_SLIPPAGE\n        ):
            raise ValueError(f"unexpected cell: {key}")
        if row["session_count"] != EXPECTED_PHASES[phase]:
            raise ValueError(f"session count mismatch: {key}")
        funnel = row["entry_funnel"]
        reasons = row["no_vote_reasons"]
        signals = int(funnel.get("signals_emitted", 0))
        intents = int(funnel.get("next_bar_intents", 0))
        fills = int(funnel.get("accepted_fills", 0))
        rr_rejected = int(reasons.get("cost_net_rr_rejected", 0))
        resolved = int(row["data_quality"]["resolved_exit_count"])
        unresolved = int(row["data_quality"]["unresolved_exit_count"])
        if min(signals, intents, fills, rr_rejected, resolved, unresolved) < 0:
            raise ValueError(f"negative funnel count: {key}")
        if fills != resolved + unresolved:
            raise ValueError(f"fill/exit mismatch: {key}")
        if intents > signals or fills > intents or rr_rejected > intents:
            raise ValueError(f"invalid funnel ordering: {key}")
        output.append({
            "phase": phase, "strategy": strategy, "slippage_bps_per_side": slip,
            "sessions": row["session_count"],
            "evaluations": int(funnel.get("evaluations", 0)),
            "signals": signals, "geometry_valid": int(funnel.get("geometry_valid", 0)),
            "next_bar_intents": intents, "cost_net_rr_rejected": rr_rejected,
            "accepted_fills": fills, "resolved": resolved, "unresolved": unresolved,
            "cost_rejection_share_of_intents": (\n                round(rr_rejected / intents, 6) if intents else None\n            ),
            "other_nonfilled_intents": intents - fills - rr_rejected,
            "no_vote_reasons_json": json.dumps(reasons, sort_keys=True),
        })
    expected = {\n        (p, s, b)\n        for p in EXPECTED_PHASES\n        for s in EXPECTED_STRATEGIES\n        for b in EXPECTED_SLIPPAGE\n    }
    if observed != expected:
        raise ValueError(f"missing result cells: {expected - observed}")
    return sorted(\n        output,\n        key=lambda r: (\n            list(EXPECTED_PHASES).index(r["phase"]),\n            r["strategy"],\n            r["slippage_bps_per_side"],\n        ),\n    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.artifact.read_text())
    rows = diagnose(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
