#!/usr/bin/env python3
"""Analyze whether ORB quality score/components predict realized profitability."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from nifty_scalper_bot.backtesting.strategy_research import summarize


def _score_bucket(score: Any) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(value):
        return "missing"
    floor = int(math.floor(value))
    return f"{floor}.0-{floor}.9"


def _group_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = summarize(trades)
    count = int(metrics["trade_count"])
    return {
        **metrics,
        "fees_per_trade": (metrics["fees"] / count) if count else None,
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        slip = str(row["slippage_bps_per_side"])
        trades = list(row.get("trades") or [])
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        branches: dict[str, list[dict[str, Any]]] = defaultdict(list)
        reasons_present: dict[str, list[dict[str, Any]]] = defaultdict(list)
        all_reasons: set[str] = set()

        for trade in trades:
            buckets[_score_bucket(trade.get("raw_setup_score"))].append(trade)
            branches[str(trade.get("entry_branch") or "missing")].append(trade)
            reasons = {
                str(reason) for reason in (trade.get("score_reasons") or []) if reason
            }
            all_reasons.update(reasons)
            for reason in reasons:
                reasons_present[reason].append(trade)

        components: dict[str, Any] = {}
        for reason in sorted(all_reasons):
            present = reasons_present[reason]
            absent = [trade for trade in trades if trade not in present]
            components[reason] = {
                "present": _group_metrics(present),
                "absent": _group_metrics(absent),
            }

        result[slip] = {
            "overall": _group_metrics(trades),
            "score_buckets": {
                name: _group_metrics(group) for name, group in sorted(buckets.items())
            },
            "entry_branches": {
                name: _group_metrics(group) for name, group in sorted(branches.items())
            },
            "score_components": components,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()

    output: dict[str, Any] = {
        "method": (
            "Descriptive score calibration only. Development 2017-2018 may be used "
            "to design a frozen candidate; validation 2019 and final 2020 must not "
            "be used to select weights or thresholds."
        )
    }
    for phase in ("development", "validation", "final"):
        path = args.study_dir / f"iterative_{phase}.json"
        rows = json.loads(path.read_text())
        output[phase] = analyze(rows)

    destination = args.study_dir / "score_calibration.json"
    destination.write_text(
        json.dumps(output, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
