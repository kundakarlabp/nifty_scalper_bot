#!/usr/bin/env python3
"""Five predeclared ORB improvement runs with production-like exit lifecycle.

This research is broker-free and never writes live settings. It reuses the
checksum-verified 2017-2020 prepared sessions and applies a causal minute-bar
proxy of production trailing/time-stop behavior. The five variants each change
one ORB setting relative to the lifecycle reference.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from nifty_scalper_bot.backtesting.strategy_research import (
    run_orb_session_research,
    summarize,
)

SLIPPAGE = (10.0, 25.0, 50.0)
BASE_OVERRIDES = {
    "ORB_PREMIUM_STOP_ATR_MULT": "0.75",
    "ORB_TARGET_RR": "2.5",
}


def candidates() -> list[dict[str, Any]]:
    """Return the static comparator, lifecycle reference and five one-change runs."""
    return [
        {
            "name": "static_reference_075_25",
            "overrides": dict(BASE_OVERRIDES),
            "lifecycle_proxy": False,
        },
        {
            "name": "lifecycle_reference_075_25",
            "overrides": dict(BASE_OVERRIDES),
            "lifecycle_proxy": True,
        },
        {
            "name": "run1_early_entry_60",
            "overrides": {**BASE_OVERRIDES, "ORB_MAX_ENTRY_MINUTES_AFTER_RANGE": "60"},
            "lifecycle_proxy": True,
        },
        {
            "name": "run2_retest_only",
            "overrides": {**BASE_OVERRIDES, "ORB_MOMENTUM_BRANCH_ENABLED": "false"},
            "lifecycle_proxy": True,
        },
        {
            "name": "run3_quality_score_6",
            "overrides": {**BASE_OVERRIDES, "ORB_QUALITY_MIN_SCORE_SHADOW": "6.0"},
            "lifecycle_proxy": True,
        },
        {
            "name": "run4_momentum_volume_1_5",
            "overrides": {**BASE_OVERRIDES, "ORB_MOMENTUM_MIN_VOLUME_RATIO": "1.5"},
            "lifecycle_proxy": True,
        },
        {
            "name": "run5_one_event_per_side",
            "overrides": {**BASE_OVERRIDES, "ORB_MAX_EVENTS_PER_SIDE": "1"},
            "lifecycle_proxy": True,
        },
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _run_task(task: tuple[str, dict[str, Any], float, list[str]]) -> dict[str, Any]:
    root, candidate, slippage, days = task
    os.environ.update(
        EXECUTION_MODE="SHADOW",
        STRATEGY_MODE="directional_scalp",
        ORB_ENABLED="true",
        BROKER_API_KEY="offline_research_unused",
        BROKER_API_SECRET="offline_research_unused",
    )
    logging.disable(logging.CRITICAL)
    trades: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    stress: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    exits: Counter[str] = Counter()

    for index, day in enumerate(days):
        result = run_orb_session_research(
            Path(root) / "sessions" / day,
            overrides=candidate["overrides"],
            slippage_bps=slippage,
            minimum_net_rr=1.5,
            lifecycle_proxy=bool(candidate["lifecycle_proxy"]),
        )
        trades.extend(result["trades"])
        unresolved.extend(result["unresolved_trades"])
        stress.extend(result["worst_case_stress_trades"])
        reasons.update(result["no_vote_reasons"])
        exits.update(result["exit_reasons"])
        if (index + 1) % 100 == 0:
            print(
                f"{candidate['name']} / {slippage} bps: "
                f"{index + 1}/{len(days)} sessions",
                flush=True,
            )

    return {
        "candidate": candidate["name"],
        "overrides": candidate["overrides"],
        "lifecycle_proxy": candidate["lifecycle_proxy"],
        "slippage_bps_per_side": slippage,
        "metrics": summarize(trades),
        "stress_metrics": summarize(stress),
        "data_quality": {
            "resolved_exit_count": len(trades),
            "unresolved_exit_count": len(unresolved),
            "unresolved_exit_rate": (
                len(unresolved) / (len(trades) + len(unresolved))
                if trades or unresolved
                else 0.0
            ),
        },
        "exit_reasons": dict(exits),
        "no_vote_reasons": dict(reasons),
        "trades": trades,
        "unresolved_trades": unresolved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")

    manifest = json.loads((args.study_dir / "data_manifest.json").read_text())
    phases = {
        "development": {"2017", "2018"},
        "validation": {"2019"},
        "final": {"2020"},
    }
    protocol = {
        "baseline": "0.75 ATR premium stop / 2.5R target",
        "minimum_net_rr": 1.5,
        "slippage_bps_per_side": list(SLIPPAGE),
        "lifecycle_proxy": {
            "activation_r": 0.75,
            "tier2_r": 1.0,
            "tier3_r": 2.0,
            "tier4_r": 3.0,
            "tier2_profit_lock": 0.40,
            "tier3_profit_lock": 0.50,
            "tier4_profit_lock": 0.60,
            "tier3_atr_mult": 1.50,
            "tier4_atr_mult": 1.00,
            "min_locked_profit_r": 0.10,
            "time_stop_minutes": 12,
            "time_stop_min_progress_r": 0.50,
            "bar_causality": "trail raised from completed bar is effective next bar",
            "live_parity": False,
        },
        "candidates": candidates(),
        "selection": "none; all five changes predeclared and reported",
        "promotion_eligible": False,
    }
    _write_json(args.study_dir / "iterative_protocol.json", protocol)

    all_results: dict[str, list[dict[str, Any]]] = {}
    for phase, years in phases.items():
        days = [row["day"] for row in manifest["sessions"] if row["day"][:4] in years]
        tasks = [
            (str(args.study_dir), candidate, slip, days)
            for candidate in candidates()
            for slip in SLIPPAGE
        ]
        results: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_task, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                print(
                    f"Completed {phase}: {row['candidate']} / "
                    f"{row['slippage_bps_per_side']} bps: {row['metrics']}",
                    flush=True,
                )
        results.sort(key=lambda row: (row["candidate"], row["slippage_bps_per_side"]))
        all_results[phase] = results
        _write_json(args.study_dir / f"iterative_{phase}.json", results)

    summary: list[dict[str, Any]] = []
    for candidate in candidates():
        name = candidate["name"]
        row: dict[str, Any] = {
            "candidate": name,
            "overrides": candidate["overrides"],
            "lifecycle_proxy": candidate["lifecycle_proxy"],
        }
        for phase in phases:
            matches = [
                item
                for item in all_results[phase]
                if item["candidate"] == name and item["slippage_bps_per_side"] == 10.0
            ]
            if matches:
                metrics = matches[0]["metrics"]
                row[phase] = {
                    "trades": metrics["trade_count"],
                    "net_pnl": metrics["net_pnl"],
                    "expectancy": metrics["expectancy"],
                    "profit_factor": metrics["profit_factor"],
                    "win_rate": metrics["win_rate"],
                    "max_realized_drawdown": metrics["max_realized_drawdown"],
                    "unresolved": matches[0]["data_quality"]["unresolved_exit_count"],
                    "exit_reasons": matches[0]["exit_reasons"],
                }
        summary.append(row)
    _write_json(args.study_dir / "iterative_summary.json", summary)


if __name__ == "__main__":
    main()
