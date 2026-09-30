"""Analyze canonical completed trades without modifying live strategy parameters."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from nifty_scalper_bot.backtesting.completed_trade_analysis import (  # noqa: E402, I001
    attribution_readiness,
    calibrate_signal_scores,
    canonicalize_completed_trades,
    chronological_post_cost_blocks,
    chronological_walk_forward,
    execution_data_quality,
    summarize_completed_trades,
    summarize_candidate_decisions,
    walk_forward_stability,
)
from nifty_scalper_bot.backtesting.research_validation import (  # noqa: E402
    combinatorial_purged_pbo,
    deflated_sharpe_ratio,
)
from nifty_scalper_bot.journal.trade_ledger import (  # noqa: E402
    load_candidate_decision_rows,
    load_trade_ledger_rows as load_canonical_trade_ledger_rows,
)


def load_trade_ledger_rows(db_path: Path) -> list[dict[str, Any]]:
    """Read completed trades through the shared canonical ledger read path."""

    resolved = db_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Trade journal not found: {resolved}")
    return load_canonical_trade_ledger_rows(
        resolved,
        limit=None,
        closed_only=True,
    )


def build_analysis(
    rows: list[dict[str, Any]],
    *,
    block_size: int,
    components: tuple[str, ...],
    allow_estimated_costs: bool = False,
    selection_trials: int | None = None,
    walk_forward_min_train: int = 20,
    walk_forward_test_trades: int = 10,
    walk_forward_min_folds: int = 2,
) -> dict[str, Any]:
    """Build a machine-readable evidence report from canonical ledger rows."""

    trades = canonicalize_completed_trades(
        rows,
        allow_estimated_costs=allow_estimated_costs,
    )
    overall = summarize_completed_trades(trades)
    blocks = chronological_post_cost_blocks(trades, block_size=block_size)
    readiness = attribution_readiness(trades, required_components=components)
    execution_quality = execution_data_quality(trades)
    walk_forward_folds = chronological_walk_forward(
        trades,
        min_train_trades=walk_forward_min_train,
        test_trades=walk_forward_test_trades,
    )
    walk_forward = walk_forward_stability(
        walk_forward_folds,
        minimum_folds=walk_forward_min_folds,
    )
    score_calibration = {
        key: asdict(
            calibrate_signal_scores(
                trades,
                score_key=key,
                minimum_trades_per_bin=10,
            )
        )
        for key in ("alpha_score", "final_score", "strategy_score")
    }
    r_values: list[float] = []
    for trade in trades:
        raw_r = trade.outcome.get("r_multiple")
        try:
            resolved_r = float(raw_r)
        except (TypeError, ValueError):
            continue
        if math.isfinite(resolved_r):
            r_values.append(resolved_r)
    dsr: dict[str, Any]
    if selection_trials is None:
        dsr = {
            "ready": False,
            "reason": "selection_trial_count_not_supplied",
        }
    elif len(r_values) < 3:
        dsr = {
            "ready": False,
            "reason": "insufficient_r_multiple_observations",
            "observations": len(r_values),
        }
    else:
        dsr = {
            "ready": True,
            **asdict(
                deflated_sharpe_ratio(
                    r_values,
                    trials=selection_trials,
                )
            ),
        }
    return {
        "dataset": {
            "canonical_completed_trades": len(trades),
            "first_closed_at": trades[0].closed_at if trades else None,
            "last_closed_at": trades[-1].closed_at if trades else None,
        },
        "overall_post_cost": asdict(overall),
        "chronological_blocks": [
            {
                "index": block.index,
                "start_trade_id": block.start_trade_id,
                "end_trade_id": block.end_trade_id,
                "start_closed_at": block.start_closed_at,
                "end_closed_at": block.end_closed_at,
                "summary": asdict(block.summary),
            }
            for block in blocks
        ],
        "walk_forward": {
            "configuration": {
                "min_train_trades": walk_forward_min_train,
                "test_trades": walk_forward_test_trades,
                "minimum_folds": walk_forward_min_folds,
            },
            "folds": [
                {
                    "index": fold.index,
                    "train_start_trade_id": fold.train_start_trade_id,
                    "train_end_trade_id": fold.train_end_trade_id,
                    "test_start_trade_id": fold.test_start_trade_id,
                    "test_end_trade_id": fold.test_end_trade_id,
                    "train_summary": asdict(fold.train_summary),
                    "test_summary": asdict(fold.test_summary),
                }
                for fold in walk_forward_folds
            ],
            "stability": asdict(walk_forward),
        },
        "execution_data_quality": asdict(execution_quality),
        "cost_evidence": {
            "broker_cost_trades": overall.broker_cost_trade_count,
            "total_trades": overall.trade_count,
            "all_broker_costed": (
                overall.broker_cost_trade_count == overall.trade_count
            ),
        },
        "score_calibration": score_calibration,
        "deflated_sharpe": dsr,
        "attribution": {
            "ready": readiness.ready,
            "coverage": {
                name: asdict(coverage) for name, coverage in readiness.coverage.items()
            },
            "blockers": list(readiness.blockers),
        },
        "parameter_change": {
            "allowed": False,
            "reason": (
                "candidate_walk_forward_required"
                if readiness.ready
                else "component_attribution_not_ready"
            ),
        },
    }


def load_candidate_returns(path: Path) -> dict[str, list[float]]:
    """Load aligned candidate post-cost return/R series from JSON."""

    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate returns JSON must be an object")
    candidates: dict[str, list[float]] = {}
    for name, values in payload.items():
        if not isinstance(values, list):
            raise ValueError(f"candidate {name!r} must be a JSON array")
        candidates[str(name)] = [float(value) for value in values]
    return candidates


def build_candidate_validation(
    candidates: dict[str, list[float]],
    *,
    n_groups: int,
    n_test_groups: int,
    purge_observations: int,
    embargo_observations: int,
) -> dict[str, Any]:
    """Build PBO and candidate-wise DSR evidence from aligned alternatives."""

    pbo = combinatorial_purged_pbo(
        candidates,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge_observations=purge_observations,
        embargo_observations=embargo_observations,
    )
    trial_count = len(candidates)
    dsr = {
        name: asdict(deflated_sharpe_ratio(values, trials=trial_count))
        for name, values in sorted(candidates.items())
        if len(values) >= 3
    }
    return {
        "pbo": asdict(pbo),
        "deflated_sharpe_by_candidate": dsr,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades-db", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--walk-forward-min-train", type=int, default=20)
    parser.add_argument("--walk-forward-test-trades", type=int, default=10)
    parser.add_argument("--walk-forward-min-folds", type=int, default=2)
    parser.add_argument(
        "--allow-estimated-costs",
        action="store_true",
        help=(
            "Explicit legacy-audit override; canonical research requires broker costs."
        ),
    )
    parser.add_argument(
        "--selection-trials",
        type=int,
        help="Actual number of parameter/strategy trials used for DSR deflation.",
    )
    parser.add_argument(
        "--candidate-returns-json",
        type=Path,
        help="JSON object of aligned candidate post-cost return/R series for PBO.",
    )
    parser.add_argument("--pbo-groups", type=int, default=6)
    parser.add_argument("--pbo-test-groups", type=int, default=3)
    parser.add_argument("--pbo-purge-observations", type=int, default=0)
    parser.add_argument("--pbo-embargo-observations", type=int, default=0)
    parser.add_argument(
        "--component",
        action="append",
        dest="components",
        help="Required attribution component; repeat as needed.",
    )
    args = parser.parse_args(argv)
    components = tuple(args.components or ("ORBPro", "SMC", "VWAPPro"))
    try:
        rows = load_trade_ledger_rows(args.trades_db)
        report = build_analysis(
            rows,
            block_size=args.block_size,
            components=components,
            allow_estimated_costs=args.allow_estimated_costs,
            selection_trials=args.selection_trials,
            walk_forward_min_train=args.walk_forward_min_train,
            walk_forward_test_trades=args.walk_forward_test_trades,
            walk_forward_min_folds=args.walk_forward_min_folds,
        )
        candidate_decisions = load_candidate_decision_rows(args.trades_db)
        report["candidate_decision_funnel"] = asdict(
            summarize_candidate_decisions(candidate_decisions)
        )
        report["candidate_validation"] = (
            build_candidate_validation(
                load_candidate_returns(args.candidate_returns_json),
                n_groups=args.pbo_groups,
                n_test_groups=args.pbo_test_groups,
                purge_observations=args.pbo_purge_observations,
                embargo_observations=args.pbo_embargo_observations,
            )
            if args.candidate_returns_json is not None
            else {
                "ready": False,
                "reason": "candidate_returns_not_supplied",
            }
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1

    print(json.dumps({"ok": True, **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
