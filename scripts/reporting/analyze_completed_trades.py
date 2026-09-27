"""Analyze canonical completed trades without modifying live strategy parameters."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from nifty_scalper_bot.backtesting.completed_trade_analysis import (  # noqa: E402
    attribution_readiness,
    canonicalize_completed_trades,
    chronological_post_cost_blocks,
    summarize_completed_trades,
)


def load_trade_ledger_rows(db_path: Path) -> list[dict[str, Any]]:
    """Read canonical completed-trade fields from the local SQLite ledger."""

    resolved = db_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Trade journal not found: {resolved}")
    uri = f"{resolved.as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT
                trade_id,
                state,
                strategy,
                closed_at,
                gross_pnl,
                estimated_costs,
                net_pnl,
                ledger_complete,
                outcome_json
            FROM trade_ledger
            WHERE state = 'CLOSED'
              AND ledger_complete = 1
            ORDER BY closed_at, trade_id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def build_analysis(
    rows: list[dict[str, Any]],
    *,
    block_size: int,
    components: tuple[str, ...],
) -> dict[str, Any]:
    """Build a machine-readable evidence report from canonical ledger rows."""

    trades = canonicalize_completed_trades(rows)
    overall = summarize_completed_trades(trades)
    blocks = chronological_post_cost_blocks(trades, block_size=block_size)
    readiness = attribution_readiness(trades, required_components=components)
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
        "attribution": {
            "ready": readiness.ready,
            "coverage": {
                name: asdict(coverage)
                for name, coverage in readiness.coverage.items()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades-db", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=20)
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
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1

    print(json.dumps({"ok": True, **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
