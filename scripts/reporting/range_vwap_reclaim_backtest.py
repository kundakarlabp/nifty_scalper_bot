"""Purpose: Run causal, non-executable range-reclaim signal research on Breeze CSV.

Responsibilities: Require the existing private Breeze exporter fields and
measure completed-futures-bar signal coverage without assuming option books,
historical instrument multipliers, or broker fills.
Safety: No broker imports or order path; zero execution-grade P&L claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    RangeVWAPReclaimStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.range_vwap_reclaim import (
    assess_range_vwap_reclaim,
)


def _parse_time(raw: str) -> datetime:
    timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timezone_aware_bar_timestamp_required")
    return timestamp


def run_futures_signal_audit(csv_path: Path) -> dict[str, Any]:
    """Audit a real CSV, abstain entirely from simulated option fills."""
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        expected = {
            "start_ist",
            "available_at_ist",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError("missing_causal_1m_futures_csv_fields")
        for raw in reader:
            start = _parse_time(str(raw["start_ist"]))
            available = _parse_time(str(raw["available_at_ist"]))
            if not timedelta(minutes=1) <= available - start <= timedelta(minutes=3):
                raise ValueError("bar_unavailable_or_start_clock_unproven")
            candle = {
                "timestamp": start.isoformat(),
                "open": raw["open"],
                "high": raw["high"],
                "low": raw["low"],
                "close": raw["close"],
                "volume": raw["volume"],
                "is_complete": True,
            }
            rows.append({"bar": candle, "start": start, "available": available})
    rows.sort(key=lambda item: item["start"])
    if len({r["start"] for r in rows}) != len(rows):
        raise ValueError("duplicate_futures_minute")
    reasons: Counter[str] = Counter()
    candidates = []
    history: list[dict[str, Any]] = []
    date = None
    config = RangeVWAPReclaimStrategyConfig()
    for item in rows:
        current_date = item["start"].date()
        if date != current_date:
            history = []
            date = current_date
        history.append(item["bar"])
        if len(history) < config.min_history_bars:
            reasons["insufficient_verified_futures_history"] += 1
            continue
        decision = assess_range_vwap_reclaim(
            history[-50:], decision_at=item["available"], config=config
        )
        if not decision["qualified"]:
            reasons[str(decision["reason"])] += 1
            continue
        candidates.append(
            {
                "available_at": item["available"].isoformat(),
                "bar_timestamp": decision["bar_timestamp"],
                "side": decision["side"],
                "futures_vwap": decision["futures_vwap"],
                "futures_atr": decision["futures_atr"],
                "range_width_atr": decision["range_width_atr"],
                "directional_efficiency": decision["directional_efficiency"],
                "unfilled_reason": "missing_point_in_time_option_book",
            }
        )
    return {
        "policy": "read_only_futures_signal_research",
        "futures_bars": len(rows),
        "evaluated": max(0, len(rows) - config.min_history_bars + 1),
        "qualified_futures_structures": len(candidates),
        "rejections": dict(sorted(reasons.items())),
        "signals": candidates,
        "option_fills": 0,
        "net_pnl": None,
        "profit_factor": None,
        "max_drawdown": None,
        "execution_grade_ready": False,
        "live_promotion_eligible": False,
        "missing_evidence": [
            "point_in_time_nifty_option_chain_and_lot_size",
            "continuous_observed_bid_ask_and_order_book",
            "historical_order_fill_and_stop_latency",
            "chronological_out_of_sample_cost_and_slippage_stress",
        ],
    }


def main() -> None:
    """Read a private futures export and write an explicitly non-live report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--futures-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing_to_overwrite_existing_research")
    result = run_futures_signal_audit(args.futures_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
