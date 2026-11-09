#!/usr/bin/env python3
"""Produce a privacy-preserving Breeze historical execution readiness report.

Responsibilities: Read audited aggregate coverage, validation and repair summaries.
Ownership: Offline research reporting only; never imports raw broker market data.
Safety: Missing executable quotes/depth/fill proof always blocks execution-grade
backtest certification and live promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nifty_scalper_bot.backtesting.execution_evidence import execution_readiness


def audit_reports(
    coverage: dict[str, Any],
    candle_validation: dict[str, Any],
    repair_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    windows = int(coverage.get("input_archives", 0))
    candles = int(coverage.get("total_bars", 0))
    if windows < 1 or candles < 1:
        raise ValueError("historical_coverage_unavailable")
    if (
        candle_validation.get("archives") != windows
        or int(candle_validation.get("sha256_mismatches", -1)) != 0
        or int(candle_validation.get("row_issue_count", -1)) != 0
    ):
        raise ValueError("historical_archive_integrity_not_verified")
    if repair_comparison is not None and (
        repair_comparison.get("original_windows") != windows
        or repair_comparison.get("repair_requested", 0) < 1
        or repair_comparison.get("invalid_candle_hashes", -1) != 0
    ):
        raise ValueError("targeted_repair_integrity_unverified")

    # Positive book/depth claims are prohibited even if the bar arrays are valid.
    evidence = execution_readiness(
        point_in_time_contracts=False,
        verified_clocks=False,
        five_level_quotes=False,
        actual_broker_fills=False,
        exit_ledger_verified=False,
        chronological_oos=False,
    )
    return {
        "study": "NIFTY_BREEZE_HISTORICAL_EXECUTION_READINESS",
        "source": "ICICI_BREEZE_1MIN_OHLCV_RESEARCH",
        "archive_windows": windows,
        "observed_bars": candles,
        "observed_index_bars": int(coverage.get("total_index_bars", 0)),
        "observed_futures_bars": int(coverage.get("total_futures_bars", 0)),
        "observed_selected_option_bars": int(coverage.get("total_option_bars", 0)),
        "contract_days": int(coverage.get("total_contract_days", 0)),
        "provider_or_validation_failed_contracts": len(coverage.get("all_failed_contracts", [])),
        "empty_contract_days": len(coverage.get("all_empty_contracts", [])),
        "missing_minute_contract_days_before_special_session_correction":
            len(coverage.get("all_missing_minute_contracts", [])),
        "source_unavailable_dates_pre_exchange_calendar_reconciliation":
            len(coverage.get("all_source_unavailable", [])),
        "archive_integrity": "SHA256_AND_OHLC_STRUCTURE_PASSED",
        "timestamps_broker_convention_independently_verified": False,
        "historical_strike_listing_and_lot_size_verified": False,
        "full_chain_or_depth_present": False,
        "one_minute_volume_is_not_order_book_liquidity": True,
        "repair": {
            "requests": int((repair_comparison or {}).get("repair_requested", 0)),
            "identical_series": int(
                (repair_comparison or {}).get("hash_states", {}).get("identical", 0)
            ),
            "net_additional_candles": int((repair_comparison or {}).get("net_bar_delta", 0)),
            "failed_requests": int((repair_comparison or {}).get("retry_failed", 0)),
        },
        "evidence_gate": evidence,
        "research_label": "CAUSAL_BAR_RESEARCH_ONLY",
        "not_execution_grade": True,
        "promotion_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--repair-comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coverage = json.loads(args.coverage.read_text())
    validation = json.loads(args.validation.read_text())
    repair = (
        json.loads(args.repair_comparison.read_text())
        if args.repair_comparison is not None else None
    )
    report = audit_reports(coverage, validation, repair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "observed_bars": report["observed_bars"],
        "execution_grade_ready": report["evidence_gate"]["execution_grade_ready"],
        "promotion_eligible": False,
    }))


if __name__ == "__main__":
    main()
