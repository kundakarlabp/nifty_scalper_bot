#!/usr/bin/env python3
"""Purpose: Run one reproducible, broker-free directional-strategy research bundle.

Responsibilities: Execute the existing VWAP and ORB component comparisons plus
the independent SMC/reversal matched-event comparison on the same archive.
Boundaries: Reads an explicit local archive; writes JSON only, never runtime
configuration, broker orders, candidate promotion or production state.
Safety: SHADOW-only; unavailable data and conflicting provenance fail closed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from nifty_scalper_bot.backtesting.failed_breakout_smc_comparison import (
    run_failed_breakout_smc_comparison,
)
from nifty_scalper_bot.backtesting.strategy_research import (
    run_orb_comparison,
    run_vwap_comparison,
)


def run_directional_research_bundle(directory: Path) -> dict[str, Any]:
    """Keep all strategy comparisons on one immutable source and non-live mode."""
    if str(os.getenv("EXECUTION_MODE", "")).strip().upper() != "SHADOW":
        raise ValueError("directional_research_bundle_requires_shadow")
    existing = os.environ.get("VWAP_RESEARCH_COMPARE_RETEST_REBREAK_ONLY")
    try:
        # This flag only affects isolated research arms; no live VWAP changes.
        os.environ["VWAP_RESEARCH_COMPARE_RETEST_REBREAK_ONLY"] = "true"
        vwap = run_vwap_comparison(directory)
    finally:
        if existing is None:
            os.environ.pop("VWAP_RESEARCH_COMPARE_RETEST_REBREAK_ONLY", None)
        else:
            os.environ["VWAP_RESEARCH_COMPARE_RETEST_REBREAK_ONLY"] = existing
    orb = run_orb_comparison(directory)
    reversal = run_failed_breakout_smc_comparison(directory)
    digests = {
        str(result["data_sha256"])
        for result in (vwap, orb, reversal)
    }
    if len(digests) != 1 or not next(iter(digests)):
        raise ValueError("directional_research_source_drift")
    return {
        "scope": "reproducible_same_archive_directional_research",
        "data_sha256": next(iter(digests)),
        "evidence_label": "RESEARCH_CANDIDATE",
        "live_equivalent": False,
        "promotion_eligible": False,
        "selected_for_live": None,
        "vwap": vwap,
        "orb": orb,
        "failed_breakout_vs_smc": reversal,
        "limitations": [
            "These studies share a source archive, not always identical fill cohorts",
            "The option archive is not a synchronized executable historical quote book",
            "Untouched out-of-sample and prospective SHADOW evidence absent",
            "No isolated component result can authorize live execution",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.archive.is_dir():
        parser.error("archive directory not found")
    result = run_directional_research_bundle(args.archive)
    if args.output.exists():
        parser.error("refusing to overwrite existing research output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
