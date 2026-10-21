#!/usr/bin/env python3
"""Fixed no-network production replay worker; never load broker credentials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write_capture_preflight(session: Path, output: Path) -> dict[str, object]:
    """Report replay evidence quality; never alter the existing replay decision."""
    from nifty_scalper_bot.backtesting.replay_fidelity import audit_session

    try:
        report: dict[str, object] = audit_session(session)
    except (ValueError, OSError, TypeError) as exc:
        # A failed *diagnostic* is not permission to change strategy/data;
        # existing runtime_session validation still determines replay fate.
        report = {
            "profile": "replay_capture_fidelity_v1",
            "state": "unavailable",
            "error_code": "replay_capture_fidelity_preflight_unavailable",
            "error_type": type(exc).__name__,
            "archive_integrity_verified": False,
            "observed_selected_context_available": False,
            "live_equivalent": False,
            "promotion_eligible": False,
            "account_pnl": None,
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "capture-fidelity.json").write_text(
        json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    args = parser.parse_args()
    from nifty_scalper_bot.backtesting.runtime_research import run_runtime_session

    # On-demand OFFLINE quality telemetry. Does not block or alter existing
    # runtime replay outcomes, and never reads live broker credentials.
    write_capture_preflight(args.session, args.output)
    try:
        run_runtime_session(args.session, args.output, slippage_bps=args.slippage_bps)
    except Exception as exc:
        # Only stable locally-owned codes are surfaced, never upstream payloads.
        code = str(exc)
        if not code.startswith("replay_") or len(code) > 100 or " " in code:
            code = "replay_runtime_failed"
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "failure.json").write_text(
            json.dumps(
                {
                    "state": "failed",
                    "error_code": code,
                    "error_type": type(exc).__name__,
                    "live_equivalent": False,
                }
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
