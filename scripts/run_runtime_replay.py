#!/usr/bin/env python3
"""Fixed no-network production replay worker; never load broker credentials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    args = parser.parse_args()
    from nifty_scalper_bot.backtesting.runtime_research import run_runtime_session

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
