#!/usr/bin/env python3
"""Poll a GitHub research request or run its fixed read-only worker.

History and completed-trade evidence are saved locally. A missing current-bot
replay adapter is a reported blocker, never replaced by the generic RSI demo.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from nifty_scalper_bot.ops.research_jobs import (  # noqa: E402
    start_job,
    write_json,
)


def run_worker(request: dict[str, Any], env_file: Path) -> dict[str, Any]:
    """Use broker read methods and the canonical completed-trade analyzer only."""
    from dotenv import load_dotenv

    load_dotenv(env_file, override=False)
    from scripts.archive_kite_history import archive_history
    from scripts.reporting.analyze_completed_trades import (
        build_analysis,
        load_trade_ledger_rows,
    )

    from nifty_scalper_bot.config.paths import get_data_dir
    from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
    from nifty_scalper_bot.instruments.active_contracts import (
        resolve_active_nifty_future_from_instruments,
    )

    directory = ROOT / "data/research" / request["id"]
    status = {**request, "state": "collecting", "backtest_completed": False}
    write_json(directory / "status.json", status)
    write_json(ROOT / "data/research/latest.json", status)
    client = ZerodhaKiteClient()
    try:
        with urlopen("http://127.0.0.1:8080/trading/status", timeout=5) as response:
            selected = json.load(response).get("selected", {})
        ce, pe = selected.get("ce"), selected.get("pe")
        if not ce or not pe:
            raise ValueError("active_option_basket_unavailable")
        future = resolve_active_nifty_future_from_instruments(
            client.instruments("NFO")
        ).symbol
        if not future:
            raise ValueError("active_nifty_future_unavailable")
        coverage = archive_history(
            client,
            ["NSE:NIFTY 50", future, ce, pe],
            date.fromisoformat(request["start"]),
            date.fromisoformat(request["end"]),
            directory / "history",
        )
        status["coverage"] = coverage
    finally:
        client.close()
    journal = get_data_dir() / "trades.db"
    if journal.is_file():
        evidence = build_analysis(
            load_trade_ledger_rows(journal),
            block_size=20,
            components=("ORBPro", "SMC", "VWAPPro"),
        )
        write_json(directory / "completed_trade_analysis.json", evidence)
        status["completed_trade_analysis"] = "completed_trade_analysis.json"
    else:
        status["ledger_analysis_blocker"] = "canonical_trade_journal_unavailable"
    status.update(
        state="blocked",
        blocker="current_bot_offline_replay_adapter_unavailable",
        explanation=(
            "Real minute history and completed-trade analysis are prerequisites. "
            "The generic RSI/demo backtest does not validate ORBPro/SMC/VWAPPro. "
            "Kite active-contract bars lack historical bid/ask depth and cannot "
            "recover expired options; no strategy profitability claim is made."
        ),
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--request-id")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if not args.worker:
        path = ROOT / "deploy/research_request.json"
        if not path.is_file():
            return 0
        print(json.dumps(start_job(ROOT, json.loads(path.read_text()))))
        return 0
    # Validate before using an identifier as a path, even for the fixed CLI.
    from nifty_scalper_bot.ops.research_jobs import validate_request

    validate_request({"id": args.request_id})
    directory = ROOT / "data/research" / args.request_id
    request = json.loads((directory / "status.json").read_text())
    try:
        if args.env_file is None or not args.env_file.is_file():
            raise FileNotFoundError("operator_env_file_unavailable")
        result = run_worker(request, args.env_file)
    except Exception as exc:
        # Exception messages may include upstream credentials; record type only.
        result = {
            **request,
            "state": "failed",
            "backtest_completed": False,
            "error_type": type(exc).__name__,
        }
    write_json(directory / "status.json", result)
    write_json(ROOT / "data/research/latest.json", result)
    print(json.dumps(result))
    return 0 if result["state"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
