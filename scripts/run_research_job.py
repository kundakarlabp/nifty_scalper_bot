#!/usr/bin/env python3
"""Poll a GitHub research request or run its fixed read-only worker.

History, strategy component bar research and completed-trade evidence are saved
locally. Modeled component research is never presented as live-pipeline parity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def safe_error_code(exc: Exception) -> str:
    """Classify known validation failures without disclosing exception payloads."""
    message = str(exc)
    codes = {
        "active_option_basket_unavailable": "active_option_basket_unavailable",
        "active_nifty_future_unavailable": "active_nifty_future_unavailable",
        "operator_env_file_unavailable": "operator_env_file_unavailable",
        "Not in current instrument master:": "current_instrument_unavailable",
        "Not a NIFTY option/context instrument:": "invalid_nifty_instrument",
        "Invalid current instrument token:": "invalid_instrument_token",
        "broker-calculated costs required": "ledger_requires_verified_costs",
        "broker-calculated costs missing": "ledger_requires_verified_costs",
        "completed trade violates gross_pnl": "ledger_net_pnl_inconsistent",
    }
    for code in (
        "research_history_unavailable",
        "research_requires_shadow_process",
        "research_strategy_evaluation_failed",
        "research_components_unavailable",
        "research_timestamp_convention_invalid",
        "research_timestamp_timezone_missing",
        "research_bar_alignment_invalid",
        "research_bar_values_invalid",
        "research_bar_geometry_invalid",
        "research_duplicate_bar_conflict",
        "research_instrument_identity_conflict",
        "research_instrument_not_nifty",
        "research_option_identity_invalid",
        "research_option_expired",
    ):
        codes[code] = code
    return next(
        (code for prefix, code in codes.items() if message.startswith(prefix)),
        "validation_failed",
    )


def load_active_basket() -> dict[str, Any]:
    """Wait briefly for background startup before querying broker history."""
    for attempt in range(30):
        try:
            with urlopen("http://127.0.0.1:8080/trading/status", timeout=5) as response:
                selected = json.load(response).get("selected", {})
            if selected.get("ce") and selected.get("pe"):
                return selected
        except (OSError, ValueError):
            pass
        if attempt < 29:
            time.sleep(2)
    raise ValueError("active_option_basket_unavailable")


def run_worker(request: dict[str, Any], env_file: Path) -> dict[str, Any]:
    """Use broker read methods and the canonical completed-trade analyzer only."""
    from dotenv import load_dotenv

    load_dotenv(env_file, override=False)
    from scripts.archive_kite_history import archive_history
    from scripts.reporting.analyze_completed_trades import (
        build_analysis,
        load_trade_ledger_rows,
    )

    from nifty_scalper_bot.backtesting.strategy_research import run_archived_research
    from nifty_scalper_bot.config.paths import get_data_dir
    from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
    from nifty_scalper_bot.instruments.active_contracts import (
        resolve_active_nifty_future_from_instruments,
    )

    directory = ROOT / "data/research" / request["id"]
    status = {
        **request,
        "state": "collecting",
        "stage": "history",
        "backtest_completed": False,
    }
    write_json(directory / "status.json", status)
    write_json(ROOT / "data/research/latest.json", status)
    client = ZerodhaKiteClient()
    try:
        selected = load_active_basket()
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
        status["stage"] = "ledger_analysis"
        write_json(directory / "status.json", status)
        write_json(ROOT / "data/research/latest.json", status)
    finally:
        client.close()
    journal = get_data_dir() / "trades.db"
    if journal.is_file():
        try:
            evidence = build_analysis(
                load_trade_ledger_rows(journal),
                block_size=20,
                components=("ORBPro", "SMC", "VWAPPro"),
            )
        except ValueError as exc:
            status["ledger_analysis_blocker"] = safe_error_code(exc)
        else:
            write_json(directory / "completed_trade_analysis.json", evidence)
            status["completed_trade_analysis"] = "completed_trade_analysis.json"
    else:
        status["ledger_analysis_blocker"] = "canonical_trade_journal_unavailable"
    status["stage"] = "strategy_bar_research"
    write_json(directory / "status.json", status)
    write_json(ROOT / "data/research/latest.json", status)
    report = run_archived_research(directory / "history")
    write_json(directory / "strategy_bar_research.json", report)
    status.update(
        state="completed",
        stage="finished",
        backtest_completed=True,
        backtest_scope=report["scope"],
        live_equivalent=False,
        evidence_label=report["evidence_label"],
        strategy_bar_research="strategy_bar_research.json",
        backtest_coverage=report["coverage"],
        backtest_summary=[
            {
                "slippage_bps_per_side": scenario["slippage_bps_per_side"],
                "strategies": {
                    name: result["metrics"]
                    for name, result in scenario["strategies"].items()
                },
            }
            for scenario in report["scenarios"]
        ],
        explanation=(
            "Completed modeled strategy component bar research; historical ATM "
            "selection, depth and full live-pipeline parity remain unverified. "
            "See report assumptions and limitations."
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
        print(json.dumps(start_job(ROOT, json.loads(path.read_text()), wait=True)))
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
        # Keep successful collection even if a later stage fails.
        previous = json.loads((directory / "status.json").read_text())
        result = {
            **previous,
            "state": "failed",
            "backtest_completed": False,
            "error_type": type(exc).__name__,
            "error_code": safe_error_code(exc),
        }
    write_json(directory / "status.json", result)
    write_json(ROOT / "data/research/latest.json", result)
    print(json.dumps(result))
    return 0 if result["state"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
