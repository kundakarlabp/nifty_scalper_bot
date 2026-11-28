"""Read-only targeted ICICI Breeze 1-minute research gap re-acquisition.

Never places orders; never changes the production data path; all source rows stay
in encrypted offline GitHub artifacts.  Original archives are immutable.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import io
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from scripts.archive_breeze_bulk import ResearchClient
from scripts.archive_breeze_history import IST, authentication_error_code, collect_contract

FIRST = dt.date(2024, 11, 6)
STOP = dt.date(2026, 11, 6)
MAX_TARGETS = 180
MAX_CALLS = 180
MAX_SECONDS = 1200
SPECIAL_EXCLUDED = {dt.date(2025, 10, 21)}


def load_targets(path: Path) -> list[dict[str, Any]]:
    """Accept only explicit, bounded contract identities from the reviewed plan."""
    raw = path.read_bytes()
    with io.StringIO(raw.decode("utf-8")) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [
            "day", "kind", "expiry", "strike", "side", "reason", "priority"
        ]:
            raise ValueError("repair_plan_schema_invalid")
        items = list(reader)
    if not 1 <= len(items) <= MAX_TARGETS:
        raise ValueError("repair_plan_size_invalid")
    selected: set[tuple[str, str, str, str, str]] = set()
    for row in items:
        day = dt.date.fromisoformat(row["day"])
        expiry = dt.date.fromisoformat(row["expiry"]) if row["expiry"] else None
        side = row["side"]
        kind = row["kind"]
        strike = int(row["strike"]) if row["strike"] else None
        if (
            not FIRST <= day < STOP
            or day >= dt.datetime.now(IST).date()
            or day in SPECIAL_EXCLUDED
            or kind not in {"cash", "futures", "options"}
            or row["priority"] not in {"1", "2", "3"}
            or row["reason"] not in {
                "provider_or_validation_failed",
                "empty_far_or_illiquid_future_check",
                "nonempty_contract_missing_minutes",
            }
            or kind == "cash" and (expiry is not None or strike is not None or side != "INDEX")
            or kind == "futures" and (expiry is None or strike is not None or side != "FUT")
            or kind == "options" and (
                expiry is None or strike is None or strike <= 0
                or strike % 50 != 0 or side not in {"CE", "PE"}
            )
            or expiry is not None and expiry < day
        ):
            raise ValueError("repair_plan_contract_invalid")
        identity = (row["day"], kind, row["expiry"], row["strike"], side)
        if identity in selected:
            raise ValueError("repair_plan_duplicate")
        selected.add(identity)
        row["day_obj"] = day
        row["expiry_obj"] = expiry
        row["strike_obj"] = strike
    # Existing partial nonempty bars first; likely unavailable contracts last.
    return sorted(items, key=lambda r: (-int(r["priority"]), r["day"], r["kind"], r["expiry"], r["side"]))


def execute_plan(client: ResearchClient, targets: list[dict[str, Any]], outdir: Path) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "source": "icici_breeze_historical_v2",
        "research_only": True,
        "original_archives_immutable": True,
        "promotion_eligible": False,
        "source_timezone_verified": False,
        "source_timestamp_convention_verified": False,
        "full_option_chain_acquired": False,
        "bid_ask_or_depth_acquired": False,
        "targets_requested": len(targets),
        "attempted": [],
        "failures": [],
        "requests_made": 0,
        "plan_walk_completed": False,
        "all_contracts_repaired": False,
    }
    try:
        for target in targets:
            identity = {
                k: target[k]
                for k in ("day", "kind", "expiry", "strike", "side", "reason")
            }
            try:
                result = collect_contract(
                    client,
                    day=target["day_obj"],
                    expiry=target["expiry_obj"],
                    strike=target["strike_obj"],
                    side=target["side"],
                    product_type=target["kind"],
                    outdir=outdir,
                    timestamp_convention="bar_start",
                )
                report["attempted"].append({**identity, **result})
            except (ValueError, TypeError, KeyError, OverflowError, OSError):
                report["failures"].append({**identity, "result": "provider_or_validation_failed"})
            report["requests_made"] = client.calls
            (outdir / "coverage.json").write_text(json.dumps(report, sort_keys=True))
            if len(report["attempted"]) % 20 == 0 and report["attempted"]:
                print(json.dumps({"attempted": len(report["attempted"]), "calls": client.calls}), flush=True)
        report["plan_walk_completed"] = True
    except RuntimeError:
        report["stopped_early"] = True
    finally:
        report["requests_made"] = client.calls
        report["bars_retrieved"] = sum(x["bars"] for x in report["attempted"])
        report["incomplete_results"] = sum(
            x["bars"] != 375 for x in report["attempted"]
        )
        report["all_contracts_repaired"] = bool(
            report["plan_walk_completed"]
            and not report["failures"]
            and not report["incomplete_results"]
        )
        (outdir / "coverage.json").write_text(json.dumps(report, sort_keys=True))
        entries = []
        for path in sorted(outdir.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                entries.append(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                    f"{path.relative_to(outdir)}\n"
                )
        (outdir / "SHA256SUMS").write_text("".join(entries))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    logging.disable(logging.CRITICAL)
    try:
        targets = load_targets(args.plan)
        if shutil.disk_usage(args.outdir).free < 3 * 1024 ** 3:
            raise RuntimeError("repair_storage_floor")
        values = [
            os.environ.get(name, "").strip()
            for name in ("BREEZE_API_KEY", "BREEZE_API_SECRET", "BREEZE_SESSION_TOKEN")
        ]
        if not all(values):
            raise ValueError("credentials_missing")
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            from breeze_connect import BreezeConnect

            sdk = BreezeConnect(api_key=values[0])
            sdk.generate_session(api_secret=values[1], session_token=values[2])
        print("Breeze research re-acquisition authentication succeeded", flush=True)
        report = execute_plan(
            ResearchClient(sdk, MAX_CALLS, time.monotonic() + MAX_SECONDS),
            targets,
            args.outdir,
        )
    except Exception as exc:
        (args.outdir / "repair-error.json").write_text(
            json.dumps({
                "reason": authentication_error_code(exc)
                if type(exc) is not RuntimeError
                else "source_or_transport_failed",
                "details_suppressed": True,
            })
        )
        raise SystemExit("Breeze repair stopped; details suppressed") from None
    print(json.dumps({
        "attempted": len(report["attempted"]),
        "requests_made": report["requests_made"],
        "bars_retrieved": report["bars_retrieved"],
        "failures": len(report["failures"]),
        "incomplete_results": report["incomplete_results"],
        "plan_walk_completed": report["plan_walk_completed"],
    }), flush=True)
    if not report["plan_walk_completed"] or report["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
