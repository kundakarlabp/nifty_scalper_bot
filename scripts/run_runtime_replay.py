#!/usr/bin/env python3
"""Fixed no-network production replay worker; never load broker credentials."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
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


def _entry_signature(report: dict[str, object]) -> tuple[tuple[object, ...], ...]:
    """Completed buy fills only; reject partial/ambiguous entries elsewhere."""
    orders = report.get("orders")
    if not isinstance(orders, list):
        raise ValueError("paired_replay_orders_missing")
    entries: list[tuple[object, ...]] = []
    for order in orders:
        if not isinstance(order, dict):
            raise ValueError("paired_replay_order_invalid")
        side = str(order.get("transaction_type") or "").upper()
        filled = int(order.get("filled_quantity") or 0)
        status = str(order.get("status") or "").strip().lower()
        # Flat positions alone do not prove a pending broker-model order is
        # terminal. An unfilled pending BUY or SELL can still change exposure.
        if status not in {"complete", "filled", "cancelled", "canceled", "rejected"}:
            if side == "BUY" and filled > 0:
                raise ValueError("paired_replay_partial_entry")
            raise ValueError("paired_replay_unresolved_order")
        if side != "BUY" or not filled:
            continue
        if status not in {"complete", "filled"}:
            raise ValueError("paired_replay_partial_entry")
        price = float(order.get("average_price") or 0)
        stamp = float(order.get("timestamp") or 0)
        if not math.isfinite(price) or price <= 0 or not math.isfinite(stamp):
            raise ValueError("paired_replay_fill_geometry_invalid")
        entries.append(
            (
                str(order.get("symbol") or ""),
                filled,
                round(price, 4),
                round(stamp, 3),
            )
        )
    return tuple(entries)


def compare_paired_reports(
    baseline: dict[str, object],
    adaptive: dict[str, object],
    fidelity: dict[str, object],
    *,
    expected_sha: str,
    slippage_bps: float,
) -> dict[str, object]:
    """Conservative quantitative comparison; never infer profitable live fills."""
    result: dict[str, object] = {
        "state": "UNRESOLVED",
        "evidence": "RESEARCH_CANDIDATE_MODELED_BROKER_NOT_EXCHANGE_FILLED",
        "slippage_bps_per_side": slippage_bps,
        "baseline_policy": "adaptive_disabled_research_control",
        "adaptive_policy": "adaptive_enabled",
        "net_delta_semantics": "adaptive_minus_legacy_control",
        "baseline_net": None,
        "adaptive_net": None,
        "net_delta": None,
        "promotion_eligible": False,
    }
    if not (
        fidelity.get("archive_integrity_verified") is True
        and fidelity.get("observed_selected_context_available") is True
    ):
        result["blocker"] = "incomplete_captured_quote_fidelity"
        return result
    shared = (
        "archive_sha256",
        "events_processed",
        "snapshots_processed",
        "captured_revision",
    )
    if (
        not expected_sha
        or any(baseline.get(k) != adaptive.get(k) for k in shared)
        or baseline.get("archive_sha256") != expected_sha
        or fidelity.get("capture_sha256") != expected_sha
    ):
        result["blocker"] = "paired_replay_input_parity_mismatch"
        return result
    if baseline.get("positions") or adaptive.get("positions"):
        result["blocker"] = "paired_replay_open_position"
        return result
    try:
        lhs = _entry_signature(baseline)
        rhs = _entry_signature(adaptive)
    except (TypeError, ValueError) as exc:
        result["blocker"] = str(exc)
        return result
    result["baseline_entry_count"] = len(lhs)
    result["adaptive_entry_count"] = len(rhs)
    result["entry_parity"] = lhs == rhs
    if not lhs or lhs != rhs:
        result["blocker"] = "paired_replay_entry_divergence_or_no_fills"
        return result
    try:
        base_account = baseline["modeled_account"]
        alt_account = adaptive["modeled_account"]
        if not isinstance(base_account, dict) or not isinstance(alt_account, dict):
            raise ValueError
        bn = float(base_account["net_equity_pnl"])
        an = float(alt_account["net_equity_pnl"])
        bf = float(base_account["fees"])
        af = float(alt_account["fees"])
        bd = float(baseline["max_equity_drawdown"])
        ad = float(adaptive["max_equity_drawdown"])
        if not all(math.isfinite(x) and x >= 0 for x in (bf, af, bd, ad)):
            raise ValueError
        if not math.isfinite(bn) or not math.isfinite(an):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        result["blocker"] = "paired_replay_account_metrics_invalid"
        return result
    result.update(
        state="MODELED_PAIRED_COMPARISON",
        baseline_net=round(bn, 2),
        adaptive_net=round(an, 2),
        net_delta=round(an - bn, 2),
        baseline_fees=round(bf, 2),
        adaptive_fees=round(af, 2),
        baseline_drawdown=round(bd, 2),
        adaptive_drawdown=round(ad, 2),
        blocker="broker_fills_and_holdout_unverified",
    )
    return result


def run_paired_replays(
    session: Path,
    output: Path,
    *,
    stress_bps: tuple[float, ...] = (10.0, 25.0, 50.0),
) -> dict[str, object]:
    """Run two isolated production-composition workers per slippage scenario."""
    from nifty_scalper_bot.storage.replay_archive import archive_fingerprint

    if (
        not stress_bps
        or len(stress_bps) > 5
        or any(not math.isfinite(v) or not 0 <= v <= 100 for v in stress_bps)
    ):
        raise ValueError("paired_replay_invalid_stress")
    output.mkdir(parents=True, exist_ok=True)
    fidelity = write_capture_preflight(session, output)
    result: dict[str, object] = {
        "scope": "offline_paired_runtime_adaptive_exit_comparison",
        "session": session.name,
        "state": "BLOCKED",
        "promotion_eligible": False,
        "live_settings_modified": False,
        "stress": [],
        "limitations": [
            "Modeled fills are not Zerodha order confirmations",
            "Only identical filled-entry paths are compared",
            "Exchange SL-LIMIT gap/queue/modification parity unverified",
            "A single session cannot supply chronological out-of-sample evidence",
        ],
    }
    output_file = output / "paired-comparison.json"
    if not (
        fidelity.get("archive_integrity_verified") is True
        and fidelity.get("observed_selected_context_available") is True
    ):
        result["blocker"] = "no_usable_archived_depth_session"
        output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    fingerprint = archive_fingerprint(session)
    result["archive_sha256"] = fingerprint
    deadline = time.monotonic() + 1500.0
    statuses: list[dict[str, object]] = []
    for bps in stress_bps:
        reports: dict[str, dict[str, object]] = {}
        for name, enabled in (("baseline", False), ("adaptive", True)):
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                result["blocker"] = "paired_replay_budget_exhausted"
                break
            destination = output / f"{name}-{bps:g}bps"
            destination.mkdir(parents=True, exist_ok=True)
            # Never inherit broker credentials or operator env files.
            env = {
                k: v
                for k, v in os.environ.items()
                if k in {"PATH", "LANG", "LC_ALL", "TZ", "VIRTUAL_ENV"}
            }
            env.update(
                REPLAY_ISOLATED_PROCESS="true",
                PYTHONPATH=str(ROOT / "src"),
                ADAPTIVE_PROFIT_LOCK_ENABLED="true" if enabled else "false",
                ENABLE_LIVE="false",
                ENABLE_LIVE_TRADING="false",
                ALLOW_REAL_BROKER="false",
                ALLOW_NETWORK="false",
                BROKER_API_KEY="offline_replay",
                BROKER_API_SECRET="offline_replay",
                BROKER_ACCESS_TOKEN="offline_replay",
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--session",
                str(session.resolve()),
                "--output",
                str(destination.resolve()),
                "--slippage-bps",
                str(bps),
            ]
            try:
                process = subprocess.run(
                    command,
                    cwd=str(ROOT),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=min(600.0, remaining),
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                result["blocker"] = "paired_replay_worker_unavailable"
                break
            path = destination / "report.json"
            if process.returncode != 0 or not path.is_file():
                result["blocker"] = "paired_replay_worker_failed"
                result["failed_scenario"] = {"policy": name, "slippage_bps": bps}
                # The isolated worker writes a bounded failure code. Preserve
                # it instead of discarding the only actionable diagnostic.
                # Never return arbitrary exception text or broker payloads.
                try:
                    failure = json.loads(
                        (destination / "failure.json").read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    failure = {}
                error_code = None
                if isinstance(failure, dict):
                    error_code = failure.get("error_code")
                if (
                    isinstance(error_code, str)
                    and error_code.startswith("replay_")
                    and len(error_code) <= 100
                    and error_code.replace("_", "").isalnum()
                ):
                    result["worker_error_code"] = error_code
                break
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result["blocker"] = "paired_replay_report_invalid"
                break
            if not isinstance(report, dict):
                result["blocker"] = "paired_replay_report_invalid"
                break
            reports[name] = report
        if len(reports) != 2:
            break
        candidate = compare_paired_reports(
            reports["baseline"],
            reports["adaptive"],
            fidelity,
            expected_sha=fingerprint,
            slippage_bps=bps,
        )
        statuses.append(candidate)
    result["stress"] = statuses
    result["state"] = (
        "MODELED_COMPARISON_ONLY"
        if statuses
        and len(statuses) == len(stress_bps)
        and all(row["state"] == "MODELED_PAIRED_COMPARISON" for row in statuses)
        else "BLOCKED"
    )
    if result["state"] != "MODELED_COMPARISON_ONLY":
        result.setdefault("blocker", "paired_replay_scenarios_incomplete")
    if archive_fingerprint(session) != fingerprint:
        result["state"] = "BLOCKED"
        result["blocker"] = "paired_replay_archive_changed"
        result["stress"] = []
    output_file.write_text(
        json.dumps(result, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--compare-adaptive", action="store_true")
    parser.add_argument(
        "--stress-bps", type=float, nargs="+", default=[10.0, 25.0, 50.0]
    )
    args = parser.parse_args()
    if args.compare_adaptive:
        try:
            report = run_paired_replays(
                args.session, args.output, stress_bps=tuple(args.stress_bps)
            )
        except (ValueError, OSError) as exc:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "paired-comparison.json").write_text(
                json.dumps(
                    {
                        "state": "BLOCKED",
                        "error_code": (
                            str(exc)
                            if str(exc).startswith("paired_replay_")
                            else "paired_replay_preflight_failed"
                        ),
                        "promotion_eligible": False,
                        "live_settings_modified": False,
                    }
                )
            )
            return 1
        return 0 if report["state"] == "MODELED_COMPARISON_ONLY" else 1

    from nifty_scalper_bot.backtesting.runtime_research import run_runtime_session

    # On-demand OFFLINE quality telemetry. Does not alter live execution,
    # strategy behavior or broker calls. Preflight IO errors are handled by
    # the same outer failure/report boundary as the existing replay worker.
    try:
        write_capture_preflight(args.session, args.output)
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
