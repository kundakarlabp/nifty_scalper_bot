"""Isolated, resource-bounded research jobs and recorded-feed replay.

Responsibilities: Manage single-flight collection and credential-free historical
replay, date-pin completed sessions, partition genuine process epochs, and admit
pinned SHADOW-only original-tape audits strictly outside market hours.
run_recorded_replays supports unchanged captured-config and paired exits research.
Safety: Never import live application entrypoints, send broker orders, invent
missing quotes or treat an invalid replay as a successful backtest.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

_COMPONENT_WORKER_BUDGET_SECONDS = 3600
_RUNTIME_REPLAY_BUDGET_SECONDS = 1500
_WORKER_WAIT_GRACE_SECONDS = 300
_WORKER_TERMINATION_GRACE_SECONDS = 10
_SYSTEMD_RESEARCH_UNIT = "niftybot-research-worker.service"
_SYSTEMD_BUSY_STATES = {"active", "activating", "reloading", "deactivating"}


def require_epoch_audit_offmarket(moment: datetime) -> None:
    """Reserve the entire bounded worker lifetime before 05:00 IST."""
    if moment.tzinfo is None:
        raise ValueError("epoch_audit_timezone_required")
    local = moment.astimezone(IST)
    if 5 <= local.hour < 17:
        raise ValueError("epoch_audit_off_market_only")
    next_protected_start = local.replace(hour=5, minute=0, second=0, microsecond=0)
    if local.hour >= 17:
        next_protected_start += timedelta(days=1)
    # Include the systemd worker's termination grace and a 5-minute buffer.
    budget_seconds = _worker_wait_timeout_seconds({"mode": "epoch_audit"}) + 300
    if (next_protected_start - local).total_seconds() <= budget_seconds:
        raise ValueError("epoch_audit_insufficient_offmarket_budget")


def validate_request(
    payload: dict[str, Any],
    *,
    today: date | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Accept bounded, dated offline requests; never commands or arbitrary paths."""
    allowed = {
        "id",
        "days",
        "mode",
        "session_date",
        "profile",
        "expected_uncompressed_sha256",
    }
    if set(payload) - allowed:
        raise ValueError("Unsupported research request field")
    job_id = payload.get("id")
    days = payload.get("days", 30)
    mode = payload.get("mode", "all")
    profile = payload.get("profile", "paired_adaptive")
    if mode not in {"all", "components", "runtime", "epoch_audit"}:
        raise ValueError("unsupported research mode")
    if profile not in {"paired_adaptive", "captured_baseline"}:
        raise ValueError("unsupported research replay profile")
    if profile != "paired_adaptive" and mode != "runtime":
        raise ValueError("captured_baseline_requires_runtime_mode")
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", job_id):
        raise ValueError("Invalid request id")
    if type(days) is not int or not 1 <= days <= 90:
        raise ValueError("days must be an integer from 1 to 90")
    clock = now or datetime.now(IST)
    if clock.tzinfo is None:
        raise ValueError("research_request_clock_requires_timezone")
    local_now = clock.astimezone(IST)
    current_date = today or local_now.date()
    pinned = payload.get("session_date")
    source_digest = payload.get("expected_uncompressed_sha256")
    if mode == "epoch_audit":
        if days != 1 or pinned is None or "profile" in payload:
            raise ValueError("epoch_audit_requires_one_pinned_day")
        if (
            type(source_digest) is not str
            or re.fullmatch(r"[a-fA-F0-9]{64}", source_digest) is None
        ):
            raise ValueError("epoch_audit_requires_source_sha256")
        # The same guard runs again in the independent worker.
        require_epoch_audit_offmarket(local_now)
    elif source_digest is not None:
        raise ValueError("source_digest_only_valid_for_epoch_audit")
    if pinned is not None:
        if mode not in {"runtime", "epoch_audit"} or days != 1:
            raise ValueError("pinned_session_requires_one_day_runtime")
        if not isinstance(pinned, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", pinned
        ):
            raise ValueError("invalid_replay_session_date")
        try:
            end = date.fromisoformat(pinned)
        except ValueError as exc:
            raise ValueError("invalid_replay_session_date") from exc
        if end > current_date or end < current_date - timedelta(days=89):
            raise ValueError("replay_session_date_out_of_range")
        if end == current_date and local_now.hour < 16:
            raise ValueError("replay_same_day_requires_post_close")
    else:
        end = current_date - timedelta(days=1)
    request = {
        "id": job_id,
        "days": days,
        "mode": mode,
        "start": (end - timedelta(days=days - 1)).isoformat(),
        "end": end.isoformat(),
    }
    if pinned is not None:
        request["session_date"] = end.isoformat()
    if "profile" in payload:
        request["profile"] = profile
    if mode == "epoch_audit":
        if not isinstance(source_digest, str):
            raise ValueError("epoch_audit_requires_source_sha256")
        request["expected_uncompressed_sha256"] = source_digest.lower()
    return request


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _worker_wait_timeout_seconds(request: dict[str, Any]) -> int:
    """Return a bounded parent wait budget aligned with requested research work."""

    mode = str(request.get("mode") or "all")
    if mode == "epoch_audit":
        return _COMPONENT_WORKER_BUDGET_SECONDS + _WORKER_WAIT_GRACE_SECONDS
    if mode == "runtime":
        return _RUNTIME_REPLAY_BUDGET_SECONDS + _WORKER_WAIT_GRACE_SECONDS
    if mode == "components":
        return _COMPONENT_WORKER_BUDGET_SECONDS + _WORKER_WAIT_GRACE_SECONDS
    return (
        _COMPONENT_WORKER_BUDGET_SECONDS
        + _RUNTIME_REPLAY_BUDGET_SECONDS
        + _WORKER_WAIT_GRACE_SECONDS
    )


def _read_job_status(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return dict(fallback)
    return value if isinstance(value, dict) else dict(fallback)


def _terminate_worker(process: Any) -> None:
    """Stop the isolated worker and its replay subprocesses after a hard timeout."""

    pid = getattr(process, "pid", None)
    if isinstance(pid, int) and pid > 0:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()
            process.wait()
            return
        try:
            process.wait(timeout=_WORKER_TERMINATION_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait()
            return
    process.kill()
    process.wait()


def _same_request_definition(existing: dict[str, Any], request: dict[str, Any]) -> bool:
    """Keep immutable job IDs bound to one mode and requested duration."""

    identity = ("id", "days", "mode")
    if not all(existing.get(key) == request.get(key) for key in identity):
        return False
    special = ("session_date", "profile", "expected_uncompressed_sha256")
    if any(key in existing or key in request for key in special):
        return all(
            existing.get(key) == request.get(key) for key in (*special, "start", "end")
        )
    # Historical in-progress research statuses did not store start/end.
    return True


def _launcher_mode() -> str:
    """Keep production research outside restartable application cgroups."""

    platform = (os.getenv("DEPLOYMENT_PLATFORM", "") or "").strip().lower()
    if platform == "aws_lightsail":
        return "systemd"
    return (os.getenv("BOT_RESEARCH_LAUNCHER", "direct") or "direct").strip().lower()


def _current_revision(root: Path) -> str:
    """Return the checked-out revision for research provenance, when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    value = (result.stdout or "").strip()
    if result.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{7,64}", value):
        return value
    return "unknown"


def _systemd_unit_state() -> str | None:
    """Return transient worker state; None means launcher state is unavailable."""

    try:
        result = subprocess.run(
            [
                "systemctl",
                "show",
                _SYSTEMD_RESEARCH_UNIT,
                "--property=LoadState",
                "--property=ActiveState",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    properties = {}
    for line in (result.stdout or "").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key.strip()] = value.strip()
    if properties.get("LoadState") == "not-found":
        return "inactive"
    state = properties.get("ActiveState")
    return state or None


def _configured_epoch_data_dir(root: Path, env_file: str) -> Path:
    """Pass only the operator's canonical DATA_DIR, never broker credentials."""
    configured = os.environ.get("DATA_DIR", "").strip()
    protected_file = Path(env_file)
    if protected_file.is_file():
        try:
            with protected_file.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line.startswith("DATA_DIR="):
                        configured = line.partition("=")[2].strip()
        except OSError as exc:
            raise ValueError("epoch_audit_data_dir_unreadable") from exc
    if configured.startswith(("'", '"')) and configured[-1:] == configured[0]:
        configured = configured[1:-1]
    if any(ch in configured for ch in ("\x00", "\r", "\n", "$", "`")):
        raise ValueError("epoch_audit_data_dir_invalid")
    if not configured:
        return (root / "data").resolve()
    location = Path(configured).expanduser()
    if not location.is_absolute():
        raise ValueError("epoch_audit_data_dir_not_absolute")
    return location.resolve()


def _launch_systemd_worker(
    root: Path,
    request: dict[str, Any],
    *,
    interpreter: Path,
    env_file: str,
) -> None:
    """Launch the fixed worker in an independent transient systemd service."""

    timeout = _worker_wait_timeout_seconds(request)
    extra_env = (
        [
            f"--setenv=DATA_DIR={_configured_epoch_data_dir(root, env_file)}"
        ]
        if request.get("mode") == "epoch_audit"
        else []
    )
    command = [
        "sudo",
        "-n",
        "systemd-run",
        f"--unit={_SYSTEMD_RESEARCH_UNIT}",
        "--collect",
        "--service-type=exec",
        f"--uid={os.getuid()}",
        f"--gid={os.getgid()}",
        f"--working-directory={root}",
        "--nice=10",
        f"--property=RuntimeMaxSec={timeout}s",
        "--property=CPUWeight=5",
        "--property=IOWeight=10",
        "--property=OOMScoreAdjust=500",
        "--property=KillMode=control-group",
        "--setenv=ENABLE_LIVE=false",
        "--setenv=ENABLE_LIVE_TRADING=false",
        "--setenv=ORDERS__ENABLE_LIVE=false",
        "--setenv=EXECUTION_MODE=SHADOW",
        "--setenv=PAPER_MODE=true",
        "--setenv=SHADOW_MODE=true",
        "--setenv=SUPABASE_TRADE_REPLICATION_ENABLED=false",
        "--setenv=SUPABASE_LOG_ARCHIVE_ENABLED=false",
        f"--setenv=PYTHONPATH={root / 'src'}:{root}",
        *extra_env,
        str(interpreter),
        str(root / "scripts/run_research_job.py"),
        "--worker",
        "--request-id",
        str(request["id"]),
        "--env-file",
        env_file,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OSError("research_systemd_launch_failed") from exc
    if result.returncode != 0:
        raise OSError("research_systemd_launch_failed")


def read_status(root: Path) -> dict[str, Any]:
    try:
        return json.loads((root / "data/research/latest.json").read_text())
    except (OSError, ValueError):
        return {"state": "not_requested", "backtest_completed": False}


def start_job(
    root: Path, payload: dict[str, Any], *, wait: bool = False
) -> dict[str, Any]:
    """Launch one bounded worker; detached production work is systemd-owned."""
    import fcntl

    request = validate_request(payload)
    env_file = os.getenv("BOT_ENV_FILE", "/home/ubuntu/.config/niftybot/niftybot.env")
    # Resolve operator provenance before taking the single-flight worker lock.
    original_data_dir = (
        _configured_epoch_data_dir(root, env_file)
        if request.get("mode") == "epoch_audit"
        else None
    )
    directory = root / "data/research"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "worker.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return {"state": "busy", "backtest_completed": False}

    launcher = _launcher_mode()
    status_file = directory / request["id"] / "status.json"
    existing: dict[str, Any] = {}
    if status_file.exists():
        existing = _read_job_status(status_file, {})
        if not _same_request_definition(existing, request):
            lock.close()
            raise ValueError("research_request_id_conflict")
        state = str(existing.get("state") or "")
        if state not in {"queued", "collecting"}:
            lock.close()
            return existing
        if launcher == "systemd" and not wait:
            unit_state = _systemd_unit_state()
            if unit_state is None:
                failed = {
                    **existing,
                    "state": "failed",
                    "error_type": "ResearchLauncherUnavailable",
                    "error_code": "research_launcher_state_unavailable",
                    "requested_work_completed": False,
                }
                write_json(status_file, failed)
                write_json(directory / "latest.json", failed)
                lock.close()
                return failed
            if unit_state in _SYSTEMD_BUSY_STATES:
                lock.close()
                return existing
        # Acquiring the launcher lock while the owning worker is absent proves
        # the previous nonterminal request is stale. Systemd mode additionally
        # verifies that the independent transient worker is no longer active.
        stale = {
            **existing,
            "state": "failed",
            "error_type": "WorkerExitedWithoutResult",
            "error_code": "stale_research_worker",
            "stale_state": state,
            "recovery_reason": (
                "nonterminal_status_without_live_worker"
                if launcher == "systemd" and not wait
                else "nonterminal_status_without_worker_lock"
            ),
            "requested_work_completed": False,
        }
        write_json(status_file, stale)
        write_json(directory / "latest.json", stale)
        lock.close()
        return stale

    if launcher not in {"direct", "systemd"}:
        failed = {
            **request,
            "state": "failed",
            "backtest_completed": False,
            "component_backtest_completed": False,
            "runtime_replay_completed": False,
            "requested_work_completed": False,
            "launch_attempt": 0,
            "error_type": "ResearchLauncherInvalid",
            "error_code": "research_launcher_invalid",
        }
        write_json(status_file, failed)
        write_json(directory / "latest.json", failed)
        lock.close()
        return failed

    if launcher == "systemd" and not wait:
        unit_state = _systemd_unit_state()
        if unit_state is None:
            failed = {
                **request,
                "state": "failed",
                "backtest_completed": False,
                "component_backtest_completed": False,
                "runtime_replay_completed": False,
                "requested_work_completed": False,
                "launch_attempt": 0,
                "launcher": "systemd",
                "worker_unit": _SYSTEMD_RESEARCH_UNIT,
                "error_type": "ResearchLauncherUnavailable",
                "error_code": "research_launcher_state_unavailable",
            }
            write_json(status_file, failed)
            write_json(directory / "latest.json", failed)
            lock.close()
            return failed
        if unit_state in _SYSTEMD_BUSY_STATES:
            lock.close()
            return {
                "state": "busy",
                "backtest_completed": False,
                "worker_unit": _SYSTEMD_RESEARCH_UNIT,
            }

    env = dict(os.environ)
    env.update(
        ENABLE_LIVE="false",
        ENABLE_LIVE_TRADING="false",
        ORDERS__ENABLE_LIVE="false",
        EXECUTION_MODE="SHADOW",
        PAPER_MODE="true",
        SHADOW_MODE="true",
        SUPABASE_TRADE_REPLICATION_ENABLED="false",
        SUPABASE_LOG_ARCHIVE_ENABLED="false",
        PYTHONPATH=str(root / "src"),
    )
    interpreter = root / ".venv/bin/python"
    if not interpreter.is_file():
        interpreter = Path(sys.executable)
    if original_data_dir is not None:
        env["DATA_DIR"] = str(original_data_dir)
    queued = {
        **request,
        "state": "queued",
        "backtest_completed": False,
        "component_backtest_completed": False,
        "runtime_replay_completed": False,
        "requested_work_completed": False,
        "launch_attempt": 1,
        "launcher": "direct" if wait else launcher,
    }
    if launcher == "systemd" and not wait:
        queued["launch_revision"] = _current_revision(root)
        queued["worker_unit"] = _SYSTEMD_RESEARCH_UNIT
    write_json(status_file, queued)
    write_json(directory / "latest.json", queued)

    if launcher == "systemd" and not wait:
        try:
            _launch_systemd_worker(
                root,
                request,
                interpreter=interpreter,
                env_file=env_file,
            )
        except (OSError, ValueError) as exc:
            queued.update(
                state="failed",
                error_type=type(exc).__name__,
                error_code=(
                    "epoch_audit_data_dir_invalid"
                    if isinstance(exc, ValueError)
                    else "research_systemd_launch_failed"
                ),
            )
            write_json(status_file, queued)
            write_json(directory / "latest.json", queued)
        finally:
            lock.close()
        return queued

    try:
        with (directory / request["id"] / "worker.log").open("a") as log:
            process = subprocess.Popen(
                [
                    str(interpreter),
                    str(root / "scripts/run_research_job.py"),
                    "--worker",
                    "--request-id",
                    request["id"],
                    "--env-file",
                    env_file,
                ],
                cwd=root,
                env=env,
                pass_fds=(lock.fileno(),),
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            if wait:
                # A systemd oneshot kills remaining children when it exits.
                # Keep that parent alive; direct/manual callers remain bounded.
                wait_timeout = _worker_wait_timeout_seconds(request)
                try:
                    process.wait(timeout=wait_timeout)
                except subprocess.TimeoutExpired:
                    _terminate_worker(process)
                    queued = _read_job_status(status_file, queued)
                    queued.update(
                        state="failed",
                        error_type="WorkerTimeout",
                        timed_out_stage=queued.get("stage"),
                        worker_timeout_seconds=wait_timeout,
                    )
                    write_json(status_file, queued)
                    write_json(directory / "latest.json", queued)
                else:
                    queued = _read_job_status(status_file, queued)
                    if queued.get("state") in {"queued", "collecting"}:
                        queued.update(
                            state="failed",
                            error_type="WorkerExitedWithoutResult",
                        )
                        write_json(status_file, queued)
                        write_json(directory / "latest.json", queued)
    except OSError as exc:
        queued.update(state="failed", error_type=type(exc).__name__)
        write_json(status_file, queued)
        write_json(directory / "latest.json", queued)
    finally:
        # Direct child holds the descriptor until it exits, including HTTP timeouts.
        lock.close()
    return queued


def new_request(days: int = 30, mode: str = "all") -> dict[str, Any]:
    return {"id": f"dashboard-{uuid.uuid4().hex}", "days": days, "mode": mode}


def partition_replay_runs(path: Path, destination: Path) -> list[tuple[str, Path]]:
    """Validate a daily archive and isolate process epochs without restoring state."""
    from nifty_scalper_bot.storage.replay_archive import (
        archive_fingerprint,
        iter_session,
    )

    run_ids: list[str] = []
    prior_run: str | None = None
    for event in iter_session(path):
        run_id = event["run_id"]
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("replay_run_id_invalid")
        if run_id != prior_run:
            if run_id in run_ids:
                raise ValueError("replay_epoch_noncontiguous")
            run_ids.append(run_id)
            if len(run_ids) > 16:
                raise ValueError("replay_epoch_count_exceeded")
            prior_run = run_id
    if len(run_ids) == 1:
        return [(run_ids[0], path)]

    source_sha = archive_fingerprint(path)
    shards = {
        run_id: destination / f"epoch-{index:02d}" / path.name.removesuffix(".gz")
        for index, run_id in enumerate(run_ids, 1)
    }
    try:
        for event in iter_session(path):
            shard = shards[event["run_id"]]
            shard.parent.mkdir(parents=True, exist_ok=True)
            with shard.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n"
                )

        for run_id, shard in shards.items():
            shutil.copyfile(
                path.parent / f"{run_id}.integrity.json",
                shard.parent / f"{run_id}.integrity.json",
            )
            # Strict capture integrity is checked here; broker positions,
            # warm-up and starting state are validated independently by each
            # child replay. An unusable restart cannot mask other valid runs.
            sum(1 for _ in iter_session(shard))
        if archive_fingerprint(path) != source_sha:
            raise ValueError("replay_archive_changed_during_partition")
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return list(shards.items())


def _run_captured_baseline_stress(
    root: Path,
    epoch_path: Path,
    output: Path,
    env: dict[str, str],
    interpreter: Path,
    deadline: float,
) -> dict[str, Any]:
    """Replay the captured configuration unchanged at three execution stresses.

    This never toggles live trailing/strategy settings. Each scenario runs the
    normal isolated production-composition replay, not the paired ablation.
    """
    from nifty_scalper_bot.storage.replay_archive import archive_fingerprint

    source_sha = archive_fingerprint(epoch_path)
    result: dict[str, Any] = {
        "profile": "captured_baseline",
        "state": "failed",
        "evidence_label": "RESEARCH_CANDIDATE",
        "archive_sha256": source_sha,
        "promotion_eligible": False,
        "slippage_stress": [],
    }
    for bps in (10, 25, 50):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            result["error_code"] = "replay_runtime_budget_exhausted"
            break
        directory = output / f"captured-baseline-{bps}bps"
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            str(interpreter),
            str(root / "scripts/run_runtime_replay.py"),
            "--session",
            str(epoch_path.resolve()),
            "--output",
            str(directory.resolve()),
            "--slippage-bps",
            str(bps),
        ]
        # No --compare-adaptive and no ADAPTIVE_PROFIT_LOCK_ENABLED override:
        # the original archived configuration selects the live policy.
        try:
            with (directory / "worker.log").open("w") as log:
                worker = subprocess.run(
                    command,
                    cwd=directory,
                    env=env,
                    stdout=log,
                    stderr=log,
                    timeout=min(600, remaining),
                    check=False,
                )
        except subprocess.TimeoutExpired:
            result["error_code"] = "replay_timeout"
            break
        except OSError:
            result["error_code"] = "replay_worker_unavailable"
            break
        try:
            scenario = json.loads(
                (directory / "report.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            scenario = None
        raw_bps = (
            scenario.get("slippage_bps_per_side")
            if isinstance(scenario, dict)
            else None
        )
        if isinstance(raw_bps, (str, int, float)) and not isinstance(raw_bps, bool):
            try:
                recorded_bps = float(raw_bps)
            except ValueError:
                recorded_bps = -1.0
        else:
            recorded_bps = -1.0
        if (
            worker.returncode != 0
            or not isinstance(scenario, dict)
            or scenario.get("archive_sha256") != source_sha
            or recorded_bps != bps
        ):
            try:
                failure = json.loads(
                    (directory / "failure.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                failure = {}
            error = failure.get("error_code") if isinstance(failure, dict) else None
            result["error_code"] = (
                error
                if isinstance(error, str)
                and re.fullmatch(r"replay_[a-z0-9_]{1,80}", error)
                else "replay_baseline_scenario_failed"
            )
            result["failed_stress_bps"] = bps
            break
        orders = scenario.get("orders")
        account = scenario.get("modeled_account")
        positions = scenario.get("positions")
        if (
            not isinstance(orders, list)
            or not isinstance(account, dict)
            or not isinstance(positions, list)
        ):
            result["error_code"] = "replay_baseline_report_invalid"
            break
        result["slippage_stress"].append(
            {
                "slippage_bps_per_side": bps,
                "archive_sha256": source_sha,
                "captured_revision": scenario.get("captured_revision"),
                "order_count": len(orders),
                "open_position_count": len(positions),
                "after_cost_modeled_equity_pnl": account.get("net_equity_pnl"),
                "modeled_fees": account.get("fees"),
                "max_equity_drawdown": scenario.get("max_equity_drawdown"),
                "risk_stops_and_fills_are_modeled": True,
                "exchange_stop_execution_proven": False,
                "result_file": f"captured-baseline-{bps}bps/report.json",
            }
        )
    if (
        len(result["slippage_stress"]) == 3
        and archive_fingerprint(epoch_path) == source_sha
    ):
        result["state"] = "completed"
    elif result.get("error_code") is None:
        result["error_code"] = "replay_archive_changed_during_stress"
    return result


def run_recorded_replays(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    """Replay each complete captured session in its own credential-free process."""
    from nifty_scalper_bot.config.paths import get_data_dir

    source = get_data_dir() / "replay_archive"
    output = root / "data/research" / request["id"] / "runtime"
    report: dict[str, Any] = {
        "scope": "production_composition_recorded_feed_replay",
        "live_equivalent": False,
        "sessions": [],
        "missing_dates": [],
        "requested_start": request["start"],
        "requested_end": request["end"],
    }
    cursor = date.fromisoformat(request["start"])
    end = date.fromisoformat(request["end"])
    report["budget_seconds"] = _RUNTIME_REPLAY_BUDGET_SECONDS
    report["budget_exhausted"] = False
    deadline = time.monotonic() + _RUNTIME_REPLAY_BUDGET_SECONDS
    while cursor <= end:
        day = cursor.isoformat()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            report["budget_exhausted"] = True
            report["deferred_from"] = day
            break
        path = source / f"{day}.jsonl"
        if not path.is_file():
            path = source / f"{day}.jsonl.gz"
        if not path.is_file():
            # Calendar dates; holidays are not assumed to be missing trading data.
            report["missing_dates"].append(day)
            cursor += timedelta(days=1)
            continue
        directory = output / day
        directory.mkdir(parents=True, exist_ok=True)
        scratch = directory / "_epochs"
        try:
            epochs = partition_replay_runs(path, scratch)
        except (OSError, ValueError) as exc:
            code = str(exc)
            report["sessions"].append(
                {
                    "date": day,
                    "state": "failed",
                    "error_code": (
                        code
                        if code.startswith("replay_") and len(code) <= 100
                        else "replay_archive_unusable"
                    ),
                }
            )
            cursor += timedelta(days=1)
            continue
        try:
            for epoch_index, (run_id, epoch_path) in enumerate(epochs, 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    report["budget_exhausted"] = True
                    report["deferred_from"] = day
                    report["deferred_epoch"] = epoch_index
                    break
                run_directory = (
                    directory
                    if len(epochs) == 1
                    else directory / f"result-epoch-{epoch_index:02d}"
                )
                run_directory.mkdir(parents=True, exist_ok=True)
                # No operator credentials or live broker access cross this boundary.
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if key in {"PATH", "LANG", "LC_ALL", "TZ", "VIRTUAL_ENV"}
                }
                env.update(
                    REPLAY_ISOLATED_PROCESS="true",
                    EXECUTION_MODE="LIVE_SIMULATION",
                    ENABLE_LIVE="false",
                    ENABLE_LIVE_TRADING="false",
                    BROKER_API_KEY="offline_replay",
                    BROKER_API_SECRET="offline_replay",
                    BROKER_ACCESS_TOKEN="offline_replay",
                    ALLOW_NETWORK="false",
                    ALLOW_REAL_BROKER="false",
                    DATA_DIR=str(run_directory.resolve()),
                    PYTHONPATH=str(root / "src"),
                )
                interpreter = root / ".venv/bin/python"
                if not interpreter.is_file():
                    interpreter = Path(sys.executable)
                if request.get("profile") == "captured_baseline":
                    baseline = _run_captured_baseline_stress(
                        root,
                        epoch_path,
                        run_directory,
                        env,
                        interpreter,
                        deadline,
                    )
                    report["sessions"].append(
                        {"date": day, "run_id": run_id, **baseline}
                    )
                    continue
                with (run_directory / "worker.log").open("w") as log:
                    try:
                        result = subprocess.run(
                            [
                                str(interpreter),
                                str(root / "scripts/run_runtime_replay.py"),
                                "--session",
                                str(epoch_path.resolve()),
                                "--output",
                                str(run_directory.resolve()),
                                "--compare-adaptive",
                                "--stress-bps",
                                "10",
                                "25",
                                "50",
                            ],
                            cwd=run_directory,
                            env=env,
                            stdout=log,
                            stderr=log,
                            timeout=min(600, remaining),
                        )
                    except subprocess.TimeoutExpired:
                        report["sessions"].append(
                            {
                                "date": day,
                                "run_id": run_id,
                                "state": "failed",
                                "error_code": "replay_timeout",
                            }
                        )
                    else:
                        evidence = run_directory / "paired-comparison.json"
                        if evidence.is_file():
                            try:
                                payload = json.loads(evidence.read_text())
                            except (OSError, ValueError):
                                payload = {}
                            success = (
                                result.returncode == 0
                                and payload.get("state") == "MODELED_COMPARISON_ONLY"
                            )
                            report["sessions"].append(
                                {
                                    "date": day,
                                    "run_id": run_id,
                                    "state": "completed" if success else "failed",
                                    "comparison_state": payload.get("state"),
                                    "error_code": payload.get("blocker")
                                    or payload.get("error_code"),
                                    "worker_error_code": payload.get(
                                        "worker_error_code"
                                    ),
                                    "failed_scenario": payload.get("failed_scenario"),
                                    "stress": payload.get("stress", []),
                                    "promotion_eligible": False,
                                }
                            )
                        else:
                            report["sessions"].append(
                                {
                                    "date": day,
                                    "run_id": run_id,
                                    "state": "failed",
                                    "error_code": "replay_worker_exited_without_report",
                                }
                            )
        finally:
            if scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)
        if report["budget_exhausted"]:
            break
        cursor += timedelta(days=1)
    completed = sum(row["state"] == "completed" for row in report["sessions"])
    failed = sum(row["state"] == "failed" for row in report["sessions"])
    report["completed_sessions"] = completed
    report["failed_sessions"] = failed
    report["available_sessions_fully_processed"] = bool(report["sessions"]) and (
        not report["budget_exhausted"] and failed == 0
    )
    report["state"] = (
        "completed"
        if completed and report["available_sessions_fully_processed"]
        else "partial" if completed else "blocked"
    )
    report["full_requested_period_covered"] = (
        False  # calendar/quote/initial-state parity remains unverified
    )
    if not report["sessions"]:
        report["blocker"] = "recorded_live_feed_unavailable"
    write_json(root / "data/research" / request["id"] / "runtime_replay.json", report)
    return report
