"""Fixed, isolated research jobs; never import the live application or place orders.

GitHub's request manifest and the dashboard share one single-flight launcher.
Collection/ledger analysis is explicitly distinct from a full strategy backtest.
"""

from __future__ import annotations

import json
import os
import re
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

_COMPONENT_WORKER_BUDGET_SECONDS = 1800
_RUNTIME_REPLAY_BUDGET_SECONDS = 1500
_WORKER_WAIT_GRACE_SECONDS = 300
_WORKER_TERMINATION_GRACE_SECONDS = 10
_SYSTEMD_RESEARCH_UNIT = "niftybot-research-worker.service"
_SYSTEMD_BUSY_STATES = {"active", "activating", "reloading", "deactivating"}


def validate_request(
    payload: dict[str, Any], *, today: date | None = None
) -> dict[str, Any]:
    """Accept bounded data requests only, never arbitrary commands or paths."""
    if set(payload) - {"id", "days", "mode"}:
        raise ValueError("Only id, days and mode are supported")
    job_id = payload.get("id")
    days = payload.get("days", 30)
    mode = payload.get("mode", "all")
    if mode not in {"all", "components", "runtime"}:
        raise ValueError("mode must be all, components or runtime")
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", job_id):
        raise ValueError("Invalid request id")
    if type(days) is not int or not 1 <= days <= 90:
        raise ValueError("days must be an integer from 1 to 90")
    end = (today or datetime.now(IST).date()) - timedelta(days=1)
    return {
        "id": job_id,
        "days": days,
        "mode": mode,
        "start": (end - timedelta(days=days - 1)).isoformat(),
        "end": end.isoformat(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _worker_wait_timeout_seconds(request: dict[str, Any]) -> int:
    """Return a bounded parent wait budget aligned with requested research work."""

    mode = str(request.get("mode") or "all")
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

    return all(existing.get(key) == request.get(key) for key in ("id", "days", "mode"))


def _launcher_mode() -> str:
    """Return the configured research launcher without silently changing semantics."""

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


def _launch_systemd_worker(
    root: Path,
    request: dict[str, Any],
    *,
    interpreter: Path,
    env_file: str,
) -> None:
    """Launch the fixed worker in an independent transient systemd service."""

    timeout = _worker_wait_timeout_seconds(request)
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
    env_file = os.getenv("BOT_ENV_FILE", "/home/ubuntu/.config/niftybot/niftybot.env")
    queued = {
        **request,
        "state": "queued",
        "backtest_completed": False,
        "component_backtest_completed": False,
        "runtime_replay_completed": False,
        "requested_work_completed": False,
        "launch_attempt": 1,
        "launcher": "direct" if wait else launcher,
        "launch_revision": _current_revision(root),
    }
    if launcher == "systemd" and not wait:
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
        except OSError as exc:
            queued.update(
                state="failed",
                error_type=type(exc).__name__,
                error_code="research_systemd_launch_failed",
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
            # Calendar dates; holidays are not assumed to be missing trading data.
            report["missing_dates"].append(day)
            cursor += timedelta(days=1)
            continue
        directory = output / day
        directory.mkdir(parents=True, exist_ok=True)
        # No operator env file or inherited credential/path settings cross here.
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
            DATA_DIR=str(directory.resolve()),
            PYTHONPATH=str(root / "src"),
        )
        interpreter = root / ".venv/bin/python"
        if not interpreter.is_file():
            interpreter = Path(sys.executable)
        with (directory / "worker.log").open("w") as log:
            try:
                result = subprocess.run(
                    [
                        str(interpreter),
                        str(root / "scripts/run_runtime_replay.py"),
                        "--session",
                        str(path.resolve()),
                        "--output",
                        str(directory.resolve()),
                    ],
                    cwd=directory,
                    env=env,
                    stdout=log,
                    stderr=log,
                    timeout=min(600, remaining),
                )
            except subprocess.TimeoutExpired:
                report["sessions"].append(
                    {"date": day, "state": "failed", "error_code": "replay_timeout"}
                )
            else:
                evidence = directory / (
                    "report.json" if result.returncode == 0 else "failure.json"
                )
                if evidence.is_file():
                    payload = json.loads(evidence.read_text())
                    report["sessions"].append(
                        {
                            "date": day,
                            "state": (
                                "completed" if result.returncode == 0 else "failed"
                            ),
                            **{
                                key: value
                                for key, value in payload.items()
                                if key not in {"orders", "runner_status"}
                            },
                        }
                    )
                else:
                    report["sessions"].append(
                        {
                            "date": day,
                            "state": "failed",
                            "error_code": "replay_worker_exited_without_report",
                        }
                    )
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
