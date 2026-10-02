"""Fixed, isolated research jobs; never import the live application or place orders.

GitHub's request manifest and the dashboard share one single-flight launcher.
Collection/ledger analysis is explicitly distinct from a full strategy backtest.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def validate_request(
    payload: dict[str, Any], *, today: date | None = None
) -> dict[str, Any]:
    """Accept bounded data requests only, never arbitrary commands or paths."""
    if set(payload) - {"id", "days"}:
        raise ValueError("Only id and days are supported")
    job_id = payload.get("id")
    days = payload.get("days", 30)
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", job_id):
        raise ValueError("Invalid request id")
    if type(days) is not int or not 1 <= days <= 90:
        raise ValueError("days must be an integer from 1 to 90")
    end = (today or datetime.now(IST).date()) - timedelta(days=1)
    return {
        "id": job_id,
        "days": days,
        "start": (end - timedelta(days=days - 1)).isoformat(),
        "end": end.isoformat(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def read_status(root: Path) -> dict[str, Any]:
    try:
        return json.loads((root / "data/research/latest.json").read_text())
    except (OSError, ValueError):
        return {"state": "not_requested", "backtest_completed": False}


def start_job(
    root: Path, payload: dict[str, Any], *, wait: bool = False
) -> dict[str, Any]:
    """Launch one detached worker; an OS lock rejects overlapping requests."""
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
    status_file = directory / request["id"] / "status.json"
    if status_file.exists():
        lock.close()
        return json.loads(status_file.read_text())
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
    queued = {**request, "state": "queued", "backtest_completed": False}
    write_json(status_file, queued)
    write_json(directory / "latest.json", queued)
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
                # Keep that parent alive; dashboard callers remain detached.
                try:
                    process.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    queued.update(state="failed", error_type="WorkerTimeout")
                    write_json(status_file, queued)
                    write_json(directory / "latest.json", queued)
                queued = json.loads(status_file.read_text())
                if queued["state"] in {"queued", "collecting"}:
                    queued.update(
                        state="failed", error_type="WorkerExitedWithoutResult"
                    )
                    write_json(status_file, queued)
                    write_json(directory / "latest.json", queued)
    except OSError as exc:
        queued.update(state="failed", error_type=type(exc).__name__)
        write_json(status_file, queued)
        write_json(directory / "latest.json", queued)
    finally:
        # Child holds the same descriptor until it exits, including HTTP timeouts.
        lock.close()
    return queued


def new_request(days: int = 30) -> dict[str, Any]:
    return {"id": f"dashboard-{uuid.uuid4().hex}", "days": days}
