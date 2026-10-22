"""Bounded system-control helpers for low-memory Lightsail recovery.

These helpers intentionally expose only a tiny allow-list of operations needed by
remote operator surfaces: restart the trading engine and restart the lightweight
console. Instance shutdown is intentionally not performed from the web process.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

_SERVICE = re.compile(r"^[A-Za-z0-9_.@:-]+$")


@dataclass(frozen=True, slots=True)
class ControlResult:
    ok: bool
    action: str
    message: str
    command: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["command"] = list(self.command)
        return data


def _valid_service_name(service: str) -> bool:
    return bool(service and _SERVICE.fullmatch(service))


def restart_service(service: str, *, action: str) -> ControlResult:
    """Confirm systemd accepted the queued restart, not merely that sudo spawned."""
    if not _valid_service_name(service):
        return ControlResult(False, action, f"invalid systemd service name: {service!r}")
    command = ("sudo", "-n", "systemctl", "restart", "--no-block", service)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
    except Exception as exc:  # noqa: BLE001 - operator boundary
        return ControlResult(
            False, action, f"restart failed: {type(exc).__name__}", command
        )
    if result.returncode != 0:
        # Do not forward sudo error output, which may contain host-local paths.
        return ControlResult(
            False,
            action,
            f"systemd rejected restart (exit {result.returncode})",
            command,
        )
    return ControlResult(
        True, action, f"restart accepted by systemd for {service}", command
    )


def restart_bot() -> ControlResult:
    return restart_service(os.getenv("BOT_SERVICE_NAME", "niftybot"), action="restart_bot")


def restart_deployer() -> ControlResult:
    """Restart only the fixed validated deployment service."""
    return restart_service(
        os.getenv("BOT_DEPLOY_SERVICE_NAME", "niftybot-autodeploy.service"),
        action="restart_deployer",
    )


def queue_validated_restart() -> ControlResult:
    """Queue an engine restart only through the validated release runner.

    LIVE new entries are held during the broker-flat release handoff.
    Pending orders, open positions or unavailable broker proof defer the
    restart and are automatically rechecked on the next release timer.
    """
    action = "restart_bot"
    config = Path(os.getenv("NIFTYBOT_CONFIG_DIR", "/home/ubuntu/.config/niftybot"))
    marker = config / "auto_release_restart_pending"
    if not config.is_dir() or config.is_symlink():
        return ControlResult(False, action, "restart marker directory unavailable")
    descriptor = None
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return ControlResult(False, action, "restart marker is not a regular file")
        os.fchmod(descriptor, 0o600)
    except OSError:
        return ControlResult(False, action, "restart marker could not be written")
    finally:
        if descriptor is not None:
            os.close(descriptor)

    triggered = restart_deployer()
    if not triggered.ok:
        return ControlResult(
            False,
            action,
            f"restart marker saved; deployment trigger failed: {triggered.message}",
            triggered.command,
        )
    return ControlResult(
        True,
        action,
        "Validated restart queued; engine will restart only after confirmed "
        "broker-flat and terminal-order handoff",
        triggered.command,
    )


def restart_console() -> ControlResult:
    return restart_service(
        os.getenv("BOT_STREAMLIT_SERVICE_NAME", "niftybot-streamlit"),
        action="restart_console",
    )


def instance_stop_disabled() -> ControlResult:
    return ControlResult(
        False,
        "stop_instance",
        "instance stop is not exposed from the bot UI; use Lightsail/AWS console or SSH for host power actions",
    )


def memory_snapshot(meminfo_path: str | os.PathLike[str] = "/proc/meminfo") -> dict[str, Any]:
    data: dict[str, int] = {}
    try:
        for line in Path(meminfo_path).read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            parts = raw.strip().split()
            if parts and parts[0].isdigit():
                data[key] = int(parts[0])
    except OSError:
        return {"available": False}
    total = data.get("MemTotal", 0)
    available = data.get("MemAvailable", 0)
    swap_total = data.get("SwapTotal", 0)
    swap_free = data.get("SwapFree", 0)
    used = max(total - available, 0)
    swap_used = max(swap_total - swap_free, 0)
    return {
        "available": bool(total),
        "mem_total_mb": round(total / 1024, 1) if total else None,
        "mem_available_mb": round(available / 1024, 1) if available else None,
        "mem_used_pct": round((used / total) * 100, 1) if total else None,
        "swap_total_mb": round(swap_total / 1024, 1) if swap_total else 0,
        "swap_used_mb": round(swap_used / 1024, 1) if swap_total else 0,
    }


__all__ = [
    "ControlResult",
    "instance_stop_disabled",
    "memory_snapshot",
    "restart_bot",
    "queue_validated_restart",
    "restart_console",
    "restart_deployer",
    "restart_service",
]
