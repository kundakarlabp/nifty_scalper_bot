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


def _popen(command: list[str]) -> None:
    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def restart_service(service: str, *, action: str) -> ControlResult:
    if not _valid_service_name(service):
        return ControlResult(False, action, f"invalid systemd service name: {service!r}")
    command = ("sudo", "systemctl", "restart", "--no-block", service)
    try:
        _popen(list(command))
    except Exception as exc:  # noqa: BLE001 - operator boundary
        return ControlResult(False, action, f"restart failed: {type(exc).__name__}: {exc}", command)
    return ControlResult(True, action, f"restart requested for {service}", command)


def restart_bot() -> ControlResult:
    return restart_service(os.getenv("BOT_SERVICE_NAME", "niftybot"), action="restart_bot")


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
    "restart_console",
    "restart_service",
]
