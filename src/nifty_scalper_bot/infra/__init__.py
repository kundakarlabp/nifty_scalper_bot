"""Infrastructure helpers (health endpoints, logging, scheduled tasks)."""

from importlib import import_module
from typing import Any

from nifty_scalper_bot.infra.scheduled_tasks import (
    run_periodic_task,
    start_background_tasks,
)

from .structured_logger import setup_structured_logging

__all__ = [
    "HealthState",
    "create_health_app",
    "setup_structured_logging",
    "run_periodic_task",
    "start_background_tasks",
]


def __getattr__(name: str) -> Any:
    """Lazily expose health module attributes to avoid import cycles."""

    if name in {"HealthState", "create_health_app"}:
        module = import_module("nifty_scalper_bot.infra.health")
        return getattr(module, name)
    raise AttributeError(f"module 'nifty_scalper_bot.infra' has no attribute {name!r}")
