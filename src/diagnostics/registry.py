from __future__ import annotations

"""Registry and helpers for diagnostic self-tests.

This module provides a lightweight registration system so individual
components can expose health checks.  Results are represented by
:class:`CheckResult` dataclasses and can be serialized to JSON for
Telegram or log consumption.
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CheckResult:
    """Outcome of a diagnostic check."""

    name: str
    ok: bool
    msg: str
    details: dict[str, Any]
    fix: str | None = None
    took_ms: int = 0


_registry: dict[str, Callable[[], CheckResult]] = {}


def register(
    name: str,
) -> Callable[[Callable[[], CheckResult]], Callable[[], CheckResult]]:
    """Decorator to register a diagnostic check under ``name``."""

    def deco(fn: Callable[[], CheckResult]) -> Callable[[], CheckResult]:
        _registry[name] = fn
        return fn

    return deco


def run(name: str) -> CheckResult:
    """Run a diagnostic check by name."""

    fn = _registry.get(name)
    if not fn:
        return CheckResult(
            name=name,
            ok=False,
            msg="unknown check",
            details={},
            fix="use /selftest to list",
        )
    t0 = time.time()
    result = fn()
    result.took_ms = int((time.time() - t0) * 1000)
    return result


def run_all() -> list[CheckResult]:
    """Run all registered diagnostic checks and return their results."""

    return [run(n) for n in sorted(_registry)]


def to_json(results: list[CheckResult]) -> str:
    """Serialize a list of :class:`CheckResult` objects to JSON."""

    return json.dumps([asdict(r) for r in results], default=str, ensure_ascii=False)


__all__ = [
    "CheckResult",
    "register",
    "run",
    "run_all",
    "to_json",
]
