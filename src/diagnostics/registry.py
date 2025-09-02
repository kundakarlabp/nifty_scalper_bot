from __future__ import annotations

"""Simple registry for diagnostic self-tests.

This module exposes a :func:`run` helper returning a :class:`CheckResult` for a
named diagnostic. The current implementation provides placeholder checks so the
/why command can surface structured results along with optional fix hints.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class CheckResult:
    """Outcome of a diagnostic check."""

    name: str
    ok: bool
    msg: str
    fix: Optional[str] = None


def _stub(name: str) -> CheckResult:
    return CheckResult(name=name, ok=True, msg="ok")


_CHECKS: Dict[str, Callable[[], CheckResult]] = {
    "data_window": lambda: _stub("data_window"),
    "atr": lambda: _stub("atr"),
    "regime": lambda: _stub("regime"),
    "micro": lambda: _stub("micro"),
    "risk_gates": lambda: _stub("risk_gates"),
}


def run(name: str) -> CheckResult:
    """Run a diagnostic check by name."""

    func = _CHECKS.get(name)
    if not func:
        return CheckResult(name=name, ok=False, msg="unknown check")
    return func()
