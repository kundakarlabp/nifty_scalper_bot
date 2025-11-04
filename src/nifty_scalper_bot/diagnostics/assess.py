"""Composable assessment helpers used by runtime and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple


@dataclass(slots=True)
class CheckResult:
    """Result of executing a single assessment check."""

    name: str
    ok: bool
    detail: str = ""
    meta: Dict[str, Any] | None = None


def _run_check(
    name: str, fn: Callable[[], Tuple[bool, str, Dict[str, Any] | None]]
) -> CheckResult:
    """Execute a check callable and normalize its response."""

    try:
        ok, detail, meta = fn()
        return CheckResult(name=name, ok=bool(ok), detail=detail, meta=meta or {})
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name=name, ok=False, detail=f"EXC: {exc!r}", meta={})


def assess_suite(
    checks: Sequence[Tuple[str, Callable[[], Tuple[bool, str, Dict[str, Any] | None]]]],
) -> List[CheckResult]:
    """Run a suite of assessment callables and collect results."""

    return [_run_check(name, fn) for (name, fn) in checks]


__all__ = ["CheckResult", "assess_suite"]
