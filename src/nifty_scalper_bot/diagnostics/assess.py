"""
Diagnostic assessment utilities.

Design principles:
- Checks may FAIL HARD or FAIL SOFT
- Soft failures must NOT block trading
- Hard failures must block trading
- No exceptions should escape this layer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple


# ============================
# Result Model
# ============================

@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    meta: Dict[str, Any] | None = None


# ============================
# Internal runner
# ============================

def _run_check(
    name: str,
    fn: Callable[[], Tuple[bool, str, Dict[str, Any] | None]],
) -> CheckResult:
    """
    Execute a single diagnostic check.

    Hard rules:
    - Exceptions are HARD failures
    - 'no_quote' is a SOFT failure (live-market normal)
    - Everything else respects `ok`
    """

    try:
        ok, detail, meta = fn()

        # ----------------------------
        # SOFT FAILURE NORMALIZATION
        # ----------------------------
        # no_quote is NORMAL in live markets
        if not ok and detail == "no_quote":
            return CheckResult(
                name=name,
                ok=True,  # allow trading
                detail="warn:no_quote",
                meta=meta or {},
            )

        # Future-proof: allow other soft warnings
        if detail.startswith("warn:"):
            return CheckResult(
                name=name,
                ok=True,
                detail=detail,
                meta=meta or {},
            )

        # Default behavior (hard pass / hard fail)
        return CheckResult(
            name=name,
            ok=bool(ok),
            detail=detail,
            meta=meta or {},
        )

    except Exception as exc:
        # HARD failure – diagnostics must never crash the bot
        return CheckResult(
            name=name,
            ok=False,
            detail=f"EXC:{exc!r}",
            meta={},
        )


# ============================
# Public API
# ============================

def assess_suite(
    checks: Iterable[
        Tuple[str, Callable[[], Tuple[bool, str, Dict[str, Any] | None]]]
    ],
) -> List[CheckResult]:
    """
    Run a suite of diagnostic checks.

    IMPORTANT:
    - Does NOT aggregate ok/fail
    - Caller decides policy (e.g. `all(result.ok)`)
    """

    results: List[CheckResult] = []

    for name, fn in checks:
        results.append(_run_check(name, fn))

    return results


__all__ = [
    "CheckResult",
    "assess_suite",
]
