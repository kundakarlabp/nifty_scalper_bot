"""Light-weight pre-trade risk gate evaluation utilities.

This module provides a light-weight risk check that can be used by
strategy runners to enforce per-trade and session level risk rules. It
is intentionally decoupled from the heavier :mod:`risk.limits` engine so
it can be unit-tested easily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AccountState:
    """Snapshot of account risk state used for gate evaluation."""

    equity_rupees: float
    dd_rupees: float
    max_daily_loss: float
    loss_streak: int


def evaluate(plan: dict, acct: AccountState, cfg: Any) -> tuple[bool, list[str]]:
    """Evaluate basic risk gates for a trade plan."""

    reasons: list[str] = []

    if acct.dd_rupees >= acct.max_daily_loss:
        reasons.append("daily_dd")

    if getattr(acct, "loss_streak", 0) >= int(
        getattr(getattr(cfg, "risk", object()), "max_consec_losses", 3)
    ):
        reasons.append("loss_streak")

    rrt = getattr(cfg, "rr_threshold", None)
    if rrt is not None and float(plan.get("rr", 0.0)) < float(rrt):
        reasons.append("rr_low")

    ok = len(reasons) == 0
    return ok, reasons


__all__ = ["AccountState", "evaluate"]
