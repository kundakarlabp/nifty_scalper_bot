"""Light-weight pre-trade risk gate evaluation utilities.

This module provides a light-weight risk check that can be used by
strategy runners to enforce per-trade and session level risk rules. It
is intentionally decoupled from the heavier :mod:`risk.limits` engine so
it can be unit-tested easily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Any


@dataclass
class AccountState:
    """Snapshot of account risk state used for gate evaluation."""

    equity_rupees: float
    dd_rupees: float
    max_daily_loss: float
    loss_streak: int


def evaluate(plan: dict, acct: AccountState, cfg: Any) -> Tuple[bool, List[str]]:
    """Evaluate basic risk gates for a trade plan.

    Parameters
    ----------
    plan:
        Mutable plan dictionary. ``risk_rupees`` is read from it and
        ``per_trade_allowed_rupees`` is written back for diagnostics.
    acct:
        Current :class:`AccountState` of the trading account.
    cfg:
        Strategy configuration object providing ``risk`` section and
        optional ``rr_threshold``.
    Returns
    -------
    ok, reasons:
        ``ok`` is ``True`` when no gates block the trade. ``reasons`` is a
        list of failing gate identifiers.
    """

    reasons: List[str] = []

    # Daily drawdown gate
    if acct.dd_rupees >= acct.max_daily_loss:
        reasons.append("daily_dd")

    # Loss streak gate
    max_losses = int(getattr(getattr(cfg, "risk", object()), "max_consec_losses", 3))
    if acct.loss_streak >= max_losses:
        reasons.append("loss_streak")

    # Optional risk-reward threshold gate
    rrt = getattr(cfg, "rr_threshold", None)
    if rrt is not None and float(plan.get("rr", 0.0)) < float(rrt):
        reasons.append("rr_low")

    # Per-trade percentage cap gate
    cap_pct = float(getattr(getattr(cfg, "risk", object()), "per_trade_pct_max", 0.6))
    allowed_rupees = max(0.0, acct.equity_rupees * cap_pct / 100.0)
    plan_risk_rupees = float(plan.get("risk_rupees", 0.0))
    plan["per_trade_allowed_rupees"] = round(allowed_rupees, 2)
    if plan_risk_rupees > allowed_rupees:
        reasons.append("per_trade_pct")

    ok = not reasons
    return ok, reasons


__all__ = ["AccountState", "evaluate"]
