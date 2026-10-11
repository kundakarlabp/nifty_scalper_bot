"""Correct stale local P&L carryover at an IST trading-day boundary.

The dedicated broker P&L layer introduced account-level Zerodha M2M truth, but
its first version could stamp a stale persisted local realised P&L with today's
trading date without clearing it.  That made historical carryover look like
current-session P&L.

The native PositionManager keeps broker daily P&L authoritative for the current trading day and
resets only stale/unverified local session accounting.  It does not alter
position, orphan, protection, exit-lifecycle or readiness safety gates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

_EPS = 1e-6


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _broker_proves_no_current_day_trading(snapshot: Mapping[str, Any]) -> bool:
    """Return True only when broker evidence supports a clean zero-trade day."""

    if snapshot.get("positions_error"):
        return False
    realized = _finite_float(snapshot.get("account_realized"))
    unrealized = _finite_float(snapshot.get("account_unrealized"))
    if realized is None:
        return False
    if abs(realized) > _EPS:
        return False
    if unrealized is not None and abs(unrealized) > _EPS:
        return False
    try:
        day_rows = int(snapshot.get("strategy_day_rows", 0) or 0)
    except (TypeError, ValueError):
        return False
    return day_rows == 0


def _reset_local_session_state(self: Any, *, today: str, reason: str) -> bool:
    """Clear stale session-only P&L while retaining durable order/trade history."""

    with getattr(self, "_lock"):
        stale_confirmed = float(getattr(self, "_local_realized_pnl", 0.0) or 0.0)
        stale_provisional = float(
            getattr(self, "_local_provisional_realized_pnl", 0.0) or 0.0
        )
        if abs(stale_confirmed) <= _EPS and abs(stale_provisional) <= _EPS:
            return False

        self._local_realized_pnl = 0.0
        self._local_provisional_realized_pnl = 0.0
        for position in getattr(self, "_positions", {}).values():
            if hasattr(position, "realized_pnl"):
                position.realized_pnl = 0.0

        self._pnl_trading_date = today
        self._session_opening_realized_baseline = 0.0
        self._baseline_established_at = datetime.now(timezone.utc)
        self._baseline_source = "zerodha_margins_m2m"
        self._pnl_product_scope = "ACCOUNT_EQUITY_FNO"
        self._broker_realized_pnl = None
        self._refresh_realized_pnl_locked()
        self._pnl_authority = "local_confirmed_ledger"
        self._broker_account_pnl_snapshot = {}
        self._broker_pnl_last_fetch_mono = 0.0

    logger = getattr(self, "_logger", None)
    warning = getattr(logger, "warning", None)
    if callable(warning):
        warning(
            "PNL_SESSION_STALE_LOCAL_RESET stale_confirmed=%.2f stale_provisional=%.2f "
            "trading_date=%s reason=%s broker_authority=zerodha_margins_m2m",
            stale_confirmed,
            stale_provisional,
            today,
            reason,
            extra={
                "event": "PNL_SESSION_STALE_LOCAL_RESET",
                "stale_confirmed": stale_confirmed,
                "stale_provisional": stale_provisional,
                "trading_date": today,
                "reason": reason,
                "broker_authority": "zerodha_margins_m2m",
            },
        )
    try:
        self.save_state()
    except Exception:
        if callable(warning):
            warning("PNL_SESSION_RESET_PERSIST_FAILED", exc_info=True)
    return True


__all__ = [
    "_broker_proves_no_current_day_trading",
    "_reset_local_session_state",
]
