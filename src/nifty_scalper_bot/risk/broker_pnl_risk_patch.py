"""Drive account-level daily P&L risk from dedicated broker authority.

The strategy fill ledger remains PositionManager's local accounting source.  When
Zerodha's account M2M P&L is fresh, RiskManager uses that broker value for the
daily capital-loss/profit circuit.  If broker P&L is unavailable, the existing
local-ledger logic is retained unchanged.
"""

from __future__ import annotations

import math
from contextlib import suppress
from typing import Any

_PATCH_APPLIED = False
_ORIGINAL_SEED_DAY_PNL: Any = None
_ORIGINAL_REFRESH_REALIZED_PNL: Any = None
_ORIGINAL_RESET_DAILY: Any = None


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _resolve_broker_realized_pnl(
    position_manager: Any,
    *,
    force: bool = False,
) -> float | None:
    """Return fresh broker account realised P&L when that capability exists."""

    refresh = getattr(position_manager, "refresh_broker_pnl_diagnostic", None)
    if force and callable(refresh):
        with suppress(Exception):
            refresh(force=True)

    getter = getattr(position_manager, "get_broker_account_realized_pnl", None)
    if not callable(getter):
        return None
    try:
        value = getter(force=False)
    except TypeError:
        try:
            value = getter()
        except Exception:
            return None
    except Exception:
        return None
    return _finite_float(value)


def _record_broker_pnl_metrics(owner: Any, value: float) -> None:
    """Mirror the canonical P&L metrics without changing risk behavior."""

    try:
        from nifty_scalper_bot.risk import risk_manager as risk_module

        risk_module.METRICS.set_live_pnl(book="primary", value=float(value))
        risk_module.METRICS.set_pnl_breakdown(
            book="primary", realized=float(value)
        )
    except Exception:
        logger = getattr(owner, "_logger", None)
        exception = getattr(logger, "exception", None)
        if callable(exception):
            exception("Broker P&L metric update failed", exc_info=True)


def _check_breach(owner: Any) -> None:
    reason = owner._switches.breach_reason()
    if reason:
        owner._trip_breaker(owner._format_switch_reason(reason))


def _patched_seed_day_pnl_from_persisted_state(self: Any) -> None:
    """Seed the daily circuit from broker account P&L, with local fallback."""

    broker_realized = _resolve_broker_realized_pnl(
        self.position_manager,
        force=True,
    )
    if broker_realized is None:
        _ORIGINAL_SEED_DAY_PNL(self)
        return

    current = float(broker_realized)
    self._last_pnl_snapshot = current
    if abs(current) >= 1e-6:
        self._switches.record_pnl(current)

    logger = getattr(self, "_logger", None)
    warning = getattr(logger, "warning", None)
    if callable(warning):
        warning(
            "DAY_PNL_SEEDED_FROM_BROKER realized=%.2f day_loss=%.2f source=zerodha_margins_m2m",
            current,
            self._switches.day_loss(),
            extra={
                "event": "DAY_PNL_SEEDED_FROM_BROKER",
                "realized": current,
                "day_loss": self._switches.day_loss(),
                "source": "zerodha_margins_m2m",
            },
        )
    _record_broker_pnl_metrics(self, current)
    _check_breach(self)


def _patched_refresh_realized_pnl(self: Any) -> None:
    """Refresh the risk circuit from broker account P&L when available."""

    broker_realized = _resolve_broker_realized_pnl(
        self.position_manager,
        force=False,
    )
    if broker_realized is None:
        _ORIGINAL_REFRESH_REALIZED_PNL(self)
        return

    current = float(broker_realized)
    previous = float(getattr(self, "_last_pnl_snapshot", 0.0) or 0.0)
    delta = current - previous
    if abs(delta) < 1e-6:
        return

    # Broker M2M is external capital truth.  Do not apply the historical
    # "zombie local restore" suppression to it; explicit risk soft-override is
    # still honored by RiskManager._trip_breaker.
    self._switches.record_pnl(delta)
    self._last_pnl_snapshot = current
    _record_broker_pnl_metrics(self, current)

    logger = getattr(self, "_logger", None)
    info = getattr(logger, "info", None)
    if callable(info):
        info(
            "DAY_PNL_REFRESHED_FROM_BROKER realized=%.2f delta=%.2f day_loss=%.2f",
            current,
            delta,
            self._switches.day_loss(),
            extra={
                "event": "DAY_PNL_REFRESHED_FROM_BROKER",
                "realized": current,
                "delta": delta,
                "day_loss": self._switches.day_loss(),
                "source": "zerodha_margins_m2m",
            },
        )
    _check_breach(self)


def _patched_reset_daily_if_needed(self: Any) -> None:
    """Keep the post-reset delta baseline aligned with broker P&L authority."""

    before = getattr(self, "_trading_day_start", None)
    _ORIGINAL_RESET_DAILY(self)
    after = getattr(self, "_trading_day_start", None)
    if before == after:
        return

    broker_realized = _resolve_broker_realized_pnl(
        self.position_manager,
        force=True,
    )
    if broker_realized is None:
        return

    current = float(broker_realized)
    self._last_pnl_snapshot = current
    if abs(current) >= 1e-6:
        self._switches.record_pnl(current)
    _record_broker_pnl_metrics(self, current)
    _check_breach(self)


def apply_patches() -> None:
    """Install broker P&L risk authority while retaining local fallback."""

    global _PATCH_APPLIED
    global _ORIGINAL_SEED_DAY_PNL
    global _ORIGINAL_REFRESH_REALIZED_PNL
    global _ORIGINAL_RESET_DAILY

    if _PATCH_APPLIED:
        return

    from nifty_scalper_bot.risk.risk_manager import RiskManager

    if getattr(RiskManager, "_broker_pnl_risk_patch", False):
        _PATCH_APPLIED = True
        return

    _ORIGINAL_SEED_DAY_PNL = RiskManager._seed_day_pnl_from_persisted_state
    _ORIGINAL_REFRESH_REALIZED_PNL = RiskManager._refresh_realized_pnl
    _ORIGINAL_RESET_DAILY = RiskManager._reset_daily_if_needed

    RiskManager._seed_day_pnl_from_persisted_state = (
        _patched_seed_day_pnl_from_persisted_state
    )
    RiskManager._refresh_realized_pnl = _patched_refresh_realized_pnl
    RiskManager._reset_daily_if_needed = _patched_reset_daily_if_needed
    RiskManager._broker_pnl_risk_patch = True
    _PATCH_APPLIED = True


__all__ = [
    "apply_patches",
    "_resolve_broker_realized_pnl",
]
