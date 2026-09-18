"""Persist entry-risk state and keep PositionManager risk state restart-safe.

This patch stays deliberately narrow:
* persist the existing PositionManager daily-entry counter in its existing JSON
  state file so an intraday process restart cannot reset max_trades_per_day;
* after a stop-loss exit, temporarily block a new option entry for the same
  underlying and option side, including a strike change;
* when a validated broker position snapshot explicitly contains realised P&L,
  reconcile the local session ledger to that broker value so future local fill
  deltas continue from broker-confirmed truth.

Protective/reducing orders remain outside the entry-only guard.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any

from nifty_scalper_bot.execution.position_snapshot import decode_position_snapshot
from nifty_scalper_bot.utils.symbols import is_strategy_instrument

# Standalone "SL" token (SL Hit, HARD_SL_BREACH, FORCED_SL_EXIT, WATCHDOG_HARD_SL)
# or an explicit STOP LOSS / STOP_LOSS phrase. "SLIPPAGE" must not match.
_PATCH_APPLIED = False
_ORIGINAL_INIT: Any = None
_ORIGINAL_CLOSE_POSITION: Any = None
_ORIGINAL_REFRESH_REALIZED_PNL: Any = None
_ORIGINAL_SYNCHRONIZE_WITH_BROKER: Any = None
_RISK_KEY = "_risk_runtime"


def _option_thesis(symbol: object) -> tuple[str, str] | None:
    text = str(symbol or "").strip().upper()
    if ":" in text:
        text = text.split(":", 1)[1]
    option_side = text[-2:] if text.endswith(("CE", "PE")) else ""
    if not option_side:
        return None
    contract = text[:-2]
    digit_at = next((index for index, char in enumerate(contract) if char.isdigit()), -1)
    if digit_at <= 0:
        return None
    underlying = contract[:digit_at]
    if not underlying.isalpha():
        return None
    return underlying, option_side


def _cooldown_seconds() -> float:
    raw = os.getenv("STOP_LOSS_REENTRY_COOLDOWN_SECONDS", "300")
    with suppress(TypeError, ValueError):
        return max(0.0, float(raw or 0.0))
    return 300.0


def _risk_state_snapshot(owner: Any) -> dict[str, Any]:
    """Return the risk runtime fragment for the canonical atomic state writer."""
    stopped = getattr(owner, "_recent_stop_thesis", None)
    circuit = getattr(owner, "_risk_circuit_state", None)
    return {
        "trades_today_date": getattr(owner, "_trades_today_date", None),
        "trades_today_count": int(getattr(owner, "_trades_today_count", 0) or 0),
        "recent_stop_thesis": dict(stopped) if isinstance(stopped, dict) else None,
        "risk_circuit": dict(circuit) if isinstance(circuit, dict) else None,
    }


def _restore_risk_state(owner: Any, state: Any) -> None:
    """Hydrate risk runtime from the already-read canonical state document."""
    if not isinstance(state, Mapping):
        return
    today = owner._trading_date_ist()
    if state.get("trades_today_date") == today:
        with suppress(TypeError, ValueError):
            owner._trades_today_date = today
            owner._trades_today_count = max(
                0, int(state.get("trades_today_count", 0) or 0)
            )
    circuit = state.get("risk_circuit")
    if isinstance(circuit, Mapping) and str(circuit.get("trading_date") or "") == today:
        owner._risk_circuit_state = dict(circuit)
    stopped = state.get("recent_stop_thesis")
    if isinstance(stopped, Mapping):
        with suppress(TypeError, ValueError):
            expires_epoch = float(stopped.get("expires_epoch", 0.0) or 0.0)
            if expires_epoch > time.time():
                owner._recent_stop_thesis = dict(stopped)


def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
    # The canonical load wrapper runs inside native __init__, so initialize
    # these fields before delegating and let that single reader hydrate them.
    self._recent_stop_thesis = None
    self._risk_circuit_state = {}
    _ORIGINAL_INIT(self, *args, **kwargs)

def _materialize_broker_positions(payload: Any) -> Any:
    """Materialize one-shot iterables while preserving broker mapping payloads."""
    if isinstance(payload, Mapping) or isinstance(payload, Sequence):
        return payload
    try:
        return list(payload)
    except TypeError:
        return payload


def _snapshot_has_authoritative_realized(payload: Any) -> bool:
    """Return True only when a managed MIS row explicitly carries realised P&L."""
    try:
        snapshot = decode_position_snapshot(payload)
    except Exception:
        return False
    for row in snapshot.rows:
        record = row.raw
        if not is_strategy_instrument(row.symbol):
            continue
        product = str(record.get("product") or "").strip().upper()
        if product != "MIS":
            continue
        if "realised" in record or "realized" in record:
            return True
    return False


def _pnl_baseline_seed_from_snapshot(payload: Any) -> tuple[bool, float, str]:
    """Resolve a safe opening realised-P&L baseline from broker truth.

    Zerodha's authoritative ``data.net`` snapshot is an empty list before any
    intraday position exists. That is positive evidence for a zero opening
    baseline, not missing P&L evidence. Non-empty snapshots are only usable
    when a managed MIS row explicitly carries ``realised``/``realized``.
    """
    try:
        snapshot = decode_position_snapshot(payload)
    except Exception:
        return False, 0.0, ""
    if not snapshot.rows:
        return True, 0.0, "validated_broker_empty_snapshot"

    total = 0.0
    seen = False
    for row in snapshot.rows:
        record = row.raw
        if not is_strategy_instrument(row.symbol):
            continue
        if str(record.get("product") or "").strip().upper() != "MIS":
            continue
        key = "realised" if "realised" in record else "realized" if "realized" in record else None
        if key is None:
            continue
        try:
            total += float(record.get(key) or 0.0)
        except (TypeError, ValueError):
            return False, 0.0, ""
        seen = True
    if seen:
        return True, total, "validated_broker_positions"
    return False, 0.0, ""


def _maybe_seed_pnl_session_baseline(self: Any, payload: Any) -> bool:
    """Initialize today's baseline only from authoritative broker evidence.

    A non-zero local ledger with no verified trading date is deliberately left
    blocked because its session provenance cannot be reconstructed safely.
    A stale dated ledger may be reset by ``establish_pnl_session_baseline`` for
    the new IST trading day, which is the existing owner of day rollover.
    """
    available, seed_value, source = _pnl_baseline_seed_from_snapshot(payload)
    if not available:
        return False
    today = self._trading_date_ist()
    with getattr(self, "_lock"):
        baseline = getattr(self, "_session_opening_realized_baseline", None)
        session_date = getattr(self, "_pnl_trading_date", None)
        local_realized = float(getattr(self, "_local_realized_pnl", 0.0) or 0.0)
    if baseline is not None and str(session_date or "") == today:
        return False
    stale_dated_state = bool(session_date) and str(session_date) != today
    if abs(local_realized) > 1e-6 and not stale_dated_state:
        self._logger.warning(
            "PNL_BASELINE_SEED_BLOCKED local_realized=%.2f session_date=%s source=%s",
            local_realized,
            session_date,
            source,
            extra={
                "event": "PNL_BASELINE_SEED_BLOCKED",
                "reason": "unverified_nonzero_local_pnl",
                "local_realized": local_realized,
                "session_date": session_date,
                "source": source,
            },
        )
        return False
    establish = getattr(self, "establish_pnl_session_baseline", None)
    if not callable(establish):
        return False
    established = bool(
        establish(
            seed_value,
            trading_date=today,
            source=source,
        )
    )
    self._logger.info(
        "PNL_SESSION_BASELINE_READY trading_date=%s baseline=%.2f source=%s established=%s",
        today,
        seed_value,
        source,
        established,
        extra={
            "event": "PNL_SESSION_BASELINE_READY",
            "trading_date": today,
            "baseline": seed_value,
            "source": source,
            "established": established,
        },
    )
    return True


def _patched_synchronize_with_broker(self: Any, broker_positions: Any) -> Any:
    """Reconcile local session P&L to explicit broker truth after a valid sync."""
    payload = _materialize_broker_positions(broker_positions)
    broker_realized_authoritative = _snapshot_has_authoritative_realized(payload)
    result = _ORIGINAL_SYNCHRONIZE_WITH_BROKER(self, payload)
    _maybe_seed_pnl_session_baseline(self, payload)
    if not broker_realized_authoritative:
        return result

    mismatch = False
    local_before = 0.0
    broker_session = 0.0
    with getattr(self, "_lock"):
        broker_realized = getattr(self, "_broker_realized_pnl", None)
        baseline = getattr(self, "_session_opening_realized_baseline", None)
        if broker_realized is None or baseline is None:
            return result
        broker_session = float(broker_realized) - float(baseline)
        local_before = float(getattr(self, "_local_realized_pnl", 0.0) or 0.0)
        mismatch = abs(local_before - broker_session) > 1.0
        self._local_realized_pnl = broker_session
        _ORIGINAL_REFRESH_REALIZED_PNL(self)
        self._pnl_authority = "validated_broker_positions"
        self._pnl_reconciliation_status = (
            "broker_authoritative_reconciled" if mismatch else "matched"
        )

    if mismatch:
        self._logger.warning(
            "PNL_BROKER_AUTHORITY_RECONCILED local_before=%.2f broker_session=%.2f adjustment=%.2f",
            local_before,
            broker_session,
            broker_session - local_before,
            extra={
                "event": "PNL_BROKER_AUTHORITY_RECONCILED",
                "local_before": local_before,
                "broker_session": broker_session,
                "adjustment": broker_session - local_before,
            },
        )
    self.save_state()
    return result


def get_risk_circuit_state(self: Any) -> dict[str, Any]:
    """Return the persisted same-day risk-circuit runtime state."""
    with getattr(self, "_lock"):
        state = getattr(self, "_risk_circuit_state", None)
        return dict(state) if isinstance(state, dict) else {}


def persist_risk_circuit_state(self: Any, **values: Any) -> None:
    """Merge and durably store risk-circuit runtime state for today."""
    with getattr(self, "_lock"):
        state = getattr(self, "_risk_circuit_state", None)
        state = dict(state) if isinstance(state, dict) else {}
        state.update(values)
        state["trading_date"] = self._trading_date_ist()
        self._risk_circuit_state = state
    self.save_state()


def apply_patches() -> None:
    global _PATCH_APPLIED
    global _ORIGINAL_INIT
    global _ORIGINAL_REFRESH_REALIZED_PNL, _ORIGINAL_SYNCHRONIZE_WITH_BROKER
    if _PATCH_APPLIED:
        return
    from nifty_scalper_bot.execution.position_manager import PositionManager

    if getattr(PositionManager, "_position_risk_state_patch", False):
        _PATCH_APPLIED = True
        return
    _ORIGINAL_INIT = PositionManager.__init__
    _ORIGINAL_REFRESH_REALIZED_PNL = PositionManager._refresh_realized_pnl_locked
    _ORIGINAL_SYNCHRONIZE_WITH_BROKER = PositionManager.synchronize_with_broker
    PositionManager.__init__ = _patched_init
    PositionManager.synchronize_with_broker = _patched_synchronize_with_broker
    PositionManager.get_risk_circuit_state = get_risk_circuit_state
    PositionManager.persist_risk_circuit_state = persist_risk_circuit_state
    PositionManager._position_risk_state_patch = True
    _PATCH_APPLIED = True


__all__ = [
    "apply_patches",
    "get_risk_circuit_state",
    "persist_risk_circuit_state",
    "_option_thesis",
    "_snapshot_has_authoritative_realized",
    "_risk_state_snapshot",
    "_restore_risk_state",
]
