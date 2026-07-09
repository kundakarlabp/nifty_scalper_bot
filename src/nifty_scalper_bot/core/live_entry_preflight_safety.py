"""Runtime patch that enforces live-entry broker/market preflight.

The patch wraps app._recompute_and_push_runtime_readiness after the app module is
loaded. It only tightens real LIVE entry arming; it does not change evaluation,
shadow/paper-mode tests, or market-data observation readiness.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from nifty_scalper_bot.execution.live_entry_preflight import (
    SelectedOptionProof,
    evaluate_live_entry_preflight,
)

LOGGER = logging.getLogger(__name__)
_PATCH_ATTR = "_live_entry_preflight_patch_applied"
_ORIGINAL_ATTR = "_live_entry_preflight_original_recompute"
_TRUTHY = {"1", "true", "yes", "y", "on", "live"}


def _env_true(name: str) -> bool:
    return str(os.getenv(name, "") or "").strip().lower() in _TRUTHY


def _get(payload: Mapping[str, Any] | object | None, key: str, default: Any = None) -> Any:
    if payload is None:
        return default
    if isinstance(payload, Mapping):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _truthy_attr(ctx: Any, *names: str, default: bool = False) -> bool:
    for name in names:
        if hasattr(ctx, name):
            return bool(getattr(ctx, name))
    return bool(default)


def _has_broker_truth_state(ctx: Any) -> bool:
    return any(
        hasattr(ctx, name)
        for name in (
            "position_reconciliation_completed",
            "position_reconciliation_failed",
            "broker_orders_reconciled",
            "order_reconciliation_completed",
            "orders_reconciled",
            "startup_order_reconciliation_completed",
            "broker_position_mismatch",
            "position_mismatch",
            "unprotected_broker_position",
            "unprotected_broker_positions",
        )
    )


def _real_live_preflight_requested(ctx: Any) -> bool:
    """Return True only for actual live entry execution, not LIVE-shaped tests."""

    explicit_required = bool(getattr(ctx, "live_entry_preflight_required", False)) or _env_true("LIVE_ENTRY_PREFLIGHT_REQUIRED")
    if explicit_required:
        return True
    mode = str(
        getattr(getattr(ctx, "settings", None), "execution_mode", None)
        or os.getenv("EXECUTION_MODE", "PAPER")
        or "PAPER"
    ).strip().upper()
    live_enabled = _env_true("ENABLE_LIVE") or _env_true("ENABLE_LIVE_TRADING")
    paper_shadow = _env_true("PAPER_MODE") or _env_true("PAPER__ENABLED") or _env_true("SHADOW_MODE")
    return bool(mode == "LIVE" and live_enabled and not paper_shadow and _has_broker_truth_state(ctx))


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    if not callable(fn):
        return None
    try:
        return fn(*args, **kwargs)
    except TypeError:
        try:
            return fn(*args)
        except Exception:
            return None
    except Exception:
        return None


def _snapshot_for_symbol(mdm: Any, symbol: str | None) -> Any:
    if not symbol or mdm is None:
        return None
    getter = getattr(mdm, "get_symbol_snapshot", None)
    return _safe_call(getter, symbol)


def _bars_for_symbol(mdm: Any, symbol: str | None) -> list[Any]:
    if not symbol or mdm is None:
        return []
    getter = getattr(mdm, "get_ohlc_bars", None)
    bars = _safe_call(getter, symbol)
    if bars is None:
        return []
    try:
        return list(bars)
    except TypeError:
        return []


def _last_bar_ts_and_close(bars: list[Any]) -> tuple[Any, float | None]:
    if not bars:
        return None, None
    last = bars[-1]
    if isinstance(last, Mapping):
        ts = last.get("timestamp") or last.get("date") or last.get("time")
        close = last.get("close")
    else:
        ts = getattr(last, "timestamp", getattr(last, "date", None))
        close = getattr(last, "close", None)
    try:
        close_float = float(close) if close is not None else None
    except (TypeError, ValueError):
        close_float = None
    return ts, close_float


def _candle_recent(last_ts: Any, *, max_age_seconds: float) -> bool:
    if last_ts is None:
        return False
    try:
        ts = pd.to_datetime(last_ts, utc=True, errors="coerce")
    except Exception:
        return False
    if pd.isna(ts):
        return False
    now = pd.Timestamp.now(tz=timezone.utc)
    return bool((now - pd.Timestamp(ts)).total_seconds() <= max(float(max_age_seconds), 0.0))


def _quote_tradable(snapshot: Any) -> bool:
    if snapshot is None:
        return False
    if bool(_get(snapshot, "tradable_quote", False)):
        return True
    bid = _get(snapshot, "bid", _get(snapshot, "best_bid", None))
    ask = _get(snapshot, "ask", _get(snapshot, "best_ask", None))
    try:
        return float(bid) > 0 and float(ask) > float(bid)
    except (TypeError, ValueError):
        return False


def _selected_option_proof(ctx: Any, symbol: str | None, *, max_age_seconds: float) -> SelectedOptionProof | None:
    if not symbol:
        return None
    mdm = getattr(ctx, "market_data_manager", None)
    snapshot = _snapshot_for_symbol(mdm, symbol)
    bars = _bars_for_symbol(mdm, symbol)
    last_ts, last_close = _last_bar_ts_and_close(bars)
    return SelectedOptionProof(
        symbol=str(symbol),
        quote_present=snapshot is not None,
        quote_tradable=_quote_tradable(snapshot),
        timestamp_quality=_get(snapshot, "timestamp_quality", None),
        timestamp_source=_get(snapshot, "timestamp_source", _get(snapshot, "source", None)),
        candle_count=len(bars),
        last_candle_ts=last_ts,
        last_candle_close=last_close,
        max_candle_age_seconds=max_age_seconds,
        candle_recent=_candle_recent(last_ts, max_age_seconds=max_age_seconds),
        now=datetime.now(timezone.utc),
    )


def _broker_orders_reconciled(ctx: Any) -> bool:
    explicit = _truthy_attr(
        ctx,
        "broker_orders_reconciled",
        "order_reconciliation_completed",
        "orders_reconciled",
        "startup_order_reconciliation_completed",
        default=False,
    )
    if explicit:
        return True
    # Backward-compatible bridge: existing startup reconciliation establishes the
    # same broker truth boundary in current runtime builds. Requiring a brand-new
    # flag would permanently block older deployments before they can set it.
    return bool(getattr(ctx, "position_reconciliation_completed", False)) and not bool(
        getattr(ctx, "position_reconciliation_failed", False)
    )


def _local_positions_match_broker(ctx: Any) -> bool:
    if bool(getattr(ctx, "position_reconciliation_failed", False)):
        return False
    if bool(getattr(ctx, "unprotected_broker_position", False)) or bool(getattr(ctx, "unprotected_broker_positions", set())):
        return False
    if bool(getattr(ctx, "broker_position_mismatch", False)) or bool(getattr(ctx, "position_mismatch", False)):
        return False
    return bool(getattr(ctx, "position_reconciliation_completed", False))


def build_context_live_entry_preflight(ctx: Any) -> dict[str, Any]:
    selected_ce = getattr(ctx, "selected_ce", None) or getattr(ctx, "atm_ce_symbol", None)
    selected_pe = getattr(ctx, "selected_pe", None) or getattr(ctx, "atm_pe_symbol", None)
    try:
        max_age = float(getattr(ctx, "live_entry_candle_max_age_seconds", 180.0) or 180.0)
    except (TypeError, ValueError):
        max_age = 180.0
    selected_options = [
        proof
        for proof in (
            _selected_option_proof(ctx, selected_ce, max_age_seconds=max_age),
            _selected_option_proof(ctx, selected_pe, max_age_seconds=max_age),
        )
        if proof is not None
    ]
    return {
        "broker_positions_fetched": bool(getattr(ctx, "position_reconciliation_completed", False))
        and not bool(getattr(ctx, "position_reconciliation_failed", False)),
        "broker_orders_reconciled": _broker_orders_reconciled(ctx),
        "local_positions_match_broker": _local_positions_match_broker(ctx),
        "selected_options": selected_options,
        "context": {"selected_ce": selected_ce, "selected_pe": selected_pe},
    }


def _apply_preflight_to_context(ctx: Any, decision: Any, *, reason: str) -> None:
    setattr(ctx, "live_entry_preflight_ready", bool(decision.ready))
    setattr(ctx, "live_entry_preflight_blockers", list(decision.blockers))
    setattr(ctx, "live_entry_preflight_primary_blocker", decision.primary_blocker)
    setattr(ctx, "live_entry_preflight_details", decision.details)
    if decision.ready:
        return
    block_reason = f"execution_not_armed:{decision.primary_blocker or 'live_entry_preflight_failed'}"
    ctx.live_orders_armed = False
    ctx.execution_armed = False
    ctx.execution_ready = False
    ctx.trading_ready = bool(getattr(ctx, "evaluation_ready", False))
    ctx.live_block_reason = block_reason
    ctx.execution_block_reason = block_reason
    runner = getattr(ctx, "strategy_runner", None)
    setter = getattr(runner, "set_runtime_readiness", None)
    if callable(setter):
        selected_ce = getattr(ctx, "selected_ce", None)
        selected_pe = getattr(ctx, "selected_pe", None)
        try:
            setter(
                data_hard_ready=bool(getattr(ctx, "data_hard_ready", False)),
                evaluation_ready=bool(getattr(ctx, "evaluation_ready", False)),
                live_orders_armed=False,
                reason=block_reason,
                selected_ce=selected_ce,
                selected_pe=selected_pe,
                execution_ready_by_symbol=dict(getattr(ctx, "execution_ready_by_symbol", {}) or {}),
            )
        except Exception:
            LOGGER.debug("LIVE_ENTRY_PREFLIGHT_RUNNER_UPDATE_FAILED", exc_info=True)
    LOGGER.warning(
        "LIVE_ENTRY_PREFLIGHT_BLOCKED primary=%s blockers=%s reason=%s",
        decision.primary_blocker,
        list(decision.blockers),
        reason,
        extra={
            "event": "LIVE_ENTRY_PREFLIGHT_BLOCKED",
            "primary_blocker": decision.primary_blocker,
            "blockers": list(decision.blockers),
            "details": decision.details,
            "reason": reason,
        },
    )


def apply_app_patch(app_module: Any) -> None:
    if bool(getattr(app_module, _PATCH_ATTR, False)):
        return
    original = getattr(app_module, "_recompute_and_push_runtime_readiness", None)
    if not callable(original):
        return
    setattr(app_module, _ORIGINAL_ATTR, original)

    async def _wrapped(ctx: Any, *, reason: str) -> None:
        await original(ctx, reason=reason)
        if not _real_live_preflight_requested(ctx) or not bool(getattr(ctx, "live_orders_armed", False)):
            return
        snapshot = build_context_live_entry_preflight(ctx)
        decision = evaluate_live_entry_preflight(snapshot)
        _apply_preflight_to_context(ctx, decision, reason=reason)

    setattr(app_module, "_recompute_and_push_runtime_readiness", _wrapped)
    setattr(app_module, _PATCH_ATTR, True)


__all__ = ["apply_app_patch", "build_context_live_entry_preflight"]
