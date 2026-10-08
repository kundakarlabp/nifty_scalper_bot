"""File purpose:
    Implement the single production order manager used by the trading runtime.

Key responsibilities:
    - Apply unresolved-exit entry blocking before broker submission.
    - Run bounded entry recovery and finalize partial-entry reconciliation.
    - Delegate unchanged order operations to ``order_manager_core``.

Operational constraints:
    - Protective exits must bypass entry blocking.
    - Recovery must remain bounded and must not create duplicate broker orders.
"""

from __future__ import annotations

import inspect
import os
import time
from contextlib import suppress
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import nifty_scalper_bot.execution.operator_control as _operator_control
from nifty_scalper_bot.execution import order_manager_core as _core
from nifty_scalper_bot.execution.entry_geometry import (
    release_prebroker_entry_reservation,
)
from nifty_scalper_bot.execution.entry_recovery import (
    _finalize_partial_entry,
    _recover_submit,
)
from nifty_scalper_bot.execution.entry_recovery import (
    current_entry_blocker as _current_entry_blocker,
)
from nifty_scalper_bot.execution.native_entry_gate import (
    NO_BLOCK,
    block_result,
    configure_provider,
)
from nifty_scalper_bot.risk.cost_model import estimate_round_trip_cost
from nifty_scalper_bot.risk.net_rr_gate import minimum_target_for_net_rr
from nifty_scalper_bot.strategies.signal_identity import order_setup_context

_EXIT_IDENTITY_KWARGS = {"linked_entry_order_id", "trade_lifecycle_id", "bracket_id"}
_CORE_PLACE_ORDER_SIGNATURE = inspect.signature(_core.OrderManager.place_order)
_EXIT_TAG_PREFIXES = ("EXIT", "EXIT_", "SL_", "TP_", "EOD_")
_REDUCE_TAG_PREFIXES = ("FLATTEN", "EXIT_FLATTEN", "SQUAREOFF", "PANIC")
_LIVE_REPRICE_QUOTE_SOURCES = {
    "ws",
    "ws_full",
    "websocket",
    "websocket_full",
    "stream",
}


def _truthy_check_risk_disabled(value: Any) -> bool:
    return value is False or str(value).strip().lower() in {"0", "false", "no", "off"}


def _normalise_protective_intent_kwargs(
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Add explicit intent only for proven risk-bypassed protective orders."""
    cleaned = dict(kwargs)
    current_intent = str(cleaned.get("intent") or "").strip().upper()
    if current_intent:
        cleaned["intent"] = current_intent
        return cleaned

    tag = str(cleaned.get("tag") or "").strip().upper()
    if not tag or not _truthy_check_risk_disabled(cleaned.get("check_risk", True)):
        return cleaned

    if tag.startswith(_REDUCE_TAG_PREFIXES):
        cleaned["intent"] = "REDUCE"
        cleaned.setdefault("strategy_name", "operator_flatten")
    elif tag.startswith(_EXIT_TAG_PREFIXES):
        cleaned["intent"] = "EXIT"
        cleaned.setdefault("strategy_name", "protective_exit")
    return cleaned


def _bind_place_order(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> dict[str, Any] | None:
    try:
        bound = _CORE_PLACE_ORDER_SIGNATURE.bind_partial(None, *args, **dict(kwargs))
        return {key: value for key, value in bound.arguments.items() if key != "self"}
    except Exception:
        return None


def _release_failed_prebroker_entry_reservation(
    manager: Any, kwargs: Mapping[str, Any]
) -> bool:
    """Release only a proven, normally returned pre-broker entry rejection."""
    try:
        released = release_prebroker_entry_reservation(manager, kwargs)
    except Exception:
        return False
    if not released:
        return False
    logger = getattr(manager, "_logger", None)
    log = getattr(logger, "info", None)
    if callable(log):
        log(
            "PREBROKER_ENTRY_RESERVATION_RELEASED symbol=%s reason=%s",
            kwargs.get("symbol"),
            (getattr(manager, "_last_order_decision", {}) or {}).get("block_reason"),
            extra={
                "event": "PREBROKER_ENTRY_RESERVATION_RELEASED",
                "symbol": kwargs.get("symbol"),
                "block_reason": (
                    getattr(manager, "_last_order_decision", {}) or {}
                ).get("block_reason"),
            },
        )
    return True


def _place_order_with_prebroker_reservation_cleanup(
    manager: Any,
    place_order: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call native placement and clean up only a normal local rejection."""
    result = place_order(*args, **kwargs)
    if result is None:
        _release_failed_prebroker_entry_reservation(manager, kwargs)
    return result


def _strip_exit_identity_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return core-compatible kwargs while retaining supported exit identity.

    Older core ``place_order`` signatures did not accept these lifecycle fields,
    so this compatibility helper used to remove them.  The canonical core now
    accepts and persists them; keep the helper/API but stop discarding identity.
    """

    cleaned = dict(kwargs)
    identity = {key: cleaned[key] for key in _EXIT_IDENTITY_KWARGS if key in cleaned}
    return cleaned, identity


def _positive_float(value: Any) -> float | None:
    with suppress(TypeError, ValueError):
        parsed = float(value)
        if parsed > 0.0:
            return parsed
    return None


def _positive_int(value: Any) -> int:
    with suppress(TypeError, ValueError):
        parsed = int(float(value))
        if parsed > 0:
            return parsed
    return 0


def _cost_adjust_entry_target(manager: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Raise only a distance-anchored BUY option target enough to preserve net RR.

    The stop, entry, quantity and final risk gate are never changed. Explicit
    technical/absolute targets are immutable. If transaction costs require more
    than the bounded uplift allowed by ``minimum_target_for_net_rr``, this helper
    returns the order unchanged so the final risk gate still rejects it.
    """
    if not bool(kwargs.get("check_risk", True)):
        return kwargs
    if str(kwargs.get("intent") or "ENTRY").strip().upper() != "ENTRY":
        return kwargs
    if str(kwargs.get("side") or "").strip().upper() != "BUY":
        return kwargs
    symbol = str(kwargs.get("symbol") or "").strip().upper()
    if not symbol.endswith(("CE", "PE")):
        return kwargs

    provenance_raw = kwargs.get("trade_provenance")
    if not isinstance(provenance_raw, Mapping):
        return kwargs
    provenance = dict(provenance_raw)
    if str(provenance.get("bracket_anchor_mode") or "").strip().lower() != "distance":
        return kwargs
    if bool(provenance.get("net_rr_target_adjusted")):
        return kwargs

    entry = _positive_float(kwargs.get("price"))
    stop = _positive_float(kwargs.get("stop_loss"))
    target = _positive_float(kwargs.get("take_profit"))
    quantity = _positive_int(kwargs.get("quantity"))
    if entry is None or stop is None or target is None or quantity <= 0:
        return kwargs
    if not (stop < entry < target):
        return kwargs

    metadata: dict[str, Any] = {}
    quote_getter = getattr(manager, "_get_latest_quote_safe", None)
    diagnostics = getattr(manager, "_extract_quote_diagnostics", None)
    if callable(quote_getter):
        with suppress(Exception):
            quote = quote_getter(symbol) or {}
            quote_info = diagnostics(quote) if callable(diagnostics) else quote
            if isinstance(quote_info, Mapping):
                bid = _positive_float(quote_info.get("bid"))
                ask = _positive_float(quote_info.get("ask"))
                if bid is not None:
                    metadata["bid"] = bid
                if ask is not None:
                    metadata["ask"] = ask

    signal = SimpleNamespace(
        symbol=symbol,
        action="BUY",
        quantity=quantity,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        metadata=metadata,
    )
    adjusted_target = minimum_target_for_net_rr(signal)
    if adjusted_target is None or adjusted_target <= target + 1e-9:
        return kwargs

    risk_points = entry - stop
    old_rr = (target - entry) / risk_points
    new_rr = (adjusted_target - entry) / risk_points
    adjusted_provenance = dict(provenance)
    adjusted_provenance["net_rr_target_adjusted"] = True
    adjusted_provenance.setdefault("original_take_profit", float(target))
    adjusted_provenance["cost_adjusted_take_profit"] = float(adjusted_target)
    adjusted_provenance["net_rr_target_adjustment_r"] = float(new_rr - old_rr)

    adjusted = dict(kwargs)
    adjusted["take_profit"] = float(adjusted_target)
    adjusted["trade_provenance"] = adjusted_provenance

    logger = getattr(manager, "_logger", None)
    log = getattr(logger, "info", None)
    if callable(log):
        log(
            "NET_RR_TARGET_ADJUSTED symbol=%s entry=%.2f stop=%.2f old_tp=%.2f new_tp=%.2f old_gross_rr=%.3f new_gross_rr=%.3f",
            symbol,
            entry,
            stop,
            target,
            adjusted_target,
            old_rr,
            new_rr,
            extra={
                "event": "NET_RR_TARGET_ADJUSTED",
                "symbol": symbol,
                "entry": entry,
                "stop_loss": stop,
                "original_take_profit": target,
                "adjusted_take_profit": adjusted_target,
                "old_gross_rr": old_rr,
                "new_gross_rr": new_rr,
            },
        )
    return adjusted


def _maybe_reprice_open_entry(
    manager: Any,
    order: Any,
    payload: Mapping[str, Any],
) -> bool:
    """Reprice one unfilled live BUY option only after all live gates re-pass."""
    enabled = str(
        os.getenv("ENTRY_OPEN_REPRICE_ENABLED", "true") or "true"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return False
    live_checker = getattr(manager, "is_live_mode", None)
    if not callable(live_checker):
        return False
    try:
        if not bool(live_checker()):
            return False
    except Exception:
        return False

    status_obj = getattr(order, "status", None)
    status = (
        str(payload.get("status") or getattr(status_obj, "name", status_obj) or "")
        .strip()
        .upper()
        .replace("_", " ")
    )
    intent = str(getattr(order, "intent", "") or "").strip().upper()
    side = str(getattr(order, "side", "") or "").strip().upper()
    symbol = str(getattr(order, "symbol", "") or "").strip().upper()
    order_type_obj = getattr(order, "order_type", None)
    order_type = (
        str(getattr(order_type_obj, "name", order_type_obj) or "").strip().upper()
    )
    filled = max(
        _positive_int(payload.get("filled_quantity")),
        _positive_int(getattr(order, "filled_quantity", 0)),
    )
    if (
        status != "OPEN"
        or intent != "ENTRY"
        or side != "BUY"
        or not symbol.endswith(("CE", "PE"))
        or order_type != "LIMIT"
        or filled > 0
    ):
        return False

    current_price = _positive_float(getattr(order, "price", None))
    order_id = str(getattr(order, "order_id", "") or "").strip()
    stop_loss = _positive_float(getattr(order, "stop_loss", None))
    take_profit = _positive_float(getattr(order, "take_profit", None))
    if (
        current_price is None
        or stop_loss is None
        or take_profit is None
        or not order_id
    ):
        return False

    provenance_raw = getattr(order, "trade_provenance", None)
    provenance = dict(provenance_raw) if isinstance(provenance_raw, Mapping) else {}
    anchor_mode = (
        str(provenance.get("bracket_anchor_mode") or "").strip().lower()
    )
    guard = provenance.get("entry_reprice_guard")
    if anchor_mode != "distance" or not isinstance(guard, Mapping):
        return False

    count = _positive_int(provenance.get("entry_open_reprice_count"))
    try:
        max_modifications = max(
            0, int(os.getenv("ENTRY_OPEN_REPRICE_MAX_MODIFICATIONS", "2") or 2)
        )
    except ValueError:
        max_modifications = 2
    if count >= max_modifications:
        return False

    now_epoch = time.time()
    try:
        min_interval = max(
            0.0,
            float(os.getenv("ENTRY_OPEN_REPRICE_MIN_INTERVAL_SECONDS", "0.35") or 0.35),
        )
    except ValueError:
        min_interval = 0.35
    last_epoch = _positive_float(provenance.get("entry_open_reprice_last_epoch"))
    if last_epoch is not None and now_epoch - last_epoch < min_interval:
        return False

    quote_getter = getattr(manager, "_get_latest_quote_safe", None)
    diagnostics = getattr(manager, "_extract_quote_diagnostics", None)
    if not callable(quote_getter):
        return False
    try:
        raw_quote = quote_getter(symbol) or {}
    except Exception:
        return False
    if not isinstance(raw_quote, Mapping):
        return False

    source = str(raw_quote.get("source") or "").strip().lower()
    if (
        source not in _LIVE_REPRICE_QUOTE_SOURCES
        or raw_quote.get("tradable_quote") is not True
        or raw_quote.get("depth_available") is not True
        or raw_quote.get("source_timestamp_valid") is not True
    ):
        return False

    quote = diagnostics(raw_quote) if callable(diagnostics) else raw_quote
    if not isinstance(quote, Mapping):
        return False
    bid = _positive_float(quote.get("bid"))
    ask = _positive_float(quote.get("ask"))
    ask_qty = _positive_int(quote.get("ask_qty"))
    spread_pct = _positive_float(quote.get("spread_pct"))
    age_ms = quote.get("age_ms")
    try:
        age = float(age_ms) if age_ms is not None else None
        max_age_ms = max(
            1.0,
            min(
                float(os.getenv("ENTRY_OPEN_REPRICE_MAX_QUOTE_AGE_MS", "1000") or 1000),
                float(guard.get("max_quote_age_ms") or 1000),
            ),
        )
        max_spread_pct = max(0.0, float(guard.get("max_spread_pct") or 0.0))
        min_depth_qty = max(
            int(getattr(order, "quantity", 0) or 0),
            int(float(guard.get("min_depth_qty") or 0)),
        )
    except (TypeError, ValueError):
        return False
    if (
        bid is None
        or ask is None
        or ask < bid
        or age is None
        or age > max_age_ms
        or spread_pct is None
        or spread_pct > max_spread_pct
        or ask_qty < min_depth_qty
        or ask <= current_price
    ):
        return False

    anchor_price = (
        _positive_float(provenance.get("entry_open_reprice_anchor_price"))
        or current_price
    )
    try:
        max_deviation_pct = max(
            0.0,
            float(os.getenv("ENTRY_OPEN_REPRICE_MAX_DEVIATION_PCT", "0.75") or 0.75),
        )
    except ValueError:
        max_deviation_pct = 0.75
    deviation_pct = (ask - anchor_price) / anchor_price * 100.0
    if deviation_pct > max_deviation_pct:
        return False

    rounder = getattr(manager, "_round_to_tick", None)
    try:
        new_price = (
            float(rounder(ask)) if callable(rounder) else round(ask / 0.05) * 0.05
        )
    except Exception:
        return False
    new_price = round(new_price, 2)
    try:
        min_ticks = max(1, int(os.getenv("ENTRY_OPEN_REPRICE_MIN_TICKS", "1") or 1))
    except ValueError:
        min_ticks = 1
    if new_price - current_price < 0.05 * min_ticks - 1e-9:
        return False

    quantity = int(getattr(order, "quantity", 0) or 0)
    lot_size = int(getattr(order, "resolved_lot_size", 0) or 0)
    requested_lots = int(getattr(order, "requested_lots", 0) or 0)
    if quantity <= 0 or lot_size <= 0:
        return False
    if requested_lots <= 0:
        if quantity % lot_size != 0:
            return False
        requested_lots = quantity // lot_size

    try:
        plan = _core.TradePlan(
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            bracket_anchor_mode="distance",
            strategy_name=str(provenance.get("strategy") or "runner"),
            signal_id=getattr(order, "signal_id", None),
            trace_id=provenance.get("trace_id"),
            tag=str(getattr(order, "tag", "") or "runner"),
            product=str(getattr(order, "product", "") or "MIS"),
            max_quote_age_ms=int(float(guard.get("max_quote_age_ms") or 1000)),
            max_spread_pct=float(guard.get("max_spread_pct") or 0.0),
            min_depth_qty=int(float(guard.get("min_depth_qty") or quantity)),
            max_signal_age_seconds=float(
                guard.get("max_signal_age_seconds") or 0.0
            ),
            max_entry_drift_pct=float(guard.get("max_entry_drift_pct") or 0.0),
            intent="ENTRY",
            intended_position_side=(
                getattr(order, "intended_position_side", None) or "LONG"
            ),
            trade_lifecycle_id=getattr(order, "trade_lifecycle_id", None),
            client_order_id=getattr(order, "client_order_id", None),
            basket_version=getattr(order, "basket_version", None),
            instrument_token=getattr(order, "instrument_token", None),
            contract_expiry=getattr(order, "contract_expiry", None),
            requested_lots=requested_lots,
            resolved_lot_size=lot_size,
            trade_provenance=dict(provenance),
        )
    except (TypeError, ValueError):
        return False

    validator = getattr(manager, "_validate_trade_plan", None)
    reanchor = getattr(manager, "_reanchor_bracket_to_price", None)
    margin_gate = getattr(manager, "_apply_entry_margin_gate", None)
    if not callable(validator) or not callable(reanchor) or not callable(margin_gate):
        return False
    try:
        validation = validator(plan)
    except Exception:
        return False
    if not bool(getattr(validation, "allowed", False)):
        return False

    try:
        repriced_plan = reanchor(plan, new_price)
    except Exception:
        return False
    repriced_stop = _positive_float(getattr(repriced_plan, "stop_loss", None))
    repriced_target = _positive_float(getattr(repriced_plan, "take_profit", None))
    if (
        repriced_stop is None
        or repriced_target is None
        or not (repriced_stop < new_price < repriced_target)
    ):
        return False

    try:
        effective_plan, sizing_rejection = margin_gate(repriced_plan, new_price)
    except Exception:
        return False
    if sizing_rejection is not None or effective_plan is None:
        return False
    if int(getattr(effective_plan, "quantity", 0) or 0) != quantity:
        return False

    modifier = getattr(manager, "modify_order", None)
    if not callable(modifier):
        return False
    try:
        modified = bool(modifier(order_id, price=new_price))
    except Exception:
        modified = False
    if not modified:
        return False

    final_provenance_raw = getattr(effective_plan, "trade_provenance", None)
    final_provenance = (
        dict(final_provenance_raw)
        if isinstance(final_provenance_raw, Mapping)
        else dict(provenance)
    )
    final_provenance["entry_open_reprice_anchor_price"] = float(anchor_price)
    final_provenance["entry_open_reprice_count"] = count + 1
    final_provenance["entry_open_reprice_last_epoch"] = now_epoch
    final_provenance["entry_open_reprice_last_price"] = new_price
    order.price = new_price
    order.stop_loss = float(repriced_stop)
    order.take_profit = float(repriced_target)
    order.trade_provenance = final_provenance

    persist_snapshot = getattr(manager, "_persist_order_snapshot", None)
    if callable(persist_snapshot):
        with suppress(Exception):
            persist_snapshot(order)
    save_orders = getattr(manager, "save_orders", None)
    if callable(save_orders):
        with suppress(Exception):
            save_orders()

    logger = getattr(manager, "_logger", None)
    log = getattr(logger, "info", None)
    if callable(log):
        log(
            "ENTRY_OPEN_REPRICED order_id=%s symbol=%s old_price=%.2f "
            "new_price=%.2f count=%s/%s",
            order_id,
            symbol,
            current_price,
            new_price,
            count + 1,
            max_modifications,
            extra={
                "event": "ENTRY_OPEN_REPRICED",
                "order_id": order_id,
                "symbol": symbol,
                "old_price": current_price,
                "new_price": new_price,
                "stop_loss": repriced_stop,
                "take_profit": repriced_target,
                "count": count + 1,
                "max_modifications": max_modifications,
                "anchor_price": anchor_price,
                "deviation_pct": deviation_pct,
                "quote_source": source,
                "spread_pct": spread_pct,
                "ask_qty": ask_qty,
            },
        )
    return True

def _enrich_trade_plan_exit_provenance(
    plan: Any,
    *,
    finalize_tp1: bool = True,
    final_entry_price: float | None = None,
) -> Any:
    """Persist exit inputs; auto-TP1 is decided only on final entry economics."""
    if str(getattr(plan, "intent", "ENTRY") or "ENTRY").upper() not in {
        "ENTRY",
        "SCALE_IN",
        "REVERSAL",
    }:
        return plan

    provenance = getattr(plan, "trade_provenance", {})
    enriched = dict(provenance) if isinstance(provenance, Mapping) else {}
    quantity = _positive_int(getattr(plan, "quantity", 0))
    lot_size = _positive_int(getattr(plan, "resolved_lot_size", 0))
    entry = _positive_float(
        final_entry_price
        if final_entry_price is not None
        else getattr(plan, "entry_price", None)
    )
    stop = _positive_float(getattr(plan, "stop_loss", None))
    target = _positive_float(getattr(plan, "take_profit", None))
    side = str(getattr(plan, "side", "BUY") or "BUY").upper()

    if lot_size > 0:
        enriched["resolved_lot_size"] = lot_size
    enriched["bracket_anchor_mode"] = str(
        getattr(plan, "bracket_anchor_mode", "distance") or "distance"
    ).strip().lower()
    enriched["entry_reprice_guard"] = {
        "max_quote_age_ms": int(getattr(plan, "max_quote_age_ms", 0) or 0),
        "max_spread_pct": float(getattr(plan, "max_spread_pct", 0.0) or 0.0),
        "min_depth_qty": int(getattr(plan, "min_depth_qty", 0) or 0),
        "max_signal_age_seconds": float(
            getattr(plan, "max_signal_age_seconds", 0.0) or 0.0
        ),
        "max_entry_drift_pct": float(
            getattr(plan, "max_entry_drift_pct", 0.0) or 0.0
        ),
    }

    risk = None
    reward = None
    if entry is not None and stop is not None and target is not None:
        risk = entry - stop if side == "BUY" else stop - entry
        reward = target - entry if side == "BUY" else entry - target
        if risk > 0.0 and reward > 0.0:
            enriched["initial_risk_points"] = float(risk)
            enriched["initial_reward_points"] = float(reward)
            enriched["initial_reward_risk"] = float(reward / risk)
        else:
            risk = reward = None

    tp1_enabled = str(
        os.getenv("ENABLE_TP1_SCALE_OUT", "true") or "true"
    ).strip().lower() in {"1", "true", "yes", "on"}
    total_lots = quantity // lot_size if lot_size > 0 else 0
    existing_tp1_price = _positive_float(enriched.get("tp1_price"))
    existing_tp1_qty = _positive_int(enriched.get("tp1_qty"))
    existing_tp1_source = str(enriched.get("tp1_source") or "").strip().lower()
    explicit_tp1 = bool(
        existing_tp1_price is not None
        and existing_tp1_qty > 0
        and existing_tp1_source != "auto_cost_gated"
    )

    if existing_tp1_source == "auto_cost_gated":
        for key in (
            "tp1_price",
            "tp1_qty",
            "tp1_incremental_cost",
            "tp1_incremental_edge_multiple",
            "tp1_min_incremental_edge_multiple",
            "tp1_final_entry_price",
        ):
            enriched.pop(key, None)
        existing_tp1_price = None
        existing_tp1_qty = 0

    tp1_status = "skipped"
    tp1_skip_reason = "unknown"
    if explicit_tp1:
        tp1_status = "armed"
        tp1_skip_reason = ""
    elif lot_size <= 0:
        tp1_skip_reason = "lot_size_unresolved"
    elif not tp1_enabled:
        tp1_skip_reason = "disabled"
    elif total_lots < 2:
        tp1_skip_reason = "single_lot"
    elif not finalize_tp1:
        tp1_status = "pending_reanchor"
        tp1_skip_reason = "pending_final_entry"
    elif risk is None or reward is None or entry is None or target is None:
        tp1_skip_reason = "invalid_geometry"
    else:
        tp1_r = _positive_float(os.getenv("TP1_R_MULT", "1.0")) or 1.0
        tp1_lots = max(1, total_lots // 2)
        tp1_qty = tp1_lots * lot_size
        tp1_price = entry + risk * tp1_r if side == "BUY" else entry - risk * tp1_r
        strictly_before_final = (
            entry < tp1_price < target if side == "BUY" else target < tp1_price < entry
        )
        if tp1_qty >= quantity:
            tp1_skip_reason = "no_remainder"
        elif not strictly_before_final:
            tp1_skip_reason = "not_before_final_target"
        else:
            two_order_cost = estimate_round_trip_cost(
                entry_price=entry,
                exit_price=target,
                quantity=quantity,
                executed_orders=2,
            )
            three_order_cost = estimate_round_trip_cost(
                entry_price=entry,
                exit_price=target,
                quantity=quantity,
                executed_orders=3,
            )
            incremental_cost = max(0.0, three_order_cost.total - two_order_cost.total)
            tp1_gross_reward = abs(tp1_price - entry) * tp1_qty
            edge_multiple = (
                tp1_gross_reward / incremental_cost
                if incremental_cost > 0.0
                else float("inf")
            )
            try:
                min_incremental_edge = max(
                    0.0,
                    float(os.getenv("TP1_MIN_INCREMENTAL_EDGE_MULTIPLE", "2.0") or 2.0),
                )
            except ValueError:
                min_incremental_edge = 2.0
            enriched["tp1_incremental_cost"] = float(incremental_cost)
            enriched["tp1_incremental_edge_multiple"] = float(edge_multiple)
            enriched["tp1_min_incremental_edge_multiple"] = float(min_incremental_edge)
            enriched["tp1_final_entry_price"] = float(entry)
            if edge_multiple < min_incremental_edge:
                tp1_skip_reason = "incremental_cost_edge_thin"
            else:
                enriched["tp1_price"] = float(tp1_price)
                enriched["tp1_qty"] = int(tp1_qty)
                enriched["tp1_source"] = "auto_cost_gated"
                tp1_status = "armed"
                tp1_skip_reason = ""

    enriched["tp1_status"] = tp1_status
    if tp1_skip_reason:
        enriched["tp1_skip_reason"] = tp1_skip_reason
    else:
        enriched.pop("tp1_skip_reason", None)

    trailing_mult = _positive_float(enriched.get("trailing_atr_mult"))
    if trailing_mult is None:
        trailing_mult = _positive_float(os.getenv("BRACKET_TRAILING_ATR_MULT", "0"))
    if trailing_mult is not None:
        enriched["trailing_atr_mult"] = float(trailing_mult)

    setattr(plan, "trade_provenance", enriched)
    return plan

def _submit_core_with_exit_provenance(manager: Any, plan: Any) -> Any:
    """Persist guard metadata; core finalizes TP1 after protected-price reanchor."""
    _enrich_trade_plan_exit_provenance(plan, finalize_tp1=False)
    return _core.OrderManager.submit_trade_plan_result(manager, plan)


class RuntimeOrderManager(_core.OrderManager):
    """Production order manager with native recovery and entry gating."""

    def _reanchor_bracket_to_price(self, plan: Any, price: float) -> Any:
        """Finalize cost-aware TP1 only after core resolves protected entry price."""
        reanchored = super()._reanchor_bracket_to_price(plan, price)
        return _enrich_trade_plan_exit_provenance(
            reanchored,
            finalize_tp1=True,
            final_entry_price=price,
        )

    def emergency_stop(self, reason: str = "telegram_emergency") -> dict[str, Any]:
        """Pause entries, cancel pending orders and flatten open exposure."""
        return _operator_control.emergency_stop(self, reason=reason)

    def engage_kill_switch(self, reason: str = "telegram_emergency") -> dict[str, Any]:
        """Compatibility alias for the canonical emergency-stop control."""
        return self.emergency_stop(reason=reason)

    def kill_switch(self, reason: str = "telegram_emergency") -> dict[str, Any]:
        """Compatibility alias for the canonical emergency-stop control."""
        return self.emergency_stop(reason=reason)

    def cancel_pending_orders(self) -> dict[str, Any]:
        """Cancel currently open or pending broker orders."""
        return _operator_control.cancel_pending_orders(self)

    def cancel_all_open_orders(self) -> dict[str, Any]:
        """Compatibility alias for canonical pending-order cancellation."""
        return self.cancel_pending_orders()

    def cancel_non_protective_orders(self) -> dict[str, Any]:
        """Compatibility alias for canonical pending-order cancellation."""
        return self.cancel_pending_orders()

    def flatten_all(
        self,
        reason: str = "telegram_flatten",
        *,
        cancel_first: bool = True,
    ) -> dict[str, Any]:
        """Cancel pending orders and flatten every non-zero exposure."""
        return _operator_control.flatten_all(
            self,
            reason=reason,
            cancel_first=cancel_first,
        )

    def flatten_positions(
        self,
        reason: str = "telegram_flatten",
        *,
        cancel_first: bool = True,
    ) -> dict[str, Any]:
        """Compatibility alias for the canonical flatten operation."""
        return self.flatten_all(reason=reason, cancel_first=cancel_first)

    def close_all_positions(
        self,
        reason: str = "telegram_flatten",
        *,
        cancel_first: bool = True,
    ) -> dict[str, Any]:
        """Compatibility alias for the canonical flatten operation."""
        return self.flatten_all(reason=reason, cancel_first=cancel_first)

    def set_trade_plan_rebuilder(
        self,
        callback: Callable[..., Any] | None,
    ) -> None:
        self._trade_plan_rebuilder = callback

    def set_unresolved_exit_provider(self, provider: Any | None) -> None:
        configure_provider(self, provider)
        # The provider is the canonical runtime bracket owner. Reconciliation
        # already reads ``order_manager._bracket_manager``; keep both references
        # aligned so a broker-flat snapshot can clear a completed exit lifecycle.
        self._bracket_manager = provider

    def _release_resolved_entry_reconciliation_blocker(
        self, blocker: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        """Release only a terminal-unfilled entry blocker proven broker-flat.

        The entry-recovery latch is intentionally fail-closed while broker truth is
        uncertain. Once the same broker order is authoritatively terminal-unfilled
        and the canonical bracket authority proves zero broker exposure for the same
        symbol, keeping the manager-global latch would block unrelated future entries
        indefinitely. Broker I/O occurs outside the OrderManager lock; the lock is
        used only for the final identity-checked state transition so a newer blocker
        cannot be cleared by an older reconciliation result.
        """
        if str(blocker.get("block_reason") or "").strip().lower() != (
            "entry_reconciliation_pending"
        ):
            return blocker
        details = blocker.get("details")
        if not isinstance(details, Mapping):
            return blocker
        order_id = str(details.get("order_id") or "").strip()
        symbol = str(details.get("symbol") or "").strip()
        if not order_id or not symbol:
            return blocker

        authority = getattr(self, "_bracket_manager", None)
        order_status = getattr(authority, "_broker_entry_order_status", None)
        broker_quantity = getattr(authority, "_broker_position_quantity", None)
        if not callable(order_status) or not callable(broker_quantity):
            return blocker

        try:
            status_payload, status_known = order_status(order_id)
        except Exception:
            return blocker
        if not status_known or not isinstance(status_payload, Mapping):
            return blocker
        status = str(status_payload.get("status") or "").strip().upper()
        if status not in {"CANCELLED", "CANCELED", "REJECTED", "EXPIRED"}:
            return blocker

        try:
            quantity = broker_quantity(symbol)
        except Exception:
            return blocker
        if quantity is None:
            return blocker
        try:
            if int(quantity) != 0:
                return blocker
        except (TypeError, ValueError):
            return blocker

        def _clear_if_current() -> Mapping[str, Any] | None:
            current = getattr(self, "_entry_lifecycle_blocker", None)
            if current is not blocker:
                return current if isinstance(current, Mapping) else None
            self._entry_lifecycle_blocker = None
            if getattr(self, "_last_order_decision", None) is blocker:
                self._last_order_decision = {}
            return None

        lock = getattr(self, "_lock", None)
        if lock is None:
            remaining = _clear_if_current()
        else:
            try:
                with lock:
                    remaining = _clear_if_current()
            except Exception:
                return blocker

        if remaining is None:
            logger = getattr(self, "_logger", None)
            log = getattr(logger, "info", None)
            if callable(log):
                log(
                    "ENTRY_RECONCILIATION_RESOLVED_FLAT "
                    "order_id=%s symbol=%s status=%s",
                    order_id,
                    symbol,
                    status,
                    extra={
                        "event": "ENTRY_RECONCILIATION_RESOLVED_FLAT",
                        "order_id": order_id,
                        "symbol": symbol,
                        "order_status": status,
                    },
                )
        return remaining

    def current_entry_blocker(self) -> Mapping[str, Any] | None:
        blocker = _current_entry_blocker(self)
        if not isinstance(blocker, Mapping):
            return None
        return self._release_resolved_entry_reconciliation_blocker(blocker)

    def _blocked(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return block_result(
            self,
            _core,
            _core.OrderManager.place_order,
            method_name,
            args,
            kwargs,
        )

    def submit_trade_plan_result(self, plan: Any) -> Any:
        _enrich_trade_plan_exit_provenance(plan, finalize_tp1=False)
        blocked = self._blocked("submit_trade_plan_result", (plan,), {})
        if blocked is not NO_BLOCK:
            return blocked
        return _recover_submit(
            _submit_core_with_exit_provenance,
            self,
            plan,
        )

    def submit_trade_plan(self, *args: Any, **kwargs: Any) -> Any:
        blocked = self._blocked("submit_trade_plan", args, kwargs)
        if blocked is not NO_BLOCK:
            return blocked
        return super().submit_trade_plan(*args, **kwargs)

    def place_managed_order_result(self, *args: Any, **kwargs: Any) -> Any:
        blocked = self._blocked("place_managed_order_result", args, kwargs)
        if blocked is not NO_BLOCK:
            return blocked
        previous = getattr(self, "_managed_strategy_name", None)
        self._managed_strategy_name = str(kwargs.get("strategy_name") or "runner")
        try:
            return super().place_managed_order_result(*args, **kwargs)
        finally:
            if previous is None:
                self.__dict__.pop("_managed_strategy_name", None)
            else:
                self._managed_strategy_name = previous

    def place_managed_order(self, *args: Any, **kwargs: Any) -> Any:
        blocked = self._blocked("place_managed_order", args, kwargs)
        if blocked is not NO_BLOCK:
            return blocked
        return super().place_managed_order(*args, **kwargs)

    def place_order(self, *args: Any, **kwargs: Any) -> Any:
        values = _bind_place_order(args, kwargs)
        if values is None:
            effective_args = args
            effective_kwargs = _normalise_protective_intent_kwargs(kwargs)
        else:
            effective_args = ()
            effective_kwargs = _normalise_protective_intent_kwargs(values)
        if effective_kwargs != (values if values is not None else dict(kwargs)):
            logger = getattr(self, "_logger", None)
            log = getattr(logger, "info", None)
            if callable(log):
                with suppress(Exception):
                    log(
                        "PROTECTIVE_ORDER_INTENT_NORMALISED symbol=%s tag=%s intent=%s",
                        effective_kwargs.get("symbol"),
                        effective_kwargs.get("tag"),
                        effective_kwargs.get("intent"),
                        extra={
                            "event": "PROTECTIVE_ORDER_INTENT_NORMALISED",
                            "symbol": effective_kwargs.get("symbol"),
                            "tag": effective_kwargs.get("tag"),
                            "intent": effective_kwargs.get("intent"),
                        },
                    )
        return _place_order_with_prebroker_reservation_cleanup(
            self,
            self._place_order_native,
            *effective_args,
            **effective_kwargs,
        )

    def _place_order_native(self, *args: Any, **kwargs: Any) -> Any:
        effective_kwargs = dict(kwargs)
        managed_strategy = getattr(self, "_managed_strategy_name", None)
        current_strategy = (
            str(effective_kwargs.get("strategy_name") or "").strip().lower()
        )
        if managed_strategy and current_strategy in {"", "manual"}:
            effective_kwargs["strategy_name"] = managed_strategy
        blocked = self._blocked("place_order", args, effective_kwargs)
        if blocked is not NO_BLOCK:
            return blocked
        effective_kwargs = _cost_adjust_entry_target(self, effective_kwargs)
        cleaned_kwargs, identity = _strip_exit_identity_kwargs(effective_kwargs)
        if identity:
            self._last_exit_identity_kwargs = dict(identity)
        # Keep setup freshness exact and call-scoped. Core still receives the
        # unchanged order kwargs; the structural stop guard can recover the
        # strategy setup only while this specific signal is under risk review.
        with order_setup_context(cleaned_kwargs.get("signal_id")):
            result = super().place_order(*args, **cleaned_kwargs)
        # A non-empty native placement result is positive broker evidence that the
        # order endpoint accepted a real runtime request.  This is telemetry only:
        # it never arms trading and never substitutes for readiness/risk gates.
        if result:
            self.order_endpoint_verified = True
            self.broker_order_endpoint_verified = True
        return result

    def _sync_filled_exit_bracket(
        self,
        order: Any,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Ask the canonical bracket owner to converge a confirmed reducing fill."""

        if order is None:
            return
        status = getattr(order, "status", None)
        status_name = str(getattr(status, "name", status) or "").strip().upper()
        intent = str(getattr(order, "intent", "") or "").strip().upper()
        if status_name != "FILLED" or intent not in {"EXIT", "REDUCE"}:
            return
        bracket_manager = getattr(self, "_bracket_manager", None)
        reconcile = getattr(bracket_manager, "reconcile_filled_exit_order", None)
        if not callable(reconcile):
            return
        try:
            reconcile(order, dict(payload or {}))
        except Exception as exc:  # noqa: BLE001 - fill is already broker truth
            logger = getattr(self, "_logger", None)
            log = getattr(logger, "error", None)
            if callable(log):
                log(
                    "EXIT_BRACKET_TERMINAL_RECONCILE_FAILED order_id=%s symbol=%s error=%s",
                    getattr(order, "order_id", ""),
                    getattr(order, "symbol", ""),
                    exc,
                    extra={
                        "event": "EXIT_BRACKET_TERMINAL_RECONCILE_FAILED",
                        "order_id": getattr(order, "order_id", ""),
                        "symbol": getattr(order, "symbol", ""),
                        "error_type": type(exc).__name__,
                    },
                )

    def _apply_broker_order_update(self, order_update: dict[str, Any]) -> Any:
        updated = super()._apply_broker_order_update(order_update)
        if updated is not None:
            try:
                _finalize_partial_entry(self, updated, order_update)
            except Exception as exc:
                logger = getattr(self, "_logger", None)
                log = getattr(logger, "error", None)
                if callable(log):
                    log(
                        "ENTRY_PARTIAL_FILL_RECONCILE_FAILED order_id=%s error=%s",
                        getattr(updated, "order_id", ""),
                        exc,
                        extra={
                            "event": "ENTRY_PARTIAL_FILL_RECONCILE_FAILED",
                            "order_id": getattr(updated, "order_id", ""),
                            "error_type": type(exc).__name__,
                        },
                    )
            _maybe_reprice_open_entry(self, updated, order_update)
        self._sync_filled_exit_bracket(updated, order_update)
        return updated

    def _update_from_response(
        self,
        order: Any,
        payload: dict[str, Any],
    ) -> Any:
        updated = super()._update_from_response(order, payload)
        try:
            _finalize_partial_entry(self, updated, payload)
        except Exception as exc:
            logger = getattr(self, "_logger", None)
            log = getattr(logger, "error", None)
            if callable(log):
                log(
                    "ENTRY_PARTIAL_FILL_RECONCILE_FAILED order_id=%s error=%s",
                    getattr(order, "order_id", ""),
                    exc,
                    extra={
                        "event": "ENTRY_PARTIAL_FILL_RECONCILE_FAILED",
                        "order_id": getattr(order, "order_id", ""),
                        "error_type": type(exc).__name__,
                    },
                )
        if updated is not None:
            _maybe_reprice_open_entry(self, updated, payload)
        self._sync_filled_exit_bracket(updated, payload)
        return updated


__all__ = [
    "RuntimeOrderManager",
    "_normalise_protective_intent_kwargs",
    "_place_order_with_prebroker_reservation_cleanup",
    "_cost_adjust_entry_target",
    "_maybe_reprice_open_entry",
    "_enrich_trade_plan_exit_provenance",
    "_strip_exit_identity_kwargs",
    "_submit_core_with_exit_provenance",
]
