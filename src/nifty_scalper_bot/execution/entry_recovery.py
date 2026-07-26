"""Bounded recovery for the authoritative runner-facing entry path.

Wraps existing OrderManager methods without replacing the class. Recovery stays
bounded, reconciles uncertain broker outcomes before retrying, keeps quantities
whole-lot, and protects partial entry fills.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import replace
from typing import Any, Callable, Mapping

from nifty_scalper_bot.execution.broker_recovery import (
    RecoveryAction,
    RecoveryDecision,
    decide_recovery,
)
from nifty_scalper_bot.utils.lot_size import resolve_lot_size


def _log(manager: Any, level: str, message: str, *args: Any, **extra: Any) -> None:
    logger = getattr(manager, "_logger", None)
    fn = getattr(logger, level, None)
    if callable(fn):
        fn(
            message,
            *args,
            extra={"event": extra.pop("event", "ENTRY_RECOVERY"), **extra},
        )


def _result_like(
    previous: Any,
    *,
    accepted: bool,
    order_id: str | None,
    reason: str,
    details: Mapping[str, Any],
    broker_attempted: bool,
) -> Any:
    cls = type(previous)
    try:
        return cls(
            accepted=accepted,
            order_id=order_id,
            reason=reason,
            details=dict(details),
            broker_attempted=broker_attempted,
        )
    except Exception:
        from nifty_scalper_bot.execution.order_manager import TradePlanSubmitResult

        return TradePlanSubmitResult(
            accepted=accepted,
            order_id=order_id,
            reason=reason,
            details=dict(details),
            broker_attempted=broker_attempted,
        )


def _failure_text(manager: Any, result: Any) -> str:
    fields = [str(getattr(result, "reason", "") or "")]
    details = getattr(result, "details", {}) or {}
    if isinstance(details, Mapping):
        fields.extend(str(value) for value in details.values())
    decision = getattr(manager, "_last_order_decision", {}) or {}
    if isinstance(decision, Mapping):
        fields.extend(str(value) for value in decision.values())
    return " | ".join(fields)


def _symbol_key(value: object) -> str:
    return str(value or "").strip().upper().split(":", 1)[-1]


def _provider(manager: Any) -> Any | None:
    return getattr(manager, "_data_hub", None) or getattr(manager, "_market_data", None)


def _fresh_quote(manager: Any, symbol: str, trace_id: str | None) -> dict[str, Any]:
    provider = _provider(manager)
    if provider is None:
        return {}
    for name in ("refresh_quote_now", "pull_quote", "refresh_quote"):
        fn = getattr(provider, name, None)
        if not callable(fn):
            continue
        try:
            refreshed = fn(symbol, trace_id=trace_id)
        except TypeError:
            try:
                refreshed = fn(symbol)
            except Exception:
                continue
        except Exception:
            continue
        if isinstance(refreshed, Mapping) and refreshed:
            return dict(refreshed)
    for name in ("get_quote", "get_latest_tick", "get_tick"):
        fn = getattr(provider, name, None)
        if callable(fn):
            try:
                quote = fn(symbol)
            except Exception:
                continue
            if isinstance(quote, Mapping) and quote:
                return dict(quote)
    return {}


def _positive(quote: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        try:
            value = float(quote.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def _manager_live_mode(manager: Any) -> bool:
    checker = getattr(manager, "is_live_mode", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return True
    execution_mode = (
        str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
    )
    live_enabled = str(
        os.getenv("ENABLE_LIVE", "") or os.getenv("ENABLE_LIVE_TRADING", "")
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    shadow_or_paper = str(os.getenv("SHADOW_MODE", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    } or str(
        os.getenv("PAPER_MODE", "") or os.getenv("PAPER__ENABLED", "")
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    return execution_mode == "LIVE" and live_enabled and not shadow_or_paper


def _is_entry_intent(plan: Any) -> bool:
    intent = str(getattr(plan, "intent", "ENTRY") or "ENTRY").strip().upper()
    return intent in {"", "ENTRY", "UNKNOWN"}


def _is_live_entry(manager: Any, plan: Any) -> bool:
    return _manager_live_mode(manager) and _is_entry_intent(plan)


def _entry_price(manager: Any, plan: Any, quote: Mapping[str, Any]) -> float | None:
    side = str(getattr(plan, "side", "BUY") or "BUY").upper()
    bid = _positive(quote, "bid", "best_bid", "buy_price")
    ask = _positive(quote, "ask", "best_ask", "offer", "sell_price")

    if _is_live_entry(manager, plan):
        # Live option entries must be executable from fresh two-sided quotes.
        # LTP is allowed for diagnostics and emergency exits elsewhere, but not
        # for retrying a new risk-increasing entry after broker rejection.
        if bid is None or ask is None or ask < bid:
            return None
        if side != "BUY":
            return None
        return float(ask)

    keys = (
        ("ask", "best_ask", "offer", "sell_price")
        if side == "BUY"
        else ("bid", "best_bid", "buy_price")
    )
    return _positive(quote, *keys) or _positive(
        quote, "ltp", "last_price", "last_traded_price", "price"
    )


def _tick_size(manager: Any, symbol: str) -> float:
    resolver = getattr(manager, "_instrument_resolver", None)
    for name in ("get_tick_size", "tick_size", "get_tick"):
        fn = getattr(resolver, name, None) if resolver is not None else None
        if callable(fn):
            try:
                value = float(fn(symbol) or 0.0)
            except Exception:
                continue
            if math.isfinite(value) and value > 0:
                return value
    try:
        return max(float(os.getenv("ORDER_TICK_SIZE", "0.05") or 0.05), 0.01)
    except ValueError:
        return 0.05


def _tick(value: float, size: float = 0.05) -> float:
    return round(round(float(value) / size) * size, 2)


def _valid_geometry(plan: Any) -> bool:
    try:
        entry, stop, target = (
            float(plan.entry_price or 0.0),
            float(plan.stop_loss or 0.0),
            float(plan.take_profit or 0.0),
        )
    except (TypeError, ValueError):
        return False
    if min(entry, stop, target) <= 0:
        return False
    return (
        stop < entry < target
        if str(plan.side).upper() == "BUY"
        else target < entry < stop
    )


def _reprice_deviation_pct(plan: Any, fresh_price: float | None) -> float | None:
    try:
        original = float(getattr(plan, "entry_price", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    return (
        None
        if fresh_price is None or original <= 0
        else abs(float(fresh_price) - original) / original * 100.0
    )


def _max_reprice_deviation_pct(manager: Any) -> float:
    for attr in (
        "_max_entry_reprice_deviation_pct",
        "_entry_recovery_max_reprice_deviation_pct",
    ):
        try:
            value = float(getattr(manager, attr))
        except (AttributeError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    for key in (
        "ENTRY_RECOVERY_MAX_REPRICE_DEVIATION_PCT",
        "MAX_ENTRY_REPRICE_DEVIATION_PCT",
    ):
        raw = os.getenv(key)
        if raw:
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value) and value > 0:
                return value
    return 2.5 if _manager_live_mode(manager) else 8.0


def _rebuild_plan(
    manager: Any,
    plan: Any,
    quote: Mapping[str, Any],
    decision: RecoveryDecision,
    *,
    quantity: int | None = None,
) -> Any | None:
    fresh = _entry_price(manager, plan, quote)
    if _is_live_entry(manager, plan) and fresh is None:
        _log(
            manager,
            "warning",
            "ENTRY_RECOVERY_LIVE_QUOTE_BLOCKED symbol=%s reason=bid_ask_required",
            getattr(plan, "symbol", ""),
            event="ENTRY_RECOVERY_LIVE_QUOTE_BLOCKED",
            symbol=getattr(plan, "symbol", ""),
            side=getattr(plan, "side", ""),
        )
        return None
    deviation = _reprice_deviation_pct(plan, fresh)
    max_deviation = _max_reprice_deviation_pct(manager)
    if deviation is not None and deviation > max_deviation:
        _log(
            manager,
            "warning",
            "ENTRY_RECOVERY_REPRICE_BLOCKED symbol=%s deviation_pct=%.2f max_pct=%.2f",
            getattr(plan, "symbol", ""),
            deviation,
            max_deviation,
            event="ENTRY_RECOVERY_REPRICE_BLOCKED",
            deviation_pct=deviation,
            max_deviation_pct=max_deviation,
        )
        return None

    callback = getattr(manager, "_trade_plan_rebuilder", None)
    if callable(callback):
        try:
            rebuilt = callback(plan, dict(quote), decision)
        except TypeError:
            rebuilt = callback(plan, dict(quote))
        if rebuilt is not None and _valid_geometry(rebuilt):
            return (
                replace(rebuilt, quantity=quantity) if quantity is not None else rebuilt
            )

    engine = getattr(manager, "_indicator_engine", None)
    for name in ("rebuild_trade_plan", "revalidate_trade_plan"):
        fn = getattr(engine, name, None) if engine is not None else None
        if not callable(fn):
            continue
        try:
            rebuilt = fn(plan=plan, quote=dict(quote), decision=decision)
        except TypeError:
            try:
                rebuilt = fn(plan, dict(quote))
            except Exception:
                continue
        except Exception:
            continue
        if rebuilt is not None and _valid_geometry(rebuilt):
            return (
                replace(rebuilt, quantity=quantity) if quantity is not None else rebuilt
            )

    try:
        old_entry, old_stop, old_target = (
            float(plan.entry_price or 0.0),
            float(plan.stop_loss or 0.0),
            float(plan.take_profit or 0.0),
        )
    except (TypeError, ValueError):
        return None
    if fresh is None or old_entry <= 0 or old_stop <= 0 or old_target <= 0:
        return None
    if str(plan.side).upper() == "BUY":
        risk, reward = old_entry - old_stop, old_target - old_entry
        stop, target = fresh - risk, fresh + reward
    else:
        risk, reward = old_stop - old_entry, old_entry - old_target
        stop, target = fresh + risk, fresh - reward
    if risk <= 0 or reward <= 0:
        return None
    size = _tick_size(manager, str(getattr(plan, "symbol", "")))
    rebuilt = replace(
        plan,
        entry_price=_tick(fresh, size),
        stop_loss=_tick(stop, size),
        take_profit=_tick(target, size),
        quantity=int(quantity if quantity is not None else plan.quantity),
    )
    return rebuilt if _valid_geometry(rebuilt) else None


def _available_margin(manager: Any) -> float | None:
    fn = getattr(manager, "_resolve_available_margin", None)
    if not callable(fn):
        return None
    try:
        value = fn()
    except Exception:
        return None
    raw = value[0] if isinstance(value, tuple) else value
    try:
        margin = float(raw or 0.0)
    except (TypeError, ValueError):
        return None
    return margin if math.isfinite(margin) and margin > 0 else None


def _instrument_lookup(manager: Any) -> Callable[[str], int] | None:
    resolver = getattr(manager, "_instrument_resolver", None)
    candidates = (
        getattr(manager, "_lot_size_for_symbol", None),
        (
            getattr(resolver, "lot_size_for_symbol", None)
            if resolver is not None
            else None
        ),
        getattr(resolver, "get_lot_size", None) if resolver is not None else None,
        getattr(resolver, "lot_size", None) if resolver is not None else None,
    )
    for candidate in candidates:
        if callable(candidate):
            fn = candidate

            def _lookup(symbol: str) -> int:
                return int(fn(symbol) or 0)

            return _lookup
    return None


def _lot_size(manager: Any, symbol: str) -> int:
    try:
        lot, _source = resolve_lot_size(symbol, _instrument_lookup(manager))
        return max(1, int(lot))
    except Exception as exc:
        if _manager_live_mode(manager):
            raise ValueError(f"lot_size_unresolved:{symbol}") from exc
        for key in (
            "NIFTY_LOT_SIZE",
            "INSTRUMENTS__NIFTY_LOT_SIZE",
            "DEFAULT_OPTION_LOT_SIZE",
        ):
            raw = os.getenv(key)
            if not raw:
                continue
            try:
                lot = int(raw)
            except ValueError:
                continue
            if lot > 0:
                return lot
        return 65 if "NIFTY" in _symbol_key(symbol) else 1


def _whole_lot_quantity(quantity: int, lot: int) -> int:
    return 0 if quantity <= 0 or lot <= 0 else (int(quantity) // int(lot)) * int(lot)


def _affordable_quantity(manager: Any, plan: Any, fresh_price: float) -> int:
    available = _available_margin(manager)
    if available is None or fresh_price <= 0:
        return 0
    lot = _lot_size(manager, str(plan.symbol))
    factor = max(float(getattr(manager, "_margin_factor", 1.0) or 1.0), 1.0)
    buffer = min(
        max(float(getattr(manager, "_margin_buffer", 0.95) or 0.95), 0.5), 0.98
    )
    per_lot = fresh_price * lot * factor
    lots = int((available * buffer) // per_lot) if per_lot > 0 else 0
    return min(int(plan.quantity), max(lots, 0) * lot)


def _freeze_quantity(manager: Any, plan: Any) -> int:
    resolver = getattr(manager, "_instrument_resolver", None)
    cap = 0
    for name in ("get_freeze_quantity", "freeze_quantity", "get_freeze_limit"):
        fn = getattr(resolver, name, None) if resolver is not None else None
        if callable(fn):
            try:
                cap = int(fn(plan.symbol) or 0)
            except Exception:
                continue
            if cap > 0:
                break
    if cap <= 0:
        cap = int(os.getenv("ORDER_FREEZE_QUANTITY", "0") or 0)
    if cap <= 0:
        return 0
    return _whole_lot_quantity(
        min(int(plan.quantity), cap), _lot_size(manager, str(plan.symbol))
    )


def _broker_orders(manager: Any) -> list[Mapping[str, Any]] | None:
    broker = getattr(manager, "_broker", None)
    for name in ("get_orders", "orders"):
        fn = getattr(broker, name, None) if broker is not None else None
        if callable(fn):
            try:
                rows = fn() or []
            except Exception:
                return None
            return [row for row in rows if isinstance(row, Mapping)]
    return None


def _broker_positions(manager: Any) -> list[Mapping[str, Any]] | None:
    broker = getattr(manager, "_broker", None)
    fn = getattr(broker, "get_positions", None) if broker is not None else None
    if not callable(fn):
        return None
    try:
        payload = fn() or []
    except Exception:
        return None
    if isinstance(payload, Mapping):
        payload = (
            payload.get("net") or payload.get("day") or payload.get("positions") or []
        )
    return [row for row in payload if isinstance(row, Mapping)]


def _reconcile_intent(
    manager: Any, plan: Any
) -> tuple[str, str | None, dict[str, Any]]:
    keys = [
        str(getattr(plan, name, "") or "") for name in ("signal_id", "trace_id", "tag")
    ]
    finder = getattr(manager, "_find_open_order", None)
    if callable(finder):
        for key in keys:
            if not key:
                continue
            try:
                order = finder(key)
            except Exception:
                order = None
            if isinstance(order, Mapping):
                oid = str(order.get("order_id") or order.get("id") or "")
                if oid:
                    return "order_found", oid, {"order": dict(order)}

    orders = _broker_orders(manager)
    if orders is None:
        return "unknown", None, {"reason": "orders_unavailable"}
    wanted_symbol = _symbol_key(plan.symbol)
    wanted = {key.upper() for key in keys if key}
    for order in orders:
        if (
            _symbol_key(order.get("tradingsymbol") or order.get("symbol"))
            != wanted_symbol
        ):
            continue
        candidates = {
            str(order.get(key) or "").upper()
            for key in ("tag", "client_order_id", "guid")
        }
        if wanted and not any(
            key == candidate or (key and candidate and key[-8:] in candidate)
            for key in wanted
            for candidate in candidates
        ):
            continue
        if str(order.get("status") or "").upper() not in {
            "REJECTED",
            "CANCELLED",
            "CANCELED",
            "EXPIRED",
        }:
            oid = str(order.get("order_id") or order.get("id") or "")
            return "order_found", oid or None, {"order": dict(order)}

    positions = _broker_positions(manager)
    if positions is None:
        return "unknown", None, {"reason": "positions_unavailable"}
    for position in positions:
        if (
            _symbol_key(position.get("tradingsymbol") or position.get("symbol"))
            != wanted_symbol
        ):
            continue
        try:
            qty = abs(
                int(
                    float(
                        position.get("quantity", position.get("net_quantity", 0)) or 0
                    )
                )
            )
        except (TypeError, ValueError):
            return "unknown", None, {"reason": "position_quantity_invalid"}
        if qty > 0:
            return "exposure_found", None, {"position": dict(position), "quantity": qty}
    return "absent", None, {}


def _annotate(result: Any, decision: RecoveryDecision, **extra: Any) -> Any:
    details = dict(getattr(result, "details", {}) or {})
    details["entry_recovery"] = {
        "failure": decision.failure.value,
        "action": decision.action.value,
        "retryable": decision.retryable,
        **extra,
    }
    try:
        result.details = details
        return result
    except Exception:
        return _result_like(
            result,
            accepted=bool(getattr(result, "accepted", False)),
            order_id=getattr(result, "order_id", None),
            reason=str(getattr(result, "reason", "unknown")),
            details=details,
            broker_attempted=bool(getattr(result, "broker_attempted", False)),
        )


def _first_effective_quantity(result: Any, plan: Any) -> int | None:
    """First gate-approved quantity for THIS submission, or None.

    Provenance is read only from the result the caller was given -- never
    from manager attributes, globals, thread-local state, latest-submission
    caches or broker error text -- so one trade can never consume another
    trade's sizing record. A retry may reduce further but must never restore
    quantity the first entry gate removed.
    Returns None when no record exists (legacy path) and raises
    _ProvenanceMismatch when a record exists but does not match this plan.
    """
    record = getattr(result, "details", None)
    if not isinstance(record, dict) or "entry_sizing_effective_quantity" not in record:
        return None
    try:
        effective = int(record.get("entry_sizing_effective_quantity", 0) or 0)
        recorded_original = int(record.get("entry_sizing_requested_quantity", 0) or 0)
        lot = int(record.get("entry_sizing_lot_size", 0) or 0)
    except (TypeError, ValueError):
        raise _ProvenanceMismatch("entry_sizing_provenance_invalid") from None

    plan_qty = int(getattr(plan, "quantity", 0) or 0)
    if (
        recorded_original != plan_qty
        or str(record.get("entry_sizing_symbol") or "") != str(
            getattr(plan, "symbol", "") or ""
        )
        or effective <= 0
        or lot <= 0
        or effective > recorded_original
        or effective % lot != 0
    ):
        raise _ProvenanceMismatch("entry_sizing_provenance_invalid")

    for record_key, plan_attr in (
        ("entry_sizing_trace_id", "trace_id"),
        ("entry_sizing_signal_id", "signal_id"),
        ("entry_sizing_trade_lifecycle_id", "trade_lifecycle_id"),
    ):
        recorded = record.get(record_key)
        actual = getattr(plan, plan_attr, None)
        if recorded is not None and actual is not None and recorded != actual:
            raise _ProvenanceMismatch("entry_sizing_provenance_invalid")
    return effective


class _ProvenanceMismatch(Exception):
    """Entry-sizing record does not belong to this plan."""


def _recover_submit(original: Callable[..., Any], manager: Any, plan: Any) -> Any:
    result = original(manager, plan)
    if (
        bool(getattr(result, "accepted", False))
        or not bool(getattr(result, "broker_attempted", False))
        or getattr(manager, "_entry_recovery_active", False)
    ):
        return result
    decision = decide_recovery(_failure_text(manager, result))
    if not decision.retryable:
        return _annotate(result, decision, outcome="terminal")

    setattr(manager, "_entry_recovery_active", True)
    try:
        if decision.reconcile_first:
            state, order_id, evidence = _reconcile_intent(manager, plan)
            if state == "order_found" and order_id:
                return _result_like(
                    result,
                    accepted=True,
                    order_id=order_id,
                    reason="broker_order_reconciled",
                    details={
                        "entry_recovery": {
                            "failure": decision.failure.value,
                            "action": decision.action.value,
                            "outcome": state,
                            **evidence,
                        }
                    },
                    broker_attempted=True,
                )
            if state == "exposure_found":
                return _result_like(
                    result,
                    accepted=False,
                    order_id=None,
                    reason="entry_exposure_reconciliation_required",
                    details={"entry_recovery": {"outcome": state, **evidence}},
                    broker_attempted=True,
                )
            if state != "absent":
                return _result_like(
                    result,
                    accepted=False,
                    order_id=None,
                    reason="entry_order_state_ambiguous",
                    details={"entry_recovery": {"outcome": state, **evidence}},
                    broker_attempted=True,
                )

        quote = _fresh_quote(manager, str(plan.symbol), getattr(plan, "trace_id", None))
        fresh = _entry_price(manager, plan, quote)
        if _is_live_entry(manager, plan) and fresh is None:
            return _annotate(result, decision, outcome="live_bid_ask_unavailable")
        quantity: int | None = None
        if decision.action is RecoveryAction.RESIZE_AND_REVALIDATE:
            quantity = _affordable_quantity(manager, plan, float(fresh or 0.0))
            if quantity < _lot_size(manager, str(plan.symbol)):
                return _annotate(result, decision, outcome="no_affordable_lot")
        elif decision.action is RecoveryAction.CAP_QUANTITY_AND_REVALIDATE:
            quantity = _freeze_quantity(manager, plan)
            if quantity <= 0 or quantity >= int(plan.quantity):
                return _annotate(result, decision, outcome="freeze_cap_unavailable")
        elif decision.action is RecoveryAction.BACKOFF_AND_REVALIDATE:
            delay = min(
                max(float(os.getenv("ENTRY_RECOVERY_BACKOFF_SECONDS", "0.25")), 0.0),
                1.0,
            )
            if delay:
                time.sleep(delay)

        rebuilt = _rebuild_plan(manager, plan, quote, decision, quantity=quantity)
        if rebuilt is not None:
            try:
                frozen = _first_effective_quantity(result, plan)
            except _ProvenanceMismatch:
                return _annotate(
                    result, decision, outcome="entry_sizing_provenance_invalid"
                )
            if frozen is not None:
                lot = _lot_size(manager, str(plan.symbol)) or 0
                capped = min(int(getattr(rebuilt, "quantity", 0) or 0), frozen)
                if capped <= 0 or (lot > 0 and capped % lot != 0):
                    return _annotate(
                        result, decision, outcome="frozen_quantity_invalid"
                    )
                if capped != int(getattr(rebuilt, "quantity", 0) or 0):
                    rebuilt = replace(rebuilt, quantity=capped)
                if lot > 0 and hasattr(rebuilt, "requested_lots"):
                    rebuilt = replace(
                        rebuilt,
                        requested_lots=capped // lot,
                        resolved_lot_size=lot,
                    )
        if rebuilt is None:
            details = {"fresh_price": fresh}
            deviation = _reprice_deviation_pct(plan, fresh)
            if deviation is not None:
                details.update(
                    {
                        "reprice_deviation_pct": round(deviation, 4),
                        "max_reprice_deviation_pct": _max_reprice_deviation_pct(
                            manager
                        ),
                    }
                )
            return _annotate(result, decision, outcome="rebuild_failed", **details)

        clearer = getattr(manager, "_clear_pending_signal", None)
        signal_id = str(getattr(plan, "signal_id", "") or "")
        if signal_id and callable(clearer):
            clearer(signal_id)
        retried = original(manager, rebuilt)
        return _annotate(
            retried,
            decision,
            outcome="retried_once",
            original_quantity=int(plan.quantity),
            retry_quantity=int(rebuilt.quantity),
            original_entry=getattr(plan, "entry_price", None),
            retry_entry=getattr(rebuilt, "entry_price", None),
        )
    finally:
        setattr(manager, "_entry_recovery_active", False)


def _is_entry_order(order: Any) -> bool:
    intent = str(getattr(order, "intent", "") or "").strip().upper()
    if intent and intent != "UNKNOWN":
        return intent == "ENTRY"
    tag = str(getattr(order, "tag", "") or "").upper()
    return not any(
        token in tag
        for token in ("EXIT", "STOP", "TARGET", "SQUARE", "GUARD", "SL_", "TP_")
    )


def quantity_to_exact_lots(quantity: int, lot_size: int) -> int | None:
    """Return exact lot count, or None when quantity is not lot-valid."""
    if quantity < 0 or lot_size <= 0:
        return None
    if quantity % lot_size != 0:
        return None
    return quantity // lot_size


def lots_to_quantity(lots: int, lot_size: int) -> int:
    """Convert lots to broker quantity without implicit coercion."""
    if lots < 0 or lot_size <= 0:
        raise ValueError("invalid_lot_conversion")
    return lots * lot_size


def _partial_state(order: Any) -> dict[str, Any]:
    state = getattr(order, "entry_lifecycle_state", None)
    if not isinstance(state, dict):
        requested_quantity = int(getattr(order, "quantity", 0) or 0)
        lot_size = int(getattr(order, "resolved_lot_size", 0) or 0)
        requested_lots = int(getattr(order, "requested_lots", 0) or 0)
        if requested_lots <= 0 and lot_size > 0:
            exact = quantity_to_exact_lots(requested_quantity, lot_size)
            requested_lots = exact or 0
        state = {
            "state": "ENTRY_OPEN",
            "requested_lots": requested_lots,
            "resolved_lot_size": lot_size,
            "requested_quantity": requested_quantity,
            "broker_filled_lots": 0,
            "broker_pending_lots": 0,
            "broker_cancelled_lots": 0,
            "broker_filled_quantity": 0,
            "broker_pending_quantity": 0,
            "broker_cancelled_quantity": 0,
            "protected_lots": 0,
            "protected_quantity": 0,
            "broker_position_lots": None,
            "broker_position_quantity": None,
            "cancel_requested_at": None,
            "terminal_confirmed_at": None,
            "last_reconcile_at": None,
            "last_error": None,
        }
        setattr(order, "entry_lifecycle_state", state)
    return state


def _payload_int(payload: Mapping[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value in (None, ""):
            continue
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            continue
    return int(default)


def _broker_position_quantity(manager: Any, symbol: str) -> int | None:
    positions = _broker_positions(manager)
    if positions is None:
        return None
    wanted = _symbol_key(symbol)
    for position in positions:
        if (
            _symbol_key(position.get("tradingsymbol") or position.get("symbol"))
            != wanted
        ):
            continue
        value = position.get("quantity", position.get("net_quantity"))
        if value in (None, ""):
            return None
        try:
            return abs(int(float(value)))
        except (TypeError, ValueError):
            return None
    return 0


def _protection_verified(
    result: Any, expected_lots: int, expected_quantity: int
) -> bool:
    if result is True:
        return True
    if result is None or result is False:
        return False
    accepted = bool(getattr(result, "accepted", False))
    protected_lots = getattr(result, "protected_lots", expected_lots)
    protected_quantity = getattr(result, "protected_quantity", expected_quantity)
    persisted = getattr(result, "persisted", True)
    try:
        return bool(
            accepted
            and int(protected_lots) == expected_lots
            and int(protected_quantity) == expected_quantity
            and persisted
        )
    except (TypeError, ValueError):
        return False


def _latch_entry_blocker(
    manager: Any,
    order: Any,
    reason: str,
    *,
    state: Mapping[str, Any],
    broker_attempted: bool = True,
) -> None:
    details = {
        "broker_attempted": broker_attempted,
        "retryable": False,
        "trade_lifecycle_id": getattr(order, "trade_lifecycle_id", None),
        "symbol": getattr(order, "symbol", None),
        "order_id": getattr(order, "order_id", None),
        "bracket_id": getattr(order, "bracket_id", None),
        "requested_quantity": state.get("requested_quantity"),
        "requested_lots": state.get("requested_lots"),
        "resolved_lot_size": state.get("resolved_lot_size"),
        "filled_quantity": state.get("broker_filled_quantity"),
        "filled_lots": state.get("broker_filled_lots"),
        "protected_quantity": state.get("protected_quantity"),
        "protected_lots": state.get("protected_lots"),
        "broker_position_quantity": state.get("broker_position_quantity"),
        "broker_position_lots": state.get("broker_position_lots"),
        "entry_lifecycle_state": state.get("state"),
    }
    blocker = {
        "block_reason": reason,
        "broker_attempted": broker_attempted,
        "details": details,
    }
    setattr(manager, "_entry_lifecycle_blocker", blocker)
    setattr(manager, "_last_order_decision", blocker)


def current_entry_blocker(manager: Any) -> Mapping[str, Any] | None:
    blocker = getattr(manager, "_entry_lifecycle_blocker", None)
    return blocker if isinstance(blocker, Mapping) else None


def _resolve_order_lot_size(manager: Any, order: Any, state: Mapping[str, Any]) -> int:
    lot_size = int(
        getattr(order, "resolved_lot_size", 0) or state.get("resolved_lot_size") or 0
    )
    if lot_size > 0:
        return lot_size
    symbol = str(getattr(order, "symbol", "") or "")
    if symbol:
        resolved = _lot_size(manager, symbol)
        if resolved > 0:
            return resolved
    raise ValueError("lot_size_unresolved")


def _finalize_partial_entry(
    manager: Any, order: Any, payload: Mapping[str, Any]
) -> None:
    if not _is_entry_order(order):
        return
    raw_status = payload.get("status") or getattr(
        getattr(order, "status", None), "name", getattr(order, "status", "")
    )
    status = str(raw_status or "").strip().upper().replace("_", " ")
    handled = {
        "OPEN",
        "PARTIALLY FILLED",
        "PARTIALLY_FILLED",
        "CANCEL PENDING",
        "CANCELLED",
        "CANCELED",
        "COMPLETE",
        "FILLED",
        "REJECTED",
        "EXPIRED",
    }
    if status not in handled:
        return
    order_id = str(getattr(order, "order_id", "") or "")
    if not order_id:
        return

    state = _partial_state(order)
    now = time.time()
    state["last_reconcile_at"] = now
    try:
        lot_size = _resolve_order_lot_size(manager, order, state)
    except ValueError:
        state["state"] = "ENTRY_RECONCILIATION_UNKNOWN"
        state["last_error"] = "lot_size_unresolved"
        _latch_entry_blocker(manager, order, "lot_size_unresolved", state=state)
        return

    requested_quantity = int(
        state.get("requested_quantity") or getattr(order, "quantity", 0) or 0
    )
    requested_lots = int(
        getattr(order, "requested_lots", 0) or state.get("requested_lots") or 0
    )
    if requested_lots <= 0:
        exact_requested = quantity_to_exact_lots(requested_quantity, lot_size)
        if exact_requested is None:
            state["state"] = "ENTRY_RECONCILIATION_UNKNOWN"
            state["last_error"] = "invalid_entry_lot_quantity"
            _latch_entry_blocker(
                manager, order, "invalid_entry_lot_quantity", state=state
            )
            return
        requested_lots = exact_requested
    requested_quantity = lots_to_quantity(requested_lots, lot_size)
    state.update(
        {
            "requested_lots": requested_lots,
            "resolved_lot_size": lot_size,
            "requested_quantity": requested_quantity,
        }
    )

    broker_filled = _payload_int(
        payload,
        "filled_quantity",
        "filled",
        default=int(getattr(order, "filled_quantity", 0) or 0),
    )
    broker_pending = _payload_int(
        payload,
        "pending_quantity",
        "pending",
        default=max(requested_quantity - broker_filled, 0),
    )
    broker_cancelled = _payload_int(
        payload,
        "cancelled_quantity",
        "cancelled",
        default=int(state.get("broker_cancelled_quantity") or 0),
    )
    filled_lots = quantity_to_exact_lots(broker_filled, lot_size)
    pending_lots = quantity_to_exact_lots(broker_pending, lot_size)
    cancelled_lots = quantity_to_exact_lots(broker_cancelled, lot_size)
    if (
        filled_lots is None
        or pending_lots is None
        or cancelled_lots is None
        or filled_lots > requested_lots
    ):
        state.update(
            {
                "state": "ENTRY_RECONCILIATION_UNKNOWN",
                "broker_filled_quantity": broker_filled,
                "broker_pending_quantity": broker_pending,
                "broker_cancelled_quantity": broker_cancelled,
                "last_error": "broker_quantity_lot_invariant_violation",
            }
        )
        _latch_entry_blocker(
            manager, order, "broker_quantity_lot_invariant_violation", state=state
        )
        _log(
            manager,
            "critical",
            "BROKER_QUANTITY_LOT_INVARIANT_VIOLATION order_id=%s symbol=%s filled=%s pending=%s cancelled=%s lot_size=%s",
            order_id,
            getattr(order, "symbol", ""),
            broker_filled,
            broker_pending,
            broker_cancelled,
            lot_size,
            event="BROKER_QUANTITY_LOT_INVARIANT_VIOLATION",
        )
        return

    previous_lots = int(state.get("broker_filled_lots") or 0)
    if filled_lots < previous_lots:
        state["state"] = "ENTRY_RECONCILIATION_UNKNOWN"
        state["last_error"] = "entry_fill_quantity_regressed"
        _latch_entry_blocker(
            manager, order, "entry_fill_quantity_regressed", state=state
        )
        return

    state.update(
        {
            "broker_filled_lots": filled_lots,
            "broker_pending_lots": pending_lots,
            "broker_cancelled_lots": cancelled_lots,
            "broker_filled_quantity": broker_filled,
            "broker_pending_quantity": broker_pending,
            "broker_cancelled_quantity": broker_cancelled,
        }
    )

    terminal = status in {
        "COMPLETE",
        "FILLED",
        "CANCELLED",
        "CANCELED",
        "REJECTED",
        "EXPIRED",
    }
    if filled_lots == 0:
        if status in {"CANCELLED", "CANCELED"}:
            state["state"] = "ENTRY_CANCELLED"
            state["terminal_confirmed_at"] = now
            setattr(manager, "_entry_lifecycle_blocker", None)
        elif status in {"REJECTED", "EXPIRED"}:
            state["state"] = "ENTRY_REJECTED"
            state["terminal_confirmed_at"] = now
            setattr(manager, "_entry_lifecycle_blocker", None)
        elif status == "CANCEL PENDING":
            state["state"] = "ENTRY_CANCEL_PENDING"
            _latch_entry_blocker(
                manager, order, "entry_reconciliation_pending", state=state
            )
        else:
            state["state"] = "ENTRY_OPEN"
        return

    broker_position = _broker_position_quantity(
        manager, str(getattr(order, "symbol", ""))
    )
    state["broker_position_quantity"] = broker_position
    if broker_position is None:
        state["state"] = "ENTRY_RECONCILIATION_UNKNOWN"
        state["last_error"] = "broker_position_unknown_with_possible_exposure"
        _latch_entry_blocker(
            manager,
            order,
            "broker_position_unknown_with_possible_exposure",
            state=state,
        )
        return
    broker_position_lots = quantity_to_exact_lots(broker_position, lot_size)
    if broker_position_lots is None:
        state["state"] = "ENTRY_RECONCILIATION_UNKNOWN"
        state["last_error"] = "broker_quantity_lot_invariant_violation"
        _latch_entry_blocker(
            manager, order, "broker_quantity_lot_invariant_violation", state=state
        )
        return
    state["broker_position_lots"] = broker_position_lots
    if broker_position_lots != filled_lots:
        state["state"] = "ENTRY_RECONCILIATION_UNKNOWN"
        state["last_error"] = "entry_reconciliation_pending"
        _latch_entry_blocker(
            manager, order, "entry_reconciliation_pending", state=state
        )
        return

    if (
        requested_lots > 1
        and 0 < filled_lots < requested_lots
        and pending_lots > 0
        and not state.get("cancel_requested_at")
    ):
        cancel = getattr(getattr(manager, "_broker", None), "cancel_order", None)
        if callable(cancel):
            try:
                cancel(order_id)
            except TypeError:
                cancel(order_id=order_id)
            except Exception as exc:  # noqa: BLE001
                state["state"] = "ENTRY_RECONCILIATION_UNKNOWN"
                state["last_error"] = f"entry_cancel_failed:{type(exc).__name__}"
                _latch_entry_blocker(manager, order, "entry_cancel_failed", state=state)
                return
            state["cancel_requested_at"] = now
            state["state"] = "ENTRY_CANCEL_PENDING"

    try:
        average = float(
            payload.get("average_price")
            or payload.get("avg_price")
            or getattr(order, "fill_price", 0.0)
            or getattr(order, "average_price", 0.0)
            or 0.0
        )
    except (TypeError, ValueError):
        average = 0.0
    if average <= 0:
        state["state"] = "ENTRY_RECONCILIATION_UNKNOWN"
        state["last_error"] = "entry_reconciliation_pending"
        _latch_entry_blocker(
            manager, order, "entry_reconciliation_pending", state=state
        )
        return

    desired_quantity = lots_to_quantity(filled_lots, lot_size)
    if int(state.get("protected_lots") or 0) > filled_lots:
        state["state"] = "ENTRY_PROTECTION_FAILED"
        state["last_error"] = "entry_protection_failed"
        _latch_entry_blocker(manager, order, "entry_protection_failed", state=state)
        return

    confirmer = getattr(
        getattr(manager, "_bracket_manager", None), "confirm_partial_entry_fill", None
    )
    if not callable(confirmer):
        state["state"] = "ENTRY_PROTECTION_FAILED"
        state["last_error"] = "entry_protection_failed"
        _latch_entry_blocker(manager, order, "entry_protection_failed", state=state)
        return
    if int(state.get("protected_lots") or 0) < filled_lots:
        result = confirmer(order_id, desired_quantity, average)
        if not _protection_verified(result, filled_lots, desired_quantity):
            state["state"] = "ENTRY_PROTECTION_FAILED"
            state["last_error"] = "entry_protection_failed"
            _latch_entry_blocker(manager, order, "entry_protection_failed", state=state)
            _log(
                manager,
                "critical",
                "ENTRY_PROTECTION_FAILED order_id=%s symbol=%s filled_lots=%s quantity=%s",
                order_id,
                getattr(order, "symbol", ""),
                filled_lots,
                desired_quantity,
                event="ENTRY_PROTECTION_FAILED",
            )
            return
        state["protected_lots"] = filled_lots
        state["protected_quantity"] = desired_quantity

    if terminal:
        state["terminal_confirmed_at"] = now
    state["state"] = (
        "ENTRY_PROTECTED"
        if filled_lots == requested_lots
        else "ENTRY_PROTECTION_PENDING"
    )
    if terminal and state["protected_lots"] == broker_position_lots:
        setattr(manager, "_entry_lifecycle_blocker", None)
        reason = (
            "entry_full_lot_protected"
            if filled_lots == requested_lots
            else "entry_whole_lot_partial_protected"
        )
    else:
        reason = "entry_reconciliation_pending"
        _latch_entry_blocker(manager, order, reason, state=state)
    setattr(
        manager,
        "_last_order_decision",
        {"block_reason": reason, "broker_attempted": True, "details": dict(state)},
    )
    _log(
        manager,
        "critical",
        "ENTRY_LOT_PROTECTION_UPDATED order_id=%s symbol=%s filled_lots=%s protected_lots=%s state=%s",
        order_id,
        getattr(order, "symbol", ""),
        filled_lots,
        state.get("protected_lots"),
        state.get("state"),
        event="ENTRY_LOT_PROTECTION_UPDATED",
    )


def install_entry_recovery(order_manager_class: type[Any]) -> None:
    """Compatibility installer for legacy tests only.

    Production uses ``RuntimeOrderManager`` direct method overrides and must not
    rely on import-time monkey-patching. This hook is retained so older focused
    tests and external test doubles can exercise the same bounded recovery helper
    without altering production package import behaviour.
    """
    if getattr(order_manager_class, "_canonical_entry_recovery_installed", False):
        return
    original_submit = getattr(order_manager_class, "submit_trade_plan_result", None)
    if callable(original_submit):
        setattr(order_manager_class, "_entry_recovery_original_submit", original_submit)

        def submit_trade_plan_result(self: Any, plan: Any) -> Any:
            return _recover_submit(original_submit, self, plan)

        setattr(
            order_manager_class, "submit_trade_plan_result", submit_trade_plan_result
        )

    original_update = getattr(order_manager_class, "_update_from_response", None)
    if callable(original_update):
        setattr(order_manager_class, "_entry_recovery_original_update", original_update)

        def update_from_response(self: Any, order: Any, payload: dict[str, Any]) -> Any:
            updated = original_update(self, order, payload)
            try:
                _finalize_partial_entry(self, updated, payload)
            except Exception as exc:  # noqa: BLE001
                _log(
                    self,
                    "error",
                    "ENTRY_PARTIAL_FILL_RECONCILE_FAILED order_id=%s error=%s",
                    getattr(order, "order_id", ""),
                    exc,
                    event="ENTRY_PARTIAL_FILL_RECONCILE_FAILED",
                )
            return updated

        setattr(order_manager_class, "_update_from_response", update_from_response)

    def set_trade_plan_rebuilder(
        self: Any, callback: Callable[..., Any] | None
    ) -> None:
        setattr(self, "_trade_plan_rebuilder", callback)

    setattr(order_manager_class, "set_trade_plan_rebuilder", set_trade_plan_rebuilder)
    setattr(order_manager_class, "_canonical_entry_recovery_installed", True)


__all__ = [
    "install_entry_recovery",
    "_recover_submit",
    "_finalize_partial_entry",
    "current_entry_blocker",
    "quantity_to_exact_lots",
    "lots_to_quantity",
]
