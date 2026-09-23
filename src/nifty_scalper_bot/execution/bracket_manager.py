"""File purpose:
    Provide the stable public API for the canonical bracket and exit lifecycle.

Key responsibilities:
    - Re-export bracket state models and helpers from ``bracket_core``.
    - Expose ``BoundBracketManager`` as the single production bracket authority.

Operational constraints:
    - This facade must not own independent bracket state or exit execution logic.
    - Entry release remains blocked until the bound runtime confirms durable closure.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

from nifty_scalper_bot.execution import bracket_core as _core

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager  # noqa: E402
from nifty_scalper_bot.execution.ownership import BoundBracketManager  # noqa: E402


_original_tick_exchange_epoch = _core.tick_exchange_epoch


def _tick_exchange_epoch_with_receipt(tick):
    """Use broker event time first, then explicit receipt time; never invent time."""
    epoch = _original_tick_exchange_epoch(tick)
    if epoch is not None:
        return epoch
    for key in (
        "last_trade_time",
        "last_traded_time",
        "last_trade_timestamp",
        "received_at",
        "received_ts",
        "received_time",
    ):
        value = tick.get(key)
        if hasattr(value, "timestamp"):
            try:
                return float(value.timestamp())
            except (TypeError, ValueError, OSError):
                continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            return value / 1000.0 if value > 1e12 else value
    return None


_core.tick_exchange_epoch = _tick_exchange_epoch_with_receipt
tick_exchange_epoch = _tick_exchange_epoch_with_receipt

_original_confirm_entry_fill = BoundBracketManager.confirm_entry_fill


def _positive_filled_quantity(value):
    try:
        quantity = int(float(value or 0))
    except (TypeError, ValueError):
        return None
    return quantity if quantity > 0 else None


def _positive_fill_price(value):
    try:
        price = float(value or 0.0)
    except (TypeError, ValueError):
        return None
    return price if price > 0.0 else None


def _reconcile_confirmed_entry_quantity(self, order_id, bracket, filled_qty):
    """Reconcile bracket and ledger to cumulative fills within the request."""
    reported = _positive_filled_quantity(filled_qty)
    if bracket is None or reported is None:
        return False
    intent = str(getattr(bracket, "entry_order_intent", "ENTRY") or "ENTRY").upper()
    if intent not in {"ENTRY", "SCALE_IN", "REVERSAL"}:
        return False
    try:
        registered = int(getattr(bracket, "quantity", 0) or 0)
        requested = int(
            getattr(bracket, "requested_entry_quantity", 0) or registered
        )
    except (TypeError, ValueError):
        return False
    if (
        registered <= 0
        or requested <= 0
        or reported > requested
        or reported == registered
    ):
        return False

    reconciler = getattr(self, "_reconcile_entry_fill_quantity", None)
    if not callable(reconciler):
        return False
    if not bool(reconciler(bracket, reported)):
        return False

    # On the first callback the ledger has not been written yet; pre-shrinking
    # makes the existing ledger layer record the actual quantity. On a later
    # duplicate callback, explicitly shrink the already-persisted ENTRY row.
    if bool(getattr(bracket, "entry_confirmed", False)):
        ledger = getattr(self, "_fill_ledger", None)
        ledger_reconcile = getattr(ledger, "reconcile_entry_quantity", None)
        fill_id_builder = getattr(self, "_entry_fill_id", None)
        if callable(ledger_reconcile) and callable(fill_id_builder):
            try:
                ledger_reconcile(
                    fill_id_builder(str(order_id)),
                    reported,
                    maximum_quantity=requested,
                )
            except Exception as exc:  # noqa: BLE001 - block release on accounting drift
                blocker = getattr(self, "_block_ledger_release", None)
                if callable(blocker):
                    blocker(
                        bracket,
                        reason="entry_fill_quantity_reconcile_failed",
                        payload={
                            "order_id": str(order_id),
                            "filled_qty": reported,
                            "error": str(exc),
                        },
                    )
                raise
        _core.LOGGER.warning(
            "FILL_LEDGER_ENTRY_QTY_RECONCILED order_id=%s symbol=%s requested=%s previous=%s filled=%s",
            order_id,
            bracket.symbol,
            requested,
            registered,
            reported,
            extra={
                "event": "FILL_LEDGER_ENTRY_QTY_RECONCILED",
                "order_id": str(order_id),
                "symbol": bracket.symbol,
                "requested_qty": requested,
                "previous_filled_qty": registered,
                "filled_qty": reported,
            },
        )
    return True


def _confirm_entry_fill_once(self, order_id, fill_price, filled_qty=None):
    """Keep repeated COMPLETE callbacks idempotent while accepting smaller fills."""
    bracket = self.get_bracket(order_id)
    was_confirmed = bool(getattr(bracket, "entry_confirmed", False))
    quantity_reconciled = _reconcile_confirmed_entry_quantity(
        self, order_id, bracket, filled_qty
    )
    try:
        price = float(fill_price)
        prior = float(bracket.entry_fill_price) if bracket is not None else None
    except (TypeError, ValueError):
        price = prior = None
    if (
        bracket is not None
        and bracket.entry_confirmed
        and prior is not None
        and price is not None
        and abs(prior - price) < 1e-9
    ):
        event = (
            "BRACKET_DUPLICATE_FILL_QTY_RECONCILED"
            if quantity_reconciled
            else "BRACKET_ACTIVATION_DUPLICATE_IGNORED"
        )
        _core.LOGGER.info(
            "%s order_id=%s symbol=%s fill_price=%.2f filled_qty=%s",
            event,
            order_id,
            bracket.symbol,
            price,
            _positive_filled_quantity(filled_qty),
            extra={
                "event": event,
                "order_id": str(order_id),
                "symbol": bracket.symbol,
                "fill_price": price,
                "filled_qty": _positive_filled_quantity(filled_qty),
            },
        )
        return True

    result = _original_confirm_entry_fill(self, order_id, fill_price, filled_qty)
    bracket = self.get_bracket(order_id)
    if (
        not was_confirmed
        and bracket is not None
        and bool(getattr(bracket, "entry_confirmed", False))
        and bool(getattr(bracket, "active", False))
    ):
        self._log_bracket_event(
            "BRACKET_ARMED",
            bracket,
            meta={
                "entry_order_id": str(order_id),
                "filled_qty": int(getattr(bracket, "quantity", 0) or 0),
                "fill_price": float(
                    getattr(bracket, "entry_fill_price", None)
                    or getattr(bracket, "entry_price", 0.0)
                    or 0.0
                ),
                "stop_price": float(getattr(bracket, "sl_trigger_price", 0.0) or 0.0),
                "target_price": float(getattr(bracket, "tp_trigger_price", 0.0) or 0.0),
            },
        )
    return result


BoundBracketManager.confirm_entry_fill = _confirm_entry_fill_once

from nifty_scalper_bot.execution.market_aware_profit_extension import (  # noqa: E402
    apply_patches as _apply_market_aware_profit_extension,
)

_apply_market_aware_profit_extension(BoundBracketManager)


_CANONICAL_BRACKET_EVENTS = {
    "BRACKET_ARMED": "bracket.armed",
    "TRAIL_UPDATED": "trail.updated",
    "EXIT_TRIGGERED": "exit.triggered",
    "EXIT_SUBMITTED": "exit.submitted",
    "EXIT_FILLED": "exit.filled",
    "BRACKET_CLOSED": "trade.closed",
}


def _correlation_for_bracket(self, bracket):
    """Resolve one durable correlation chain without creating parallel state."""
    merged = {}
    order_id = str(getattr(bracket, "entry_order_id", "") or "")

    manager = getattr(self, "order_manager", None)
    cache = getattr(manager, "_canonical_trade_correlation", {})
    if isinstance(cache, Mapping):
        cached = cache.get(order_id)
        if isinstance(cached, Mapping):
            merged.update(cached)

    orders = getattr(manager, "_orders", {})
    if isinstance(orders, Mapping):
        order = orders.get(order_id)
        provenance = getattr(order, "trade_provenance", None)
        if isinstance(provenance, Mapping):
            merged.update(provenance)
        signal_id = getattr(order, "signal_id", None)
        if signal_id:
            merged.setdefault("signal_id", signal_id)

    provenance = getattr(bracket, "trade_provenance", None)
    if isinstance(provenance, Mapping):
        merged.update(provenance)

    trade_id = str(
        merged.get("trade_id")
        or getattr(bracket, "trade_lifecycle_id", "")
        or getattr(bracket, "bracket_id", "")
        or order_id
    )
    trace_id = str(merged.get("trace_id") or "")
    signal_id = str(merged.get("signal_id") or trace_id or "")
    strategy = str(
        merged.get("strategy")
        or merged.get("strategy_name")
        or getattr(bracket, "tag", "")
        or ""
    )

    if trade_id:
        merged["trade_id"] = trade_id
    if signal_id:
        merged["signal_id"] = signal_id
    if trace_id:
        merged["trace_id"] = trace_id
    if strategy:
        merged["strategy"] = strategy

    durable = getattr(bracket, "trade_provenance", None)
    if not isinstance(durable, dict):
        durable = dict(durable or {}) if isinstance(durable, Mapping) else {}
        bracket.trade_provenance = durable
    for key in ("trade_id", "signal_id", "trace_id", "strategy"):
        if merged.get(key) not in (None, ""):
            durable.setdefault(key, merged[key])
    return merged


def _log_bracket_event_correlated(self, event_type, bracket, *, meta=None):
    """Write canonical bracket events into the existing TradeJournal queue."""
    journal = getattr(self, "_trade_journal", None)
    if journal is None:
        return

    metadata = _correlation_for_bracket(self, bracket)
    metadata.update(dict(meta or {}))
    metadata.setdefault("bracket_id", str(getattr(bracket, "bracket_id", "") or ""))
    metadata.setdefault(
        "entry_order_id", str(getattr(bracket, "entry_order_id", "") or "")
    )
    metadata.setdefault(
        "trade_lifecycle_id",
        str(getattr(bracket, "trade_lifecycle_id", "") or ""),
    )
    event_name = _CANONICAL_BRACKET_EVENTS.get(str(event_type))
    if event_name:
        metadata["event_name"] = event_name

    exit_order_id = str(
        metadata.get("exit_order_id")
        or getattr(bracket, "exit_order_id", "")
        or getattr(bracket, "pending_exit_order_id", "")
        or ""
    )
    journal_order_id = (
        exit_order_id
        if str(event_type) in {"EXIT_SUBMITTED", "EXIT_FILLED"} and exit_order_id
        else str(getattr(bracket, "entry_order_id", "") or "")
    )
    price = (
        _positive_fill_price(metadata.get("fill_price"))
        or _positive_fill_price(metadata.get("exit_price"))
        or _positive_fill_price(getattr(bracket, "last_ltp", None))
        or _positive_fill_price(getattr(bracket, "entry_fill_price", None))
        or _positive_fill_price(getattr(bracket, "entry_price", None))
        or 0.0
    )

    try:
        journal.log_event(
            {
                "event_type": str(event_type),
                "timestamp": time.time(),
                "symbol": str(getattr(bracket, "symbol", "") or ""),
                "side": str(getattr(bracket, "side", "") or ""),
                "qty": int(getattr(bracket, "remaining_quantity", 0) or 0),
                "price": float(price),
                "order_id": journal_order_id or None,
                "meta": metadata,
            }
        )
        if str(event_type) == "EXIT_TRIGGERED":
            setattr(
                bracket,
                "_canonical_exit_trigger_journaled_at",
                float(getattr(bracket, "exit_triggered_at", 0.0) or time.time()),
            )
    except Exception as exc:  # noqa: BLE001
        _core.LOGGER.error(
            "CANONICAL_BRACKET_JOURNAL_FAILED event=%s bracket_id=%s error=%s",
            event_type,
            getattr(bracket, "bracket_id", ""),
            exc,
        )


BoundBracketManager._log_bracket_event = _log_bracket_event_correlated

_original_reconcile_pending_entry = BoundBracketManager._reconcile_pending_entry


def _reconcile_pending_entry_broker_evidence(self, bracket):
    """Activate pending entries only from quantitative broker fill evidence."""
    if bracket.entry_confirmed or bracket.monitoring_only:
        return
    age = time.time() - float(bracket.created_at or 0.0)
    if age < self._pending_entry_reconcile_after_sec:
        return

    status, status_known = self._broker_entry_order_status(bracket.entry_order_id)
    status_text = str(status.get("status") or "").strip().upper()
    if status_text in _core._FILLED_STATUSES:
        filled_qty = _positive_filled_quantity(
            status.get("filled_quantity") or status.get("filled")
        )
        fill_price = _positive_fill_price(
            status.get("average_price")
            or status.get("avg_price")
            or status.get("fill_price")
        )
        if filled_qty is not None and fill_price is not None:
            self.confirm_entry_fill(
                bracket.entry_order_id,
                fill_price,
                filled_qty,
            )
            return
        _core.LOGGER.warning(
            "PENDING_ENTRY_FILL_EVIDENCE_INCOMPLETE entry_order_id=%s symbol=%s status=%s filled_qty=%s fill_price=%s",
            bracket.entry_order_id,
            bracket.symbol,
            status_text,
            filled_qty,
            fill_price,
            extra={
                "event": "PENDING_ENTRY_FILL_EVIDENCE_INCOMPLETE",
                "entry_order_id": bracket.entry_order_id,
                "symbol": bracket.symbol,
                "broker_status": status_text,
                "filled_quantity": filled_qty,
                "fill_price": fill_price,
            },
        )
        return

    terminal_unfilled = status_text in _core._CANCELLED_STATUSES
    authoritatively_absent = (
        status_known
        and not status_text
        and age >= self._pending_entry_stale_after_sec
    )
    if not (terminal_unfilled or authoritatively_absent):
        return
    if not self._position_flat_for_symbol(bracket.symbol):
        return
    _core.LOGGER.warning(
        "PENDING_ENTRY_RECONCILED_FLAT entry_order_id=%s symbol=%s order_status=%s age_s=%.1f",
        bracket.entry_order_id,
        bracket.symbol,
        status_text or "ABSENT",
        age,
        extra={
            "event": "PENDING_ENTRY_RECONCILED_FLAT",
            "entry_order_id": bracket.entry_order_id,
            "symbol": bracket.symbol,
            "order_status": status_text or "ABSENT",
            "age_seconds": age,
        },
    )
    self.unregister_bracket(bracket.entry_order_id)


BoundBracketManager._reconcile_pending_entry = _reconcile_pending_entry_broker_evidence

_original_get_broker_order_status = BoundBracketManager._get_broker_order_status


def _get_broker_order_status_with_fill_evidence(self, order_id):
    """Hide terminal fill state until broker quantity and price prove execution."""
    status = _original_get_broker_order_status(self, order_id)
    if not isinstance(status, Mapping):
        return status
    status_text = str(status.get("status") or "").strip().upper()
    if status_text not in _core._FILLED_STATUSES:
        return status

    filled_qty = _positive_filled_quantity(
        status.get("filled_quantity") or status.get("filled")
    )
    fill_price = _positive_fill_price(
        status.get("average_price")
        or status.get("avg_price")
        or status.get("fill_price")
    )
    expected_qty = 0
    orders = getattr(getattr(self, "order_manager", None), "_orders", {})
    if isinstance(orders, Mapping):
        local_order = orders.get(str(order_id))
        try:
            expected_qty = int(getattr(local_order, "quantity", 0) or 0)
        except (TypeError, ValueError):
            expected_qty = 0

    complete = (
        filled_qty is not None
        and fill_price is not None
        and (expected_qty <= 0 or filled_qty >= expected_qty)
    )
    if not complete:
        _core.LOGGER.warning(
            "EXIT_FILL_EVIDENCE_INCOMPLETE order_id=%s status=%s filled_qty=%s expected_qty=%s fill_price=%s",
            order_id,
            status_text,
            filled_qty,
            expected_qty,
            fill_price,
            extra={
                "event": "EXIT_FILL_EVIDENCE_INCOMPLETE",
                "order_id": str(order_id),
                "broker_status": status_text,
                "filled_quantity": filled_qty,
                "expected_quantity": expected_qty,
                "fill_price": fill_price,
            },
        )
        sanitized = dict(status)
        sanitized["status"] = "OPEN"
        return sanitized

    evidence = self.__dict__.setdefault("_canonical_exit_fill_evidence", {})
    evidence[str(order_id)] = {
        "exit_order_id": str(order_id),
        "filled_qty": int(filled_qty),
        "fill_price": float(fill_price),
        "broker_status": status_text,
    }
    return status


BoundBracketManager._get_broker_order_status = _get_broker_order_status_with_fill_evidence

_original_process_exit_state = BoundBracketManager._process_exit_state


def _process_exit_state_with_trigger_event(self, bracket, action, *, now):
    """Fill the legacy trigger-journal gap before the existing submit state machine."""
    triggered_at = float(getattr(bracket, "exit_triggered_at", 0.0) or 0.0)
    journaled_at = float(
        getattr(bracket, "_canonical_exit_trigger_journaled_at", 0.0) or 0.0
    )
    if (
        getattr(bracket, "exit_state", None)
        == _core.BracketExitLifecycle.EXIT_TRIGGERED.value
        and triggered_at > 0.0
        and abs(journaled_at - triggered_at) > 1e-9
    ):
        self._log_bracket_event(
            "EXIT_TRIGGERED",
            bracket,
            meta={
                "reason": str(action.get("reason") or bracket.exit_reason or "EXIT"),
                "qty": int(action.get("qty") or bracket.remaining_quantity or 0),
            },
        )
    return _original_process_exit_state(self, bracket, action, now=now)


BoundBracketManager._process_exit_state = _process_exit_state_with_trigger_event

_original_submit_exit_order = BoundBracketManager.submit_exit_order


def _submit_exit_order_with_event(
    self,
    symbol,
    qty,
    reason,
    bracket_id,
    preferred_order_type="LIMIT",
    correlation_tag=None,
):
    result = _original_submit_exit_order(
        self,
        symbol,
        qty,
        reason,
        bracket_id,
        preferred_order_type=preferred_order_type,
        correlation_tag=correlation_tag,
    )
    bracket = self.get_bracket(bracket_id)
    order_id = str(getattr(result, "order_id", "") or "")
    accepted = bool(getattr(result, "accepted", False) or getattr(result, "submitted", False))
    if bracket is not None and accepted and order_id:
        self._log_bracket_event(
            "EXIT_SUBMITTED",
            bracket,
            meta={
                "exit_order_id": order_id,
                "reason": str(reason or ""),
                "qty": int(qty or 0),
                "preferred_order_type": str(preferred_order_type or ""),
                "correlation_tag": correlation_tag,
            },
        )
    return result


BoundBracketManager.submit_exit_order = _submit_exit_order_with_event

_original_close_bracket = BoundBracketManager._close_bracket


def _close_bracket_with_fill_event(
    self,
    bracket,
    *,
    close_source,
    exit_price=None,
):
    if str(close_source) == "broker_fill":
        order_id = str(
            getattr(bracket, "exit_order_id", None)
            or getattr(bracket, "pending_exit_order_id", None)
            or ""
        )
        evidence_map = getattr(self, "_canonical_exit_fill_evidence", {})
        evidence = evidence_map.get(order_id) if isinstance(evidence_map, Mapping) else None
        already = str(
            getattr(bracket, "_canonical_exit_fill_journaled_order_id", "") or ""
        )
        if isinstance(evidence, Mapping) and order_id and already != order_id:
            self._log_bracket_event(
                "EXIT_FILLED",
                bracket,
                meta={
                    **dict(evidence),
                    "exit_price": evidence.get("fill_price"),
                    "reason": str(getattr(bracket, "exit_reason", "") or ""),
                },
            )
            setattr(bracket, "_canonical_exit_fill_journaled_order_id", order_id)
    return _original_close_bracket(
        self,
        bracket,
        close_source=close_source,
        exit_price=exit_price,
    )


BoundBracketManager._close_bracket = _close_bracket_with_fill_event

_original_virtual_modify_sl = BoundBracketManager._virtual_modify_sl
_original_apply_trailing_math = BoundBracketManager._apply_trailing_math
_original_update_trailing_sl = BoundBracketManager.update_trailing_sl
_original_move_sl_to_breakeven = BoundBracketManager._move_sl_to_breakeven


def _emit_trail_event(self, bracket, old_sl, source):
    new_sl = float(getattr(bracket, "sl_trigger_price", 0.0) or 0.0)
    if abs(new_sl - float(old_sl or 0.0)) <= 1e-9:
        return
    self._log_bracket_event(
        "TRAIL_UPDATED",
        bracket,
        meta={
            "old_sl": float(old_sl or 0.0),
            "new_sl": new_sl,
            "ltp": float(getattr(bracket, "last_ltp", 0.0) or 0.0),
            "trail_revision": int(getattr(bracket, "trail_revision", 0) or 0),
            "source": source,
        },
    )


def _virtual_modify_sl_with_event(self, order_id, price):
    bracket = None
    with self._lock:
        for candidate in self._brackets.values():
            if candidate.virtual_sl_id == order_id:
                bracket = candidate
                break
        old_sl = float(getattr(bracket, "sl_trigger_price", 0.0) or 0.0)
    changed = _original_virtual_modify_sl(self, order_id, price)
    if changed and bracket is not None:
        _emit_trail_event(self, bracket, old_sl, "adaptive_controller")
    return changed


def _apply_trailing_math_with_event(self, bracket):
    old_sl = float(getattr(bracket, "sl_trigger_price", 0.0) or 0.0)
    changed = _original_apply_trailing_math(self, bracket)
    if changed:
        _emit_trail_event(self, bracket, old_sl, "fallback_trailing")
    return changed


def _update_trailing_sl_with_event(self, symbol, new_sl):
    with self._lock:
        before = {
            entry_id: float(getattr(self._brackets.get(entry_id), "sl_trigger_price", 0.0) or 0.0)
            for entry_id in list(self._symbol_map.get(symbol, []))
            if self._brackets.get(entry_id) is not None
        }
    result = _original_update_trailing_sl(self, symbol, new_sl)
    with self._lock:
        after = [
            (entry_id, self._brackets.get(entry_id))
            for entry_id in before
            if self._brackets.get(entry_id) is not None
        ]
    for entry_id, bracket in after:
        _emit_trail_event(self, bracket, before[entry_id], "manual_update")
    return result


def _move_sl_to_breakeven_with_event(self, bracket):
    old_sl = float(getattr(bracket, "sl_trigger_price", 0.0) or 0.0)
    result = _original_move_sl_to_breakeven(self, bracket)
    _emit_trail_event(self, bracket, old_sl, "breakeven")
    return result


BoundBracketManager._virtual_modify_sl = _virtual_modify_sl_with_event
BoundBracketManager._apply_trailing_math = _apply_trailing_math_with_event
BoundBracketManager.update_trailing_sl = _update_trailing_sl_with_event
BoundBracketManager._move_sl_to_breakeven = _move_sl_to_breakeven_with_event

_original_on_tick = BoundBracketManager.on_tick


def _capture_same_tick_cached_quote(self, symbol, ltp, exchange_ts):
    """Recover executable depth from the cached SSOT without mixing tick identities."""
    source = getattr(self, "_market_data", None)
    getter = getattr(source, "get_latest_tick", None) if source is not None else None
    if not callable(getter):
        return
    try:
        cached = getter(symbol)
    except Exception:
        return
    if not isinstance(cached, Mapping):
        return
    try:
        cached_ltp = float(
            cached.get("ltp")
            or cached.get("last_price")
            or cached.get("price")
            or 0.0
        )
        current_ltp = float(ltp)
    except (TypeError, ValueError):
        return
    if cached_ltp <= 0.0 or current_ltp <= 0.0 or abs(cached_ltp - current_ltp) > 1e-9:
        return
    if exchange_ts is not None:
        cached_ts = tick_exchange_epoch(cached)
        try:
            current_ts = float(exchange_ts)
        except (TypeError, ValueError):
            return
        if cached_ts is None or abs(float(cached_ts) - current_ts) > 0.001:
            return
    self._capture_exit_quote(_core.normalize_symbol(symbol), cached)


def _on_tick_with_cached_executable_quote(
    self, symbol, ltp, exchange_ts=None, *, defer_submission=False
):
    """Preserve executable bid/ask when legacy callers forward only LTP."""
    _capture_same_tick_cached_quote(self, symbol, ltp, exchange_ts)
    return _original_on_tick(
        self,
        symbol,
        ltp,
        exchange_ts,
        defer_submission=defer_submission,
    )


BoundBracketManager.on_tick = _on_tick_with_cached_executable_quote

BracketManager = BoundBracketManager

__all__ = sorted(
    {
        *[name for name in dir(_core) if not name.startswith("_")],
        "BoundBracketManager",
        "BracketManager",
        "RuntimeBracketManager",
    }
)
