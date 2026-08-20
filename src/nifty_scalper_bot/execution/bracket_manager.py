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
        quantity = int(value or 0)
    except (TypeError, ValueError):
        return None
    return quantity if quantity > 0 else None


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


def _reanchor_confirmed_fill_geometry(bracket, fill_price):
    """Preserve an explicit bracket anchor policy at the broker fill price."""
    if bracket is None:
        return
    try:
        price = float(fill_price)
        old_entry = float(bracket.entry_price or 0.0)
    except (TypeError, ValueError):
        return
    if price <= 0.0 or old_entry <= 0.0 or abs(price - old_entry) < 1e-12:
        return

    provenance = getattr(bracket, "trade_provenance", {}) or {}
    raw_anchor_mode = provenance.get("bracket_anchor_mode")
    if raw_anchor_mode is None:
        # Direct/legacy registrations never declared an anchor contract. Keep
        # their historical percentage-rescale behavior; the production
        # TradePlan path always persists an explicit mode before registration.
        return
    anchor_mode = str(raw_anchor_mode).strip().lower()
    if anchor_mode not in {"distance", "absolute_level"}:
        return

    delta = price - old_entry
    if anchor_mode == "distance":
        if float(bracket.sl_trigger_price or 0.0) > 0.0:
            bracket.sl_trigger_price = _core._round_to_tick(
                float(bracket.sl_trigger_price) + delta
            )
        if float(bracket.tp_trigger_price or 0.0) > 0.0:
            bracket.tp_trigger_price = _core._round_to_tick(
                float(bracket.tp_trigger_price) + delta
            )
        for level in getattr(bracket, "tp_levels", ()):
            if not getattr(level, "executed", False):
                level.price = _core._round_to_tick(float(level.price) + delta)

    # Pre-setting entry makes the legacy core fill handler see no entry delta, so
    # it cannot percentage-rescale explicitly anchored levels a second time.
    bracket.entry_price = price
    bracket.initial_sl_trigger_price = float(bracket.sl_trigger_price or 0.0)
    _core.LOGGER.info(
        "BRACKET_FILL_REANCHORED order_id=%s symbol=%s mode=%s old_entry=%.2f fill=%.2f sl=%.2f tp=%.2f",
        bracket.entry_order_id,
        bracket.symbol,
        anchor_mode,
        old_entry,
        price,
        float(bracket.sl_trigger_price or 0.0),
        float(bracket.tp_trigger_price or 0.0),
        extra={
            "event": "BRACKET_FILL_REANCHORED",
            "order_id": str(bracket.entry_order_id),
            "symbol": bracket.symbol,
            "anchor_mode": anchor_mode,
            "old_entry": old_entry,
            "fill_price": price,
        },
    )


def _confirm_entry_fill_once(self, order_id, fill_price, filled_qty=None):
    """Keep repeated COMPLETE callbacks idempotent while accepting smaller fills."""
    bracket = self.get_bracket(order_id)
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
    _reanchor_confirmed_fill_geometry(bracket, fill_price)
    return _original_confirm_entry_fill(self, order_id, fill_price, filled_qty)


BoundBracketManager.confirm_entry_fill = _confirm_entry_fill_once

from nifty_scalper_bot.execution.market_aware_profit_extension import (  # noqa: E402
    apply_patches as _apply_market_aware_profit_extension,
)

_apply_market_aware_profit_extension(BoundBracketManager)

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
