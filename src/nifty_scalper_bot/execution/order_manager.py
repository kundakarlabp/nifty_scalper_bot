"""File purpose:
    Provide the stable public order-execution API used by the strategy runner.

Key responsibilities:
    - Re-export public order models and helpers from ``order_manager_core``.
    - Expose ``RuntimeOrderManager`` as the single production ``OrderManager``.

Operational constraints:
    - This facade must not add a second execution path or duplicate order state.
    - Runtime recovery and entry gating remain owned by ``RuntimeOrderManager``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

from nifty_scalper_bot.execution import order_manager_core as _core
from nifty_scalper_bot.execution.readiness import resolve_quote_bid_ask_spread

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

from nifty_scalper_bot.execution import (  # noqa: E402,I001
    runtime_order_manager as _runtime,
)

_original_enrich_trade_plan_exit_provenance = (
    _runtime._enrich_trade_plan_exit_provenance
)


def _enrich_trade_plan_exit_provenance(plan):
    """Carry the TradePlan bracket anchor contract into durable provenance."""
    plan = _original_enrich_trade_plan_exit_provenance(plan)
    try:
        provenance = dict(getattr(plan, "trade_provenance", {}) or {})
    except (TypeError, ValueError):
        provenance = {}
    provenance.setdefault(
        "bracket_anchor_mode",
        str(getattr(plan, "bracket_anchor_mode", "distance") or "distance"),
    )
    setattr(plan, "trade_provenance", provenance)
    return plan


# RuntimeOrderManager methods resolve this module-global helper at call time.
# Patch that one helper rather than introducing a second submission path.
_runtime._enrich_trade_plan_exit_provenance = _enrich_trade_plan_exit_provenance
RuntimeOrderManager = _runtime.RuntimeOrderManager

_original_confirm_fill_fast = RuntimeOrderManager._confirm_fill_fast
_original_log_trade_event = RuntimeOrderManager._log_trade_event
_original_extract_quote_diagnostics = RuntimeOrderManager._extract_quote_diagnostics
_original_get_latest_quote_safe = RuntimeOrderManager._get_latest_quote_safe


def _positive_execution_quantity(value):
    try:
        quantity = int(float(value or 0))
    except (TypeError, ValueError):
        return 0
    return quantity if quantity > 0 else 0


def _positive_execution_price(value):
    try:
        price = float(value or 0.0)
    except (TypeError, ValueError):
        return None
    return price if price > 0.0 else None


def _broker_fill_evidence(payload):
    """Return quantitative broker fill evidence or ``(0, None)``."""
    if not isinstance(payload, Mapping):
        return 0, None
    quantity = _positive_execution_quantity(
        payload.get("filled_quantity") or payload.get("filled")
    )
    price = _positive_execution_price(
        payload.get("average_price")
        or payload.get("avg_price")
        or payload.get("fill_price")
    )
    return quantity, price


def _confirm_fill_fast_broker_evidence(self, order_id, timeout_ms=2000):
    """Confirm fast fills only from positive broker quantity and execution price."""
    start = time.monotonic()
    backoff_ms = 50.0
    max_backoff_ms = 300.0
    attempts = 0

    self._logger.debug("Fast broker-evidence fill check started for %s", order_id)

    while (time.monotonic() - start) * 1000.0 < float(timeout_ms):
        attempts += 1
        try:
            status = None
            getter = getattr(self._broker, "get_order_status", None)
            if callable(getter):
                status = getter(order_id)
            else:
                history_getter = getattr(self._broker, "order_history", None)
                if callable(history_getter):
                    history = history_getter(order_id)
                    if isinstance(history, list) and history:
                        status = history[-1]

            if not isinstance(status, Mapping) or not status:
                time.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 1.5, max_backoff_ms)
                continue

            status_text = str(status.get("status") or "").strip().upper()
            if status_text in {"COMPLETE", "FILLED"}:
                filled_qty, fill_price = _broker_fill_evidence(status)
                if filled_qty > 0 and fill_price is not None:
                    elapsed_ms = (time.monotonic() - start) * 1000.0
                    self._logger.info(
                        "BROKER_FILL_CONFIRMED order_id=%s qty=%s price=%.2f "
                        "elapsed_ms=%.0f attempts=%s",
                        order_id,
                        filled_qty,
                        fill_price,
                        elapsed_ms,
                        attempts,
                        extra={
                            "event": "BROKER_FILL_CONFIRMED",
                            "order_id": str(order_id),
                            "filled_quantity": filled_qty,
                            "fill_price": fill_price,
                            "elapsed_ms": elapsed_ms,
                            "attempts": attempts,
                        },
                    )
                    self.on_order_update(dict(status))
                    return True

                self._logger.warning(
                    "BROKER_FILL_EVIDENCE_INCOMPLETE order_id=%s status=%s "
                    "filled_qty=%s fill_price=%s",
                    order_id,
                    status_text,
                    filled_qty,
                    fill_price,
                    extra={
                        "event": "BROKER_FILL_EVIDENCE_INCOMPLETE",
                        "order_id": str(order_id),
                        "broker_status": status_text,
                        "filled_quantity": filled_qty,
                        "fill_price": fill_price,
                    },
                )
            elif status_text in {"REJECTED", "CANCELLED", "CANCELED", "EXPIRED"}:
                self._logger.warning(
                    "Order %s terminal-unfilled: %s",
                    order_id,
                    status_text,
                )
                self.on_order_update(dict(status))
                return False

        except Exception as exc:
            self._logger.debug(
                "Fast fill evidence check failed order_id=%s attempt=%s error=%s",
                order_id,
                attempts,
                exc,
            )

        time.sleep(backoff_ms / 1000.0)
        backoff_ms = min(backoff_ms * 1.5, max_backoff_ms)

    self._logger.warning(
        "BROKER_FILL_CONFIRM_TIMEOUT order_id=%s timeout_ms=%s attempts=%s",
        order_id,
        timeout_ms,
        attempts,
        extra={
            "event": "BROKER_FILL_CONFIRM_TIMEOUT",
            "order_id": str(order_id),
            "timeout_ms": int(timeout_ms),
            "attempts": attempts,
        },
    )
    return False


def _log_trade_event_broker_truth(
    self,
    event_type,
    *,
    symbol,
    side,
    qty,
    price,
    order_id=None,
    meta=None,
):
    """Keep entry-fill journal facts quantitative and cache lifecycle identity."""
    metadata = dict(meta or {})

    if event_type == "ORDER_SUBMITTED" and order_id:
        correlation = {
            key: metadata.get(key)
            for key in ("trade_id", "signal_id", "trace_id", "strategy")
            if metadata.get(key) not in (None, "")
        }
        if correlation:
            cache = self.__dict__.setdefault("_canonical_trade_correlation", {})
            cache[str(order_id)] = correlation

    if event_type == "ORDER_FILL_CONFIRMED":
        order = getattr(self, "_orders", {}).get(str(order_id)) if order_id else None
        filled_qty = _positive_execution_quantity(
            getattr(order, "filled_quantity", 0) if order is not None else 0
        )
        fill_price = _positive_execution_price(
            (
                getattr(order, "fill_price", None)
                or getattr(order, "average_price", None)
            )
            if order is not None
            else None
        )
        if filled_qty <= 0 or fill_price is None:
            self._logger.error(
                "ENTRY_FILL_JOURNAL_SUPPRESSED_UNCONFIRMED order_id=%s "
                "filled_qty=%s fill_price=%s",
                order_id,
                filled_qty,
                fill_price,
                extra={
                    "event": "ENTRY_FILL_JOURNAL_SUPPRESSED_UNCONFIRMED",
                    "order_id": str(order_id or ""),
                    "filled_quantity": filled_qty,
                    "fill_price": fill_price,
                },
            )
            return

        qty = filled_qty
        price = fill_price
        metadata["filled_quantity"] = filled_qty
        metadata["fill_price"] = fill_price
        metadata["broker_confirmed_fill"] = True

    return _original_log_trade_event(
        self,
        event_type,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        order_id=order_id,
        meta=metadata,
    )


RuntimeOrderManager._confirm_fill_fast = _confirm_fill_fast_broker_evidence
RuntimeOrderManager._log_trade_event = _log_trade_event_broker_truth


def _depth_top_quantity(quote, side):
    """Return positive top-level executable quantity from Zerodha FULL depth."""
    if not isinstance(quote, Mapping):
        return 0
    depth = quote.get("depth")
    if not isinstance(depth, Mapping):
        return 0
    levels = depth.get(side)
    if not isinstance(levels, (list, tuple)) or not levels:
        return 0
    top = levels[0]
    if not isinstance(top, Mapping):
        return 0
    for key in ("quantity", "qty"):
        value = top.get(key)
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 0


def _extract_quote_diagnostics_canonical(self, quote):
    """Extend core diagnostics with the existing canonical Zerodha depth resolver."""
    diagnostics = dict(_original_extract_quote_diagnostics(self, quote))
    if not isinstance(quote, Mapping):
        return diagnostics

    current_bid = float(diagnostics.get("bid") or 0.0)
    current_ask = float(diagnostics.get("ask") or 0.0)
    if current_bid <= 0.0 or current_ask <= 0.0:
        bid, ask, spread_pct, _source = resolve_quote_bid_ask_spread(dict(quote))
        if bid is not None and ask is not None and bid > 0.0 and ask > bid:
            diagnostics["bid"] = float(bid)
            diagnostics["ask"] = float(ask)
            diagnostics["spread"] = float(ask - bid)
            if spread_pct is not None:
                diagnostics["spread_pct"] = float(spread_pct)

    try:
        bid_qty = int(diagnostics.get("bid_qty") or 0)
    except (TypeError, ValueError):
        bid_qty = 0
    try:
        ask_qty = int(diagnostics.get("ask_qty") or 0)
    except (TypeError, ValueError):
        ask_qty = 0
    if bid_qty <= 0:
        bid_qty = _depth_top_quantity(quote, "buy")
        diagnostics["bid_qty"] = bid_qty
    if ask_qty <= 0:
        ask_qty = _depth_top_quantity(quote, "sell")
        diagnostics["ask_qty"] = ask_qty
    diagnostics["depth_qty"] = max(0, bid_qty) + max(0, ask_qty)
    return diagnostics


RuntimeOrderManager._extract_quote_diagnostics = _extract_quote_diagnostics_canonical


def _quote_age_ms(manager, quote):
    """Return a finite quote age from the existing canonical diagnostics."""
    if not isinstance(quote, Mapping):
        return None
    try:
        age = manager._extract_quote_diagnostics(quote).get("age_ms")
        parsed = float(age) if age is not None else None
    except (AttributeError, TypeError, ValueError):
        return None
    if parsed is None or not math.isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def _quote_execution_rank(manager, quote):
    """Rank cached quote evidence without treating LTP-only data as executable."""
    if not isinstance(quote, Mapping):
        return 0
    try:
        diagnostics = manager._extract_quote_diagnostics(quote)
        bid = float(diagnostics.get("bid") or 0.0)
        ask = float(diagnostics.get("ask") or 0.0)
        bid_qty = int(diagnostics.get("bid_qty") or 0)
        ask_qty = int(diagnostics.get("ask_qty") or 0)
    except (AttributeError, TypeError, ValueError):
        return 0
    if bid <= 0.0 or ask < bid:
        return 0
    return 2 if bid_qty > 0 and ask_qty > 0 else 1


def _get_latest_quote_freshest_cached(self, symbol):
    """Prefer executable cached evidence, then freshness, preserving stale guards."""
    primary = _original_get_latest_quote_safe(self, symbol)
    best = primary if isinstance(primary, Mapping) else None
    best_age = _quote_age_ms(self, best)
    best_rank = _quote_execution_rank(self, best)
    normalized_symbol = _core.normalize_symbol(symbol)
    seen_providers: set[int] = set()

    for attr in (
        "_market_data_manager",
        "market_data_manager",
        "_market_data",
        "_data_hub",
        "data_hub",
    ):
        provider = getattr(self, attr, None)
        if provider is None or id(provider) in seen_providers:
            continue
        seen_providers.add(id(provider))
        getter = getattr(provider, "get_latest_tick", None)
        if not callable(getter):
            continue
        try:
            candidate = getter(symbol)
        except Exception:
            continue
        if not isinstance(candidate, Mapping) or not candidate:
            continue
        candidate_symbol = str(candidate.get("symbol") or "").strip()
        if (
            candidate_symbol
            and _core.normalize_symbol(candidate_symbol) != normalized_symbol
        ):
            continue
        try:
            diagnostics = self._extract_quote_diagnostics(candidate)
            candidate_ltp = float(diagnostics.get("ltp") or 0.0)
        except (AttributeError, TypeError, ValueError):
            continue
        if candidate_ltp <= 0.0:
            continue
        candidate_age = _quote_age_ms(self, candidate)
        if candidate_age is None:
            continue
        candidate_rank = _quote_execution_rank(self, candidate)
        if (
            best is None
            or candidate_rank > best_rank
            or (
                candidate_rank == best_rank
                and (best_age is None or candidate_age < best_age)
            )
        ):
            best = candidate
            best_age = candidate_age
            best_rank = candidate_rank

    return dict(best) if isinstance(best, Mapping) else None


RuntimeOrderManager._get_latest_quote_safe = _get_latest_quote_freshest_cached
OrderManager = RuntimeOrderManager

__all__ = sorted(
    {
        *[name for name in dir(_core) if not name.startswith("_")],
        "OrderManager",
        "RuntimeOrderManager",
    }
)
