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

_original_handle_bracket_update = RuntimeOrderManager._handle_bracket_update


def _handle_bracket_update_single_owner(self, order, previous_status, payload):
    """Prevent legacy broker SL/TP children when the virtual bracket owner is bound.

    ``order_manager_core`` still contains a compatibility auto-bracket path that
    creates real broker stop/target child orders after a filled entry. Production
    already binds the canonical virtual BracketManager, which independently owns
    SL, TP1/final target and trailing. Running both creates two reducing exit
    owners for the same position. Suppress only *new* legacy bracket creation;
    existing legacy bracket state is still delegated to the original handler so
    restart/reconciliation can close or cancel already-live broker children.
    """

    bracket_manager = getattr(self, "_bracket_manager", None)
    if bracket_manager is not None:
        order_id = str(getattr(order, "order_id", "") or "")
        bracket_index = getattr(self, "_bracket_index", {})
        legacy_brackets = getattr(self, "_brackets", {})
        entry_id = bracket_index.get(order_id) if isinstance(bracket_index, Mapping) else None
        if not entry_id and isinstance(legacy_brackets, Mapping) and order_id in legacy_brackets:
            entry_id = order_id

        status = getattr(order, "status", None)
        has_exit_geometry = bool(
            getattr(order, "stop_loss", None) or getattr(order, "take_profit", None)
        )
        if not entry_id and status == _core.OrderStatus.FILLED and has_exit_geometry:
            logger = getattr(self, "_logger", None)
            log = getattr(logger, "info", None)
            if callable(log):
                log(
                    "LEGACY_BROKER_BRACKET_SUPPRESSED order_id=%s symbol=%s canonical_owner=BracketManager",
                    order_id,
                    getattr(order, "symbol", None),
                    extra={
                        "event": "LEGACY_BROKER_BRACKET_SUPPRESSED",
                        "order_id": order_id,
                        "symbol": getattr(order, "symbol", None),
                        "canonical_owner": "BracketManager",
                    },
                )
            return None

    return _original_handle_bracket_update(self, order, previous_status, payload)


RuntimeOrderManager._handle_bracket_update = _handle_bracket_update_single_owner

_original_extract_quote_diagnostics = RuntimeOrderManager._extract_quote_diagnostics
_original_get_latest_quote_safe = RuntimeOrderManager._get_latest_quote_safe


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
