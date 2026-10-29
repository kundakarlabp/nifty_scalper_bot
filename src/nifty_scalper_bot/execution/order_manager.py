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

_original_extract_quote_diagnostics = RuntimeOrderManager._extract_quote_diagnostics


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


def _cached_quote(provider, symbol):
    """Read cached quote evidence without triggering broker/HTTP pulls."""
    getter = getattr(provider, "get_latest_tick", None)
    if callable(getter):
        try:
            quote = getter(symbol)
        except Exception:
            quote = None
        if isinstance(quote, Mapping) and quote:
            return quote

    getter = getattr(provider, "get_quote", None)
    if callable(getter):
        try:
            quote = getter(symbol, allow_pull=False)
        except TypeError:
            quote = None
        except Exception:
            quote = None
        if isinstance(quote, Mapping) and quote:
            return quote

    getter = getattr(provider, "get_tick", None)
    if callable(getter):
        try:
            quote = getter(symbol)
        except Exception:
            quote = None
        if isinstance(quote, Mapping) and quote:
            return quote
    return None


def _get_latest_quote_freshest_cached(self, symbol):
    """Prefer executable cached evidence, then freshness; never pull implicitly."""
    best = None
    best_age = None
    best_rank = 0
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
        candidate = _cached_quote(provider, symbol)
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

# Single production submission path: a timed-out entry must not be silently
# repriced upward by a later new TradePlan for the SAME deterministic setup.
# Existing per-plan freshness guards alone cannot compare those two attempts.
_original_submit_trade_plan_result = RuntimeOrderManager.submit_trade_plan_result


def _submit_trade_plan_with_setup_price_guard(self, plan):
    live = False
    checker = getattr(self, "is_live_mode", None)
    if callable(checker):
        try:
            live = bool(checker())
        except Exception:
            live = True
    intent = str(getattr(plan, "intent", "ENTRY") or "ENTRY").upper()
    signal_id = str(getattr(plan, "signal_id", "") or "")
    symbol = str(getattr(plan, "symbol", "") or "")
    buy_entry = live and intent == "ENTRY" and str(
        getattr(plan, "side", "") or ""
    ).upper() == "BUY" and bool(signal_id and symbol)
    if not buy_entry:
        return _original_submit_trade_plan_result(self, plan)

    anchors = self.__dict__.setdefault("_entry_attempt_price_by_setup", {})
    key = (signal_id, symbol)
    now = time.time()
    prior = anchors.get(key)
    if prior is not None:
        first_price, first_at = prior
        age = now - first_at
        try:
            price = float(getattr(plan, "entry_price", 0.0))
            cap = float(getattr(plan, "max_entry_drift_pct", 2.0) or 2.0)
        except (ValueError, TypeError):
            price, cap = float("nan"), 2.0
        if 0 <= age <= 900 and first_price > 0:
            drift_pct = (price - first_price) * 100.0 / first_price
            if not math.isfinite(price) or drift_pct > max(0.0, cap):
                logger = getattr(self, "_logger", None)
                warning = getattr(logger, "warning", None)
                if callable(warning):
                    warning(
                        "ENTRY_SAME_SETUP_PRICE_CHASE_BLOCKED symbol=%s drift_pct=%s limit_pct=%s",
                        symbol, drift_pct, cap,
                        extra={
                            "event": "ENTRY_SAME_SETUP_PRICE_CHASE_BLOCKED",
                            "signal_id": signal_id,
                            "symbol": symbol,
                            "drift_pct": drift_pct,
                            "limit_pct": cap,
                            "age_seconds": age,
                        },
                    )
                return _core.TradePlanSubmitResult(
                    accepted=False,
                    order_id=None,
                    reason="same_setup_adverse_reprice",
                    details={"drift_pct": drift_pct, "limit_pct": cap},
                    broker_attempted=False,
                )

    result = _original_submit_trade_plan_result(self, plan)
    if bool(getattr(result, "broker_attempted", False)) and key not in anchors:
        try:
            first_price = float(getattr(plan, "entry_price", 0.0))
        except (ValueError, TypeError):
            first_price = 0.0
        if math.isfinite(first_price) and first_price > 0:
            if len(anchors) >= 512:
                anchors.pop(next(iter(anchors)))
            anchors[key] = (first_price, now)
    return result


RuntimeOrderManager.submit_trade_plan_result = _submit_trade_plan_with_setup_price_guard
OrderManager = RuntimeOrderManager

__all__ = sorted(
    {
        *[name for name in dir(_core) if not name.startswith("_")],
        "OrderManager",
        "RuntimeOrderManager",
    }
)
