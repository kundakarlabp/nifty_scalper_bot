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

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

from nifty_scalper_bot.execution import runtime_order_manager as _runtime  # noqa: E402


_original_enrich_trade_plan_exit_provenance = _runtime._enrich_trade_plan_exit_provenance


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

_original_get_latest_quote_safe = RuntimeOrderManager._get_latest_quote_safe


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


def _get_latest_quote_freshest_cached(self, symbol):
    """Prefer the freshest already-cached quote without weakening stale guards."""
    primary = _original_get_latest_quote_safe(self, symbol)
    best = primary if isinstance(primary, Mapping) else None
    best_age = _quote_age_ms(self, best)
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
        if best is None or best_age is None or candidate_age < best_age:
            best = candidate
            best_age = candidate_age

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
