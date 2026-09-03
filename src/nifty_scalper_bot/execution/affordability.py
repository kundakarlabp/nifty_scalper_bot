"""Minimum executable-lot affordability for live readiness.

This module evaluates one supplied contract and never chooses a strike. Callers
may use the result only after strategy, side, expiry and liquidity ranking; the
authoritative order-manager margin gate still runs immediately before submission.
"""

from __future__ import annotations

import math
from contextlib import suppress
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class MinimumLotAffordability:
    symbol: str
    affordable: bool
    determinate: bool
    reason: str
    available: float | None
    required: float | None
    executable_capacity: float | None
    entry_price: float | None
    lot_size: int | None
    margin_factor: float
    margin_buffer: float
    balance_source: str
    per_trade_risk_budget: float | None = None
    remaining_daily_risk_budget: float | None = None
    effective_one_lot_risk_budget: float | None = None
    max_stop_distance_one_lot: float | None = None
    daily_loss_headroom: float | None = None
    risk_budget_semantics: str = "per_trade_sizing"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _field(source: Mapping[str, Any] | object | None, *names: str) -> Any:
    if source is None:
        return None
    for name in names:
        if isinstance(source, Mapping):
            value = source.get(name)
        else:
            value = getattr(source, name, None)
        if value not in (None, ""):
            return value
    return None


def _finite_float(value: Any, *, minimum: float | None = None) -> float | None:
    with suppress(TypeError, ValueError):
        parsed = float(value)
        if math.isfinite(parsed) and (minimum is None or parsed >= minimum):
            return parsed
    return None


def _available_balance(
    data_hub: Any | None, fallback_balance: Any | None
) -> tuple[float | None, str]:
    getter = getattr(data_hub, "get_available_balance", None)
    if callable(getter):
        try:
            raw = getter(force=False)
        except TypeError:
            raw = getter()
        except Exception:
            raw = None
        parsed = _finite_float(raw, minimum=0.0)
        if parsed is not None:
            return parsed, "data_hub"
    parsed_fallback = _finite_float(fallback_balance, minimum=0.0)
    if parsed_fallback is not None:
        return parsed_fallback, "broker_balance_snapshot"
    return None, "unavailable"


def _risk_budget_snapshot(
    order_manager: Any | None, lot_size: int
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return actual sizing risk plus independent daily-loss headroom telemetry.

    ``RiskManager.check_order()`` and ``suggest_position_size()`` size against the
    configured per-trade risk budget. The daily-loss switch is a realised-PnL
    circuit breaker; its remaining headroom is not a pre-trade stop-distance
    budget. Keep both visible, but do not collapse them into a fictitious tighter
    execution limit.
    """
    manager = getattr(order_manager, "_risk_manager", None)
    if manager is None or lot_size <= 0:
        return None, None, None, None

    balance = _finite_float(getattr(manager, "account_balance", None), minimum=0.0)
    settings = getattr(manager, "settings", None)
    risk_pct = _finite_float(
        getattr(settings, "per_trade_risk_pct", None), minimum=0.0
    )
    per_trade = (
        balance * risk_pct / 100.0
        if balance is not None and balance > 0.0 and risk_pct is not None
        else None
    )

    remaining = None
    switches = getattr(manager, "_switches", None)
    if switches is not None:
        max_day_loss = _finite_float(
            getattr(switches, "max_day_loss", None), minimum=0.0
        )
        if max_day_loss is not None and max_day_loss > 0.0:
            reader = getattr(switches, "day_loss", None)
            if callable(reader):
                try:
                    current = max(float(reader() or 0.0), 0.0)
                    remaining = max(max_day_loss - current, 0.0)
                except Exception:
                    remaining = 0.0
            else:
                remaining = 0.0

    effective = per_trade
    max_stop_distance = (
        effective / float(lot_size)
        if effective is not None and lot_size > 0
        else None
    )
    return per_trade, remaining, effective, max_stop_distance


def evaluate_minimum_lot_affordability(
    *,
    symbol: str,
    quote: Mapping[str, Any] | object | None,
    order_manager: Any | None,
    data_hub: Any | None = None,
    fallback_balance: Any | None = None,
) -> MinimumLotAffordability:
    """Evaluate whether one supplied BUY option lot is executable.

    The estimate mirrors the MarginEngine fallback path: ask premium × lot size ×
    margin factor, while the configured margin buffer reserves cash by reducing
    executable balance. Broker submission still performs the authoritative final
    margin check.
    """

    normalized_symbol = str(symbol or "").strip()
    raw_factor = _finite_float(
        getattr(order_manager, "_margin_factor", None), minimum=0.0
    )
    margin_factor = max(raw_factor or 1.0, 1.0)
    raw_buffer = _finite_float(
        getattr(order_manager, "_margin_buffer", None), minimum=0.0
    )
    margin_buffer = min(raw_buffer if raw_buffer and raw_buffer > 0 else 1.0, 1.0)

    ask = _finite_float(_field(quote, "ask", "best_ask"), minimum=0.0)
    if ask is None or ask <= 0:
        return MinimumLotAffordability(
            normalized_symbol,
            False,
            False,
            "executable_quote_unavailable",
            None,
            None,
            None,
            None,
            None,
            margin_factor,
            margin_buffer,
            "unresolved",
        )

    resolver = getattr(order_manager, "resolve_lot_size", None)
    lot_size = 0
    if callable(resolver):
        with suppress(Exception):
            lot_size = int(resolver(normalized_symbol) or 0)
    if lot_size <= 0:
        return MinimumLotAffordability(
            normalized_symbol,
            False,
            False,
            "lot_size_unresolved",
            None,
            None,
            None,
            ask,
            None,
            margin_factor,
            margin_buffer,
            "unresolved",
        )

    available, balance_source = _available_balance(data_hub, fallback_balance)
    if available is None:
        return MinimumLotAffordability(
            normalized_symbol,
            False,
            False,
            "available_balance_unavailable",
            None,
            None,
            None,
            ask,
            lot_size,
            margin_factor,
            margin_buffer,
            balance_source,
        )

    required = ask * lot_size * margin_factor
    executable_capacity = available * margin_buffer
    affordable = bool(required > 0 and executable_capacity >= required)
    (
        per_trade_risk_budget,
        remaining_daily_risk_budget,
        effective_one_lot_risk_budget,
        max_stop_distance_one_lot,
    ) = _risk_budget_snapshot(order_manager, lot_size)
    return MinimumLotAffordability(
        normalized_symbol,
        affordable,
        True,
        "affordable" if affordable else "minimum_lot_unaffordable",
        available,
        required,
        executable_capacity,
        ask,
        lot_size,
        margin_factor,
        margin_buffer,
        balance_source,
        per_trade_risk_budget=per_trade_risk_budget,
        remaining_daily_risk_budget=remaining_daily_risk_budget,
        effective_one_lot_risk_budget=effective_one_lot_risk_budget,
        max_stop_distance_one_lot=max_stop_distance_one_lot,
        daily_loss_headroom=remaining_daily_risk_budget,
        risk_budget_semantics="per_trade_sizing",
    )


__all__ = [
    "MinimumLotAffordability",
    "evaluate_minimum_lot_affordability",
]
