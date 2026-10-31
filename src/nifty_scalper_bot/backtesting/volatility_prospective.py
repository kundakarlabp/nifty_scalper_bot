"""Audit completed P4/P5 paper or broker fills without hypothetical profits.

Responsibility: Deterministic after-cost leg-level accounting for prospective
research. The source is an externally captured decision and actual recorded
fills; this module does not collect data, submit orders, or grant LIVE approval.
Unfilled, ambiguous, unpaired or stale decisions are not profit observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Mapping

from nifty_scalper_bot.risk.cost_model import estimate_order_cost

FILL_SOURCE = Literal["BROKER", "PAPER"]


@dataclass(frozen=True, slots=True)
class ProspectiveLegFill:
    """One explicitly final cumulative order fill with its own provenance."""

    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price: float
    filled_at: datetime
    source: FILL_SOURCE
    status: Literal["FILLED"] = "FILLED"


@dataclass(frozen=True, slots=True)
class ProspectiveTradeResult:
    """One fully matched paired-trade after-cost result, never LIVE eligible."""

    parent_id: str
    structure: str
    source: FILL_SOURCE
    gross_pnl_rupees: float
    fees_rupees: float
    net_pnl_rupees: float
    entry_slippage_rupees: float
    live_eligible: bool = False


def reconcile_prospective_trade(
    *,
    parent: Mapping[str, object],
    decision_at: datetime,
    quote_observed_at: datetime,
    quote_source: str,
    executable_entry_quotes: Mapping[str, float],
    entry_fills: Mapping[str, ProspectiveLegFill],
    exit_fills: Mapping[str, ProspectiveLegFill],
) -> ProspectiveTradeResult:
    """Require one final entry and exit on every parent role and compute cash P&L.

    Quotes are decision-time executable bid/ask prices, not hypothetical fills.
    Each role must have a broker-confirmed or explicitly paper-simulated actual
    fill. Missing fills are rejected rather than assigned zero transaction cost.
    """
    if (
        decision_at.tzinfo is None
        or quote_observed_at.tzinfo is None
        or not timedelta(0) <= decision_at - quote_observed_at <= timedelta(seconds=2)
        or quote_source != "FULL"
    ):
        raise ValueError("prospective_decision_quote_unverified")
    parent_id = str(parent.get("parent_id") or "").strip()
    structure = str(parent.get("structure") or "")
    legs = parent.get("legs")
    expected = (
        {"long_call", "long_put"}
        if structure == "STRADDLE"
        else (
            {"long_put", "long_call", "short_put", "short_call"}
            if structure == "IRON_CONDOR"
            else set()
        )
    )
    if (
        not parent_id
        or not expected
        or not isinstance(legs, Mapping)
        or set(legs) != expected
        or set(entry_fills) != expected
        or set(exit_fills) != expected
        or set(executable_entry_quotes) != expected
    ):
        raise ValueError("prospective_parent_or_leg_set_invalid")

    ids: set[str] = set()
    sources: set[FILL_SOURCE] = set()
    cash = 0.0
    fees = 0.0
    slippage = 0.0
    for role in sorted(expected):
        leg = legs[role]
        if not isinstance(leg, Mapping):
            raise ValueError("prospective_parent_leg_invalid")
        planned_side = "BUY" if role.startswith("long_") else "SELL"
        symbol = str(leg.get("symbol") or "")
        qty = leg.get("quantity")
        quote = executable_entry_quotes[role]
        if (
            leg.get("side") != planned_side
            or type(qty) is not int
            or qty <= 0
            or not isinstance(quote, (int, float))
            or isinstance(quote, bool)
            or not math.isfinite(quote)
            or quote <= 0
        ):
            raise ValueError("prospective_parent_leg_invalid")
        entry = entry_fills[role]
        exit_fill = exit_fills[role]
        for fill, expected_side in (
            (entry, planned_side),
            (exit_fill, "SELL" if planned_side == "BUY" else "BUY"),
        ):
            if (
                not isinstance(fill, ProspectiveLegFill)
                or not fill.order_id
                or fill.order_id in ids
                or fill.symbol != symbol
                or fill.side != expected_side
                or type(fill.quantity) is not int
                or fill.quantity != qty
                or not math.isfinite(fill.price)
                or fill.price <= 0
                or fill.status != "FILLED"
                or fill.source not in {"BROKER", "PAPER"}
                or fill.filled_at.tzinfo is None
            ):
                raise ValueError("prospective_fill_contract_invalid")
            ids.add(fill.order_id)
            sources.add(fill.source)
            fees += estimate_order_cost(turnover=fill.price * qty, side=fill.side)
            cash += (1 if fill.side == "SELL" else -1) * fill.price * qty
        if not (decision_at <= entry.filled_at <= exit_fill.filled_at):
            raise ValueError("prospective_fill_chronology_invalid")
        slippage += (
            (entry.price - quote) * qty
            if planned_side == "BUY"
            else (quote - entry.price) * qty
        )
    if len(sources) != 1:
        raise ValueError("prospective_fill_sources_mixed")
    return ProspectiveTradeResult(
        parent_id=parent_id,
        structure=structure,
        source=next(iter(sources)),
        gross_pnl_rupees=cash,
        fees_rupees=fees,
        net_pnl_rupees=cash - fees,
        entry_slippage_rupees=slippage,
    )
