"""Round-trip transaction cost model for NSE option buy trades (Zerodha).

Rates current as of June 2026 (post Budget 2026-27, effective 1 Apr 2026):
- Brokerage: flat Rs 20 per executed order (buy + sell = Rs 40 round trip)
- STT: 0.15% of premium, sell side only
- NSE exchange transaction charge: 0.03553% of premium, both sides
- SEBI charges: Rs 10 per crore (0.0001%), both sides
- GST: 18% on (brokerage + exchange txn charge + SEBI charges)
- Stamp duty: 0.003% of premium, buy side only

All rates are env-overridable so future Budget changes need no code edit:
COST_BROKERAGE_PER_ORDER, COST_STT_SELL_PCT, COST_EXCH_TXN_PCT,
COST_SEBI_PCT, COST_GST_PCT, COST_STAMP_BUY_PCT, MIN_EDGE_MULTIPLE.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from nifty_scalper_bot.config.env_utils import parse_float_env


@dataclass(slots=True)
class CostBreakdown:
    """Args: cost components in rupees. Returns: breakdown object. Raises: none."""

    brokerage: float
    stt: float
    exchange_txn: float
    sebi: float
    gst: float
    stamp_duty: float
    half_spread_slippage: float
    total: float
    cost_per_unit: float  # total cost expressed in option-premium points per unit


def _rate(name: str, default: float) -> float:
    return parse_float_env(os.getenv(name), default)


def estimate_round_trip_cost(
    *,
    entry_price: float,
    exit_price: float,
    quantity: int,
    half_spread: float = 0.0,
    executed_orders: int = 2,
) -> CostBreakdown:
    """Args: entry/exit premium per unit, total quantity, half-spread per unit.
    Returns: full round-trip cost breakdown in rupees. Raises: none.

    quantity is total units (lots * lot_size). half_spread models the
    bid/ask cost paid once on each side when crossing the spread.
    """
    qty = max(1, int(quantity))
    buy_value = max(0.0, float(entry_price)) * qty
    sell_value = max(0.0, float(exit_price)) * qty

    brokerage = max(2, int(executed_orders)) * _rate("COST_BROKERAGE_PER_ORDER", 20.0)
    stt = sell_value * _rate("COST_STT_SELL_PCT", 0.0015)
    exchange_txn = (buy_value + sell_value) * _rate("COST_EXCH_TXN_PCT", 0.0003553)
    sebi = (buy_value + sell_value) * _rate("COST_SEBI_PCT", 0.000001)
    gst = (brokerage + exchange_txn + sebi) * _rate("COST_GST_PCT", 0.18)
    stamp_duty = buy_value * _rate("COST_STAMP_BUY_PCT", 0.00003)
    slippage = 2.0 * max(0.0, float(half_spread)) * qty

    total = brokerage + stt + exchange_txn + sebi + gst + stamp_duty + slippage
    return CostBreakdown(
        brokerage=brokerage,
        stt=stt,
        exchange_txn=exchange_txn,
        sebi=sebi,
        gst=gst,
        stamp_duty=stamp_duty,
        half_spread_slippage=slippage,
        total=total,
        cost_per_unit=total / qty,
    )


def passes_cost_edge_gate(
    *,
    entry_price: float,
    target_price: float,
    quantity: int,
    half_spread: float = 0.0,
) -> tuple[bool, float, CostBreakdown]:
    """Args: entry, target premium, quantity, half-spread per unit.
    Returns: (allowed, edge_multiple, breakdown). Raises: none.

    Allowed when expected gross profit at target covers the round-trip
    cost by at least MIN_EDGE_MULTIPLE (default 2.0). Costs are estimated
    conservatively at the target exit (highest premium => highest charges).
    """
    qty = max(1, int(quantity))
    breakdown = estimate_round_trip_cost(
        entry_price=entry_price,
        exit_price=target_price,
        quantity=qty,
        half_spread=half_spread,
    )
    gross_profit = max(0.0, (float(target_price) - float(entry_price))) * qty
    min_multiple = _rate("MIN_EDGE_MULTIPLE", 2.0)
    edge_multiple = (gross_profit / breakdown.total) if breakdown.total > 0 else 0.0
    return edge_multiple >= min_multiple, edge_multiple, breakdown
