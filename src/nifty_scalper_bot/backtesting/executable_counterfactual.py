"""Research-only post-decision executable quote labeling for missed NIFTY setups.

Responsibilities: Label provenance-verified, exact-contract full-depth quotes
with conservative entry slippage, recorded stop/target and canonical costs.
Boundary: Research-only counterfactuals; no live or broker imports.
Safety: A slipped entry outside the recorded protective geometry is unresolved.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nifty_scalper_bot.backtesting.setup_opportunity_analysis import _timestamp
from nifty_scalper_bot.risk.cost_model import estimate_round_trip_cost


@dataclass(frozen=True, slots=True)
class ExecutableCounterfactual:
    status: str
    reason: str
    entry_ts: float | None = None
    exit_ts: float | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    gross_pnl: float | None = None
    estimated_fees: float | None = None
    net_pnl: float | None = None
    net_r: float | None = None


def evaluate_executable_counterfactual(
    observations: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    decision_ts: float,
    stop_price: float,
    target_price: float,
    quantity: int,
    horizon_seconds: float = 600.0,
    max_entry_lag_seconds: float = 2.0,
    max_quote_gap_seconds: float = 10.0,
    adverse_slippage_bps: float = 0.0,
) -> ExecutableCounterfactual:
    """Return modeled net outcome, or UNRESOLVED without synthetic prices.

    Inputs must use the recorded stop/target and price-at-time quote snapshots.
    'captured_full_depth' is an independently verified provenance assertion
    supplied after raw archive fidelity validation, never inferred here.
    """
    if (
        not symbol.startswith("NFO:")
        or not symbol.endswith(("CE", "PE"))
        or isinstance(quantity, bool)
        or not isinstance(quantity, int)
        or quantity <= 0
        or not all(
            math.isfinite(float(x))
            for x in (
                decision_ts,
                stop_price,
                target_price,
                horizon_seconds,
                max_entry_lag_seconds,
                max_quote_gap_seconds,
                adverse_slippage_bps,
            )
        )
        or stop_price <= 0
        or target_price <= stop_price
        or min(horizon_seconds, max_entry_lag_seconds, max_quote_gap_seconds) <= 0
        or adverse_slippage_bps < 0
    ):
        raise ValueError("invalid executable-counterfactual contract")

    def unresolved(reason: str) -> ExecutableCounterfactual:
        return ExecutableCounterfactual("UNRESOLVED", reason)

    quotes: list[tuple[float, float, float]] = []
    for row in observations:
        if str(row.get("symbol") or "") != symbol:
            continue
        try:
            ts = _timestamp(row.get("timestamp"), field="quote timestamp")
            if ts < decision_ts or ts > decision_ts + horizon_seconds:
                continue
            if row.get("captured_full_depth") is not True or row.get("stale_data_used"):
                return unresolved("quote_provenance_or_freshness_unverified")
            bid, ask = float(row["bid"]), float(row["ask"])
            bid_size, ask_size = float(row["bid_size"]), float(row["ask_size"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return unresolved("incomplete_executable_quote")
        if (
            not all(math.isfinite(x) for x in (bid, ask, bid_size, ask_size))
            or bid <= 0
            or ask < bid
            or min(bid_size, ask_size) < quantity
        ):
            return unresolved("invalid_quote_or_insufficient_top_size")
        quotes.append((ts, bid, ask))

    if not quotes:
        return unresolved("no_captured_contract_quotes")
    quotes.sort(key=lambda row: row[0])
    for before, after in zip(quotes, quotes[1:]):
        if before[0] == after[0] and before[1:] != after[1:]:
            return unresolved("same_timestamp_quote_order_ambiguous")

    entry_ts, _, ask = quotes[0]
    if entry_ts - decision_ts > max_entry_lag_seconds:
        return unresolved("next_executable_ask_too_late")
    if not stop_price < ask < target_price:
        return unresolved("recorded_stop_target_incompatible_with_ask")
    slippage = adverse_slippage_bps / 10_000.0
    entry = ask * (1 + slippage)
    if not stop_price < entry < target_price:
        return unresolved("slippage_invalidates_entry_geometry")

    reason = "time_exit"
    last_ts = entry_ts
    exit_ts, exit_bid = entry_ts, quotes[0][1]
    for ts, bid, _ask in quotes:
        if ts - last_ts > max_quote_gap_seconds:
            return unresolved("unobserved_quote_gap")
        last_ts, exit_ts, exit_bid = ts, ts, bid
        if bid <= stop_price:
            reason = "stop"
            break
        if bid >= target_price:
            reason = "target"
            break
    else:
        if exit_ts < decision_ts + horizon_seconds - max_quote_gap_seconds:
            return unresolved("forward_horizon_incomplete")

    exit_price = exit_bid * (1 - slippage)
    if exit_price <= 0:
        return unresolved("slippage_exceeds_executable_exit")
    # Canonical charges; observed bid/ask crossing is already priced in.
    fees = estimate_round_trip_cost(
        entry_price=entry,
        exit_price=exit_price,
        quantity=quantity,
        half_spread=0.0,
    ).total
    gross = (exit_price - entry) * quantity
    risk_rupees = (entry - stop_price) * quantity + fees
    if risk_rupees <= 0:
        return unresolved("invalid_cost_inclusive_risk")
    return ExecutableCounterfactual(
        status="MODELED",
        reason=reason,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        entry_price=round(entry, 6),
        exit_price=round(exit_price, 6),
        gross_pnl=round(gross, 4),
        estimated_fees=round(fees, 4),
        net_pnl=round(gross - fees, 4),
        net_r=round((gross - fees) / risk_rupees, 6),
    )
