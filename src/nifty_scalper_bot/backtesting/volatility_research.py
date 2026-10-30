"""Offline, fail-closed NIFTY volatility-structure research; no broker actions.

Responsibilities: Evaluate quote-consistent entry geometry for four-leg defined-risk
credit condors and debit straddles with the canonical per-order cost estimator.
Ownership: Research-only option strategy screening, not StrategyManager/execution.
Safety: Requires explicit SHADOW mode and synchronized FULL-depth quotes; never
registers a strategy, creates orders, or certifies historical/live profitability.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from nifty_scalper_bot.risk.cost_model import estimate_order_cost

_IST = ZoneInfo("Asia/Kolkata")
_QUALITY_WINDOW = timedelta(seconds=2)
_MAX_SKEW = timedelta(milliseconds=500)
_FORECAST_WINDOW = timedelta(seconds=60)


@dataclass(frozen=True, slots=True)
class VolatilityOptionQuote:
    """Immutable, broker-identified research observation of one NIFTY option."""

    symbol: str
    instrument_token: int
    option_type: Literal["CE", "PE"]
    strike: int
    expiry: date
    lot_size: int
    bid: float
    ask: float
    bid_quantity: int
    ask_quantity: int
    observed_at: datetime
    source: Literal["FULL", "ARCHIVED_FULL"]


@dataclass(frozen=True, slots=True)
class CondorAssessment:
    """Expiry payoff is bounded before fees; exit and settlement fees are unknown."""

    credit_per_unit: float
    gross_max_profit_rupees: float
    gross_max_loss_rupees: float
    entry_fees_rupees: float
    max_call_width: int
    max_put_width: int
    evidence_label: str = "RESEARCH_CANDIDATE"
    live_eligible: bool = False
    net_loss_bound_certified: bool = False


@dataclass(frozen=True, slots=True)
class LongStraddleAssessment:
    """Premium-at-risk calculation, not an IV trading recommendation."""

    debit_per_unit: float
    lower_expiry_breakeven: float
    upper_expiry_breakeven: float
    premium_plus_entry_fees_rupees: float
    forecast_minus_implied_vol: float
    evidence_label: str = "RESEARCH_CANDIDATE"
    live_eligible: bool = False


def _validated(
    quotes: tuple[VolatilityOptionQuote, ...],
    *,
    at: datetime,
) -> int:
    """Fail closed on mode, expiry, instrument, crossed/stale and unfilled depth."""
    if os.getenv("EXECUTION_MODE", "").strip().upper() != "SHADOW":
        raise ValueError("volatility_research_requires_shadow_process")
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError("research_clock_not_timezone_aware")
    market_time = at.astimezone(_IST)
    if not (time(9, 15) <= market_time.time() < time(15, 30)):
        raise ValueError("research_outside_market_session")
    if len(quotes) not in {2, 4}:
        raise ValueError("research_structure_leg_count_invalid")

    first = quotes[0]
    if (
        type(first.lot_size) is not int
        or first.lot_size <= 0
        or not isinstance(first.expiry, date)
        or first.expiry <= market_time.date()
    ):
        raise ValueError("research_expiry_or_lot_invalid")

    identifiers: set[str] = set()
    tokens: set[int] = set()
    stamps: list[datetime] = []
    for quote in quotes:
        if (
            quote.expiry != first.expiry
            or quote.lot_size != first.lot_size
            or quote.option_type not in {"CE", "PE"}
            or type(quote.strike) is not int
            or quote.strike <= 0
            or quote.strike % 50 != 0
            or not quote.symbol.startswith("NFO:NIFTY")
            or not quote.symbol.endswith(f"{quote.strike}{quote.option_type}")
            or type(quote.instrument_token) is not int
            or quote.instrument_token <= 0
            or quote.symbol in identifiers
            or quote.instrument_token in tokens
            or quote.source not in {"FULL", "ARCHIVED_FULL"}
            or quote.observed_at.tzinfo is None
            or quote.observed_at.utcoffset() is None
            or type(quote.bid_quantity) is not int
            or type(quote.ask_quantity) is not int
            or quote.bid_quantity < first.lot_size
            or quote.ask_quantity < first.lot_size
        ):
            raise ValueError("research_contract_or_depth_unverified")
        if (
            not math.isfinite(quote.bid)
            or not math.isfinite(quote.ask)
            or not (0 <= quote.bid < quote.ask)
        ):
            raise ValueError("research_crossed_or_nonfinite_book")
        age = at - quote.observed_at
        if not timedelta(0) <= age <= _QUALITY_WINDOW:
            raise ValueError("research_quote_stale_or_future")
        if quote.observed_at.astimezone(_IST).date() != market_time.date():
            raise ValueError("research_cross_session_quote")
        identifiers.add(quote.symbol)
        tokens.add(quote.instrument_token)
        stamps.append(quote.observed_at)
    if max(stamps) - min(stamps) > _MAX_SKEW:
        raise ValueError("research_multileg_quote_skew")
    return first.lot_size


def assess_credit_iron_condor(
    *,
    long_put: VolatilityOptionQuote,
    short_put: VolatilityOptionQuote,
    short_call: VolatilityOptionQuote,
    long_call: VolatilityOptionQuote,
    at: datetime,
) -> CondorAssessment:
    """Estimate point-in-time four-leg *gross* bounded expiry payoff.

    Entry crosses short bids and protective wing asks. Orders are not placed,
    fills are not assumed and margin, partial leg execution, Greeks, settlement
    taxes and true net worst-case losses are NOT modeled or certified.
    """
    quotes = (long_put, short_put, short_call, long_call)
    quantity = _validated(quotes, at=at)
    if (
        long_put.option_type != "PE"
        or short_put.option_type != "PE"
        or short_call.option_type != "CE"
        or long_call.option_type != "CE"
        or not (
            long_put.strike
            < short_put.strike
            < short_call.strike
            < long_call.strike
        )
    ):
        raise ValueError("research_condor_wings_invalid")

    credit = short_put.bid + short_call.bid - long_put.ask - long_call.ask
    call_width = long_call.strike - short_call.strike
    put_width = short_put.strike - long_put.strike
    if not math.isfinite(credit) or not 0 < credit < max(call_width, put_width):
        raise ValueError("research_condor_credit_invalid")
    entry_fees = sum(
        estimate_order_cost(turnover=price * quantity, side=side)
        for price, side in (
            (long_put.ask, "BUY"),
            (short_put.bid, "SELL"),
            (short_call.bid, "SELL"),
            (long_call.ask, "BUY"),
        )
    )
    return CondorAssessment(
        credit_per_unit=credit,
        gross_max_profit_rupees=credit * quantity,
        gross_max_loss_rupees=(max(call_width, put_width) - credit) * quantity,
        entry_fees_rupees=entry_fees,
        max_call_width=call_width,
        max_put_width=put_width,
    )


def assess_long_atm_straddle(
    *,
    call: VolatilityOptionQuote,
    put: VolatilityOptionQuote,
    at: datetime,
    implied_annual_vol: float,
    forecast_annual_vol: float,
    forecast_asof: datetime,
) -> LongStraddleAssessment:
    """Describe executable-side premium and a *diagnostic* forecast/IV gap.

    Same-expiry matched-strike quotes are required. Forecast provenance and
    calibration cannot be established by this function; no entry signal emitted.
    """
    quantity = _validated((call, put), at=at)
    if call.option_type != "CE" or put.option_type != "PE":
        raise ValueError("research_straddle_sides_invalid")
    if call.strike != put.strike:
        raise ValueError("research_straddle_strikes_mismatched")
    if forecast_asof.tzinfo is None or forecast_asof.utcoffset() is None:
        raise ValueError("research_forecast_clock_invalid")
    if not timedelta(0) <= at - forecast_asof <= _FORECAST_WINDOW:
        raise ValueError("research_forecast_stale_or_future")
    if not all(
        math.isfinite(value) and 0 < value <= 3.0
        for value in (implied_annual_vol, forecast_annual_vol)
    ):
        raise ValueError("research_volatility_input_invalid")
    debit = call.ask + put.ask
    fees = sum(
        estimate_order_cost(turnover=quote.ask * quantity, side="BUY")
        for quote in (call, put)
    )
    return LongStraddleAssessment(
        debit_per_unit=debit,
        lower_expiry_breakeven=call.strike - debit,
        upper_expiry_breakeven=call.strike + debit,
        premium_plus_entry_fees_rupees=debit * quantity + fees,
        forecast_minus_implied_vol=forecast_annual_vol - implied_annual_vol,
    )


__all__ = [
    "VolatilityOptionQuote",
    "CondorAssessment",
    "LongStraddleAssessment",
    "assess_credit_iron_condor",
    "assess_long_atm_straddle",
]
