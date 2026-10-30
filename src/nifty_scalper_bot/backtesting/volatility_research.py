"""Offline, fail-closed NIFTY volatility-structure research; no broker actions.

Responsibilities: Evaluate quote-consistent entry geometry and conservative
portfolio budget assumptions for condors and straddles using canonical costs.
Ownership: Research-only option strategy screening and read-only cache mapping,
not StrategyManager/execution.
Safety: Requires explicit SHADOW mode and synchronized FULL-depth quotes; never
registers a strategy, creates orders, or certifies historical/live profitability.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Literal, Mapping, cast
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
            long_put.strike < short_put.strike < short_call.strike < long_call.strike
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
    spot: float,
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
    if not math.isfinite(spot) or spot <= 0 or abs(call.strike - spot) > 25.0:
        raise ValueError("research_straddle_not_atm")
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


@dataclass(frozen=True, slots=True)
class PortfolioBudgetScreen:
    """Conservative research liquidity/risk screen; never live authorization."""

    research_budget_fits: bool
    reason: str
    planning_loss_reserve_rupees: float
    planning_margin_reserve_rupees: float
    live_eligible: bool = False
    net_loss_bound_certified: bool = False


def screen_volatility_portfolio_budget(
    *,
    assessment: CondorAssessment | LongStraddleAssessment,
    broker_required_margin: float | None,
    broker_free_margin: float | None,
    remaining_trade_risk: float | None,
    remaining_daily_risk: float | None,
    exit_settlement_fee_reserve: float | None,
    stressed_slippage_reserve: float | None,
) -> PortfolioBudgetScreen:
    """Check assumed portfolio risk and actual broker margin numbers.

    Caller must independently establish timestamped broker margin authority,
    unencumbered availability, pending obligations and a defensible stress
    scenario. This pure screen cannot certify naked-leg recovery or safe LIVE
    short-selling, even when its assumptions fit a supplied budget.
    """
    if os.getenv("EXECUTION_MODE", "").strip().upper() != "SHADOW":
        raise ValueError("volatility_research_requires_shadow_process")
    if not isinstance(assessment, (CondorAssessment, LongStraddleAssessment)):
        raise ValueError("research_structure_not_supported")

    inputs = (
        broker_required_margin,
        broker_free_margin,
        remaining_trade_risk,
        remaining_daily_risk,
        exit_settlement_fee_reserve,
        stressed_slippage_reserve,
    )
    if any(
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in inputs
    ):
        return PortfolioBudgetScreen(False, "research_budget_evidence_missing", 0, 0)

    assert broker_required_margin is not None
    assert broker_free_margin is not None
    assert remaining_trade_risk is not None
    assert remaining_daily_risk is not None
    assert exit_settlement_fee_reserve is not None
    assert stressed_slippage_reserve is not None

    gross_loss = (
        assessment.gross_max_loss_rupees
        if isinstance(assessment, CondorAssessment)
        else assessment.premium_plus_entry_fees_rupees
    )
    entry_cost = (
        assessment.entry_fees_rupees
        if isinstance(assessment, CondorAssessment)
        else 0.0
    )
    if (
        not math.isfinite(gross_loss)
        or not math.isfinite(entry_cost)
        or gross_loss <= 0
        or entry_cost < 0
        or broker_required_margin <= 0
        or exit_settlement_fee_reserve <= 0
        or stressed_slippage_reserve <= 0
    ):
        return PortfolioBudgetScreen(
            False, "research_stress_or_margin_unverified", 0, 0
        )

    loss_reserve = (
        gross_loss
        + entry_cost
        + exit_settlement_fee_reserve
        + stressed_slippage_reserve
    )
    margin_reserve = broker_required_margin + (
        exit_settlement_fee_reserve + stressed_slippage_reserve
    )
    if not math.isfinite(loss_reserve) or not math.isfinite(margin_reserve):
        return PortfolioBudgetScreen(False, "research_budget_evidence_missing", 0, 0)
    if loss_reserve > min(remaining_trade_risk, remaining_daily_risk):
        return PortfolioBudgetScreen(
            False, "research_risk_budget_insufficient", loss_reserve, margin_reserve
        )
    if margin_reserve > broker_free_margin:
        return PortfolioBudgetScreen(
            False, "research_free_margin_insufficient", loss_reserve, margin_reserve
        )
    return PortfolioBudgetScreen(
        True, "research_assumptions_fit_budget", loss_reserve, margin_reserve
    )


def quote_from_canonical_cache(
    *,
    symbol: str,
    instrument_manager: Any,
    market_data_manager: Any,
) -> VolatilityOptionQuote:
    """Read one verified NIFTY instrument plus its existing WS FULL book.

    This creates no subscription, cache, reservation or order. The caller must
    assess the *entire* 2/4-leg snapshot against a common timestamp afterwards.
    """
    if os.getenv("EXECUTION_MODE", "").strip().upper() != "SHADOW":
        raise ValueError("volatility_research_requires_shadow_process")
    normalized_symbol = str(symbol or "").strip().upper()
    if not normalized_symbol.startswith("NFO:NIFTY"):
        raise ValueError("research_contract_or_depth_unverified")
    try:
        contract = instrument_manager.lookup(normalized_symbol)
        tick = market_data_manager.get_quote(normalized_symbol)
    except Exception as exc:
        raise ValueError("research_canonical_source_unavailable") from exc
    if not isinstance(contract, Mapping) or not isinstance(tick, Mapping):
        raise ValueError("research_canonical_source_unavailable")

    option_type = str(contract.get("instrument_type") or "").upper()
    bare = normalized_symbol.partition(":")[2]
    if (
        str(contract.get("exchange") or "").upper() != "NFO"
        or str(contract.get("name") or "").upper() != "NIFTY"
        or str(contract.get("tradingsymbol") or "").upper() != bare
        or option_type not in {"CE", "PE"}
        or str(tick.get("source") or "").lower()
        not in {"ws", "ws_full", "full", "websocket"}
        or str(tick.get("bid_ask_source") or "").lower() != "ws_full"
        or tick.get("depth_available") is not True
    ):
        raise ValueError("research_contract_or_depth_unverified")

    depth = tick.get("depth")
    if not isinstance(depth, Mapping):
        raise ValueError("research_contract_or_depth_unverified")
    buy = depth.get("buy")
    sell = depth.get("sell")
    if not isinstance(buy, list) or not isinstance(sell, list) or not buy or not sell:
        raise ValueError("research_contract_or_depth_unverified")
    if not isinstance(buy[0], Mapping) or not isinstance(sell[0], Mapping):
        raise ValueError("research_contract_or_depth_unverified")

    try:
        token = int(contract["instrument_token"])
        tick_token = int(tick["instrument_token"])
        strike_number = float(contract["strike"])
        lot_size = int(contract["lot_size"])
        bid = float(buy[0]["price"])
        ask = float(sell[0]["price"])
        bid_quantity = int(buy[0]["quantity"])
        ask_quantity = int(sell[0]["quantity"])
        top_bid = float(tick["bid"])
        top_ask = float(tick["ask"])
        expiry_value = contract["expiry"]
        if isinstance(expiry_value, datetime):
            expiry = expiry_value.date()
        elif isinstance(expiry_value, date):
            expiry = expiry_value
        else:
            expiry = date.fromisoformat(str(expiry_value))
        raw_time = tick.get("exchange_timestamp") or tick.get("timestamp")
        observed = (
            raw_time
            if isinstance(raw_time, datetime)
            else datetime.fromisoformat(str(raw_time))
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("research_contract_or_depth_unverified") from exc

    if (
        token <= 0
        or token != tick_token
        or not math.isfinite(strike_number)
        or not strike_number.is_integer()
        or lot_size <= 0
        or not all(math.isfinite(x) for x in (bid, ask, top_bid, top_ask))
        or abs(bid - top_bid) > 0.001
        or abs(ask - top_ask) > 0.001
        or observed.tzinfo is None
        or observed.utcoffset() is None
    ):
        raise ValueError("research_contract_or_depth_unverified")
    return VolatilityOptionQuote(
        symbol=normalized_symbol,
        instrument_token=token,
        option_type=cast(Literal["CE", "PE"], option_type),
        strike=int(strike_number),
        expiry=expiry,
        lot_size=lot_size,
        bid=bid,
        ask=ask,
        bid_quantity=bid_quantity,
        ask_quantity=ask_quantity,
        observed_at=observed,
        source="FULL",
    )


__all__ = [
    "VolatilityOptionQuote",
    "PortfolioBudgetScreen",
    "screen_volatility_portfolio_budget",
    "quote_from_canonical_cache",
    "CondorAssessment",
    "LongStraddleAssessment",
    "assess_credit_iron_condor",
    "assess_long_atm_straddle",
]
