"""Fail-closed, read-only NIFTY option execution model.

Responsibilities: Model marketable orders only from observed five-level quotes,
historical fee inputs, causal latency, and exact point-in-time contract identity.
Ownership: Canonical offline backtesting evidence, not runtime execution.
Safety: Never place broker orders, infer OHLCV bid/ask, or grant LIVE promotion;
depth-modeled fills are never confirmed exchange/broker fills.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Literal, Sequence

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: float
    quantity: int


@dataclass(frozen=True, slots=True)
class ObservedBook:
    symbol: str
    token: str
    expiry: date
    lot_size: int
    observed_at: datetime
    bid: tuple[BookLevel, ...]
    ask: tuple[BookLevel, ...]
    source: Literal["ARCHIVED_FULL", "BROKER_FULL"]
    full_depth_captured: bool
    historical_identity_verified: bool
    clock_convention_verified: bool


@dataclass(frozen=True, slots=True)
class OrderIntent:
    symbol: str
    token: str
    expiry: date
    lot_size: int
    decision_at: datetime
    side: Side
    quantity: int
    latency_ms: int
    short_margin_verified: bool = False
    position_effect: Literal["OPEN", "CLOSE"] = "OPEN"
    closing_position_verified: bool = False


@dataclass(frozen=True, slots=True)
class ModeledExecution:
    status: Literal["UNRESOLVED", "PARTIAL", "MODELED_DEPTH"]
    reason: str
    filled_quantity: int = 0
    remaining_quantity: int = 0
    price: float | None = None
    fees: float | None = None
    observed_at: datetime | None = None
    evidence_tier: str = "OBSERVED_DEPTH_MODEL_NOT_BROKER_FILL"
    live_eligible: bool = False


def _positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _aware(value: datetime) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _valid_book(book: ObservedBook) -> bool:
    if (
        not book.symbol or not book.token or not _aware(book.observed_at)
        or not book.full_depth_captured or not book.historical_identity_verified
        or not book.clock_convention_verified
        or book.source not in {"ARCHIVED_FULL", "BROKER_FULL"}
        or type(book.lot_size) is not int or book.lot_size <= 0
        or len(book.bid) != 5 or len(book.ask) != 5
    ):
        return False
    for levels in (book.bid, book.ask):
        if any(not _positive(x.price) or type(x.quantity) is not int or x.quantity <= 0 for x in levels):
            return False
    return (
        all(book.bid[i].price > book.bid[i + 1].price for i in range(4))
        and all(book.ask[i].price < book.ask[i + 1].price for i in range(4))
        and book.bid[0].price < book.ask[0].price
    )


def historical_option_fee(
    *, day: date, side: Side, price: float, quantity: int,
    exchange_rate: float, broker: str = "ZERODHA", brokerage: float = 20.0,
) -> float:
    """Fee for one NSE option order, not exercise or final contract-note cost.

    Caller provides date-verified decimal exchange fee (0.03503% => 0.0003503).
    2024-11-06 through 2026-03-31: sell STT 0.10% premium; from 2026-04-01:
    0.15% premium. Brokerage, GST, SEBI fees and stamp charged per order.
    No unsupported assumption about when the exchange fee changed.
    """
    if (
        broker != "ZERODHA" or day < date(2024, 11, 6)
        or side not in ("BUY", "SELL")
        or not _positive(price) or type(quantity) is not int or quantity <= 0
        or not _positive(brokerage) or type(exchange_rate) not in (int, float)
        or not math.isfinite(exchange_rate) or not 0 < exchange_rate < 0.01
    ):
        raise ValueError("historical_fee_inputs_unverified")
    turnover = price * quantity
    stt = turnover * (0.001 if day < date(2026, 4, 1) else 0.0015) if side == "SELL" else 0.0
    exchange = turnover * exchange_rate
    sebi = turnover * 0.000001
    gst = (brokerage + exchange + sebi) * 0.18
    stamp = turnover * 0.00003 if side == "BUY" else 0.0
    return brokerage + stt + exchange + sebi + gst + stamp


def _tick_outward(price: float, side: Side, tick: float) -> float:
    if not _positive(tick):
        raise ValueError("invalid_tick")
    units = Decimal(str(price)) / Decimal(str(tick))
    mode = ROUND_CEILING if side == "BUY" else ROUND_FLOOR
    return float(units.to_integral_value(rounding=mode) * Decimal(str(tick)))


def replay_market_order(
    intent: OrderIntent,
    books: Sequence[ObservedBook],
    *,
    historical_exchange_rate: float,
    max_quote_delay_ms: int = 1500,
    adverse_slippage_bps: float = 0.0,
    tick_size: float = 0.05,
    cash_available: float | None = None,
    limit_price: float | None = None,
) -> ModeledExecution:
    """Walk an observed 5-level book with causal latency and partial-fill accounting.

    Does not extrapolate beyond level five, infer queue priority, fill from LTP,
    claim atomic multi-leg execution, or synthesize option strikes.
    """
    def reject(reason: str) -> ModeledExecution:
        return ModeledExecution("UNRESOLVED", reason, remaining_quantity=max(intent.quantity, 0))

    if (
        not _aware(intent.decision_at) or not intent.symbol or not intent.token
        or intent.side not in ("BUY", "SELL")
        or intent.position_effect not in ("OPEN", "CLOSE")
        or type(intent.quantity) is not int or intent.quantity <= 0
        or type(intent.lot_size) is not int or intent.lot_size <= 0
        or intent.quantity % intent.lot_size != 0
        or type(intent.latency_ms) is not int or intent.latency_ms < 0
        or type(max_quote_delay_ms) is not int or max_quote_delay_ms < 0
        or not _positive(tick_size)
        or not isinstance(intent.expiry, date)
        or intent.expiry < intent.decision_at.date()
        or type(adverse_slippage_bps) not in (int, float)
        or not math.isfinite(adverse_slippage_bps) or adverse_slippage_bps < 0
        or cash_available is not None and (
            type(cash_available) not in (int, float)
            or not math.isfinite(cash_available) or cash_available < 0
        )
        or limit_price is not None and not _positive(limit_price)
    ):
        return reject("invalid_order_or_contract_identity")
    if intent.position_effect == "CLOSE" and not intent.closing_position_verified:
        return reject("closing_position_ledger_unverified")
    if (
        intent.side == "SELL" and intent.position_effect == "OPEN"
        and not intent.short_margin_verified
    ):
        return reject("short_margin_evidence_missing")
    if not books:
        return reject("no_historical_quote_depth")
    available_at = intent.decision_at + timedelta(milliseconds=intent.latency_ms)
    deadline = available_at + timedelta(milliseconds=max_quote_delay_ms)
    best: ObservedBook | None = None
    seen: set[datetime] = set()
    for book in books:
        if not _valid_book(book):
            return reject("invalid_or_unverified_depth")
        if (
            book.symbol != intent.symbol or book.token != intent.token
            or book.expiry != intent.expiry or book.lot_size != intent.lot_size
        ):
            return reject("point_in_time_contract_mismatch")
        if book.observed_at in seen:
            return reject("ambiguous_quote_order")
        seen.add(book.observed_at)
        if available_at <= book.observed_at <= deadline and (
            best is None or book.observed_at < best.observed_at
        ):
            best = book
    if best is None:
        return reject("no_fresh_quote_after_latency")
    depth = best.ask if intent.side == "BUY" else best.bid
    outstanding = intent.quantity
    pieces: list[tuple[int, float]] = []
    for level in depth:
        if outstanding == 0:
            break
        adjusted = level.price * (
            1 + adverse_slippage_bps / 10000.0
            if intent.side == "BUY" else 1 - adverse_slippage_bps / 10000.0
        )
        if adjusted <= 0:
            break
        price = _tick_outward(adjusted, intent.side, tick_size)
        if limit_price is not None and (
            intent.side == "BUY" and price > limit_price
            or intent.side == "SELL" and price < limit_price
        ):
            break
        piece = min(outstanding, level.quantity)
        pieces.append((piece, price))
        outstanding -= piece
    if not pieces:
        return reject("no_executable_depth_within_limit")
    filled = intent.quantity - outstanding
    average = sum(n * p for n, p in pieces) / filled
    try:
        fee = historical_option_fee(
            day=intent.decision_at.date(), side=intent.side,
            price=average, quantity=filled, exchange_rate=historical_exchange_rate
        )
    except ValueError:
        return reject("historical_exchange_fee_unverified")
    if intent.side == "BUY" and cash_available is not None:
        if average * filled + fee > cash_available:
            return reject("insufficient_cash_including_fees")
    return ModeledExecution(
        status="MODELED_DEPTH" if outstanding == 0 else "PARTIAL",
        reason="observed_depth_model_only" if outstanding == 0 else "insufficient_observed_depth",
        filled_quantity=filled,
        remaining_quantity=outstanding,
        price=round(average, 8),
        fees=round(fee, 8),
        observed_at=best.observed_at,
    )


def classify_barrier(
    *, opening: float, high: float, low: float, closing: float,
    stop: float, target: float, direction: Literal["LONG", "SHORT"] = "LONG",
) -> str:
    """OHLC cannot establish target-vs-stop intraminute ordering or exit fills."""
    if (
        any(not _positive(v) for v in (opening, high, low, closing, stop, target))
        or low > min(opening, closing) or high < max(opening, closing) or low > high
    ):
        raise ValueError("invalid_bar")
    if direction == "LONG":
        if stop >= target:
            raise ValueError("invalid_barrier_geometry")
        if opening <= stop:
            return "STOP_GAP"
        if opening >= target:
            return "TARGET_GAP"
        st, tg = low <= stop, high >= target
    elif direction == "SHORT":
        if target >= stop:
            raise ValueError("invalid_barrier_geometry")
        if opening >= stop:
            return "STOP_GAP"
        if opening <= target:
            return "TARGET_GAP"
        st, tg = high >= stop, low <= target
    else:
        raise ValueError("invalid_direction")
    return (
        "AMBIGUOUS_UNRESOLVED" if st and tg
        else "STOP_TOUCHED_UNPRICED" if st
        else "TARGET_TOUCHED_UNPRICED" if tg
        else "NO_TOUCH"
    )


def execution_readiness(*, point_in_time_contracts: bool, verified_clocks: bool,
                        five_level_quotes: bool, actual_broker_fills: bool,
                        exit_ledger_verified: bool, chronological_oos: bool) -> dict[str, object]:
    missing = [
        key for key, ok in {
            "point_in_time_contract_identity": point_in_time_contracts,
            "verified_timestamp_and_latency": verified_clocks,
            "observed_five_level_quotes": five_level_quotes,
            "actual_order_ack_and_partial_fills": actual_broker_fills,
            "completed_exit_cost_ledger": exit_ledger_verified,
            "chronological_out_of_sample": chronological_oos,
        }.items() if not ok
    ]
    return {"execution_grade_ready": not missing, "missing_evidence": missing,
            "ohlcv_may_substitute_for_quotes": False, "automatic_live_promotion": False}
