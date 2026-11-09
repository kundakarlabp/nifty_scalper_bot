"""Deterministic, broker-free basket fill rehearsal for offline volatility research.

This module never sends/cancels orders, certifies broker margin or permits LIVE.
All generated fills have PAPER provenance. Real production linked-parent state,
broker execution and risk controls remain owned by their existing modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Mapping

Side = Literal["BUY", "SELL"]
Phase = Literal["ENTRY", "OPEN", "EXIT", "FLAT", "RECOVERY"]


@dataclass(frozen=True)
class PaperLeg:
    role: str
    symbol: str
    token: int
    side: Side
    quantity: int
    expiry: str
    strike: int
    option_type: Literal["CE", "PE"]


@dataclass(frozen=True)
class PaperQuote:
    symbol: str
    token: int
    bid: float
    ask: float
    bid_quantity: int
    ask_quantity: int
    observed_at: datetime
    source: str = "FULL"


@dataclass(frozen=True)
class PaperBudget:
    available_margin: float
    required_margin: float
    stress_loss_including_costs: float
    remaining_trade_risk: float
    remaining_daily_risk: float
    observed_at: datetime
    source: str = "BROKER_READ_ONLY"


@dataclass(frozen=True)
class PaperIntent:
    intent_id: str
    role: str
    side: Side
    quantity: int
    phase: Literal["ENTRY", "EXIT"]


@dataclass(frozen=True)
class PaperFill:
    event_id: str
    intent_id: str
    role: str
    side: Side
    quantity: int
    price: float
    observed_at: datetime
    source: str = "PAPER"


def _positive_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _aware(value: datetime) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _role_sets(structure: str) -> dict[str, tuple[Side, str]]:
    if structure == "STRADDLE":
        return {"long_call": ("BUY", "CE"), "long_put": ("BUY", "PE")}
    if structure == "IRON_CONDOR":
        return {
            "long_put": ("BUY", "PE"),
            "short_put": ("SELL", "PE"),
            "short_call": ("SELL", "CE"),
            "long_call": ("BUY", "CE"),
        }
    if structure == "BULL_CALL_SPREAD":
        return {"long_call": ("BUY", "CE"), "short_call": ("SELL", "CE")}
    if structure == "BEAR_PUT_SPREAD":
        return {"long_put": ("BUY", "PE"), "short_put": ("SELL", "PE")}
    raise ValueError("paper_structure_unsupported")


def _check_legs(structure: str, legs: Mapping[str, PaperLeg]) -> None:
    expected = _role_sets(structure)
    if set(legs) != set(expected):
        raise ValueError("paper_roles_invalid")
    first = next(iter(legs.values()))
    if not isinstance(first, PaperLeg):
        raise ValueError("paper_leg_identity_invalid")
    symbols: set[str] = set()
    tokens: set[int] = set()
    for role, leg in legs.items():
        if (
            not isinstance(leg, PaperLeg)
            or leg.role != role
            or (leg.side, leg.option_type) != expected[role]
            or not leg.symbol.startswith("NFO:NIFTY")
            or not leg.symbol.endswith(f"{leg.strike}{leg.option_type}")
            or type(leg.token) is not int
            or leg.token <= 0
            or type(leg.quantity) is not int
            or leg.quantity <= 0
            or type(leg.strike) is not int
            or leg.strike <= 0
            or leg.expiry != first.expiry
            or leg.quantity != first.quantity
            or leg.symbol in symbols
            or leg.token in tokens
        ):
            raise ValueError("paper_leg_identity_invalid")
        symbols.add(leg.symbol)
        tokens.add(leg.token)
    if structure == "STRADDLE" and legs["long_call"].strike != legs["long_put"].strike:
        raise ValueError("paper_strike_geometry_invalid")
    if structure == "IRON_CONDOR" and not (
        legs["long_put"].strike
        < legs["short_put"].strike
        < legs["short_call"].strike
        < legs["long_call"].strike
    ):
        raise ValueError("paper_strike_geometry_invalid")
    if structure == "BULL_CALL_SPREAD" and not (
        legs["long_call"].strike < legs["short_call"].strike
    ):
        raise ValueError("paper_strike_geometry_invalid")
    if structure == "BEAR_PUT_SPREAD" and not (
        legs["short_put"].strike < legs["long_put"].strike
    ):
        raise ValueError("paper_strike_geometry_invalid")


def _check_quote(leg: PaperLeg, quote: PaperQuote, side: Side, now: datetime, quantity: int) -> float:
    if (
        not _aware(now)
        or not isinstance(quote, PaperQuote)
        or quote.source != "FULL"
        or quote.symbol != leg.symbol
        or quote.token != leg.token
        or not _aware(quote.observed_at)
        or not timedelta(0) <= now - quote.observed_at <= timedelta(seconds=2)
        or not _positive_finite(quote.ask)
        or not _positive_finite(quote.bid)
        or quote.bid >= quote.ask
        or type(quote.bid_quantity) is not int
        or type(quote.ask_quantity) is not int
        or quote.bid_quantity < 0
        or quote.ask_quantity < 0
    ):
        raise ValueError("paper_quote_unverified")
    depth = quote.ask_quantity if side == "BUY" else quote.bid_quantity
    if quantity > depth:
        raise ValueError("paper_depth_insufficient")
    return quote.ask if side == "BUY" else quote.bid


class PaperBasketRehearsal:
    """Research-only state machine; callers supply explicitly labeled paper fills.

    Sequential entry buys protection before selling any short legs; exit buys
    shorts back before closing longs. Partial fills remain pending until their
    original intent is completed or cancelled. No optimistic automatic fills.
    """

    def __init__(
        self,
        *,
        parent_id: str,
        structure: str,
        legs: Mapping[str, PaperLeg],
        budget: PaperBudget,
        at: datetime,
        mode: str = "PAPER",
    ) -> None:
        if mode != "PAPER" or not parent_id or ":" in parent_id:
            raise ValueError("paper_mode_or_parent_invalid")
        _check_legs(structure, legs)
        if (
            not isinstance(budget, PaperBudget)
            or budget.source != "BROKER_READ_ONLY"
            or not _aware(at)
            or not _aware(budget.observed_at)
            or not timedelta(0) <= at - budget.observed_at <= timedelta(seconds=20)
            or not all(
                _positive_finite(x)
                for x in (
                    budget.available_margin,
                    budget.required_margin,
                    budget.stress_loss_including_costs,
                    budget.remaining_trade_risk,
                    budget.remaining_daily_risk,
                )
            )
            or budget.required_margin > budget.available_margin
            or budget.stress_loss_including_costs > min(
                budget.remaining_trade_risk, budget.remaining_daily_risk
            )
        ):
            raise ValueError("paper_margin_or_risk_unverified")
        self.parent_id = parent_id
        self.structure = structure
        self.legs = dict(legs)
        self.phase: Phase = "ENTRY"
        self.entry = {role: 0 for role in legs}
        self.exit = {role: 0 for role in legs}
        self.pending: PaperIntent | None = None
        self.fills: list[PaperFill] = []
        self._seen: dict[str, PaperFill] = {}
        self._sequence = 0
        self.live_eligible = False

    def next_intent(self) -> PaperIntent | None:
        if self.pending is not None:
            role = self.pending.role
            target = self.legs[role].quantity if self.phase == "ENTRY" else self.entry[role]
            done = self.entry[role] if self.phase == "ENTRY" else self.exit[role]
            return PaperIntent(self.pending.intent_id, role, self.pending.side, target - done, self.pending.phase)
        if self.phase not in {"ENTRY", "EXIT"}:
            return None
        # Deterministic hedge-first entry, short-first close ordering.
        roles = sorted(self.legs, key=lambda r: (self.legs[r].side != "BUY", r))
        if self.phase == "EXIT":
            roles = sorted(self.legs, key=lambda r: (self.legs[r].side != "SELL", r))
        for role in roles:
            leg = self.legs[role]
            total = leg.quantity if self.phase == "ENTRY" else self.entry[role]
            done = self.entry[role] if self.phase == "ENTRY" else self.exit[role]
            if done >= total:
                continue
            if self.phase == "ENTRY" and leg.side == "SELL":
                hedge_role = "long_put" if leg.option_type == "PE" else "long_call"
                if self.entry[hedge_role] < total:
                    raise ValueError("paper_short_hedge_unverified")
            side: Side = leg.side if self.phase == "ENTRY" else ("SELL" if leg.side == "BUY" else "BUY")
            self._sequence += 1
            self.pending = PaperIntent(f"{self.parent_id}:{self._sequence}", role, side, total - done, self.phase)
            return self.pending
        self.phase = "OPEN" if self.phase == "ENTRY" else "FLAT"
        return None

    def apply_paper_fill(
        self,
        intent: PaperIntent,
        quote: PaperQuote,
        *,
        quantity: int,
        event_id: str,
        at: datetime,
    ) -> PaperFill:
        if event_id in self._seen:
            existing = self._seen[event_id]
            if existing.intent_id != intent.intent_id or existing.quantity != quantity:
                raise ValueError("paper_duplicate_event_conflict")
            return existing
        current = self.next_intent()
        if not event_id or current is None or intent != current:
            raise ValueError("paper_intent_unverified")
        if type(quantity) is not int or not 0 < quantity <= current.quantity:
            raise ValueError("paper_fill_quantity_invalid")
        price = _check_quote(self.legs[current.role], quote, current.side, at, quantity)
        fill = PaperFill(event_id, intent.intent_id, current.role, current.side, quantity, price, at)
        if self.phase == "ENTRY":
            self.entry[current.role] += quantity
        else:
            self.exit[current.role] += quantity
        self.fills.append(fill)
        self._seen[event_id] = fill
        if quantity == current.quantity:
            self.pending = None
        return fill

    def cancel_pending(self) -> None:
        """Local simulation of cancellation; never assumes a broker cancel."""
        if self.pending is None:
            raise ValueError("paper_no_pending_intent")
        self.pending = None
        self.phase = "RECOVERY"

    def request_group_exit(self) -> None:
        if self.pending is not None or self.phase not in {"OPEN", "RECOVERY", "ENTRY"}:
            raise ValueError("paper_group_exit_requires_reconciled_intent")
        self.phase = "EXIT"

    def simulate_atomic_snapshot(
        self,
        quotes: Mapping[str, PaperQuote],
        *,
        at: datetime,
        exchange_all_or_none_capability: bool,
    ) -> bool:
        """Paper AON rehearsal only; a broker claim must be verified elsewhere."""
        if not exchange_all_or_none_capability or self.pending is not None:
            return False
        if self.phase not in {"ENTRY", "EXIT"}:
            return False
        phase = self.phase
        ordered = sorted(self.legs)
        actions: list[tuple[str, Side, int, float]] = []
        timestamps: list[datetime] = []
        for role in ordered:
            leg = self.legs[role]
            remaining = leg.quantity if phase == "ENTRY" else self.entry[role]
            if phase == "EXIT":
                remaining -= self.exit[role]
            if phase == "ENTRY" and self.entry[role] != 0:
                return False
            side: Side = leg.side if phase == "ENTRY" else ("BUY" if leg.side == "SELL" else "SELL")
            if remaining <= 0 or role not in quotes:
                return False
            try:
                price = _check_quote(leg, quotes[role], side, at, remaining)
            except ValueError:
                return False
            actions.append((role, side, remaining, price))
            timestamps.append(quotes[role].observed_at)
        if max(timestamps) - min(timestamps) > timedelta(milliseconds=500):
            return False
        for role, side, quantity, price in actions:
            self._sequence += 1
            intent_id = f"{self.parent_id}:{self._sequence}"
            event_id = f"paper-atomic-{intent_id}"
            fill = PaperFill(event_id, intent_id, role, side, quantity, price, at)
            if phase == "ENTRY":
                self.entry[role] += quantity
            else:
                self.exit[role] += quantity
            self.fills.append(fill)
            self._seen[event_id] = fill
        self.phase = "OPEN" if phase == "ENTRY" else "FLAT"
        return True