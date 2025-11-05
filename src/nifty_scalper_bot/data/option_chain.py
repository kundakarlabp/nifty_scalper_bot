"""Utilities for working with NIFTY option chains."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math
from statistics import median
from typing import Literal

_TICK_SIZE = 0.05


def _round_to_tick(value: float) -> float:
    """Round a price to the nearest NSE option tick size."""

    rounded = round(value / _TICK_SIZE) * _TICK_SIZE
    return round(rounded + 1e-12, 2)


@dataclass(slots=True)
class OptionQuote:
    """Single strike snapshot for a NIFTY weekly option."""

    strike: int
    ce_bid: float
    ce_ask: float
    pe_bid: float
    pe_ask: float
    ce_oi: int
    pe_oi: int
    ltt_ns: int

    def __post_init__(self) -> None:
        self.ce_bid = _round_to_tick(self.ce_bid)
        self.ce_ask = _round_to_tick(self.ce_ask)
        self.pe_bid = _round_to_tick(self.pe_bid)
        self.pe_ask = _round_to_tick(self.pe_ask)
        self.ce_oi = int(self.ce_oi)
        self.pe_oi = int(self.pe_oi)
        if self.ce_bid < 0 or self.pe_bid < 0:
            raise ValueError("Bid prices must be non-negative")
        if self.ce_ask <= 0 or self.pe_ask <= 0:
            raise ValueError("Ask prices must be positive")

    @property
    def ce_spread(self) -> float:
        return round(self.ce_ask - self.ce_bid, 2)

    @property
    def pe_spread(self) -> float:
        return round(self.pe_ask - self.pe_bid, 2)


@dataclass(slots=True)
class OptionChainSnapshot:
    """Point-in-time NIFTY chain snapshot with helper utilities.

    Example:
        >>> quotes = [
        ...     OptionQuote(22750, 105.1, 106.5, 94.8, 95.6, 45000, 52000, 1700000000),
        ...     OptionQuote(22800, 92.5, 93.9, 108.1, 109.5, 62000, 48000, 1700000100),
        ... ]
        >>> chain = OptionChainSnapshot(
        ...     symbol="NIFTY", spot=22785.0, expiry=date(2024, 5, 2), quotes=quotes
        ... )
        >>> chain.atm_strike()
        22800
    """

    symbol: Literal["NIFTY"]
    spot: float
    expiry: date
    quotes: list[OptionQuote] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.quotes = sorted(self.quotes, key=lambda q: q.strike)

    def atm_strike(self) -> int:
        """Return the nearest 50-point strike to the current spot."""

        nearest = int(round(self.spot / 50.0) * 50)
        return nearest

    def filter_liquid(
        self, *, min_oi: int = 20_000, max_spread: float = 3.0
    ) -> "OptionChainSnapshot":
        """Return a filtered snapshot with only liquid strikes."""

        filtered = [
            quote
            for quote in self.quotes
            if quote.ce_oi >= min_oi
            and quote.pe_oi >= min_oi
            and quote.ce_spread <= max_spread
            and quote.pe_spread <= max_spread
        ]
        return OptionChainSnapshot(
            symbol=self.symbol, spot=self.spot, expiry=self.expiry, quotes=filtered
        )

    def best_ce_otm(self, k: int) -> OptionQuote:
        """Return the k-th call strike above the ATM level."""

        if k < 0:
            raise ValueError("k must be non-negative")
        atm = self.atm_strike()
        candidates = [quote for quote in self.quotes if quote.strike > atm]
        if k >= len(candidates):
            raise IndexError("Not enough OTM call strikes available")
        return candidates[k]

    def best_pe_otm(self, k: int) -> OptionQuote:
        """Return the k-th put strike below the ATM level."""

        if k < 0:
            raise ValueError("k must be non-negative")
        atm = self.atm_strike()
        candidates = [quote for quote in reversed(self.quotes) if quote.strike < atm]
        if k >= len(candidates):
            raise IndexError("Not enough OTM put strikes available")
        return candidates[k]

    def sanity_check(self) -> None:
        """Validate chain consistency and raise if checks fail."""

        ltt_values: list[int] = []
        spreads: list[float] = []
        for quote in self.quotes:
            if any(
                math.isnan(value)
                for value in (
                    quote.ce_bid,
                    quote.ce_ask,
                    quote.pe_bid,
                    quote.pe_ask,
                )
            ):
                raise ValueError("NaN detected in quote")
            if not quote.ce_bid < quote.ce_ask:
                raise ValueError("Call bid/ask inverted")
            if not quote.pe_bid < quote.pe_ask:
                raise ValueError("Put bid/ask inverted")
            ce_spread = quote.ce_spread
            pe_spread = quote.pe_spread
            spreads.extend([ce_spread, pe_spread])
            for spread in (ce_spread, pe_spread):
                if spread < 0:
                    raise ValueError("Negative spread detected")
                scaled = spread / _TICK_SIZE
                if abs(round(scaled) - scaled) > 1e-6:
                    raise ValueError("Spread not aligned to tick size")
            ltt_values.append(quote.ltt_ns)
        if ltt_values and any(x > y for x, y in zip(ltt_values, ltt_values[1:])):
            raise ValueError("Quote timestamps not monotonic")
        if spreads and median(spreads) <= 0:
            raise ValueError("Median spread must be positive")


__all__ = ["OptionQuote", "OptionChainSnapshot"]
