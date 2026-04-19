"""Utilities for working with NIFTY option chains."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Literal

from nifty_scalper_bot.utils.symbols import canonical

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


__all__ = ["OptionQuote", "OptionChainSnapshot", "OptionsChainManager"]


# ---------------------------------------------------------------------------
# Phase 7: OptionsChainManager — single source for live options chain data
# Refreshes every 30–60 seconds; never fetches per tick.
# ---------------------------------------------------------------------------

import logging
import threading
import time
from typing import Any

_OCM_LOGGER = logging.getLogger("nifty_scalper_bot.data.option_chain")

_NIFTY_STRIKE_STEP = 50  # NIFTY options are quoted in 50-point increments


class OptionsChainManager:
    """Manages NIFTY options chain data with periodic refresh.

    Primary source: Zerodha instruments + LTP + OI via broker client.
    Refresh interval: every 30–60 seconds (configurable).
    Never fetches on every tick — callers use cached results.

    Usage::

        mgr = OptionsChainManager(broker_client, mdm, refresh_interval=30)
        mgr.start()
        strike = mgr.get_atm_strike()
        oi     = mgr.get_oi_data()
        pain   = mgr.get_max_pain()
    """

    def __init__(
        self,
        broker_client: Any,
        market_data_manager: Any,
        refresh_interval: float = 30.0,
        underlying_symbol: str = "NSE:NIFTY",
    ) -> None:
        self._broker = broker_client
        self._mdm = market_data_manager
        self._refresh_interval = max(refresh_interval, 10.0)
        self._underlying_symbol = canonical(underlying_symbol)

        self._lock = threading.Lock()
        self._snapshot: OptionChainSnapshot | None = None
        self._last_refresh: float = 0.0
        self._spot_price: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background refresh thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._refresh_loop,
            name="options_chain_refresh",
            daemon=True,
        )
        self._thread.start()
        _OCM_LOGGER.info(
            "OptionsChainManager started — refresh_interval=%.0fs", self._refresh_interval
        )

    def stop(self) -> None:
        """Stop background refresh thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Public query methods (Phase 7 spec)
    # ------------------------------------------------------------------

    def get_atm_strike(self) -> int | None:
        """Return the at-the-money strike nearest to current spot.

        Returns:
            int | None: Nearest 50-point strike, or None if spot unavailable.
        """
        spot = self._get_spot()
        if not spot or spot <= 0:
            _OCM_LOGGER.warning("get_atm_strike: spot unavailable")
            return None
        atm = int(round(spot / _NIFTY_STRIKE_STEP) * _NIFTY_STRIKE_STEP)
        _OCM_LOGGER.debug("get_atm_strike: spot=%.2f atm=%d", spot, atm)
        return atm

    def get_strikes_around_atm(self, n: int = 5) -> list[int]:
        """Return n strikes above and below ATM (2n+1 total).

        Args:
            n: Number of strikes on each side of ATM.

        Returns:
            Sorted list of strike prices around ATM.
        """
        atm = self.get_atm_strike()
        if atm is None:
            return []
        return [atm + (i * _NIFTY_STRIKE_STEP) for i in range(-n, n + 1)]

    def get_oi_data(self) -> dict[int, dict[str, int]]:
        """Return open interest data keyed by strike.

        Returns:
            Dict mapping strike → {"ce_oi": int, "pe_oi": int}.
            Empty dict if no snapshot available.
        """
        with self._lock:
            snap = self._snapshot
        if snap is None:
            return {}
        return {
            q.strike: {"ce_oi": q.ce_oi, "pe_oi": q.pe_oi}
            for q in snap.quotes
        }

    def get_max_pain(self) -> int | None:
        """Compute max pain strike — the strike where total OI loss is minimum.

        Max pain = strike where writers (shorts) lose least = buyers lose most.
        At expiry, price gravitates toward this level.

        Returns:
            int | None: Max pain strike price, or None if data unavailable.
        """
        with self._lock:
            snap = self._snapshot
        if not snap or not snap.quotes:
            return None

        strikes = [q.strike for q in snap.quotes]
        if not strikes:
            return None

        min_pain = float("inf")
        max_pain_strike = strikes[0]

        for candidate in strikes:
            pain = 0.0
            for q in snap.quotes:
                # CE writers lose when spot > strike
                if candidate > q.strike:
                    pain += (candidate - q.strike) * q.ce_oi
                # PE writers lose when spot < strike
                if candidate < q.strike:
                    pain += (q.strike - candidate) * q.pe_oi
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = candidate

        _OCM_LOGGER.debug("get_max_pain: strike=%d pain=%.0f", max_pain_strike, min_pain)
        return max_pain_strike

    def get_snapshot(self) -> OptionChainSnapshot | None:
        """Return the latest cached chain snapshot (thread-safe)."""
        with self._lock:
            return self._snapshot

    # ------------------------------------------------------------------
    # Internal refresh logic
    # ------------------------------------------------------------------

    def _refresh_loop(self) -> None:
        """Background thread: refresh chain every refresh_interval seconds."""
        while not self._stop_event.is_set():
            try:
                self._refresh_once()
            except Exception as exc:
                _OCM_LOGGER.warning("OptionsChainManager refresh error: %s", exc)
            self._stop_event.wait(timeout=self._refresh_interval)

    def _refresh_once(self) -> None:
        """Fetch fresh chain data from broker and update snapshot."""
        spot = self._get_spot()
        if not spot or spot <= 0:
            _OCM_LOGGER.warning("_refresh_once: spot unavailable, skipping")
            return

        atm = int(round(spot / _NIFTY_STRIKE_STEP) * _NIFTY_STRIKE_STEP)
        strikes_to_fetch = [
            atm + (i * _NIFTY_STRIKE_STEP) for i in range(-10, 11)
        ]

        quotes: list[OptionQuote] = []
        for strike in strikes_to_fetch:
            try:
                q = self._fetch_strike_quote(strike, spot)
                if q is not None:
                    quotes.append(q)
            except Exception as exc:
                _OCM_LOGGER.debug("strike %d fetch error: %s", strike, exc)

        if not quotes:
            _OCM_LOGGER.warning("_refresh_once: no quotes fetched")
            return

        from datetime import date as _date
        snap = OptionChainSnapshot(
            symbol="NIFTY",
            spot=spot,
            expiry=_date.today(),
            quotes=quotes,
        )
        with self._lock:
            self._snapshot = snap
            self._last_refresh = time.time()
            self._spot_price = spot

        _OCM_LOGGER.info(
            "OptionsChainManager refreshed: spot=%.2f atm=%d quotes=%d",
            spot, atm, len(quotes),
        )

    def _fetch_strike_quote(self, strike: int, spot: float) -> OptionQuote | None:
        """Fetch CE and PE quotes for a single strike from broker."""
        if not self._broker:
            return None

        # Build Zerodha-style symbol names
        from datetime import date as _date
        today = _date.today()
        # Weekly expiry format: NIFTY{YY}{MON}{DD}{STRIKE}{CE/PE}
        # For simplicity we query via instruments cache if broker supports it
        ce_ltp = pe_ltp = 0.0
        ce_oi = pe_oi = 0

        try:
            # Try broker quote API — works with Zerodha NFO symbols
            # Attempt via MDM if available (avoids direct broker calls per tick)
            if self._mdm:
                ce_sym = f"NFO:NIFTY{today.strftime('%y%b').upper()}{strike}CE"
                pe_sym = f"NFO:NIFTY{today.strftime('%y%b').upper()}{strike}PE"
                ce_price = self._mdm.get_ltp(ce_sym) if hasattr(self._mdm, "get_ltp") else None
                pe_price = self._mdm.get_ltp(pe_sym) if hasattr(self._mdm, "get_ltp") else None
                if ce_price:
                    ce_ltp = float(ce_price)
                if pe_price:
                    pe_ltp = float(pe_price)
        except Exception as exc:
            _OCM_LOGGER.debug("_fetch_strike_quote MDM error strike=%d: %s", strike, exc)

        # Require at least CE ltp to form a quote
        if ce_ltp <= 0 and pe_ltp <= 0:
            return None

        # Build minimal valid OptionQuote (bid/ask estimated from LTP ±0.05)
        tick = 0.05
        return OptionQuote(
            strike=strike,
            ce_bid=max(ce_ltp - tick, tick),
            ce_ask=ce_ltp + tick if ce_ltp > 0 else tick * 2,
            pe_bid=max(pe_ltp - tick, tick),
            pe_ask=pe_ltp + tick if pe_ltp > 0 else tick * 2,
            ce_oi=ce_oi,
            pe_oi=pe_oi,
            ltt_ns=int(time.time() * 1e9),
        )

    def _get_spot(self) -> float:
        """Get NIFTY spot price from MDM, falling back to cached value."""
        if self._mdm:
            try:
                p = (
                    self._mdm.get_ltp(self._underlying_symbol)
                    if hasattr(self._mdm, "get_ltp")
                    else self._mdm.get_latest_price(self._underlying_symbol)
                )
                if p and float(p) > 0:
                    return float(p)
            except Exception:
                pass
        with self._lock:
            return self._spot_price
