"""Thread-safe in-memory market state for WS and polling integration."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
import time
from typing import Any


@dataclass(frozen=True, slots=True)
class MarketStateSnapshot:
    """Immutable market state snapshot. Args: none. Returns: snapshot. Raises: none."""

    spot_ltp: float | None
    option_ltps: dict[int, float]
    last_ws_update: float | None
    last_poll_update: float | None
    ohlc_cache: dict[tuple[int, str], tuple[float, list[dict[str, Any]]]]


class MarketState:
    """Central thread-safe state for market prices and OHLC cache."""

    def __init__(self) -> None:
        """Initialize empty market state. Args: none. Returns: none. Raises: none."""

        self._lock = RLock()
        self._spot_ltp: float | None = None
        self._option_ltps: dict[int, float] = {}
        self._last_ws_update: float | None = None
        self._last_poll_update: float | None = None
        self._ohlc_cache: dict[tuple[int, str], tuple[float, list[dict[str, Any]]]] = {}

    def update_tick(self, token: int, price: float, *, source: str = "ws") -> None:
        """Update token LTP. Args: token/price/source. Returns: none. Raises: ValueError."""

        normalized_token = int(token)
        normalized_price = float(price)
        if normalized_price <= 0:
            raise ValueError("price must be positive")
        now = time.time()
        with self._lock:
            self._option_ltps[normalized_token] = normalized_price
            if self._spot_ltp is None and normalized_token == 256265:
                self._spot_ltp = normalized_price
            if source == "poll":
                self._last_poll_update = now
            else:
                self._last_ws_update = now

    def set_spot_ltp(self, price: float, *, source: str = "ws") -> None:
        """Set spot LTP directly. Args: price/source. Returns: none. Raises: ValueError."""

        normalized_price = float(price)
        if normalized_price <= 0:
            raise ValueError("price must be positive")
        now = time.time()
        with self._lock:
            self._spot_ltp = normalized_price
            if source == "poll":
                self._last_poll_update = now
            else:
                self._last_ws_update = now

    def mark_poll_update(self) -> None:
        """Mark poll update timestamp. Args: none. Returns: none. Raises: none."""

        with self._lock:
            self._last_poll_update = time.time()

    def set_ohlc_cache(
        self, token: int, interval: str, rows: list[dict[str, Any]]
    ) -> None:
        """Store OHLC cache entry. Args: token/interval/rows. Returns: none. Raises: none."""

        with self._lock:
            self._ohlc_cache[(int(token), str(interval))] = (time.time(), list(rows))

    def get_ohlc_cache(
        self,
        token: int,
        interval: str,
        *,
        ttl: int,
    ) -> list[dict[str, Any]] | None:
        """Read OHLC cache by token+interval. Args: token/interval/ttl. Returns: rows|None. Raises: none."""

        with self._lock:
            cache = self._ohlc_cache.get((int(token), str(interval)))
            if cache is None:
                return None
            updated_at, rows = cache
            if (time.time() - updated_at) > max(0, int(ttl)):
                return None
            return list(rows)

    def get_snapshot(self) -> MarketStateSnapshot:
        """Return immutable market snapshot. Args: none. Returns: snapshot. Raises: none."""

        with self._lock:
            return MarketStateSnapshot(
                spot_ltp=self._spot_ltp,
                option_ltps=dict(self._option_ltps),
                last_ws_update=self._last_ws_update,
                last_poll_update=self._last_poll_update,
                ohlc_cache={
                    key: (value[0], list(value[1]))
                    for key, value in self._ohlc_cache.items()
                },
            )
