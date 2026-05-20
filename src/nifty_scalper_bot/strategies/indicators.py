"""Utility classes for calculating technical indicators.

This module provides a :class:`PriceHistory` container for efficiently storing
OHLCV (open, high, low, close, volume) data and an :class:`IndicatorEngine`
that calculates common technical indicators from the stored price history.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, time, timedelta, timezone
import logging
import os
import threading
from typing import Any, Callable, Deque, Dict, Iterable, Literal, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from nifty_scalper_bot.utils.logging import get_logger, log_throttled

PriceInput = float | Mapping[str, float] | Sequence[float]

_INDIA_TZ = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN = time(hour=9, minute=15)
_MARKET_CLOSE = time(hour=15, minute=30)


LOGGER = get_logger(__name__)

_FEATURE_CAPABILITIES: dict[str, bool] = {
    'vwap': True,
    'bollinger': True,
    'orb_levels': True,
    'cpr_levels': False,
    'structure_flags': False,
    'order_book_depth': True,
}


def supports_feature(name: str) -> bool:
    """Args: name. Returns: feature support status. Raises: none."""
    return bool(_FEATURE_CAPABILITIES.get(str(name).strip().lower(), False))


class PriceHistory:
    """Store OHLCV data efficiently.

    The container keeps only the most recent ``max_length`` entries using
    :class:`collections.deque` to guarantee ``O(1)`` append operations. When the
    container is full, the oldest entries are automatically discarded. Prices
    can be provided either as a single closing price or as OHLC tuples/mappings.
    """

    def __init__(self, max_length: int = 1500):
        """Initialize with max history length.

        Default raised 500→1500: Zerodha returns up to 5 trading days of 1-min
        bars (≈5×375=1875 bars). 500 silently discarded ~1375 bars, degrading
        ATR/VWAP/EMA accuracy. 1500 holds a full 4-day window comfortably.
        """
        if max_length <= 0:
            msg = "max_length must be positive"
            raise ValueError(msg)
        self._max_length = max_length
        self._opens: Deque[float] = deque(maxlen=max_length)
        self._highs: Deque[float] = deque(maxlen=max_length)
        self._lows: Deque[float] = deque(maxlen=max_length)
        self._closes: Deque[float] = deque(maxlen=max_length)
        self._volumes: Deque[int] = deque(maxlen=max_length)
        self._timestamps: Deque[datetime] = deque(maxlen=max_length)
        self._is_complete: Deque[bool] = deque(maxlen=max_length)
        self._is_provisional: Deque[bool] = deque(maxlen=max_length)

    def add_tick(
        self,
        price: PriceInput,
        volume: int,
        timestamp: datetime,
        *,
        is_complete: bool = True,
        is_provisional: bool = False,
    ) -> None:
        """Add a new price tick.

        Parameters
        ----------
        price:
            Either the closing price or a mapping/sequence representing an OHLC
            bar. Supported layouts:

            * ``float``: the value is used for open, high, low and close.
            * ``Mapping``: expected keys include ``open``, ``high``, ``low`` and
              ``close``. Missing values default to the closing price.
            * ``Sequence``: interpreted as ``(open, high, low, close)`` when at
              least three values are available. Missing entries again default to
              the closing price.
        volume:
            Traded volume for the bar.
        timestamp:
            Timestamp of the bar. Naive timestamps are assumed to be in UTC.
        """
        open_price, high_price, low_price, close_price = self._normalize_price(price)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp = timestamp.astimezone(timezone.utc).replace(microsecond=0)
        if self._timestamps:
            last_ts = self._timestamps[-1]
            if timestamp < last_ts:
                return
            if timestamp == last_ts:
                self._opens.pop()
                self._highs.pop()
                self._lows.pop()
                self._closes.pop()
                self._volumes.pop()
                self._timestamps.pop()
                self._is_complete.pop()
                self._is_provisional.pop()
        self._opens.append(open_price)
        self._highs.append(high_price)
        self._lows.append(low_price)
        self._closes.append(close_price)
        self._volumes.append(int(volume))
        self._timestamps.append(timestamp)
        self._is_complete.append(bool(is_complete))
        self._is_provisional.append(bool(is_provisional))

    def get_closes(self, count: int | None = None) -> list[float]:
        """Get closing prices."""
        return self._get_tail(self._closes, count)

    def get_highs(self, count: int | None = None) -> list[float]:
        """Get high prices."""
        return self._get_tail(self._highs, count)

    def get_opens(self, count: int | None = None) -> list[float]:
        """Get open prices."""
        return self._get_tail(self._opens, count)

    def get_lows(self, count: int | None = None) -> list[float]:
        """Get low prices."""
        return self._get_tail(self._lows, count)

    def get_volumes(self, count: int | None = None) -> list[int]:
        """Get volumes."""
        return self._get_tail(self._volumes, count)

    def get_timestamps(self, count: int | None = None) -> list[datetime]:
        """Return timestamps for stored bars."""
        return self._get_tail(self._timestamps, count)

    def get_completeness(self, count: int | None = None) -> list[bool]:
        """Return completeness flags for stored bars."""
        return self._get_tail(self._is_complete, count)

    def get_provisional_flags(self, count: int | None = None) -> list[bool]:
        """Return provisional flags for stored bars."""
        return self._get_tail(self._is_provisional, count)

    @property
    def last_timestamp(self) -> datetime | None:
        """Return the timestamp of the most recent bar."""
        return self._timestamps[-1] if self._timestamps else None

    def __len__(self) -> int:
        """Return number of bars in history."""
        return len(self._closes)

    def _normalize_price(self, price: PriceInput) -> tuple[float, float, float, float]:
        if isinstance(price, Mapping):
            close_value = price.get("close")
            if close_value is None:
                close_value = price.get("c")
            if close_value is None:
                close_value = price.get("price")
            if close_value is None:
                msg = "close price is required in mapping input"
                raise ValueError(msg)
            close_price = float(close_value)
            open_price = float(price.get("open", close_price))
            high_price = float(price.get("high", close_price))
            low_price = float(price.get("low", close_price))
        elif isinstance(price, Sequence) and not isinstance(
            price, (str, bytes, bytearray)
        ):
            if not price:
                msg = "price sequence must not be empty"
                raise ValueError(msg)
            close_price = float(price[-1])
            open_price = float(price[0]) if len(price) > 0 else close_price
            high_price = float(price[1]) if len(price) > 1 else close_price
            low_price = float(price[2]) if len(price) > 2 else close_price
        else:
            close_price = float(price)
            open_price = high_price = low_price = close_price
        return open_price, high_price, low_price, close_price

    def _get_tail(self, source: Deque, count: int | None) -> list:
        if count is None or count >= len(source):
            return list(source)
        if count <= 0:
            return []
        return list(source)[-count:]


class IndicatorEngine:
    """Calculate technical indicators from price data."""

    def __init__(self):
        """Initialize indicator engine."""
        self._histories: Dict[str, PriceHistory] = {}
        self._cache: Dict[str, Dict[str, tuple[Any, datetime]]] = {}
        self._runtime_context: Dict[str, Dict[str, Any]] = {}
        self._last_valid_vwap: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._logger = get_logger(__name__)

    def set_runtime_context(self, symbol: str, context: Mapping[str, Any], *, merge: bool = True) -> None:
        """Store runtime context. Args: symbol, context. Returns: None. Raises: Exception."""
        try:
            if not symbol:
                return
            allowed_context_keys = {
                "contract_side", "underlying", "spot_symbol", "spot_price",
                "futures_symbol", "futures_price", "quote_age_s", "spread_pct",
                "bid", "ask", "best_bid", "best_ask",
                "bid_qty", "ask_qty", "buy_qty", "sell_qty",
                "depth", "depth_available", "tradable_quote",
                "spread", "mid", "bid_ask_source", "tick_direction",
                "data_age_seconds",
                "selected_ce",
                "selected_pe",
                "atm_strike",
                "is_selected_option",
                "strike_distance_from_atm",
                "selected_side",
                "option_role",
            }
            with self._lock:
                symbol_context = self._runtime_context.setdefault(symbol, {}) if merge else {}
                for key, value in dict(context).items():
                    if key in allowed_context_keys:
                        symbol_context[key] = value
                self._runtime_context[symbol] = symbol_context
                self._cache.pop(symbol, None)
        except Exception as e:
            self._logger.error("Failure in IndicatorEngine.set_runtime_context: %s", e)
            raise

    def set_indicators(self, symbol: str, indicators: Mapping[str, Any]) -> None:
        """Backward-compatible alias for runtime context setter. Args: symbol, indicators. Returns: None. Raises: Exception."""
        self.set_runtime_context(symbol, indicators)

    def get_runtime_context(self, symbol: str, default: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Fetch runtime context. Args: symbol. Returns: context copy. Raises: Exception."""
        try:
            with self._lock:
                return dict(self._runtime_context.get(symbol, dict(default or {})))
        except Exception as e:
            self._logger.error("Failure in IndicatorEngine.get_runtime_context: %s", e)
            raise

    def clear_runtime_context(self, symbol: str | None = None) -> None:
        """Clear runtime context for one/all symbols. Args: symbol. Returns: None. Raises: Exception."""
        try:
            with self._lock:
                if symbol is None:
                    self._runtime_context.clear()
                else:
                    self._runtime_context.pop(symbol, None)
        except Exception as e:
            self._logger.error("Failure in IndicatorEngine.clear_runtime_context: %s", e)
            raise

    def update_price(
        self,
        symbol: str,
        price: PriceInput,
        volume: int = 0,
        timestamp: datetime | None = None,
        *,
        is_complete: bool = True,
        is_provisional: bool = False,
    ) -> None:
        """Update price for symbol and invalidate cache."""
        with self._lock:
            history = self._histories.setdefault(symbol, PriceHistory())
            timestamp = timestamp or datetime.now(timezone.utc)
            history.add_tick(
                price,
                volume,
                timestamp,
                is_complete=is_complete,
                is_provisional=is_provisional,
            )
            self._cache.pop(symbol, None)

    def ingest_bar(self, symbol: str, bar: Any) -> None:
        """Ingest normalized/live bar. Args: symbol, bar. Returns: None. Raises: Exception."""
        try:
            if isinstance(bar, Mapping):
                ts_value = bar.get("timestamp") or bar.get("start")
                open_price = float(bar["open"])
                high_price = float(bar["high"])
                low_price = float(bar["low"])
                close_price = float(bar["close"])
                volume = int(bar.get("volume", 0) or 0)
            else:
                ts_value = getattr(bar, "timestamp", None) or getattr(bar, "start", None)
                open_price = float(getattr(bar, "open"))
                high_price = float(getattr(bar, "high"))
                low_price = float(getattr(bar, "low"))
                close_price = float(getattr(bar, "close"))
                volume = int(getattr(bar, "volume", 0) or 0)

            if isinstance(ts_value, datetime):
                ts = ts_value
            elif isinstance(ts_value, (int, float)):
                ts = datetime.fromtimestamp(float(ts_value), tz=timezone.utc)
            elif isinstance(ts_value, str):
                ts = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
            else:
                raise ValueError(f"Unsupported bar timestamp: {ts_value!r}")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            payload_ohlc = {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
            }
            self.update_price(
                symbol,
                payload_ohlc,
                volume=volume,
                timestamp=ts,
                is_complete=True,
                is_provisional=False,
            )
        except Exception as e:
            self._logger.error(
                "Failure in IndicatorEngine.ingest_bar: %s",
                e,
                extra={"event": "INDICATOR_INGEST_BAR_FAILED", "symbol": symbol},
            )
            raise


    def ingest_historical_bar(self, symbol: str, bar: Mapping[str, Any]) -> int:
        """Args: symbol/bar mapping. Returns: new history count. Raises: Exception."""
        try:
            self.ingest_bar(symbol, bar)
            return self.history_count(symbol)
        except Exception as e:
            self._logger.error("Failure in IndicatorEngine.ingest_historical_bar: %s", e)
            raise

    def replace_history(
        self,
        symbol: str,
        bars: Iterable[Mapping[str, Any]],
        source: str = "historical_reseed",
        min_bars: int = 1,
    ) -> int:
        """Args: symbol/bars/source/min_bars. Returns: final history count. Raises: Exception."""
        try:
            normalized_rows: dict[datetime, dict[str, Any]] = {}
            for row in bars or ():
                if not isinstance(row, Mapping):
                    continue
                ts_value = (
                    row.get("timestamp")
                    or row.get("start")
                    or row.get("date")
                    or row.get("time")
                )
                if ts_value is None:
                    continue
                if isinstance(ts_value, datetime):
                    ts = ts_value
                elif isinstance(ts_value, (int, float)):
                    ts = datetime.fromtimestamp(float(ts_value), tz=timezone.utc)
                elif isinstance(ts_value, str):
                    ts = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                else:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ts = ts.astimezone(timezone.utc).replace(microsecond=0)
                open_price = float(row.get("open", row.get("close", 0.0)) or 0.0)
                high_price = float(row.get("high", row.get("close", 0.0)) or 0.0)
                low_price = float(row.get("low", row.get("close", 0.0)) or 0.0)
                close_price = float(row.get("close", row.get("open", 0.0)) or 0.0)
                volume = int(float(row.get("volume", 0) or 0))

                if min(open_price, high_price, low_price, close_price) <= 0:
                    continue

                normalized_rows[ts] = {
                    "timestamp": ts,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }

            selected_rows = sorted(normalized_rows.values(), key=lambda item: item["timestamp"])
            if not selected_rows:
                self._logger.warning(
                    "INDICATOR_HISTORY_RESEED_EMPTY symbol=%s min_bars=%d source=%s",
                    symbol,
                    max(1, int(min_bars or 1)),
                    source,
                )
                with self._lock:
                    existing = self._histories.get(symbol)
                    existing_count = len(existing) if existing is not None else 0
                return existing_count
            new_history = PriceHistory()
            for bar in selected_rows:
                new_history.add_tick(
                    {
                        "open": bar["open"],
                        "high": bar["high"],
                        "low": bar["low"],
                        "close": bar["close"],
                    },
                    volume=int(bar["volume"]),
                    timestamp=bar["timestamp"],
                    is_complete=True,
                    is_provisional=False,
                )
            with self._lock:
                self._histories[symbol] = new_history
                self._cache.pop(symbol, None)
                final_count = len(new_history)
            event_name = (
                "INDICATOR_HISTORY_RESEEDED"
                if final_count >= max(1, int(min_bars or 1))
                else "INDICATOR_HISTORY_RESEED_SHORT"
            )
            self._logger.info(
                "%s symbol=%s bars=%d min_bars=%d source=%s",
                event_name,
                symbol,
                final_count,
                max(1, int(min_bars or 1)),
                source,
            )
            return final_count
        except Exception as e:
            self._logger.exception(
                "INDICATOR_HISTORY_RESEED_FAILED symbol=%s source=%s error=%s",
                symbol,
                source,
                e,
            )
            raise

    def history_count(self, symbol: str) -> int:
        """Return symbol history length. Args: symbol. Returns: count. Raises: none."""
        with self._lock:
            history = self._histories.get(symbol)
            return len(history) if history is not None else 0

    def get_history(self, symbol: str, count: int | None = None, *, field: Literal["close", "bars"] = "close") -> list[Any]:
        """Args: symbol, count. Returns: close-price history list. Raises: Exception."""
        LOGGER.debug(
            "Entered IndicatorEngine.get_history",
            extra={"event": "indicator_engine_get_history_enter", "symbol": symbol},
        )
        try:
            with self._lock:
                history = self._histories.get(symbol)
                if history is None:
                    try:
                        from nifty_scalper_bot.utils.market_hours import (
                            is_market_open_now,
                        )

                        market_open_now = is_market_open_now()
                    except Exception:
                        market_open_now = True
                    if market_open_now:
                        log_throttled(
                            LOGGER,
                            f"indicator_history_missing:{symbol}",
                            "Condition met: indicator_history_missing",
                            interval_sec=60.0,
                            level=logging.INFO,
                            extra={
                                "event": "indicator_engine_history_missing",
                                "symbol": symbol,
                            },
                        )
                    else:
                        log_throttled(
                            LOGGER,
                            f"indicator_history_missing_offmarket:{symbol}",
                            "Condition met: indicator_history_missing (market_closed)",
                            interval_sec=900.0,
                            level=logging.DEBUG,
                            extra={
                                "event": "indicator_engine_history_missing",
                                "symbol": symbol,
                                "market_session_state": "closed",
                            },
                        )
                    return []
                if field == "bars":
                    opens = history.get_opens(count)
                    highs = history.get_highs(count)
                    lows = history.get_lows(count)
                    closes = history.get_closes(count)
                    volumes = history.get_volumes(count)
                    timestamps = history.get_timestamps(count)
                    complete = history.get_completeness(count)
                    provisional = history.get_provisional_flags(count)
                    bars = [
                        {"open": o, "high": h, "low": l, "close": c, "volume": v, "timestamp": ts, "is_complete": ic, "is_provisional": ip}
                        for o, h, l, c, v, ts, ic, ip in zip(opens, highs, lows, closes, volumes, timestamps, complete, provisional)
                    ]
                else:
                    bars = history.get_closes(count)
            LOGGER.debug(
                "Condition met: indicator_history_resolved",
                extra={
                    "event": "indicator_engine_history_resolved",
                    "symbol": symbol,
                    "bars": len(bars),
                },
            )
            return bars
        except Exception as e:  # noqa: BLE001
            LOGGER.error("Failure in IndicatorEngine.get_history: %s", e, exc_info=True)
            return []

    def get_indicators(
        self, symbol: str, names: Iterable[str] | None = None
    ) -> dict[str, float | tuple[float, float, float] | None]:
        """Args: symbol, names. Returns: indicators dict. Raises: Exception."""
        LOGGER.debug(
            "Entered IndicatorEngine.get_indicators",
            extra={"event": "indicator_engine_get_indicators_enter", "symbol": symbol},
        )
        try:
            with self._lock:
                indicators: dict[str, float | tuple[float, float, float] | None] = {}
                indicators["rsi"] = self.get_rsi(symbol)
                indicators["ema"] = self.get_ema(symbol)
                indicators["sma"] = self.get_sma(symbol)
            macd = self.get_macd(symbol)
            if macd is not None:
                (
                    indicators["macd"],
                    indicators["macd_signal"],
                    indicators["macd_histogram"],
                ) = macd
            else:
                indicators["macd"] = indicators["macd_signal"] = indicators[
                    "macd_histogram"
                ] = None
            bands = self.get_bollinger_bands(symbol)
            if bands is not None:
                (
                    indicators["bollinger_upper"],
                    indicators["bollinger_middle"],
                    indicators["bollinger_lower"],
                ) = bands
            else:
                indicators["bollinger_upper"] = indicators["bollinger_middle"] = (
                    indicators["bollinger_lower"]
                ) = None
            indicators["atr"] = self.get_atr(symbol)
            indicators["atr_trend"] = self.get_atr_trend(symbol)
            indicators["volatility_index"] = self.get_volatility_index(symbol)
            indicators["vwap"] = self.get_vwap(symbol)
            try:
                self._augment_session_metrics(symbol, indicators)
            except Exception as e:
                __import__("logging").getLogger(__name__).exception(
                    "[CRITICAL] unhandled exception", exc_info=True
                )
                raise  # Session metrics are non-critical — don't crash pipeline
            # ✅ FIX S1: Expose latest-bar OHLC for strategies that need it
            history = self._histories.get(symbol)
            if history and len(history) > 0:
                closes = history.get_closes(1)
                highs = history.get_highs(1)
                lows = history.get_lows(1)
                opens = (
                    history.get_opens(1) if hasattr(history, "get_opens") else closes
                )
                if closes:
                    indicators.setdefault("close", float(closes[-1]))
                if highs:
                    indicators.setdefault("high", float(highs[-1]))
                if lows:
                    indicators.setdefault("low", float(lows[-1]))
                if opens:
                    indicators.setdefault("open", float(opens[-1]))
            _always_include = {
                "vwap",
                "atr",
                "volume",
                "avg_volume",
                "rsi",
                "high",
                "low",
                "close",
                "open",
                "bollinger_upper",
                "bollinger_lower",
                "bollinger_middle",
                "minutes_since_open",
                "minutes_until_close",
                "volume_spike_ratio",
                "bar_range",
            }
            name_set: set[str] = set()
            if names is None:
                log_throttled(
                    LOGGER,
                    "indicator_names_defaulted_global",
                    "indicator_names_defaulted: using default indicator set",
                    interval_sec=300.0,
                    level=logging.DEBUG,
                    extra={
                        "event": "indicator_engine_names_defaulted",
                        "symbol": symbol,
                    },
                )
            else:
                try:
                    name_set = {str(name) for name in names if name is not None}
                except TypeError as e:
                    log_throttled(
                        LOGGER,
                        f"indicator_names_invalid_{symbol}",
                        "Condition met: indicator_names_invalid",
                        interval_sec=60.0,
                        extra={
                            "event": "indicator_engine_names_invalid",
                            "symbol": symbol,
                        },
                    )
                    LOGGER.error(
                        "Failure in IndicatorEngine.get_indicators: %s",
                        e,
                        extra={
                            "event": "indicator_engine_names_error",
                            "symbol": symbol,
                        },
                        exc_info=e,
                    )
                    name_set = set()
            all_names = name_set | _always_include
            requested: dict[str, float | tuple[float, float, float] | None] = {}
            for name in all_names:
                key = str(name)
                if key in indicators:
                    requested[key] = indicators[key]
            context = self._runtime_context.get(symbol, {})
            for key, value in context.items():
                requested.setdefault(key, value)
            if "tick_direction" in all_names or "tick_direction" in requested:
                runtime_tick = str(context.get("tick_direction") or "").upper()
                if runtime_tick:
                    requested["tick_direction"] = runtime_tick
                    requested["tick_direction_source"] = "runtime_context"
                elif history is not None:
                    closes = history.get_closes(2)
                    if len(closes) >= 2:
                        if closes[-1] > closes[-2]:
                            requested["tick_direction"] = "UP"
                        elif closes[-1] < closes[-2]:
                            requested["tick_direction"] = "DOWN"
                        else:
                            requested["tick_direction"] = "FLAT"
                        requested["tick_direction_source"] = "history_close_delta"
            return requested
        except Exception as e:
            LOGGER.error(
                "Failure in IndicatorEngine.get_indicators: %s",
                e,
                extra={
                    "event": "indicator_engine_get_indicators_error",
                    "symbol": symbol,
                },
                exc_info=e,
            )
            return {}

    def get_rsi(self, symbol: str, period: int = 14) -> float | None:
        """Calculate RSI. Returns None if insufficient data."""
        with self._lock:
            history = self._histories.get(symbol)
            if history is None or len(history) < period + 1:
                return None
            last_timestamp = history.last_timestamp
            if last_timestamp is None:
                return None
            cache_key = f"rsi_{period}"
            cached = self._get_cached(symbol, cache_key, last_timestamp)
            if cached is not None:
                return cached  # type: ignore[return-value]
            prices = history.get_closes(period + 1)
            value = self._calculate_rsi(prices, period)
            self._set_cache(symbol, cache_key, value, last_timestamp)
            return value

    def get_ema(self, symbol: str, period: int = 20) -> float | None:
        """Calculate EMA. Returns None if insufficient data."""
        with self._lock:
            history = self._histories.get(symbol)
            if history is None or len(history) < period:
                return None
            last_timestamp = history.last_timestamp
            if last_timestamp is None:
                return None
            cache_key = f"ema_{period}"
            cached = self._get_cached(symbol, cache_key, last_timestamp)
            if cached is not None:
                return cached  # type: ignore[return-value]
            prices = history.get_closes()
            value = self._calculate_ema(prices, period)
            self._set_cache(symbol, cache_key, value, last_timestamp)
            return value

    def get_sma(self, symbol: str, period: int = 20) -> float | None:
        """Calculate SMA. Returns None if insufficient data."""
        with self._lock:
            history = self._histories.get(symbol)
        if history is None or len(history) < period:
            return None
        last_timestamp = history.last_timestamp
        if last_timestamp is None:
            return None
        cache_key = f"sma_{period}"
        cached = self._get_cached(symbol, cache_key, last_timestamp)
        if cached is not None:
            return cached  # type: ignore[return-value]
        prices = history.get_closes(period)
        value = self._calculate_sma(prices, period)
        self._set_cache(symbol, cache_key, value, last_timestamp)
        return value

    def get_macd(
        self,
        symbol: str,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[float, float, float] | None:
        """Calculate MACD. Returns (macd, signal, histogram) or None."""
        with self._lock:
            history = self._histories.get(symbol)
        if history is None or len(history) < slow + signal:
            return None
        last_timestamp = history.last_timestamp
        if last_timestamp is None:
            return None
        cache_key = f"macd_{fast}_{slow}_{signal}"
        cached = self._get_cached(symbol, cache_key, last_timestamp)
        if cached is not None:
            return cached  # type: ignore[return-value]
        prices = history.get_closes()
        macd = self._calculate_macd(prices, fast, slow, signal)
        if macd is None:
            return None
        self._set_cache(symbol, cache_key, macd, last_timestamp)
        return macd

    def get_bollinger_bands(
        self,
        symbol: str,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[float, float, float] | None:
        """Calculate Bollinger Bands. Returns (upper, middle, lower) or None."""
        with self._lock:
            history = self._histories.get(symbol)
        if history is None or len(history) < period:
            return None
        last_timestamp = history.last_timestamp
        if last_timestamp is None:
            return None
        cache_key = f"bollinger_{period}_{std_dev}"
        cached = self._get_cached(symbol, cache_key, last_timestamp)
        if cached is not None:
            return cached  # type: ignore[return-value]
        prices = history.get_closes(period)
        bands = self._calculate_bollinger_bands(prices, period, std_dev)
        self._set_cache(symbol, cache_key, bands, last_timestamp)
        return bands

    def get_atr(self, symbol: str, period: int = 14) -> float | None:
        """Calculate ATR. Returns None if insufficient data."""
        with self._lock:
            history = self._histories.get(symbol)
            if history is None or len(history) < period + 1:
                return None
            last_timestamp = history.last_timestamp
            if last_timestamp is None:
                return None
            cache_key = f"atr_{period}"
            cached = self._get_cached(symbol, cache_key, last_timestamp)
            if cached is not None:
                return cached  # type: ignore[return-value]
            highs = history.get_highs(period + 1)
            lows = history.get_lows(period + 1)
            closes = history.get_closes(period + 1)
            value = self._calculate_atr(highs, lows, closes, period)
            self._set_cache(symbol, cache_key, value, last_timestamp)
            return value

    def get_vwap(self, symbol: str, period: int = 20) -> float | None:
        """Calculate VWAP. Returns None if insufficient data. NEVER raises."""
        try:
            with self._lock:
                history = self._histories.get(symbol)
            if history is None or len(history) == 0:
                return None
            effective_period = min(period, len(history))
            if effective_period < 3:
                return None

            last_timestamp = history.last_timestamp
            if last_timestamp is None:
                return None
            cache_key = f"vwap_{effective_period}"
            cached = self._get_cached(symbol, cache_key, last_timestamp)
            if cached is not None:
                return cached  # type: ignore[return-value]
            prices = history.get_closes(effective_period)
            volumes = history.get_volumes(effective_period)
            provisional_flags = history.get_provisional_flags(effective_period)
            if provisional_flags and len(provisional_flags) == len(volumes):
                filtered_pairs = [
                    (float(price), int(volume))
                    for price, volume, provisional in zip(
                        prices, volumes, provisional_flags, strict=False
                    )
                    if not bool(provisional)
                ]
                if filtered_pairs:
                    prices = [pair[0] for pair in filtered_pairs]
                    volumes = [pair[1] for pair in filtered_pairs]

            # Pre-check: skip calculation if all volumes are zero
            if not volumes or all(v == 0 for v in volumes):
                return self._last_valid_vwap.get(symbol)

            value = self._calculate_vwap(prices, volumes)
            if value is None:
                # Return last known good VWAP to avoid signal dropout
                return self._last_valid_vwap.get(symbol)
            self._last_valid_vwap[symbol] = float(value)
            self._set_cache(symbol, cache_key, value, last_timestamp)
            return value
        except Exception:
            return self._last_valid_vwap.get(symbol)

    def get_atr_trend(self, symbol: str, period: int = 14) -> float | None:
        """Return ATR trend ratio for *symbol*.

        Args:
            symbol: Canonical instrument identifier.
            period: ATR lookback used for ratio calculation.

        Returns:
            float | None: Ratio of current ATR to prior-period ATR when available.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered IndicatorEngine.get_atr_trend",
            extra={"event": "indicator_atr_trend", "symbol": symbol},
        )
        with self._lock:
            history = self._histories.get(symbol)
        if history is None:
            LOGGER.info(
                "Condition met: atr_trend_missing_history",
                extra={"event": "indicator_atr_trend_no_history", "symbol": symbol},
            )
            return None
        if len(history) < (period * 2) + 1:
            LOGGER.debug(
                "Condition met: insufficient bars for atr_trend",
                extra={
                    "event": "indicator_atr_trend_insufficient",
                    "symbol": symbol,
                    "bars": len(history),
                    "required": (period * 2) + 1,
                },
            )
            return None
        last_timestamp = history.last_timestamp
        if last_timestamp is None:
            LOGGER.debug(
                "Condition met: atr_trend_no_timestamp",
                extra={"event": "indicator_atr_trend_no_ts", "symbol": symbol},
            )
            return None
        cache_key = f"atr_trend_{period}"
        cached = self._get_cached(symbol, cache_key, last_timestamp)
        if cached is not None:
            return float(cached)
        try:
            current_atr = self.get_atr(symbol, period=period)
            if current_atr is None:
                LOGGER.debug(
                    "Condition met: atr_trend_current_missing",
                    extra={
                        "event": "indicator_atr_trend_current_missing",
                        "symbol": symbol,
                    },
                )
                return None
            highs = history.get_highs((period * 2) + 1)
            lows = history.get_lows((period * 2) + 1)
            closes = history.get_closes((period * 2) + 1)
            past_highs = highs[: period + 1]
            past_lows = lows[: period + 1]
            past_closes = closes[: period + 1]
            past_atr = self._calculate_atr(past_highs, past_lows, past_closes, period)
            if past_atr <= 0:
                LOGGER.debug(
                    "Condition met: atr_trend_past_nonpositive",
                    extra={
                        "event": "indicator_atr_trend_past_nonpositive",
                        "symbol": symbol,
                        "past_atr": past_atr,
                    },
                )
                return None
            trend = float(current_atr / past_atr)
            self._set_cache(symbol, cache_key, trend, last_timestamp)
            LOGGER.debug(
                "Condition met: atr_trend_computed",
                extra={
                    "event": "indicator_atr_trend_computed",
                    "symbol": symbol,
                    "trend": trend,
                },
            )
            return trend
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in IndicatorEngine.get_atr_trend: %s",
                exc,
                extra={"event": "indicator_atr_trend_error", "symbol": symbol},
                exc_info=exc,
            )
            return None

    def get_volatility_index(self, symbol: str, window: int = 20) -> float | None:
        """Return annualised realised volatility for *symbol*.

        Args:
            symbol: Canonical instrument identifier.
            window: Number of bars used when calculating realised volatility.

        Returns:
            float | None: Annualised volatility percentage when sufficient data.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered IndicatorEngine.get_volatility_index",
            extra={"event": "indicator_vol_index", "symbol": symbol},
        )
        with self._lock:
            history = self._histories.get(symbol)
        if history is None:
            LOGGER.info(
                "Condition met: volatility_index_missing_history",
                extra={"event": "indicator_vol_index_no_history", "symbol": symbol},
            )
            return None
        if len(history) <= window:
            LOGGER.debug(
                "Condition met: insufficient bars for volatility_index",
                extra={
                    "event": "indicator_vol_index_insufficient",
                    "symbol": symbol,
                    "bars": len(history),
                    "required": window + 1,
                },
            )
            return None
        last_timestamp = history.last_timestamp
        if last_timestamp is None:
            LOGGER.debug(
                "Condition met: volatility_index_no_timestamp",
                extra={"event": "indicator_vol_index_no_ts", "symbol": symbol},
            )
            return None
        cache_key = f"volatility_index_{window}"
        cached = self._get_cached(symbol, cache_key, last_timestamp)
        if cached is not None:
            return float(cached)
        try:
            closes = history.get_closes(window + 1)
            if len(closes) <= window:
                LOGGER.debug(
                    "Condition met: volatility_index_closes_short",
                    extra={
                        "event": "indicator_vol_index_closes_short",
                        "symbol": symbol,
                        "closes": len(closes),
                    },
                )
                return None
            returns = [
                (float(closes[idx]) / float(closes[idx - 1])) - 1.0
                for idx in range(1, len(closes))
            ]
            if len(returns) < window:
                LOGGER.debug(
                    "Condition met: volatility_index_returns_short",
                    extra={
                        "event": "indicator_vol_index_returns_short",
                        "symbol": symbol,
                        "returns": len(returns),
                    },
                )
                return None
            recent = returns[-window:]
            mean_return = sum(recent) / window
            variance = sum((value - mean_return) ** 2 for value in recent) / window
            volatility = float((variance**0.5) * (252.0**0.5) * 100.0)
            self._set_cache(symbol, cache_key, volatility, last_timestamp)
            LOGGER.debug(
                "Condition met: volatility_index_computed",
                extra={
                    "event": "indicator_vol_index_computed",
                    "symbol": symbol,
                    "volatility": volatility,
                },
            )
            return volatility
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in IndicatorEngine.get_volatility_index: %s",
                exc,
                extra={"event": "indicator_vol_index_error", "symbol": symbol},
                exc_info=exc,
            )
            return None

    def is_ready(self, symbol: str, min_bars: int = 50) -> bool:
        """Check if enough data exists for indicator calculation."""
        return self.has_min_bars(symbol, min_bars)

    def has_min_bars(self, symbol: str, min_bars: int) -> bool:
        """Return whether the internal history satisfies *min_bars*."""
        history = self._histories.get(symbol)
        if history is None or len(history) < min_bars:
            return False
        return self._validate_history_integrity(symbol, history, min_bars=min_bars)

    def _validate_history_integrity(
        self,
        symbol: str,
        history: PriceHistory,
        *,
        min_bars: int,
    ) -> bool:
        """Validate indicator inputs before strategy evaluation."""
        timestamps = history.get_timestamps(min_bars)
        if len(timestamps) < min_bars:
            LOGGER.error(
                "Condition met: indicator_integrity_short_history",
                extra={
                    "event": "indicator_integrity_short_history",
                    "symbol": symbol,
                    "have": len(timestamps),
                    "need": min_bars,
                },
            )
            return False
        if any(ts is None for ts in timestamps):
            LOGGER.error(
                "Condition met: indicator_integrity_missing_timestamp",
                extra={
                    "event": "indicator_integrity_missing_timestamp",
                    "symbol": symbol,
                },
            )
            return False
        completeness = history.get_completeness(min_bars)
        provisional_flags = history.get_provisional_flags(min_bars)
        if not all(completeness):
            LOGGER.debug(
                "Condition met: indicator_integrity_incomplete_candle",
                extra={
                    "event": "indicator_integrity_incomplete_candle",
                    "symbol": symbol,
                },
            )
            return False
        if any(provisional_flags):
            log_throttled(
                LOGGER,
                f"indicator_integrity_provisional_tail:{symbol}",
                "indicator_integrity_provisional_tail",
                interval_sec=float(os.getenv("INDICATOR_INTEGRITY_LOG_THROTTLE_SECONDS", "120")),
                level=logging.DEBUG,
                extra={
                    "event": "indicator_integrity_provisional_tail",
                    "symbol": symbol,
                },
            )
            return False
        for prev, curr in zip(timestamps, timestamps[1:]):
            if curr <= prev:
                LOGGER.error(
                    "Condition met: indicator_integrity_non_monotonic",
                    extra={
                        "event": "indicator_integrity_non_monotonic",
                        "symbol": symbol,
                        "prev": prev.isoformat(),
                        "curr": curr.isoformat(),
                    },
                )
                return False
            gap_seconds = (curr - prev).total_seconds()
            # FIX S12-1: Skip gaps that span non-market hours (overnight / weekend).
            # Zerodha historical data has ~63,900-second gaps between 15:30 and 09:15
            # next day (17h45m) plus ~176,400s over weekends.  The previous 600s
            # threshold rejected EVERY multi-day history, making has_min_bars()
            # return False for all 20 symbols → valid_symbols=[] → EXECUTION_ENABLED
            # never set → zero trades on every startup.
            # Fix: only flag gaps that fall entirely within a single market session
            # (i.e. both endpoints are within 09:15-15:30 IST on the same day AND
            # neither endpoint is outside market hours).  A gap is an intra-session
            # anomaly only when prev is already past open AND curr is still before
            # close on the same session date.  Overnight/weekend gaps are expected.
            prev_ist = prev.astimezone(_INDIA_TZ)
            curr_ist = curr.astimezone(_INDIA_TZ)
            prev_time = prev_ist.time().replace(tzinfo=None)
            curr_time = curr_ist.time().replace(tzinfo=None)
            prev_in_session = _MARKET_OPEN <= prev_time <= _MARKET_CLOSE
            curr_in_session = _MARKET_OPEN <= curr_time <= _MARKET_CLOSE
            same_day = prev_ist.date() == curr_ist.date()
            intra_session_gap = prev_in_session and curr_in_session and same_day
            if intra_session_gap and gap_seconds > 600.0:
                # Genuine intra-session data hole — could corrupt indicators
                log_throttled(
                    LOGGER,
                    f"indicator_gap_{symbol}",
                    "indicator_gap_detected: %s gap=%.0fs",
                    interval_sec=300.0,
                    level=logging.WARNING,
                    extra={
                        "event": "indicator_gap_detected",
                        "symbol": symbol,
                        "gap_seconds": gap_seconds,
                        "prev": prev.isoformat(),
                        "curr": curr.isoformat(),
                    },
                )
                # Log but do NOT return False — intra-session gaps in historical
                # data are common for illiquid options and must not block strategy
                # evaluation.  The strategy itself gates on recent candle staleness.
        return True

    def ensure_min_bars(
        self,
        symbol: str,
        min_bars: int,
        hydrate: Callable[[str, int], list[dict[str, Any]]] | None = None,
    ) -> bool:
        """Hydrate missing bars for *symbol* via callback before evaluation."""
        if self.has_min_bars(symbol, min_bars):
            return True
        if hydrate is None:
            return False
        for bar in hydrate(symbol, min_bars):
            try:
                self.update_price(
                    symbol,
                    {
                        "open": float(bar.get("open", 0.0) or 0.0),
                        "high": float(bar.get("high", 0.0) or 0.0),
                        "low": float(bar.get("low", 0.0) or 0.0),
                        "close": float(bar.get("close", 0.0) or 0.0),
                    },
                    volume=int(bar.get("volume", 0) or 0),
                    timestamp=bar.get("timestamp"),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure in IndicatorEngine.ensure_min_bars: %s",
                    exc,
                    extra={
                        "event": "indicator_engine_hydrate_bar_failed",
                        "symbol": symbol,
                    },
                    exc_info=exc,
                )
        return self.has_min_bars(symbol, min_bars)

    def get_opening_range(
        self, symbol: str, minutes: int = 30
    ) -> tuple[float, float, bool] | None:
        """Return the session opening range high/low for *symbol*.

        Args:
            symbol: Canonical instrument identifier.
            minutes: Opening range window in minutes.

        Returns:
            Tuple containing ``(high, low, ready)`` when sufficient data exists,
            otherwise ``None``.

        Raises:
            None.
        """

        history = self._histories.get(symbol)
        if history is None or len(history) == 0:
            return None
        return self._compute_opening_range(history, minutes)

    def is_nr7(
        self, symbol: str, lookback: int = 7
    ) -> tuple[bool, float | None, float | None]:
        """Return whether the latest bar is the narrowest of the lookback.

        Args:
            symbol: Canonical instrument identifier.
            lookback: Number of bars to evaluate for NR7 detection.

        Returns:
            Tuple ``(flag, current_range, min_range)`` containing NR7 status.

        Raises:
            None.
        """

        history = self._histories.get(symbol)
        if history is None or len(history) < lookback:
            return False, None, None
        highs = history.get_highs(lookback)
        lows = history.get_lows(lookback)
        if len(highs) < lookback or len(lows) < lookback:
            return False, None, None
        ranges = [
            float(high_value) - float(low_value)
            for high_value, low_value in zip(highs, lows)
        ]
        if not ranges:
            return False, None, None
        current_range = ranges[-1]
        min_range = min(ranges)
        return current_range <= min_range, current_range, min_range

    def _augment_session_metrics(self, symbol: str, indicators: dict[str, Any]) -> None:
        """Populate volume and session context derived metrics.

        Args:
            symbol: Canonical instrument identifier.
            indicators: Indicator dictionary to augment in-place.

        Returns:
            None.

        Raises:
            None.
        """

        history = self._histories.get(symbol)
        indicators.setdefault("volume", None)
        indicators.setdefault("avg_volume", None)
        indicators.setdefault("volume_spike_ratio", None)
        indicators.setdefault("minutes_since_open", None)
        indicators.setdefault("minutes_until_close", None)
        indicators.setdefault("market_time", None)
        indicators.setdefault("orb_high", None)
        indicators.setdefault("orb_low", None)
        indicators.setdefault("orb_ready", False)
        indicators.setdefault("nr7", False)
        indicators.setdefault("nr7_range", None)
        indicators.setdefault("nr7_min_range", None)
        indicators.setdefault("bar_range", None)
        if history is None or len(history) == 0:
            return

        volumes = history.get_volumes()
        provisional_flags = history.get_provisional_flags()
        if volumes:
            last_volume = (
                float(volumes[-1]) if not provisional_flags or not provisional_flags[-1] else 0.0
            )
            indicators["volume"] = last_volume
            upper_symbol = str(symbol).upper()
            is_option = upper_symbol.endswith("CE") or upper_symbol.endswith("PE")
            zipped = list(zip(volumes, provisional_flags, strict=False)) if provisional_flags else [
                (volume, False) for volume in volumes
            ]
            recent = zipped[-20:]
            if is_option:
                tail = [float(volume) for volume, provisional in recent if not provisional and float(volume) > 0.0]
                if not tail:
                    tail = [
                        float(volume)
                        for volume, provisional in zipped
                        if not provisional and float(volume) > 0.0
                    ][-20:]
            else:
                tail = [float(volume) for volume, provisional in recent if not provisional]
            if tail:
                avg_volume = sum(float(v) for v in tail) / len(tail)
                indicators["avg_volume"] = avg_volume
                indicators["average_volume"] = avg_volume
                if avg_volume > 0:
                    indicators["volume_spike_ratio"] = last_volume / avg_volume
                else:
                    indicators["volume_spike_ratio"] = 0.0
            else:
                indicators["volume_spike_ratio"] = 0.0

        timestamps = history.get_timestamps()
        highs = history.get_highs()
        lows = history.get_lows()
        if not timestamps:
            return
        latest_ts = timestamps[-1]
        _session_open, _session_close, minutes_since_open, minutes_until_close = (
            self._session_context(latest_ts)
        )
        indicators["minutes_since_open"] = max(minutes_since_open, 0.0)
        indicators["minutes_until_close"] = max(minutes_until_close, 0.0)
        indicators["market_time"] = latest_ts.astimezone(_INDIA_TZ).time()

        opening_range = self._compute_opening_range(history, 30)
        if opening_range is not None:
            orb_high, orb_low, ready = opening_range
            indicators["orb_high"] = orb_high
            indicators["orb_low"] = orb_low
            indicators["orb_ready"] = ready

        if len(highs) >= 1 and len(lows) >= 1:
            indicators["bar_range"] = float(highs[-1]) - float(lows[-1])

        if len(highs) >= 7 and len(lows) >= 7:
            recent_highs = highs[-7:]
            recent_lows = lows[-7:]
            ranges = [
                float(high_value) - float(low_value)
                for high_value, low_value in zip(recent_highs, recent_lows)
            ]
            if ranges:
                current_range = ranges[-1]
                min_range = min(ranges)
                indicators["nr7"] = current_range <= min_range
                indicators["nr7_range"] = current_range
                indicators["nr7_min_range"] = min_range

    def _compute_opening_range(
        self, history: PriceHistory, window_minutes: int
    ) -> tuple[float, float, bool] | None:
        """Compute opening range values from *history*.

        Args:
            history: Price history container for the symbol.
            window_minutes: Opening range window in minutes.

        Returns:
            Tuple ``(high, low, ready)`` when enough data is present else ``None``.

        Raises:
            None.
        """
        timestamps = history.get_timestamps()
        highs = history.get_highs()
        lows = history.get_lows()
        if not timestamps or not highs or not lows:
            return None
        latest_ts = timestamps[-1]
        session_open, _ = self._session_bounds(latest_ts)
        cutoff = session_open + timedelta(minutes=window_minutes)
        high_values: list[float] = []
        low_values: list[float] = []
        for ts, high, low in zip(timestamps, highs, lows):
            if ts < session_open:
                continue
            if ts > cutoff:
                break
            high_values.append(float(high))
            low_values.append(float(low))
        if not high_values or not low_values:
            return None
        ready = latest_ts >= cutoff
        return max(high_values), min(low_values), ready

    def _session_bounds(self, timestamp: datetime) -> tuple[datetime, datetime]:
        """Return market session open/close bounds for *timestamp*.

        Args:
            timestamp: Timestamp to align with the trading session.

        Returns:
            Tuple containing opening and closing datetimes in UTC.

        Raises:
            None.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        local_ts = timestamp.astimezone(_INDIA_TZ)
        open_dt = local_ts.replace(
            hour=_MARKET_OPEN.hour,
            minute=_MARKET_OPEN.minute,
            second=0,
            microsecond=0,
        )
        close_dt = local_ts.replace(
            hour=_MARKET_CLOSE.hour,
            minute=_MARKET_CLOSE.minute,
            second=0,
            microsecond=0,
        )
        if local_ts.time() < _MARKET_OPEN:
            open_dt -= timedelta(days=1)
            close_dt -= timedelta(days=1)
        return open_dt.astimezone(timezone.utc), close_dt.astimezone(timezone.utc)

    def _session_context(
        self, timestamp: datetime
    ) -> tuple[datetime, datetime, float, float]:
        """Return session metadata derived from *timestamp*.

        Args:
            timestamp: Timestamp to evaluate.

        Returns:
            Tuple ``(session_open, session_close, minutes_since_open,``
            ``minutes_until_close)``.

        Raises:
            None.
        """
        session_open, session_close = self._session_bounds(timestamp)
        minutes_since_open = (timestamp - session_open).total_seconds() / 60.0
        minutes_until_close = (session_close - timestamp).total_seconds() / 60.0
        return session_open, session_close, minutes_since_open, minutes_until_close

    def _calculate_rsi(self, prices: list[float], period: int) -> float:
        """Internal RSI calculation.

        The Relative Strength Index is computed as::

            RSI = 100 - 100 / (1 + RS)

        where ``RS`` is the ratio of the average gain to the average loss over
        the specified ``period``.
        """
        closes = np.asarray(prices, dtype=float)
        deltas = np.diff(closes)
        gains = np.clip(deltas, a_min=0.0, a_max=None)
        losses = np.clip(-deltas, a_min=0.0, a_max=None)
        avg_gain = gains[-period:].mean()
        avg_loss = losses[-period:].mean()
        if np.isclose(avg_loss, 0.0):
            if np.isclose(avg_gain, 0.0):
                return 50.0
            return 100.0
        if np.isclose(avg_gain, 0.0):
            return 0.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)

    def _calculate_ema(self, prices: list[float], period: int) -> float:
        """Internal EMA calculation using Wilder's smoothing constant.

        The Exponential Moving Average is updated recursively as::

            EMA_t = EMA_{t-1} + \alpha (P_t - EMA_{t-1})

        where ``P_t`` is the price at time ``t`` and ``\alpha = 2 / (period + 1)``.
        """
        closes = np.asarray(prices, dtype=float)
        if closes.size < period:
            raise ValueError("Insufficient data for EMA calculation")
        alpha = 2.0 / (period + 1.0)
        ema = closes[:period].mean()
        for value in closes[period:]:
            ema += alpha * (value - ema)
        return float(ema)

    def _calculate_sma(self, prices: list[float], period: int) -> float:
        """Internal SMA calculation.

        The Simple Moving Average is the arithmetic mean of the last ``period``
        closing prices::

            SMA = (P_{t-period+1} + ... + P_t) / period
        """
        closes = np.asarray(prices[-period:], dtype=float)
        return float(closes.mean())

    def _calculate_macd(
        self,
        prices: list[float],
        fast: int,
        slow: int,
        signal: int,
    ) -> tuple[float, float, float] | None:
        """Internal MACD calculation.

        The Moving Average Convergence Divergence line is defined as the
        difference between the fast and slow EMAs::

            MACD_t = EMA^{fast}_t - EMA^{slow}_t

        The signal line is an EMA of the MACD series and the histogram is the
        deviation ``MACD_t - Signal_t``.
        """
        closes = np.asarray(prices, dtype=float)
        if closes.size < slow:
            return None
        fast_series = self._ema_series(closes, fast)
        slow_series = self._ema_series(closes, slow)
        macd_series = fast_series - slow_series
        valid_macd = macd_series[~np.isnan(macd_series)]
        if valid_macd.size < signal:
            return None
        signal_series = self._ema_series(valid_macd, signal)
        macd_value = float(valid_macd[-1])
        signal_value = float(signal_series[~np.isnan(signal_series)][-1])
        histogram = macd_value - signal_value
        return macd_value, signal_value, histogram

    def _calculate_bollinger_bands(
        self,
        prices: list[float],
        period: int,
        std_dev: float,
    ) -> tuple[float, float, float]:
        """Internal Bollinger Bands calculation.

        Bollinger Bands consist of a middle SMA and symmetric bands at ``k``
        standard deviations::

            Middle_t = SMA_t
            Upper_t = Middle_t + k * sigma_t
            Lower_t = Middle_t - k * sigma_t

        where ``sigma_t`` is the population standard deviation of the last
        ``period`` closes.
        """
        closes = np.asarray(prices[-period:], dtype=float)
        middle = closes.mean()
        deviation = closes.std(ddof=0)
        upper = middle + std_dev * deviation
        lower = middle - std_dev * deviation
        return float(upper), float(middle), float(lower)

    def _calculate_atr(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        period: int,
    ) -> float:
        """Internal ATR calculation.

        The Average True Range is the mean of the True Range over ``period``
        bars. The True Range for bar ``t`` is::

            TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)

        and the ATR is ``ATR = mean(TR_{t-period+1..t})``.
        """
        high_arr = np.asarray(highs, dtype=float)
        low_arr = np.asarray(lows, dtype=float)
        close_arr = np.asarray(closes, dtype=float)
        true_ranges = []
        for idx in range(1, len(high_arr)):
            high = high_arr[idx]
            low = low_arr[idx]
            prev_close = close_arr[idx - 1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
        if len(true_ranges) < period:
            raise ValueError("Insufficient data for ATR calculation")
        atr_values = np.asarray(true_ranges[-period:], dtype=float)
        return float(atr_values.mean())

    def _calculate_vwap(self, prices: list[float], volumes: list[int]) -> float | None:
        """Internal VWAP calculation.

        Returns None when data is invalid and never raises.
        """
        try:
            if not prices or not volumes or len(prices) != len(volumes):
                return None
            cum_pv = 0.0
            cum_vol = 0.0
            for price, volume in zip(prices, volumes, strict=False):
                v = float(volume)
                if v <= 0:
                    v = 1.0
                cum_pv += float(price) * v
                cum_vol += v
            if cum_vol <= 0:
                return None
            result = cum_pv / cum_vol
            if not (result > 0):  # catches NaN and negative
                return None
            return result
        except Exception:
            return None  # NEVER crash the indicator pipeline

    def _ema_series(self, values: Iterable[float], period: int) -> np.ndarray:
        """Return the EMA series for the supplied values.

        The helper mirrors :meth:`_calculate_ema` but retains all intermediate
        values which is necessary for indicators like MACD that depend on EMA of
        EMA-derived series.
        """
        series = np.asarray(list(values), dtype=float)
        result = np.empty_like(series)
        result[:] = np.nan
        if series.size < period:
            return result
        alpha = 2.0 / (period + 1.0)
        ema = series[:period].mean()
        result[period - 1] = ema
        for idx in range(period, series.size):
            ema += alpha * (series[idx] - ema)
            result[idx] = ema
        return result

    def _get_cached(self, symbol: str, key: str, timestamp: datetime) -> Any | None:
        with self._lock:
            symbol_cache = self._cache.get(symbol)
            if symbol_cache is None:
                return None
            cached = symbol_cache.get(key)
            if cached is None:
                return None
            value, cached_timestamp = cached
            if cached_timestamp == timestamp:
                return value
            return None

    def _set_cache(
        self, symbol: str, key: str, value: Any, timestamp: datetime
    ) -> None:
        with self._lock:
            symbol_cache = self._cache.setdefault(symbol, {})
            symbol_cache[key] = (value, timestamp)

    def compute_atr(self, symbol: str, period: int = 14) -> object | None:
        """
        Compute ATR for volatility-adaptive trailing.

        ✅ PRODUCTION FIX:
        - Reduced minimum bars from 25 to 10
        - Added fallback ATR calculation (1.5% of price)
        """
        # Lazy import to avoid circular dependency

        hist = self._histories.get(symbol)

        # ✅ FIX: Reduced minimum bars requirement
        min_bars = max(period + 2, 10)  # Reduced from 25 to 10

        if not hist:
            return None

        # ✅ FIX: Fallback for insufficient bars
        if len(hist) < min_bars:
            # Try to estimate ATR from available data
            try:
                closes = hist.get_closes()
                if closes and len(closes) >= 2:
                    # Use recent price volatility as proxy
                    recent_closes = closes[-min(len(closes), 5) :]
                    if len(recent_closes) >= 2:
                        price_range = max(recent_closes) - min(recent_closes)
                        avg_price = sum(recent_closes) / len(recent_closes)
                        if avg_price > 0:
                            # Estimate ATR as price range or 1.5% of price
                            estimated_atr = max(price_range, avg_price * 0.015)
                            self._logger.debug(
                                f"ATR estimated for {symbol}: {estimated_atr:.2f} "
                                f"(bars={len(hist)}, min_needed={min_bars})"
                            )
                            return estimated_atr
            except Exception as e:
                __import__("logging").getLogger(__name__).exception(
                    "[CRITICAL] unhandled exception", exc_info=True
                )
                raise
            return None

        try:
            lookback = period + 20
            closes = np.array(hist.get_closes(lookback), dtype=float)
            highs = np.array(hist.get_highs(lookback), dtype=float)
            lows = np.array(hist.get_lows(lookback), dtype=float)

            if len(closes) < period:
                # ✅ FIX: Fallback to percentage-based ATR
                if len(closes) > 0:
                    avg_price = float(np.mean(closes[-5:]))
                    return avg_price * 0.015 if avg_price > 0 else None
                return None

            # Calculate True Range (TR)
            tr_values = []
            for i in range(1, len(closes)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
                tr_values.append(max(tr, 0.01))

            if not tr_values:
                return None

            tr_array = np.array(tr_values)

            # Current ATR
            current_atr = float(np.mean(tr_array[-period:]))

            # ✅ FIX: Ensure ATR is reasonable (at least 0.5% of price)
            if current_atr <= 0 and len(closes) > 0:
                current_atr = float(closes[-1]) * 0.015

            return current_atr

        except Exception as e:
            self._logger.error(f"ATR Compute Failed for {symbol}: {e}")
            # ✅ FIX: Return fallback instead of None
            try:
                hist = self._histories.get(symbol)
                if hist:
                    closes = hist.get_closes(5)
                    if closes:
                        return float(closes[-1]) * 0.015
            except Exception as e:
                __import__("logging").getLogger(__name__).exception(
                    "[CRITICAL] unhandled exception", exc_info=True
                )
                raise
            return None

    def calculate_slope(
        self, symbol: str, indicator_name: str = "close", period: int = 5
    ) -> float:
        try:
            # ✅ FIX: Use self._histories (plural)
            history = self._histories.get(symbol)

            if not history or len(history) < period:
                return 0.0

            # Use getters based on indicator_name
            if indicator_name == "close":
                values = history.get_closes(period)
            elif indicator_name == "high":
                values = history.get_highs(period)
            elif indicator_name == "low":
                values = history.get_lows(period)
            else:
                return 0.0

            # 3. Normalize to Basis Points (Price agnostic)
            start = values[0]
            if start == 0:
                return 0.0

            # Convert to relative change (100 -> 100.1 = +10)
            y = [(v - start) / start * 10000 for v in values]
            x = list(range(len(y)))

            # 4. Linear Regression (Least Squares)
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_xx = sum(xi**2 for xi in x)

            slope_m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)

            # 5. Convert to Degrees
            import math

            return math.degrees(math.atan(slope_m))

        except Exception as e:
            self._logger.debug(f"Slope calc failed for {symbol}: {e}")
            return 0.0

    def get_latest(self, symbol: str) -> dict[str, float | None]:
        """Args: symbol. Returns: latest indicator snapshot. Raises: Exception."""
        try:
            # 1. Fetch Indicators required by DynamicTPController
            # Use getattr to be safe if methods are missing, defaulting to None
            get_rsi = getattr(self, "get_rsi", None)
            get_adx = getattr(self, "get_adx", None)

            rsi_val = get_rsi(symbol) if callable(get_rsi) else None
            adx_val = get_adx(symbol) if callable(get_adx) else None

            # 2. Fetch ATR (Already implemented)
            atr_snap = self.compute_atr(symbol)
            if atr_snap is None:
                atr_val = None
            else:
                atr_val = getattr(
                    atr_snap, "current_atr", getattr(atr_snap, "value", atr_snap)
                )
                try:
                    atr_val = float(atr_val)
                except (TypeError, ValueError):
                    atr_val = None

            with self._lock:
                history = self._histories.get(symbol)
                latest_close = None
                if history is not None and len(history) > 0:
                    closes = history.get_closes(1)
                    if closes:
                        latest_close = float(closes[-1])

            return {
                "rsi": rsi_val,
                "adx": adx_val,
                "atr": atr_val,
                "close": latest_close,
                "ltp": latest_close,
            }
        except Exception as e:
            # Prevent indicator calculation errors from crashing the Order Manager
            self._logger.error(f"get_latest failed for {symbol}: {e}")
            return {}

    def get_latest_price(self, symbol: str) -> float | None:
        """Args: symbol. Returns: latest close price or None. Raises: Exception."""
        try:
            data = self.get_latest(symbol)
            if not data:
                return None
            close_value = data.get("close")
            if close_value is None:
                return None
            return float(close_value)
        except Exception as e:
            self._logger.error("Failure in IndicatorEngine.get_latest_price: %s", e)
            return None


    def supports_feature(self, feature_name: str) -> bool:
        """Args: feature_name. Returns: support flag. Raises: none."""
        return supports_feature(feature_name)
