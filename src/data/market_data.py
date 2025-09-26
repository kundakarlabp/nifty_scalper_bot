"""Lightweight market-data helpers for the scalper strategy."""

from __future__ import annotations

import logging
import time
from threading import RLock
from typing import Any, Iterable, Callable, Mapping, Sequence, cast

from kiteconnect import KiteTicker

from src.config import settings

logger = logging.getLogger(__name__)


DepthFetcher = Callable[[str], Mapping[str, object]]


class KiteDataFeed:
    """Minimal helper for replaying Kite subscriptions on reconnects."""

    def __init__(self, ticker: object, *, tokens: Sequence[int] | None = None) -> None:
        self.ticker = ticker
        self.tokens: list[int] = list(tokens or [])

    def on_connect(self, ws: object, response: object) -> None:  # pragma: no cover - io
        """Resubscribe tokens after websocket reconnect."""

        if not self.tokens:
            return
        try:
            subscribe = getattr(ws, "subscribe", None)
            if callable(subscribe):
                subscribe(self.tokens)
            mode_fn = getattr(ws, "set_mode", None)
            if callable(mode_fn):
                mode_full = getattr(ws, "MODE_FULL", getattr(self.ticker, "MODE_FULL", "full"))
                mode_fn(mode_full, self.tokens)
            logger.info("Resubscribed to %d tokens", len(self.tokens))
        except Exception:  # pragma: no cover - defensive logging
            logger.warning("Resubscribe on reconnect failed", exc_info=True)

    def subscribe(self, tokens: Iterable[int]) -> None:
        self.tokens = list(tokens)


def _extract_best_ask(payload: Mapping[str, object] | None) -> float:
    """Return the best ask from ``payload`` if available."""

    if not isinstance(payload, Mapping):
        return 0.0
    try:
        depth = cast(Mapping[str, object], payload.get("depth") or {})
        sell_levels = depth.get("sell") if isinstance(depth, Mapping) else None
        if isinstance(sell_levels, Mapping):
            sell_levels = list(sell_levels.values())
        if isinstance(sell_levels, Iterable):
            levels = list(sell_levels)
        else:
            levels = []
        if levels:
            prices: list[float] = []
            for level in levels:
                if not isinstance(level, Mapping):
                    continue
                price_raw: Any = level.get("price", 0.0)
                try:
                    price = float(price_raw)
                except (TypeError, ValueError):
                    continue
                if price > 0:
                    prices.append(price)
            if prices:
                return min(prices)
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("depth parsing failed", exc_info=True)
    ask_raw: Any = payload.get("ask", 0.0)
    if isinstance(ask_raw, (int, float, str)):
        try:
            ask = float(ask_raw)
        except (TypeError, ValueError):
            ask = 0.0
    else:
        ask = 0.0
    return ask if ask > 0 else 0.0


def _diagnose_ask_fallback_reason(payload: Mapping[str, object] | None) -> str:
    """Return a human readable reason describing why the best ask was missing."""

    if not isinstance(payload, Mapping):
        return "quote_invalid"

    reasons: list[str] = []
    depth = payload.get("depth")
    if not isinstance(depth, Mapping):
        reasons.append("depth_missing")
    else:
        sell_levels = depth.get("sell")
        if sell_levels is None:
            reasons.append("sell_missing")
        else:
            if isinstance(sell_levels, Mapping):
                iterable = list(sell_levels.values())
            else:
                iterable = sell_levels
            if not isinstance(iterable, Iterable):
                reasons.append("sell_invalid")
            else:
                levels = [level for level in iterable if isinstance(level, Mapping)]
                if not levels:
                    reasons.append("sell_empty")
                else:
                    positive_prices = []
                    for level in levels:
                        price_raw = level.get("price")
                        try:
                            price = float(cast(Any, price_raw))
                        except (TypeError, ValueError):
                            continue
                        if price > 0:
                            positive_prices.append(price)
                    if not positive_prices:
                        reasons.append("sell_prices_non_positive")

    ask_raw = payload.get("ask")
    if ask_raw is None:
        reasons.append("ask_missing")
    else:
        try:
            ask = float(cast(Any, ask_raw))
        except (TypeError, ValueError):
            reasons.append("ask_invalid")
        else:
            if ask <= 0:
                reasons.append("ask_non_positive")

    if not reasons:
        return "unknown"
    return ",".join(reasons)


def _extract_tick_size(payload: Mapping[str, object] | None) -> float:
    """Return the tick size from the quote payload."""

    if not isinstance(payload, Mapping):
        return 0.05
    tick_raw: Any = payload.get("tick_size") or payload.get("tick") or 0.05
    try:
        tick = float(tick_raw)
    except (TypeError, ValueError):
        tick = 0.05
    return tick if tick > 0 else 0.05


def _extract_ltp(payload: Mapping[str, object] | None) -> float:
    """Return the last traded price from the quote payload."""

    if not isinstance(payload, Mapping):
        return 0.0
    for key in ("last_price", "ltp", "last_traded_price"):
        raw: Any = payload.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.0


def _default_depth_fetcher(symbol: str) -> Mapping[str, object]:  # pragma: no cover
    """Fetch quote depth using the live order executor if available."""

    from src.execution.order_executor import fetch_quote_with_depth

    runner = None
    try:
        from src.strategies.runner import StrategyRunner
    except Exception:  # pragma: no cover - import guard during tests
        runner = None
    else:
        runner = StrategyRunner.get_singleton()

    tsym = symbol.split(":", 1)[-1]
    kite = None
    if runner is not None:
        executor = getattr(runner, "order_executor", None)
        kite = getattr(executor, "kite", None)
    return fetch_quote_with_depth(kite, tsym)


def get_best_ask(symbol: str, *, depth_fetcher: DepthFetcher | None = None) -> float:
    """Return the best ask price for ``symbol``.

    Parameters
    ----------
    symbol:
        The broker identifier for the option contract, e.g. ``"NFO:NIFTY"``.
    depth_fetcher:
        Optional callable that receives ``symbol`` and returns a mapping with
        ``ask`` or level-2 ``depth`` data.  When omitted the helper falls back to
        :func:`fetch_quote_with_depth` via the active :class:`StrategyRunner`
        instance if present.

    Returns
    -------
    float
        The best ask price.

    Raises
    ------
    ValueError
        If the depth fetcher is unavailable or does not provide a usable ask
        price.
    """

    if not symbol:
        raise ValueError("symbol must be provided")

    fetcher = depth_fetcher or _default_depth_fetcher
    quote = fetcher(symbol)
    ask = _extract_best_ask(quote)
    if ask <= 0:
        fallback = _extract_ltp(quote)
        reason = _diagnose_ask_fallback_reason(quote)
        if fallback <= 0:
            raise ValueError(
                f"best ask unavailable for {symbol} (reason={reason})"
            )
        logger.warning(
            "Depth unavailable for %s, using last traded price %.2f (reason=%s)",
            symbol,
            fallback,
            reason,
        )
        ask = fallback

    tick = _extract_tick_size(quote)
    buffered = round(ask + tick, 2)
    if buffered <= 0:
        raise ValueError(f"invalid buffered ask for {symbol}")
    return buffered


class MarketData:
    """Manage Kite ticker subscriptions with reconnection safeguards."""

    def __init__(self, kite: Any, tokens: list[int]):
        self.kite = kite
        self.ticker = KiteTicker(kite.api_key, kite.access_token)
        self._tokens = list(tokens)
        self._tlock = RLock()
        self._last_connect_mono = 0.0
        self._reconnect_debounce_s = float(
            getattr(settings, "RECONNECT_DEBOUNCE_S", 10.0)
        )

        self.ticker.on_connect = self._on_connect
        self.ticker.on_ticks = self._on_ticks
        self.ticker.on_error = self._on_error
        self.ticker.on_close = self._on_close

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Return an ``NFO:``-prefixed symbol understood by Kite APIs."""

        return symbol if symbol.startswith("NFO:") else f"NFO:{symbol}"

    @staticmethod
    def _round_to_tick(price: float, tick: float = 0.05) -> float:
        """Round ``price`` to the nearest tradable ``tick``."""

        if tick <= 0:
            return round(price, 2)
        steps = round(price / tick)
        return round(steps * tick, 2)

    def get_option_price(self, symbol: str) -> float:
        """Return the last traded price for ``symbol`` rounded to tick size."""

        quoted = self._normalize_symbol(symbol)
        quote: Mapping[str, object] | None = None
        try:
            payload = self.kite.quote(quoted)
        except Exception:  # pragma: no cover - network/io
            logger.exception("ltp_fetch_failed %s", symbol)
            raise
        if isinstance(payload, Mapping):
            raw = payload.get(quoted)
            if isinstance(raw, Mapping):
                quote = raw
        price = _extract_ltp(quote)
        if price <= 0:
            raise ValueError(f"invalid option price for {symbol}")
        tick = _extract_tick_size(quote)
        return self._round_to_tick(price, tick)

    def get_marketable_ask(self, symbol: str) -> float:
        """Return an aggressive limit price to cross the spread for ``symbol``."""

        quoted = self._normalize_symbol(symbol)
        quote: Mapping[str, object] | None = None
        try:
            payload = self.kite.quote(quoted)
        except Exception:  # pragma: no cover - network/io
            logger.warning("depth_unavailable %s; falling back to LTP", symbol)
            tick = 0.05
            ltp = self.get_option_price(symbol)
            return self._round_to_tick(ltp + tick, tick)

        if isinstance(payload, Mapping):
            raw = payload.get(quoted)
            if isinstance(raw, Mapping):
                quote = raw

        tick = _extract_tick_size(quote)
        ask = _extract_best_ask(quote)
        if ask > 0:
            return self._round_to_tick(ask + tick, tick)

        logger.warning("depth_unavailable %s; falling back to LTP", symbol)
        ltp = self.get_option_price(symbol)
        return self._round_to_tick(ltp + tick, tick)

    def _snapshot_tokens(self) -> list[int]:
        with self._tlock:
            return list(self._tokens)

    def add_tokens(self, new_tokens: list[int]) -> None:
        with self._tlock:
            for token in new_tokens:
                if token not in self._tokens:
                    self._tokens.append(token)
        try:
            self.ticker.subscribe(new_tokens)
            self.ticker.set_mode(self.ticker.MODE_FULL, new_tokens)
        except Exception:  # pragma: no cover - network/io
            logger.exception("subscribe_failed")

    def connect(self) -> None:  # pragma: no cover - io
        """Establish the Kite ticker websocket connection."""

        try:
            self.ticker.connect()
        except Exception:
            logger.exception("ticker_connect_failed")
            raise

    def _on_connect(self, ws: Any, _response: Any) -> None:  # pragma: no cover - io
        now = time.monotonic()
        if now - self._last_connect_mono < self._reconnect_debounce_s:
            return
        self._last_connect_mono = now

        tokens = self._snapshot_tokens()
        if not tokens:
            return
        try:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            logger.info("ws_resubscribe tokens=%d", len(tokens))
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("ws_resubscribe_failed")

    def _on_ticks(self, _ws: Any, _ticks: Any) -> None:  # pragma: no cover - io
        pass

    def _on_error(self, ws: Any, code: int, reason: str) -> None:  # pragma: no cover
        logger.error("ticker_error code=%s reason=%s", code, reason)

    def _on_close(self, ws: Any, code: int, reason: str) -> None:  # pragma: no cover
        logger.warning("ticker_close code=%s reason=%s", code, reason)


__all__ = ["KiteDataFeed", "MarketData", "get_best_ask"]

