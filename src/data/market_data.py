"""Lightweight market-data helpers for the scalper strategy."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Callable, Mapping, Sequence, cast

log = logging.getLogger(__name__)


DepthFetcher = Callable[[str], Mapping[str, object]]


def _round_to_tick(x: float, tick: float = 0.05) -> float:
    """Snap ``x`` to the nearest valid price increment."""

    try:
        value = float(x)
        step = float(tick)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return round(float(x), 2)
    if step <= 0:
        return round(value, 2)
    return round(round(value / step) * step, 2)


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
            log.info("Resubscribed to %d tokens", len(self.tokens))
        except Exception:  # pragma: no cover - defensive logging
            log.warning("Resubscribe on reconnect failed", exc_info=True)

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
        log.debug("depth parsing failed", exc_info=True)
    ask_raw: Any = payload.get("ask", 0.0)
    if isinstance(ask_raw, (int, float, str)):
        try:
            ask = float(ask_raw)
        except (TypeError, ValueError):
            ask = 0.0
    else:
        ask = 0.0
    return ask if ask > 0 else 0.0


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
    tick = _extract_tick_size(quote)
    if ask <= 0:
        fallback = _extract_ltp(quote)
        if fallback <= 0:
            raise ValueError(f"best ask unavailable for {symbol}")
        log.warning("depth_unavailable %s; falling back to LTP", symbol)
        ask = fallback

    buffered = _round_to_tick(ask + tick, tick)
    if buffered <= 0:
        raise ValueError(f"invalid buffered ask for {symbol}")
    return buffered


__all__ = ["KiteDataFeed", "get_best_ask"]

