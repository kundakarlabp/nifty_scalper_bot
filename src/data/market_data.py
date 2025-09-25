"""Lightweight market-data helpers for the scalper strategy."""

from __future__ import annotations

import logging
from typing import Callable, Mapping

log = logging.getLogger(__name__)


DepthFetcher = Callable[[str], Mapping[str, object]]


def _extract_best_ask(payload: Mapping[str, object] | None) -> float:
    """Return the best ask from ``payload`` if available."""

    if not isinstance(payload, Mapping):
        return 0.0
    try:
        depth = payload.get("depth") or {}
        sell_levels = depth.get("sell") if isinstance(depth, Mapping) else None
        if isinstance(sell_levels, Mapping):
            sell_levels = list(sell_levels.values())
        levels = list(sell_levels) if sell_levels else []
        if levels:
            prices: list[float] = []
            for level in levels:
                if not isinstance(level, Mapping):
                    continue
                price_raw = level.get("price", 0.0)
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
    ask_raw = payload.get("ask", 0.0)
    if isinstance(ask_raw, (int, float, str)):
        try:
            ask = float(ask_raw)
        except (TypeError, ValueError):
            ask = 0.0
    else:
        ask = 0.0
    return ask if ask > 0 else 0.0


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
        raise ValueError(f"best ask unavailable for {symbol}")
    return ask


__all__ = ["get_best_ask"]

