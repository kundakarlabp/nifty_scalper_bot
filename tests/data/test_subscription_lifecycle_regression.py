"""Regression coverage for authoritative live subscription ownership."""

from __future__ import annotations

from collections import defaultdict

from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.strategies.runner import StrategyRunner


class _MdmStub:
    def __init__(self) -> None:
        self._token_by_symbol: dict[str, int] = {}
        self._symbol_to_token: dict[str, int] = {}
        self.callbacks: defaultdict[str, set[object]] = defaultdict(set)

    def subscribe(self, symbol: str, callback: object) -> None:
        self.callbacks[symbol].add(callback)

    def unsubscribe(self, symbol: str, callback: object) -> None:
        self.callbacks[symbol].discard(callback)

    def has_subscription(self, symbol: str, callback: object) -> bool:
        return callback in self.callbacks[symbol]


class _DataHubProbe:
    def __init__(self, *, attached: bool) -> None:
        self.attached = attached

    def has_tick_subscription(
        self,
        _symbol: str,
        _callback: object,
        *,
        token: int | None = None,
    ) -> bool:
        del token
        return self.attached


def _runner_with_datahub_probe(symbol: str, *, attached: bool) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._data_hub = _DataHubProbe(attached=attached)
    runner._datahub_registered_symbols = {symbol}
    runner._active_symbols = {symbol}
    runner._callbacks = {}
    runner._mdm_callback_registered = False
    return runner


def test_unsubscribe_keeps_mdm_delegate_for_token_listener() -> None:
    """One consumer leaving must not cut MDM delivery for another token listener."""
    mdm = _MdmStub()
    hub = DataHub(mdm, defer_live_symbol_subscriptions=False)
    symbol = "NFO:NIFTY26AUG24600CE"
    token = 123

    def leaving_listener(_tick) -> None:
        return None

    def remaining_listener(_tick) -> None:
        return None

    hub._token_by_symbol[symbol] = token
    hub._symbol_by_token[token] = symbol
    hub._tick_subscribers[symbol].add(leaving_listener)
    hub._tick_subscribers_by_token[token].update(
        {leaving_listener, remaining_listener}
    )
    hub._mdm_subscribed_symbols.add(symbol)
    mdm.subscribe(symbol, hub.ingest_tick_sync)

    hub.unsubscribe_ticks(symbol, leaving_listener)

    assert remaining_listener in hub._tick_subscribers_by_token[token]
    assert mdm.has_subscription(symbol, hub.ingest_tick_sync) is True


def test_runner_readiness_rejects_stale_local_registration() -> None:
    """Runner readiness must reflect the concrete DataHub callback, not a stale set."""
    symbol = "NFO:NIFTY26AUG24600CE"
    runner = _runner_with_datahub_probe(symbol, attached=False)

    assert runner._runner_delivery_ready_for_symbol(symbol) is False
    assert symbol not in runner._datahub_registered_symbols


def test_runner_readiness_accepts_actual_callback() -> None:
    """The concrete DataHub callback is authoritative in both directions."""
    symbol = "NFO:NIFTY26AUG24600PE"
    runner = _runner_with_datahub_probe(symbol, attached=True)
    runner._datahub_registered_symbols.clear()

    assert runner._runner_delivery_ready_for_symbol(symbol) is True
    assert symbol in runner._datahub_registered_symbols
