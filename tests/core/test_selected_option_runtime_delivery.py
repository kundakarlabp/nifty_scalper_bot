from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import cast

from nifty_scalper_bot.core import app
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.strategies.runner import StrategyRunner


class _MdmStub:
    def __init__(self) -> None:
        self.tracked: list[str] = []
        self.registered: list[tuple[str, int]] = []
        self.requested: list[tuple[int, str]] = []

    def ensure_tracking(self, symbol: str, **_kwargs) -> None:
        self.tracked.append(symbol)

    def register_symbol(self, symbol: str, token: int) -> None:
        self.registered.append((symbol, token))

    def request_token_subscription(self, token: int, *, symbol: str) -> bool:
        self.requested.append((token, symbol))
        return True


class _RunnerStub:
    def __init__(self) -> None:
        self.added: list[str] = []
        self.subscribed: list[str] = []
        self._tracked_symbols: set[str] = set()

    def add_symbol(self, symbol: str) -> None:
        self.added.append(symbol)

    def _subscribe_symbol(self, symbol: str) -> None:
        self.subscribed.append(symbol)

    def on_datahub_tick(self, _tick) -> None:
        return None

    def has_datahub_subscription(self, symbol: str, token: int | None = None) -> bool:
        del token
        return symbol in self.subscribed


class _DeliveryMdmStub(_MdmStub):
    def __init__(self) -> None:
        super().__init__()
        self.callbacks: defaultdict[str, set[object]] = defaultdict(set)

    def subscribe(self, symbol: str, callback: object) -> None:
        self.callbacks[symbol].add(callback)

    def has_subscription(self, symbol: str, callback: object) -> bool:
        return callback in self.callbacks[symbol]


class _DeliveryHubStub:
    def __init__(self, mdm: _DeliveryMdmStub) -> None:
        self._mdm = mdm
        self.callbacks: defaultdict[str, set[object]] = defaultdict(set)
        self.subscribe_calls: list[tuple[str, object, int | None, bool]] = []

    def ingest_tick_sync(self, _tick) -> None:
        return None

    def has_tick_subscription(self, symbol, callback, *, token=None) -> bool:
        del token
        return callback in self.callbacks[symbol]

    def subscribe_ticks(self, symbol, callback, *, token=None, force_live=False):
        self.subscribe_calls.append((symbol, callback, token, force_live))
        self.callbacks[symbol].add(callback)
        self._mdm.subscribe(symbol, self.ingest_tick_sync)


class _DeliveryRunnerStub:
    def __init__(self, hub: _DeliveryHubStub) -> None:
        self._data_hub = hub
        self._tracked_symbols: set[str] = set()
        self.subscribe_calls: list[str] = []

    def on_datahub_tick(self, _tick) -> None:
        return None

    def has_datahub_subscription(self, symbol: str, token: int | None = None) -> bool:
        return self._data_hub.has_tick_subscription(
            symbol, self.on_datahub_tick, token=token
        )

    def _subscribe_symbol(self, symbol: str) -> None:
        self.subscribe_calls.append(symbol)
        self._data_hub.subscribe_ticks(symbol, self.on_datahub_tick)


def test_datahub_subscription_probe_checks_the_specific_callback() -> None:
    hub = DataHub(_MdmStub(), defer_live_symbol_subscriptions=True)
    symbol = "NFO:NIFTY26AUG24600CE"
    token = 101

    def runner_callback(_tick) -> None:
        return None

    def unrelated_callback(_tick) -> None:
        return None

    hub.subscribe_ticks(symbol, unrelated_callback, token=token)
    assert hub.has_tick_subscription(symbol, runner_callback, token=token) is False

    hub.subscribe_ticks(symbol, runner_callback, token=token)
    assert hub.has_tick_subscription(symbol, runner_callback, token=token) is True

    hub.unsubscribe_ticks(symbol, runner_callback)
    assert hub.has_tick_subscription(symbol, runner_callback, token=token) is False


def test_runner_repairs_stale_internal_datahub_registration() -> None:
    symbol = "NFO:NIFTY26AUG24600CE"

    class Hub:
        def __init__(self) -> None:
            self.callbacks = defaultdict(set)
            self.subscribe_calls = 0

        def has_tick_subscription(self, sym, callback, *, token=None):
            del token
            return callback in self.callbacks[sym]

        def subscribe_ticks(self, sym, callback, **_kwargs):
            self.subscribe_calls += 1
            self.callbacks[sym].add(callback)

    runner = StrategyRunner.__new__(StrategyRunner)
    runner._data_hub = Hub()
    runner._datahub_registered_symbols = {symbol}  # stale bookkeeping only

    runner._subscribe_symbol(symbol)

    assert runner._data_hub.subscribe_calls == 1
    assert runner.has_datahub_subscription(symbol) is True


def test_selected_pair_role_promotion_reasserts_live_delivery() -> None:
    mdm = _MdmStub()
    runner = _RunnerStub()

    class Hub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object, int | None, bool]] = []

        def subscribe_ticks(self, symbol, callback, *, token=None, force_live=False):
            self.calls.append((symbol, callback, token, force_live))

    hub = Hub()
    ce = "NFO:NIFTY26AUG24600CE"
    pe = "NFO:NIFTY26AUG24600PE"
    runner._tracked_symbols.update({ce, pe})
    ctx = SimpleNamespace(
        active_symbol_tokens={ce: 101, pe: 102},
        market_data_manager=mdm,
        strategy_runner=runner,
        data_hub=hub,
        instrument_manager=None,
        broker_client=None,
    )

    result = app._ensure_selected_option_runtime_delivery(
        ctx, selected_ce=ce, selected_pe=pe, reason="dynamic_basket_committed"
    )

    assert result == {ce: True, pe: True}
    assert runner.added == []
    assert runner.subscribed == [ce, pe]
    assert [(sym, token, force) for sym, _cb, token, force in hub.calls] == [
        (ce, 101, True),
        (pe, 102, True),
    ]
    assert mdm.requested == [(101, ce), (102, pe)]


def test_selected_pair_liveness_noop_when_delivery_edges_exist() -> None:
    symbol = "NFO:NIFTY26AUG24600CE"
    token = 101
    mdm = _DeliveryMdmStub()
    hub = _DeliveryHubStub(mdm)
    runner = _DeliveryRunnerStub(hub)
    runner._tracked_symbols.add(symbol)
    hub.callbacks[symbol].add(runner.on_datahub_tick)
    mdm.callbacks[symbol].add(hub.ingest_tick_sync)
    ctx = SimpleNamespace(
        active_symbol_tokens={symbol: token},
        market_data_manager=mdm,
        strategy_runner=runner,
        data_hub=hub,
    )

    result = app._ensure_selected_option_runtime_delivery(
        cast(app.BotContext, ctx),
        selected_ce=symbol,
        selected_pe=None,
        reason="periodic_liveness",
    )

    assert result == {symbol: True}
    assert runner.subscribe_calls == []
    assert hub.subscribe_calls == []
    assert mdm.requested == []


def test_selected_pair_liveness_repairs_only_missing_mdm_delegate() -> None:
    symbol = "NFO:NIFTY26AUG24600PE"
    token = 102
    mdm = _DeliveryMdmStub()
    hub = _DeliveryHubStub(mdm)
    runner = _DeliveryRunnerStub(hub)
    runner._tracked_symbols.add(symbol)
    hub.callbacks[symbol].add(runner.on_datahub_tick)
    ctx = SimpleNamespace(
        active_symbol_tokens={symbol: token},
        market_data_manager=mdm,
        strategy_runner=runner,
        data_hub=hub,
    )

    result = app._ensure_selected_option_runtime_delivery(
        cast(app.BotContext, ctx),
        selected_ce=None,
        selected_pe=symbol,
        reason="periodic_liveness",
    )

    assert result == {symbol: True}
    assert runner.subscribe_calls == []
    assert len(hub.subscribe_calls) == 1
    assert mdm.has_subscription(symbol, hub.ingest_tick_sync) is True


def test_selected_pair_liveness_repairs_only_missing_runner_callback() -> None:
    symbol = "NFO:NIFTY26AUG24600CE"
    token = 103
    mdm = _DeliveryMdmStub()
    hub = _DeliveryHubStub(mdm)
    runner = _DeliveryRunnerStub(hub)
    runner._tracked_symbols.add(symbol)
    mdm.callbacks[symbol].add(hub.ingest_tick_sync)
    ctx = SimpleNamespace(
        active_symbol_tokens={symbol: token},
        market_data_manager=mdm,
        strategy_runner=runner,
        data_hub=hub,
    )

    result = app._ensure_selected_option_runtime_delivery(
        cast(app.BotContext, ctx),
        selected_ce=symbol,
        selected_pe=None,
        reason="periodic_liveness",
    )

    assert result == {symbol: True}
    assert runner.subscribe_calls == [symbol]
    assert len(hub.subscribe_calls) == 1
    assert hub.has_tick_subscription(symbol, runner.on_datahub_tick, token=token)
