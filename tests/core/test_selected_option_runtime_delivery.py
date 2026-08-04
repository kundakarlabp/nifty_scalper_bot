from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

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

    def add_symbol(self, symbol: str) -> None:
        self.added.append(symbol)

    def on_datahub_tick(self, _tick) -> None:
        return None

    def has_datahub_subscription(self, symbol: str, token: int | None = None) -> bool:
        del token
        return symbol in self.added


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
    assert runner.added == [ce, pe]
    assert [(sym, token, force) for sym, _cb, token, force in hub.calls] == [
        (ce, 101, True),
        (pe, 102, True),
    ]
    assert mdm.requested == [(101, ce), (102, pe)]
