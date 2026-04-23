from __future__ import annotations

from typing import Any, Callable

from nifty_scalper_bot.data.data_hub import DataHub


class _StubMdm:
    def __init__(self) -> None:
        self.subscribed: list[str] = []
        self.callbacks: dict[str, list[Callable[[dict[str, Any]], None]]] = {}

    def subscribe(
        self, symbol: str, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        self.subscribed.append(symbol)
        self.callbacks.setdefault(symbol, []).append(callback)


def test_subscribe_ticks_deferred_registers_without_live_subscribe() -> None:
    mdm = _StubMdm()
    hub = DataHub(market_data_manager=mdm)

    def callback(_tick: dict[str, Any]) -> None:
        return None

    hub.subscribe_ticks('NSE:NIFTY', callback)

    assert mdm.subscribed == []
    assert 'NSE:NIFTY' in hub._pending_live_symbols  # noqa: SLF001
    assert callback in hub._tick_subscribers['NSE:NIFTY']  # noqa: SLF001


def test_flush_pending_live_subscriptions_subscribes_unique_symbols_once() -> None:
    mdm = _StubMdm()
    hub = DataHub(market_data_manager=mdm)

    hub.subscribe_ticks('NSE:NIFTY', lambda _tick: None)
    hub.subscribe_ticks('NSE:NIFTY', lambda _tick: None)
    hub.subscribe_ticks('NFO:NIFTY26APR25500CE', lambda _tick: None)

    flushed = hub.flush_pending_live_subscriptions()

    assert flushed == 2
    assert mdm.subscribed == ['NFO:NIFTY26APR25500CE', 'NSE:NIFTY']
    assert hub._pending_live_symbols == set()  # noqa: SLF001


def test_duplicate_listeners_same_symbol_keep_fanout_with_single_live_subscribe(
) -> None:
    mdm = _StubMdm()
    hub = DataHub(market_data_manager=mdm)
    received_one: list[dict[str, Any]] = []
    received_two: list[dict[str, Any]] = []

    hub.subscribe_ticks('NSE:NIFTY', lambda tick: received_one.append(tick))
    hub.subscribe_ticks('NSE:NIFTY', lambda tick: received_two.append(tick))
    hub.flush_pending_live_subscriptions()
    live_callback = mdm.callbacks['NSE:NIFTY'][0]
    live_callback({'symbol': 'NSE:NIFTY', 'ltp': 123.0, 'timestamp': 1})

    assert mdm.subscribed == ['NSE:NIFTY']
    assert len(received_one) == 1
    assert len(received_two) == 1
