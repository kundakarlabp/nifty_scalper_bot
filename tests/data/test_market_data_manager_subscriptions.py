from __future__ import annotations

from typing import Any

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.utils.market_hours import MarketState


class DummyBroker:
    def get_quote(self, symbol: str) -> dict[str, Any]:
        return {'symbol': symbol, 'ltp': 0.0}

    def instruments(self, _exchange: str) -> list[dict[str, Any]]:
        return []


class DummyBrokerQuoteList(DummyBroker):
    def quote(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        symbol = symbols[0]
        return {symbol: {'last_price': 123.0, 'bid': 122.0, 'ask': 124.0}}


class DummyBrokerGetQuoteList(DummyBroker):
    def get_quote(self, symbols):  # noqa: ANN001
        if isinstance(symbols, list):
            symbol = symbols[0]
            return {symbol: {'last_price': 123.0, 'bid': 122.0, 'ask': 124.0}}
        return {'last_price': 123.0, 'bid': 122.0, 'ask': 124.0}


class DummyResolver:
    def __init__(self) -> None:
        self._map = {'NSE:NIFTY': 256265}

    def resolve(self, symbol: str) -> int | None:
        return self._map.get(symbol)


class DummyWebSocket:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.connected = True

    def set_tokens(self, tokens) -> bool:  # noqa: ANN001
        self.calls.append(list(tokens))
        return True

    def is_connected(self) -> bool:
        return self.connected


def test_request_symbol_subscription_reconciles_once() -> None:
    ws = DummyWebSocket()
    mdm = MarketDataManager(DummyBroker(), ws, resolver=DummyResolver())

    changed = mdm.request_symbol_subscription('NSE:NIFTY')

    assert changed is True
    assert ws.calls == [[256265]]


def test_request_symbol_subscription_no_duplicate_transport_call() -> None:
    ws = DummyWebSocket()
    mdm = MarketDataManager(DummyBroker(), ws, resolver=DummyResolver())

    mdm.request_symbol_subscription('NSE:NIFTY')
    changed = mdm.request_symbol_subscription('NSE:NIFTY')

    assert changed is False
    assert ws.calls == [[256265]]


def test_request_token_subscriptions_batch_reconciles_once() -> None:
    ws = DummyWebSocket()
    mdm = MarketDataManager(DummyBroker(), ws, resolver=DummyResolver())

    added = mdm.request_token_subscriptions([101, 202, 101])

    assert added == 2
    assert ws.calls == [[101, 202]]


def test_request_token_subscriptions_idempotent() -> None:
    ws = DummyWebSocket()
    mdm = MarketDataManager(DummyBroker(), ws, resolver=DummyResolver())

    first = mdm.request_token_subscriptions([101, 202])
    second = mdm.request_token_subscriptions([202, 101])

    assert first == 2
    assert second == 0
    assert ws.calls == [[101, 202]]


def test_request_subscription_without_ws_retains_desired_set() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None, resolver=DummyResolver())

    changed = mdm.request_symbol_subscription('NSE:NIFTY')

    assert changed is True
    assert mdm._desired_tokens == {256265}  # noqa: SLF001


def test_request_token_unsubscription_reconciles_transport() -> None:
    ws = DummyWebSocket()
    mdm = MarketDataManager(DummyBroker(), ws, resolver=DummyResolver())
    mdm.request_token_subscriptions([101, 202])

    changed = mdm.request_token_unsubscription(101)

    assert changed is True
    assert ws.calls[-1] == [202]


def test_request_token_unsubscriptions_batch_reconciles_once() -> None:
    ws = DummyWebSocket()
    mdm = MarketDataManager(DummyBroker(), ws, resolver=DummyResolver())
    mdm.request_token_subscriptions([101, 202, 303])

    removed = mdm.request_token_unsubscriptions([202, 303, 303])

    assert removed == 2
    assert ws.calls[-1] == [101]


def test_request_token_unsubscriptions_missing_token_is_harmless() -> None:
    ws = DummyWebSocket()
    mdm = MarketDataManager(DummyBroker(), ws, resolver=DummyResolver())
    mdm.request_token_subscriptions([101])
    before_calls = list(ws.calls)

    removed = mdm.request_token_unsubscriptions([999])

    assert removed == 0
    assert ws.calls == before_calls


def test_desired_token_snapshot_and_count_are_stable() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None, resolver=DummyResolver())
    mdm.request_token_subscriptions([303, 101, 202, 202])

    assert mdm.desired_token_count() == 3
    assert mdm.desired_tokens_snapshot() == [101, 202, 303]


def test_ensure_subscription_uses_request_api() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None, resolver=DummyResolver())
    calls: list[str] = []

    def _request(symbol: str) -> bool:
        calls.append(symbol)
        return True

    mdm.request_symbol_subscription = _request  # type: ignore[method-assign]
    mdm._ensure_subscription('NSE:NIFTY')  # noqa: SLF001

    assert calls == ['NSE:NIFTY']


def test_pending_subscriptions_flush_on_connect() -> None:
    ws = DummyWebSocket()
    ws.connected = False
    mdm = MarketDataManager(DummyBroker(), ws, resolver=DummyResolver())

    changed = mdm.request_symbol_subscription('NSE:NIFTY')
    assert changed is True
    assert ws.calls == []
    assert 256265 in mdm._pending_subscription_tokens  # noqa: SLF001

    ws.connected = True
    mdm.set_ws_connected(True)
    assert ws.calls == [[256265]]


def test_ltp_stale_thresholds_are_symbol_specific(monkeypatch) -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None, resolver=DummyResolver())
    monkeypatch.setattr(
        'nifty_scalper_bot.data.market_data_manager.get_market_state',
        lambda: MarketState.OPEN,
    )
    assert mdm._ltp_stale_threshold_for_symbol('NSE:NIFTY') == 120.0  # noqa: SLF001
    assert mdm._ltp_stale_threshold_for_symbol('NFO:NIFTY26APR23800PE') == 30.0  # noqa: SLF001

    monkeypatch.setattr(
        'nifty_scalper_bot.data.market_data_manager.get_market_state',
        lambda: MarketState.CLOSED,
    )
    assert mdm._ltp_stale_threshold_for_symbol('NSE:NIFTY') == 3600.0  # noqa: SLF001


class DummyPollingFallback:
    def __init__(self) -> None:
        self.subscribed: list[list[int]] = []
        self.started = False

    def subscribe(self, tokens: list[int]) -> None:
        self.subscribed.append(tokens)

    def is_running(self) -> bool:
        return self.started

    def start(self) -> None:
        self.started = True


def test_normalize_tick_does_not_fabricate_bid_ask() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None, resolver=DummyResolver())

    tick = mdm._normalize_tick('NFO:NIFTY26APR23800CE', {'last_price': 123.0})  # noqa: SLF001

    assert tick is not None
    assert tick['bid'] is None
    assert tick['ask'] is None
    assert tick['tradable_quote'] is False
    assert tick['bid_ask_source'] == 'missing'


def test_snapshot_marks_ltp_only_quote_not_tradable() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None, resolver=DummyResolver())
    mdm._emit_tick(  # noqa: SLF001
        'NFO:NIFTY26APR23800CE',
        {'symbol': 'NFO:NIFTY26APR23800CE', 'last_price': 123.0, 'ltp': 123.0, 'source': 'rest_ltp'},
        source='rest_ltp',
    )

    snap = mdm.get_symbol_snapshot('NFO:NIFTY26APR23800CE')

    assert snap.bid is None
    assert snap.ask is None
    assert snap.tradable_quote is False


def test_snapshot_unknown_bid_ask_source_is_not_tradable() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None, resolver=DummyResolver())
    mdm._emit_tick(  # noqa: SLF001
        'NFO:NIFTY26APR23800CE',
        {
            'symbol': 'NFO:NIFTY26APR23800CE',
            'last_price': 123.0,
            'ltp': 123.0,
            'bid': 122.0,
            'ask': 124.0,
            'bid_ask_source': 'unknown_source',
            'source': 'custom',
        },
        source='custom',
    )
    snap = mdm.get_symbol_snapshot('NFO:NIFTY26APR23800CE')
    assert snap.tradable_quote is False


def test_broker_quote_helper_supports_quote_and_get_quote_variants() -> None:
    quote_mdm = MarketDataManager(DummyBrokerQuoteList(), websocket=None, resolver=DummyResolver())
    quote_payload = quote_mdm._broker_quote(quote_mdm._broker, 'NFO:NIFTY26APR23800CE')  # type: ignore[arg-type]  # noqa: SLF001
    assert quote_payload is not None
    assert float(quote_payload['bid']) == 122.0

    get_quote_mdm = MarketDataManager(DummyBrokerGetQuoteList(), websocket=None, resolver=DummyResolver())
    get_quote_payload = get_quote_mdm._broker_quote(get_quote_mdm._broker, 'NFO:NIFTY26APR23800CE')  # type: ignore[arg-type]  # noqa: SLF001
    assert get_quote_payload is not None
    assert float(get_quote_payload['ask']) == 124.0


def test_ensure_fresh_tick_uses_polling_fallback_owner(monkeypatch) -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None, resolver=DummyResolver())
    fallback = DummyPollingFallback()
    mdm.set_polling_fallback_streamer(fallback)
    monkeypatch.setattr(mdm, 'get_latest_tick', lambda _symbol: None)
    monkeypatch.setattr(mdm, 'time_since_last_tick', lambda _symbol: 999.0)

    import asyncio

    asyncio.run(mdm.ensure_fresh_tick('NSE:NIFTY'))

    assert fallback.subscribed




def test_request_token_subscription_sets_ws_tokens_before_connect() -> None:
    ws = DummyWebSocket()
    ws.connected = False
    ws._tokens: set[int] = set()

    def _set_tokens(tokens) -> bool:  # noqa: ANN001
        ws.calls.append(list(tokens))
        ws._tokens = set(tokens)
        return True

    ws.set_tokens = _set_tokens  # type: ignore[method-assign]
    mdm = MarketDataManager(DummyBroker(), ws, resolver=DummyResolver())

    changed = mdm.request_token_subscription(256265, symbol='NSE:NIFTY')

    assert changed is True
    assert ws.calls[-1] == [256265]
    assert 256265 in ws._tokens

def test_start_websocket_reconciles_tokens_before_connect() -> None:
    class WsWithStart(DummyWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.start_calls = 0

        def start(self) -> None:
            self.start_calls += 1

    ws = WsWithStart()
    mdm = MarketDataManager(DummyBroker(), ws, resolver=DummyResolver())
    mdm.request_token_subscription(256265, symbol='NSE:NIFTY')
    ws.calls.clear()

    mdm.start_websocket()

    assert ws.start_calls == 1
    assert ws.calls == [[256265]]


def test_spot_ws_health_logs_first_tick_missing_not_stale(monkeypatch, caplog) -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None, resolver=DummyResolver())
    mdm.register_symbol('NSE:NIFTY', 256265)
    monkeypatch.setattr(
        'nifty_scalper_bot.data.market_data_manager.get_market_state',
        lambda: MarketState.OPEN,
    )

    with caplog.at_level('WARNING'):
        mdm._monitor_spot_ws_health()  # noqa: SLF001

    assert 'MDM_WS_FIRST_TICK_MISSING' in caplog.text
    assert 'MDM_WS_TICK_STALE' not in caplog.text


def test_spot_ws_health_logs_stale_only_after_first_tick(monkeypatch, caplog) -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None, resolver=DummyResolver())
    mdm.register_symbol('NSE:NIFTY', 256265)
    mdm._last_tick_wallclock['NSE:NIFTY'] = 1.0  # noqa: SLF001
    mdm._last_tick_time['NSE:NIFTY'] = 1.0  # noqa: SLF001
    monkeypatch.setattr(
        'nifty_scalper_bot.data.market_data_manager.get_market_state',
        lambda: MarketState.OPEN,
    )
    monkeypatch.setattr(
        'nifty_scalper_bot.data.market_data_manager.time.time',
        lambda: 1000.0,
    )

    with caplog.at_level('WARNING'):
        mdm._monitor_spot_ws_health()  # noqa: SLF001

    assert 'MDM_WS_TICK_STALE' in caplog.text
