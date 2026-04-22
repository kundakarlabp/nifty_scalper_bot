from __future__ import annotations

from typing import Any

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class DummyBroker:
    def get_quote(self, symbol: str) -> dict[str, Any]:
        return {'symbol': symbol, 'ltp': 0.0}

    def instruments(self, _exchange: str) -> list[dict[str, Any]]:
        return []


class DummyResolver:
    def __init__(self) -> None:
        self._map = {'NSE:NIFTY': 256265}

    def resolve(self, symbol: str) -> int | None:
        return self._map.get(symbol)


class DummyWebSocket:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def set_tokens(self, tokens) -> bool:  # noqa: ANN001
        self.calls.append(list(tokens))
        return True


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


def test_ensure_subscription_uses_request_api() -> None:
    mdm = MarketDataManager(DummyBroker(), websocket=None, resolver=DummyResolver())
    calls: list[str] = []

    def _request(symbol: str) -> bool:
        calls.append(symbol)
        return True

    mdm.request_symbol_subscription = _request  # type: ignore[method-assign]
    mdm._ensure_subscription('NSE:NIFTY')  # noqa: SLF001

    assert calls == ['NSE:NIFTY']
