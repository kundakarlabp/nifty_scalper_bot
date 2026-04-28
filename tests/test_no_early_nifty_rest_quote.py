from __future__ import annotations

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class FakeBroker:
    def __init__(self) -> None:
        self.called = False

    def get_quote(self, symbol: str):  # noqa: ANN201
        self.called = True
        raise AssertionError('REST quote should not be called before WS')


def test_no_early_nifty_rest_quote(monkeypatch) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE', 'true')
    monkeypatch.setenv('MARKETDATA_SUPPRESS_REST_BEFORE_WS', 'true')
    monkeypatch.setenv('MARKETDATA_REST_FALLBACK_MODE', 'after_ws_degraded')
    broker = FakeBroker()
    mdm = MarketDataManager(broker=broker, websocket=None)
    mdm._ws_started = False  # noqa: SLF001 - test state setup
    assert mdm.get_latest_price('NSE:NIFTY') is None
    assert broker.called is False
