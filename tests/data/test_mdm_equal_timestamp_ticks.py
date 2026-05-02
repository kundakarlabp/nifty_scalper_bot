from __future__ import annotations

from typing import Any

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class DummyBroker:
    def get_instrument_token(self, _symbol: str) -> int:
        return 123


class DummyWebSocket:
    on_tick = None


def _base_tick(ts: str, price: float, volume: int) -> dict[str, Any]:
    return {
        'symbol': 'NSE:NIFTY23',
        'timestamp': ts,
        'ltp': price,
        'last_price': price,
        'instrument_token': 123,
        'volume_traded_today': volume,
    }


def test_older_timestamp_dropped() -> None:
    mdm = MarketDataManager(DummyBroker(), DummyWebSocket())
    mdm._process_queued_tick(_base_tick('2026-01-01T10:00:01Z', 100.0, 10))  # noqa: SLF001
    mdm._process_queued_tick(_base_tick('2026-01-01T10:00:00Z', 101.0, 11))  # noqa: SLF001
    assert mdm.get_latest_tick('NSE:NIFTY23')['ltp'] == 100.0


def test_equal_timestamp_same_price_volume_dropped() -> None:
    mdm = MarketDataManager(DummyBroker(), DummyWebSocket())
    mdm._process_queued_tick(_base_tick('2026-01-01T10:00:01Z', 100.0, 10))  # noqa: SLF001
    mdm._process_queued_tick(_base_tick('2026-01-01T10:00:01Z', 100.0, 10))  # noqa: SLF001
    assert mdm.get_latest_tick('NSE:NIFTY23')['ltp'] == 100.0


def test_equal_timestamp_changed_price_accepted() -> None:
    mdm = MarketDataManager(DummyBroker(), DummyWebSocket())
    mdm._process_queued_tick(_base_tick('2026-01-01T10:00:01Z', 100.0, 10))  # noqa: SLF001
    mdm._process_queued_tick(_base_tick('2026-01-01T10:00:01Z', 101.0, 10))  # noqa: SLF001
    assert mdm.get_latest_tick('NSE:NIFTY23')['ltp'] == 101.0


def test_equal_timestamp_changed_volume_accepted() -> None:
    mdm = MarketDataManager(DummyBroker(), DummyWebSocket())
    mdm._process_queued_tick(_base_tick('2026-01-01T10:00:01Z', 100.0, 10))  # noqa: SLF001
    mdm._process_queued_tick(_base_tick('2026-01-01T10:00:01Z', 100.0, 11))  # noqa: SLF001
    assert mdm.get_latest_tick('NSE:NIFTY23')['volume'] == 11
