from datetime import datetime, timezone

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class DummyBroker:
    def historical_data(self, token, from_date, to_date, interval):
        return [[datetime(2026, 1, 1, tzinfo=timezone.utc), 1, 2, 0.5, 1.5, 100]]


@pytest.mark.asyncio
async def test_normalize_history_row_list() -> None:
    row = [datetime(2026, 1, 1), 1, 2, 0.5, 1.5, 100]
    bar = MarketDataManager.normalize_history_row('NSE:NIFTY', row)
    assert bar is not None
    assert bar['symbol'] == 'NSE:NIFTY'
    assert bar['open'] == 1.0
    assert bar['volume'] == 100
    assert bar['timestamp'].tzinfo is not None


def test_normalize_history_row_dict() -> None:
    row = {
        'date': datetime(2026, 1, 1),
        'open': 1,
        'high': 2,
        'low': 0.5,
        'close': 1.5,
        'volume': 100,
    }
    bar = MarketDataManager.normalize_history_row('NSE:NIFTY', row)
    assert bar is not None
    assert bar['close'] == 1.5
    assert bar['source'] == 'historical'


def test_normalize_history_row_invalid() -> None:
    assert MarketDataManager.normalize_history_row('NSE:NIFTY', {'open': 1}) is None


@pytest.mark.asyncio
async def test_fetch_history_returns_normalized_dicts_only() -> None:
    manager = MarketDataManager(broker=DummyBroker())
    manager._token_by_symbol['NSE:NIFTY'] = 256265
    bars = await manager.fetch_history('NSE:NIFTY', 'minute', 2)
    assert bars
    assert isinstance(bars[0], dict)
    assert 'open' in bars[0]
    assert not isinstance(bars[0], (list, tuple))
