from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


class _Logger:
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass


class _Broker:
    def __init__(self, rows_by_key: dict[object, list[dict[str, object]]]) -> None:
        self.rows_by_key = rows_by_key
        self.calls: list[object] = []

    def historical_data(self, key, *_args):
        self.calls.append(key)
        return list(self.rows_by_key.get(key, []))


def _row(ts: datetime | None = None) -> dict[str, object]:
    return {
        "date": ts or datetime(2026, 5, 1, 9, 15, tzinfo=timezone.utc),
        "open": 1,
        "high": 2,
        "low": 1,
        "close": 2,
        "volume": 10,
    }


def _mdm(broker: _Broker) -> MarketDataManager:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = _Logger()
    mdm._canonical_symbol = lambda s: str(s).strip().upper()
    mdm._token_by_symbol = {"NFO:NIFTY26MAY23300CE": 12345}
    mdm._resolver = None
    mdm._broker = broker
    mdm._history_lock = asyncio.Lock()
    mdm._history_min_interval_sec = 0.0
    mdm._last_history_request_ts = 0.0
    return mdm


@pytest.mark.asyncio
async def test_historical_fetch_zero_rows_uses_tradingsymbol_fallback() -> None:
    broker = _Broker({12345: [], "NFO:NIFTY26MAY23300CE": [_row()]})
    mdm = _mdm(broker)

    rows = await mdm.fetch_history("NFO:NIFTY26MAY23300CE", "minute", 2)

    assert len(rows) == 1
    assert broker.calls[:2] == [12345, "NFO:NIFTY26MAY23300CE"]


@pytest.mark.asyncio
async def test_historical_fetch_zero_rows_remains_failure_not_success() -> None:
    broker = _Broker({})
    mdm = _mdm(broker)

    rows = await mdm.fetch_history("NFO:NIFTY26MAY23300CE", "minute", 2)

    assert rows == []
    assert len(broker.calls) >= 2


@pytest.mark.asyncio
async def test_spot_historical_fetch_uses_nifty_token_fallback() -> None:
    broker = _Broker({256265: [_row()]})
    mdm = _mdm(broker)
    mdm._token_by_symbol = {}

    rows = await mdm.fetch_history("NSE:NIFTY", "minute", 2)

    assert len(rows) == 1
    assert broker.calls[0] == 256265


def test_duplicate_rows_not_accepted_as_new_when_cache_is_short() -> None:
    ts = datetime(2026, 5, 1, 9, 15, tzinfo=timezone.utc)
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = _Logger()
    mdm._canonical_symbol = lambda s: s
    mdm._bar_symbol_key = lambda s: s
    mdm._ohlc = {"NFO:XCE": [{"timestamp": ts.isoformat(), "open": 1, "high": 2, "low": 1, "close": 2}]}
    mdm.get_ohlc_bars = lambda _s: list(mdm._ohlc["NFO:XCE"])
    mdm.ingest_historical_bar = lambda row: mdm._ohlc["NFO:XCE"].append(dict(row))

    accepted = mdm.ingest_historical_ohlc("NFO:XCE", [_row(ts)])

    assert accepted == 0
    assert len(mdm._ohlc["NFO:XCE"]) == 1
