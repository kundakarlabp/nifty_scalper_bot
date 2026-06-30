from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.utils.errors import BrokerAuthenticationError, BrokerError


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _Broker:
    def __init__(
        self,
        rows_by_key: dict[object, list[dict[str, object]]] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self.rows_by_key = rows_by_key or {}
        self.exc = exc
        self.calls: list[tuple[object, object, object, object]] = []

    def historical_data(self, key, from_date, to_date, interval):
        self.calls.append((key, from_date, to_date, interval))
        if self.exc is not None:
            raise self.exc
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
async def test_historical_fetch_zero_rows_retries_wide_with_same_token_only() -> None:
    broker = _Broker({12345: []})
    mdm = _mdm(broker)

    rows = await mdm.fetch_history("NFO:NIFTY26MAY23300CE", "minute", 2)

    assert rows == []
    assert [call[0] for call in broker.calls] == [12345, 12345]
    assert all(not isinstance(call[0], str) for call in broker.calls)


@pytest.mark.asyncio
async def test_historical_fetch_zero_rows_remains_failure_not_success() -> None:
    broker = _Broker({})
    mdm = _mdm(broker)

    rows = await mdm.fetch_history("NFO:NIFTY26MAY23300CE", "minute", 2)

    assert rows == []
    assert [call[0] for call in broker.calls] == [12345, 12345]


@pytest.mark.asyncio
async def test_spot_historical_fetch_uses_seeded_nifty_token() -> None:
    broker = _Broker({256265: [_row()]})
    mdm = _mdm(broker)
    mdm._token_by_symbol = {"NSE:NIFTY": 256265}

    rows = await mdm.fetch_history("NSE:NIFTY", "minute", 2)

    assert len(rows) == 1
    assert broker.calls[0][0] == 256265


def test_duplicate_rows_not_accepted_as_new_when_cache_is_short() -> None:
    ts = datetime(2026, 5, 1, 9, 15, tzinfo=timezone.utc)
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = _Logger()
    mdm._canonical_symbol = lambda s: s
    mdm._bar_symbol_key = lambda s: s
    mdm._ohlc = {
        "NFO:XCE": [
            {"timestamp": ts.isoformat(), "open": 1, "high": 2, "low": 1, "close": 2}
        ]
    }
    mdm.get_ohlc_bars = lambda _s: list(mdm._ohlc["NFO:XCE"])
    mdm.ingest_historical_bar = lambda row: mdm._ohlc["NFO:XCE"].append(dict(row))

    accepted = mdm.ingest_historical_ohlc("NFO:XCE", [_row(ts)])

    assert accepted == 0
    assert len(mdm._ohlc["NFO:XCE"]) == 1


@pytest.mark.asyncio
async def test_historical_fetch_without_token_makes_no_broker_call() -> None:
    broker = _Broker({12345: [_row()]})
    mdm = _mdm(broker)
    mdm._token_by_symbol = {}

    rows = await mdm.fetch_history("NFO:NIFTY26MAY23300CE", "minute", 2)

    assert rows == []
    assert broker.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_token", ["NFO:NIFTY26JUNFUT", None, 0, -1])
async def test_historical_fetch_invalid_token_values_do_not_call_broker(
    bad_token: object,
) -> None:
    broker = _Broker({12345: [_row()]})
    mdm = _mdm(broker)
    mdm._token_by_symbol = {"NFO:NIFTY26MAY23300CE": bad_token}

    rows = await mdm.fetch_history("NFO:NIFTY26MAY23300CE", "minute", 2)

    assert rows == []
    assert broker.calls == []


@pytest.mark.asyncio
async def test_historical_fetch_numeric_string_token_is_converted() -> None:
    broker = _Broker({12345: [_row()]})
    mdm = _mdm(broker)
    mdm._token_by_symbol = {"NFO:NIFTY26MAY23300CE": "12345"}

    rows = await mdm.fetch_history("NFO:NIFTY26MAY23300CE", "minute", 2)

    assert len(rows) == 1
    assert [call[0] for call in broker.calls] == [12345]


@pytest.mark.asyncio
async def test_historical_fetch_reraises_authentication_failure() -> None:
    broker = _Broker(exc=BrokerAuthenticationError("invalid session"))
    mdm = _mdm(broker)

    with pytest.raises(BrokerAuthenticationError):
        await mdm.fetch_history("NFO:NIFTY26MAY23300CE", "minute", 2)

    assert [call[0] for call in broker.calls] == [12345]


@pytest.mark.asyncio
async def test_historical_fetch_generic_broker_error_returns_failure() -> None:
    broker = _Broker(exc=BrokerError("InputException: invalid instrument token"))
    mdm = _mdm(broker)

    rows = await mdm.fetch_history("NFO:NIFTY26MAY23300CE", "minute", 2)

    assert rows == []
    assert [call[0] for call in broker.calls] == [12345, 12345]


@pytest.mark.asyncio
async def test_historical_fetch_normalizes_orders_and_deduplicates_rows() -> None:
    early = datetime(2026, 5, 1, 9, 15, tzinfo=timezone.utc)
    late = datetime(2026, 5, 1, 9, 16, tzinfo=timezone.utc)
    broker = _Broker({12345: [_row(late), _row(early), _row(early)]})
    mdm = _mdm(broker)

    rows = await mdm.fetch_history("NFO:NIFTY26MAY23300CE", "minute", 2)

    assert [row["timestamp"] for row in rows] == [early, late]
    assert [call[0] for call in broker.calls] == [12345]
