from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import threading

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _mdm() -> MarketDataManager:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._lock = threading.RLock()
    mdm._cache_len = 250
    mdm._ohlc = defaultdict(lambda: deque(maxlen=mdm._cache_len))
    return mdm


def _bar(
    i: int,
    *,
    source: str = "historical",
    provisional: bool = False,
    close: float | None = None,
) -> dict:
    ts = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc) + timedelta(minutes=i)
    return {
        "symbol": "NFO:NIFTY26JUN24000CE",
        "timestamp": ts,
        "open": 100 + i,
        "high": 101 + i,
        "low": 99 + i,
        "close": float(close if close is not None else 100 + i),
        "volume": i,
        "source": source,
        "provisional": provisional,
        "synthetic": False,
        "timestamp_quality": "broker",
    }


def test_canonical_getter_merges_history_and_live_dedupes_overlaps() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    for i in range(50):
        mdm._ohlc[key].append(_bar(i, source="historical"))
    for i in range(35, 50):
        mdm._ohlc[symbol].append(_bar(i, source="ws_candle", close=1000 + i))

    bars = mdm.get_ohlc_bars(symbol)

    assert len(bars) == 50
    assert [bar["timestamp"] for bar in bars] == sorted(
        bar["timestamp"] for bar in bars
    )
    assert len({bar["timestamp"] for bar in bars}) == 50
    assert bars[-1]["close"] == 1049.0


def test_canonical_getter_does_not_let_provisional_current_candle_overwrite_history() -> (
    None
):
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    mdm._ohlc[key].append(_bar(1, source="historical", close=101.0))
    mdm._ohlc[key].append(_bar(1, source="ws_candle", provisional=True, close=999.0))

    bars = mdm.get_ohlc_bars(symbol)

    assert len(bars) == 1
    assert bars[0]["close"] == 101.0
    assert bars[0]["source"] == "historical"


def test_completed_live_candle_precedes_historical_on_same_timestamp() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    mdm._ohlc[key].append(_bar(2, source="historical", close=102.0))
    mdm._ohlc[key].append(_bar(2, source="ws_candle", close=202.0))

    bars = mdm.get_ohlc_bars(symbol)

    assert len(bars) == 1
    assert bars[0]["close"] == 202.0
    assert bars[0]["source"] == "ws_candle"


def test_canonical_getter_applies_limit_after_merge_and_does_not_mutate_storage() -> (
    None
):
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    for i in range(10):
        mdm._ohlc[key].append(_bar(i, source="historical"))
    before = list(mdm._ohlc[key])

    bars = mdm.get_ohlc_bars(symbol, limit=3)

    assert [bar["timestamp"] for bar in bars] == [
        before[-3]["timestamp"],
        before[-2]["timestamp"],
        before[-1]["timestamp"],
    ]
    assert list(mdm._ohlc[key]) == before


def _custom_bar(
    timestamp: object,
    *,
    source: str,
    close: float,
    provisional: bool = False,
    synthetic: bool = False,
    timestamp_quality: str = "broker",
) -> dict:
    return {
        "symbol": "NFO:NIFTY26JUN24000CE",
        "timestamp": timestamp,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1,
        "source": source,
        "provisional": provisional,
        "synthetic": synthetic,
        "timestamp_quality": timestamp_quality,
    }


def _assert_single_live_winner(mdm: MarketDataManager, symbol: str) -> None:
    bars = mdm.get_ohlc_bars(symbol)
    assert len(bars) == 1
    assert bars[0]["source"] == "ws_candle"
    assert bars[0]["synthetic"] is True
    assert bars[0]["timestamp_quality"] == "exchange_timestamp"
    assert bars[0]["provisional"] is False
    assert bars[0]["close"] == 222.0


def test_naive_ist_history_and_aware_ist_live_timestamp_dedupe_with_metadata() -> None:
    from zoneinfo import ZoneInfo

    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    mdm._ohlc[key].append(
        _custom_bar(
            datetime(2026, 1, 1, 9, 15),
            source="historical",
            close=111.0,
            timestamp_quality="broker_naive",
        )
    )
    mdm._ohlc[key].append(
        _custom_bar(
            datetime(2026, 1, 1, 9, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
            source="ws_candle",
            close=222.0,
            synthetic=True,
            timestamp_quality="exchange_timestamp",
        )
    )

    _assert_single_live_winner(mdm, symbol)


def test_naive_ist_history_and_equivalent_utc_live_timestamp_dedupe() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    mdm._ohlc[key].append(
        _custom_bar(datetime(2026, 1, 1, 9, 15), source="historical", close=111.0)
    )
    mdm._ohlc[key].append(
        _custom_bar(
            datetime(2026, 1, 1, 3, 45, 45, tzinfo=timezone.utc),
            source="ws_candle",
            close=222.0,
            synthetic=True,
            timestamp_quality="exchange_timestamp",
        )
    )

    _assert_single_live_winner(mdm, symbol)


def test_iso_timestamp_with_ist_offset_dedupes_to_same_minute() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    mdm._ohlc[key].append(
        _custom_bar("2026-01-01T09:15:00+05:30", source="historical", close=111.0)
    )
    mdm._ohlc[key].append(
        _custom_bar(
            "2026-01-01T03:45:40Z",
            source="ws_candle",
            close=222.0,
            synthetic=True,
            timestamp_quality="exchange_timestamp",
        )
    )

    _assert_single_live_winner(mdm, symbol)


def test_unix_epoch_timestamp_dedupes_to_same_minute() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    epoch = datetime(2026, 1, 1, 3, 45, 10, tzinfo=timezone.utc).timestamp()
    mdm._ohlc[key].append(_custom_bar(epoch, source="historical", close=111.0))
    mdm._ohlc[key].append(
        _custom_bar(
            datetime(2026, 1, 1, 9, 15, 55),
            source="ws_candle",
            close=222.0,
            synthetic=True,
            timestamp_quality="exchange_timestamp",
        )
    )

    _assert_single_live_winner(mdm, symbol)


def test_invalid_timestamp_is_rejected_deterministically() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    mdm._ohlc[key].append(
        _custom_bar("not-a-timestamp", source="historical", close=111.0)
    )
    mdm._ohlc[key].append(
        _custom_bar(
            datetime(2026, 1, 1, 9, 15),
            source="ws_candle",
            close=222.0,
            synthetic=True,
            timestamp_quality="exchange_timestamp",
        )
    )

    bars = mdm.get_ohlc_bars(symbol)

    assert len(bars) == 1
    assert bars[0]["source"] == "ws_candle"
    assert bars[0]["timestamp_quality"] == "exchange_timestamp"


def test_canonical_and_raw_key_merge_order_is_deterministic_and_canonical_authoritative() -> (
    None
):
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    raw_symbol = "  NFO:NIFTY26JUN24000CE  "
    canonical_key = mdm._bar_symbol_key(symbol)
    mdm._ohlc[canonical_key].append(
        _custom_bar(
            datetime(2026, 1, 1, 9, 15),
            source="historical",
            close=111.0,
            synthetic=False,
            timestamp_quality="canonical_history",
        )
    )
    mdm._ohlc[raw_symbol].append(
        _custom_bar(
            datetime(2026, 1, 1, 9, 15),
            source="historical",
            close=999.0,
            synthetic=True,
            timestamp_quality="legacy_raw_history",
        )
    )

    first = mdm.get_ohlc_bars(raw_symbol)
    second = mdm.get_ohlc_bars(raw_symbol)

    assert first == second
    assert len(first) == 1
    assert first[0]["close"] == 111.0
    assert first[0]["timestamp_quality"] == "canonical_history"
    assert first[0]["synthetic"] is False


def test_repeated_canonical_getter_calls_keep_same_live_winner_metadata() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    mdm._ohlc[key].append(
        _custom_bar(
            datetime(2026, 1, 1, 9, 15),
            source="historical",
            close=111.0,
            synthetic=False,
            timestamp_quality="broker_history",
        )
    )
    mdm._ohlc[key].append(
        _custom_bar(
            datetime(2026, 1, 1, 9, 15, 30),
            source="ws_candle",
            close=222.0,
            synthetic=True,
            timestamp_quality="exchange_timestamp",
            provisional=False,
        )
    )

    first = mdm.get_ohlc_bars(symbol)
    second = mdm.get_ohlc_bars(symbol)

    assert first == second
    assert len(first) == 1
    assert first[0]["source"] == "ws_candle"
    assert first[0]["synthetic"] is True
    assert first[0]["timestamp_quality"] == "exchange_timestamp"
    assert first[0]["provisional"] is False
