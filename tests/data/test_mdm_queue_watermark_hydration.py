from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager

SYMBOL = "NSE:NIFTY"
OTHER_SYMBOL = "NSE:BANKNIFTY"


def _tick(identifier: str, symbol: str, minute: int | str) -> dict:
    timestamp = (
        minute
        if isinstance(minute, str)
        else datetime(2026, 1, 1, 9, minute, tzinfo=timezone.utc)
    )
    return {"id": identifier, "symbol": symbol, "timestamp": timestamp}


def _storage_mdm() -> MarketDataManager:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    mdm._lock = None
    mdm._cache_len = 100
    mdm._ohlc = defaultdict(lambda: deque(maxlen=100))
    mdm._engines = {}
    mdm._candle_metrics = defaultdict(float)
    mdm._candle_queue_watermarks = {}
    mdm._last_history_import_result = None
    mdm._last_historical_ts = {}
    mdm._tick_queue = asyncio.Queue(maxsize=100)
    mdm._bar_symbol_key = lambda s: str(s)
    mdm._canonical_symbol = lambda s: str(s)
    return mdm


async def _run_consumer_until_join(mdm: MarketDataManager) -> None:
    consumer = asyncio.create_task(mdm._consume_ticks())
    await asyncio.wait_for(mdm._tick_queue.join(), timeout=1.0)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer


@pytest.mark.asyncio
async def test_consumer_discards_stale_ticks_and_preserves_retained_order() -> None:
    mdm = _storage_mdm()
    mdm._candle_queue_watermarks[SYMBOL] = pd.Timestamp(
        datetime(2026, 1, 1, 9, 16, tzinfo=timezone.utc)
    )
    processed: list[str] = []
    mdm._process_queued_tick = lambda raw: processed.append(raw["id"])

    for tick in [
        _tick("stale", SYMBOL, 15),
        _tick("equal", SYMBOL, 16),
        _tick("newer-1", SYMBOL, 17),
        _tick("newer-2", SYMBOL, 18),
    ]:
        mdm._tick_queue.put_nowait(tick)

    assert mdm._tick_queue._unfinished_tasks == mdm._tick_queue.qsize()
    await _run_consumer_until_join(mdm)

    assert processed == ["newer-1", "newer-2"]
    assert mdm._candle_metrics["queued_ticks_purged_after_hydration"] == 2
    assert mdm._tick_queue._unfinished_tasks == mdm._tick_queue.qsize() == 0


@pytest.mark.asyncio
async def test_queue_join_completes_after_retained_ticks_are_consumed() -> None:
    mdm = _storage_mdm()
    mdm._candle_queue_watermarks[SYMBOL] = pd.Timestamp(
        datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)
    )
    processed: list[str] = []
    mdm._process_queued_tick = lambda raw: processed.append(raw["id"])

    for minute in (16, 17, 18):
        mdm._tick_queue.put_nowait(_tick(f"newer-{minute}", SYMBOL, minute))

    await _run_consumer_until_join(mdm)

    assert processed == ["newer-16", "newer-17", "newer-18"]
    assert mdm._tick_queue._unfinished_tasks == 0


@pytest.mark.asyncio
async def test_concurrent_producer_has_no_tick_loss_or_reorder() -> None:
    mdm = _storage_mdm()
    mdm._candle_queue_watermarks[SYMBOL] = pd.Timestamp(
        datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)
    )
    processed: list[str] = []
    mdm._process_queued_tick = lambda raw: processed.append(raw["id"])

    mdm._tick_queue.put_nowait(_tick("same-symbol-new", SYMBOL, 16))
    mdm._tick_queue.put_nowait(_tick("other-stale-minute", OTHER_SYMBOL, 14))

    consumer = asyncio.create_task(mdm._consume_ticks())
    await asyncio.sleep(0)
    mdm._tick_queue.put_nowait(_tick("producer-new", SYMBOL, 17))
    await asyncio.wait_for(mdm._tick_queue.join(), timeout=1.0)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert processed == ["same-symbol-new", "other-stale-minute", "producer-new"]
    assert mdm._tick_queue._unfinished_tasks == mdm._tick_queue.qsize() == 0


@pytest.mark.asyncio
async def test_invalid_timestamp_tick_follows_consumer_policy_not_watermark_purge() -> (
    None
):
    mdm = _storage_mdm()
    mdm._candle_queue_watermarks[SYMBOL] = pd.Timestamp(
        datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)
    )
    processed: list[str] = []
    mdm._process_queued_tick = lambda raw: processed.append(raw["id"])
    mdm._tick_queue.put_nowait(_tick("invalid-ts", SYMBOL, "not-a-timestamp"))

    await _run_consumer_until_join(mdm)

    assert processed == ["invalid-ts"]
    assert mdm._candle_metrics["queued_ticks_purged_after_hydration"] == 0
