from __future__ import annotations

import asyncio
import threading

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _make_mdm():
    mdm = MarketDataManager(kite=None)
    mdm._symbol_by_token[1] = "NFO:NIFTY26JUN24000CE"
    mdm._symbol_to_token["NFO:NIFTY26JUN24000CE"] = 1
    return mdm


async def _stop_mdm(mdm):
    await mdm.stop_tick_drain(timeout=1.0)
    task = getattr(mdm, "_tick_consumer_task", None)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        mdm._tick_consumer_task = None


@pytest.mark.asyncio
async def test_threadsafe_tick_ingress_coalesces_and_loop_runs(monkeypatch):
    mdm = _make_mdm()
    processed = []
    monkeypatch.setattr(
        mdm, "_process_queued_tick", lambda raw: processed.append(dict(raw))
    )
    loop = asyncio.get_running_loop()
    mdm.set_event_loop(loop)

    def submit():
        for i in range(50_000):
            mdm._enqueue_tick_threadsafe({
                "instrument_token": 1,
                "last_price": i,
                "timestamp": i,
            })

    thread = threading.Thread(target=submit)
    thread.start()
    unrelated_ticks = 0
    while thread.is_alive():
        unrelated_ticks += 1
        await asyncio.sleep(0)
    thread.join()
    await asyncio.sleep(0.05)
    stats = mdm.get_tick_pressure_stats()
    assert stats["drain_callbacks_scheduled"] < 1_000
    assert stats["coalesced_total"] > 0
    assert stats["pending_max_seen"] < 50_000
    assert unrelated_ticks > 0
    assert processed
    assert processed[-1]["last_price"] == 49_999
    await _stop_mdm(mdm)


@pytest.mark.asyncio
async def test_selected_ticks_span_multiple_budget_batches_without_loss(monkeypatch):
    mdm = _make_mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    mdm.set_open_position_symbols([symbol])
    processed = []
    monkeypatch.setattr(mdm, "_tick_drain_batch_size", 3)
    monkeypatch.setattr(mdm, "_tick_drain_budget_s", 0.000001)
    monkeypatch.setattr(
        mdm, "_process_queued_tick", lambda raw: processed.append(raw["last_price"])
    )
    loop = asyncio.get_running_loop()
    mdm.set_event_loop(loop)
    for i in range(10):
        mdm._enqueue_tick_threadsafe({
            "instrument_token": 1,
            "last_price": i,
            "timestamp": i,
        })
    await mdm.drain_pending_ticks(timeout=2.0)
    assert processed == list(range(10))
    stats = mdm.get_tick_pressure_stats()
    assert stats["unexplained_loss"] == 0
    assert stats["max_active_drains"] == 1
    await _stop_mdm(mdm)


@pytest.mark.asyncio
async def test_direct_push_tick_uses_bounded_drain(monkeypatch):
    mdm = _make_mdm()
    processed = []
    monkeypatch.setattr(
        mdm, "_process_queued_tick", lambda raw: processed.append(raw["last_price"])
    )
    mdm.set_event_loop(asyncio.get_running_loop())
    await mdm.push_tick({"instrument_token": 1, "last_price": 10, "timestamp": 1})
    await mdm.drain_pending_ticks(timeout=2.0)
    assert processed == [10]
    await _stop_mdm(mdm)


@pytest.mark.asyncio
async def test_malformed_ticks_are_bounded_and_accounted(monkeypatch):
    mdm = _make_mdm()
    mdm.set_event_loop(asyncio.get_running_loop())
    for _ in range(100):
        mdm._enqueue_tick_threadsafe({"last_price": 1})
    await asyncio.sleep(0.01)
    stats = mdm.get_tick_pressure_stats()
    assert stats["pending_ticks"] == 0
    assert stats["dropped_by_reason"]["missing_symbol_token"] == 100
    assert stats["unexplained_loss"] == 0
    await _stop_mdm(mdm)


def _wire_symbols(mdm):
    mapping = {
        1: "NFO:NIFTY26JUN24000CE",
        2: "NFO:NIFTY26JUN24000PE",
        3: "NSE:NIFTY",
        4: "NFO:NIFTY26JUNFUT",
        5: "NFO:NIFTY26JUN25000CE",
    }
    for token, symbol in mapping.items():
        mdm._symbol_by_token[token] = symbol
        mdm._token_to_symbol[token] = symbol
        mdm._symbol_to_token[symbol] = token
        mdm._token_by_symbol[symbol] = token
    mdm.set_active_contract_basket({
        "all_tokens": list(mapping),
        "token_by_symbol": {symbol: token for token, symbol in mapping.items()},
        "spot_symbol": "NSE:NIFTY",
        "spot_token": 3,
        "futures_symbol": "NFO:NIFTY26JUNFUT",
        "selected_ce": "NFO:NIFTY26JUN24000CE",
        "selected_pe": "NFO:NIFTY26JUN24000PE",
        "option_symbols": ["NFO:NIFTY26JUN24000CE", "NFO:NIFTY26JUN24000PE"],
    })
    return mapping


def _ws_tick(
    token: int,
    price: float,
    second: int,
    volume: int | None = None,
    *,
    bid: float | None = None,
    ask: float | None = None,
):
    tick = {
        "instrument_token": token,
        "last_price": price,
        "timestamp": f"2026-06-30T09:15:{second:02d}+00:00",
        "exchange_timestamp": f"2026-06-30T09:15:{second:02d}+00:00",
    }
    if volume is not None:
        tick["volume_traded_today"] = volume
    if bid is not None and ask is not None:
        tick["depth"] = {
            "buy": [{"price": bid, "quantity": 100}],
            "sell": [{"price": ask, "quantity": 100}],
        }
    return tick


@pytest.mark.asyncio
async def test_spot_and_future_protected_context_under_backlog(monkeypatch):
    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    monkeypatch.setattr(mdm, "_tick_drain_batch_size", 2)
    monkeypatch.setattr(mdm, "_tick_drain_budget_s", 0.000001)
    mdm.set_event_loop(asyncio.get_running_loop())

    assert mdm._tick_priority("NSE:NIFTY") == (2, "spot_future_context")
    assert mdm._tick_priority("NFO:NIFTY26JUNFUT") == (2, "spot_future_context")

    for tick in [
        _ws_tick(3, 24000, 1),
        _ws_tick(3, 24020, 2),
        _ws_tick(3, 23990, 3),
        _ws_tick(3, 24010, 4),
        _ws_tick(4, 24100, 1, 1000),
        _ws_tick(4, 24130, 2, 1050),
        _ws_tick(4, 24080, 3, 1075),
        _ws_tick(4, 24120, 4, 1100),
    ]:
        mdm._enqueue_tick_threadsafe(tick)
    await mdm.drain_pending_ticks(timeout=3.0)

    spot = mdm._engines["NSE:NIFTY"].current_candle
    fut = mdm._engines["NFO:NIFTY26JUNFUT"].current_candle
    assert spot["open"] == 24000
    assert spot["high"] == 24020
    assert spot["low"] == 23990
    assert spot["close"] == 24010
    assert fut["open"] == 24100
    assert fut["high"] == 24130
    assert fut["low"] == 24080
    assert fut["close"] == 24120
    assert fut["volume"] == 100
    latest = mdm.get_latest_tick("NFO:NIFTY26JUNFUT")
    assert latest["volume_cumulative"] == 1100
    stats = mdm.get_tick_pressure_stats()
    assert stats["unexplained_loss"] == 0
    assert stats["max_active_drains"] == 1
    await _stop_mdm(mdm)


@pytest.mark.asyncio
async def test_open_position_stop_loss_crossing_tick_is_not_silently_lost(monkeypatch):
    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    symbol = "NFO:NIFTY26JUN24000CE"
    mdm.set_open_position_symbols([symbol])
    monkeypatch.setattr(mdm, "_tick_drain_batch_size", 2)
    monkeypatch.setattr(mdm, "_tick_drain_budget_s", 0.000001)
    seen_prices: list[float] = []
    mdm.subscribe(symbol, lambda tick: seen_prices.append(float(tick["ltp"])))
    mdm.set_event_loop(asyncio.get_running_loop())

    for i, price in enumerate([101.0, 99.0, 94.5, 98.0], start=1):
        mdm._enqueue_tick_threadsafe(
            _ws_tick(1, price, i, bid=price - 0.1, ask=price + 0.1)
        )
    await mdm.drain_pending_ticks(timeout=3.0)

    assert 94.5 in seen_prices
    assert min(seen_prices) == 94.5
    stats = mdm.get_tick_pressure_stats()
    assert stats["unexplained_loss"] == 0
    assert stats["max_active_drains"] == 1
    await _stop_mdm(mdm)
