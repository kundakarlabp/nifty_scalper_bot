from __future__ import annotations

import asyncio
import threading
import time

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
            mdm._enqueue_tick_threadsafe(
                {
                    "instrument_token": 1,
                    "last_price": i,
                    "timestamp": i,
                }
            )

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
        mdm._enqueue_tick_threadsafe(
            {
                "instrument_token": 1,
                "last_price": i,
                "timestamp": i,
            }
        )
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
    mdm.set_active_contract_basket(
        {
            "all_tokens": list(mapping),
            "token_by_symbol": {symbol: token for token, symbol in mapping.items()},
            "spot_symbol": "NSE:NIFTY",
            "spot_token": 3,
            "futures_symbol": "NFO:NIFTY26JUNFUT",
            "selected_ce": "NFO:NIFTY26JUN24000CE",
            "selected_pe": "NFO:NIFTY26JUN24000PE",
            "option_symbols": ["NFO:NIFTY26JUN24000CE", "NFO:NIFTY26JUN24000PE"],
        }
    )
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


def test_old_basket_symbol_removed_from_watchdog_critical_set():
    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    old = "NFO:NIFTY26JUN25000CE"
    mdm._active_subscribed_symbols.add(old)
    mdm._tracked_symbols.add(old)
    mdm._symbols_with_tick.add(old)
    mdm._last_valid_live_tick_mono[old] = 1.0

    mdm.reconcile_active_subscriptions({1, 2, 3, 4})

    assert old not in mdm._required_live_symbols()
    assert old not in mdm._active_subscribed_symbols
    assert old not in mdm._tracked_symbols
    assert old not in mdm._symbols_with_tick


def test_open_position_symbol_remains_required_after_basket_drift():
    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    old = "NFO:NIFTY26JUN25000CE"
    mdm.set_open_position_symbols([old])
    mdm._active_subscribed_symbols.add(old)
    mdm._tracked_symbols.add(old)
    mdm._symbols_with_tick.add(old)

    mdm.reconcile_active_subscriptions({1, 2, 3, 4})

    assert old in mdm._required_live_symbols()
    assert old in mdm._active_subscribed_symbols
    assert old in mdm._tracked_symbols


def test_one_stale_noncritical_option_does_not_restart_full_ws(monkeypatch):
    from nifty_scalper_bot.utils import market_hours

    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    stale = "NFO:NIFTY26JUN24000CE"
    now = time.monotonic()
    mdm._symbols_with_tick.update(mdm._required_live_symbols())
    for sym in mdm._required_live_symbols():
        mdm._last_valid_live_tick_mono[sym] = now
    mdm._last_valid_live_tick_mono[stale] = now - 120.0
    mdm._zombie_tick_threshold_sec = 60.0
    mdm._last_hb_mono = now
    rest_requests = []
    monkeypatch.setattr(market_hours, "is_market_open", lambda: True)
    monkeypatch.setattr(mdm, "_is_ws_healthy", lambda: True)
    monkeypatch.setattr(mdm, "_monitor_spot_ws_health", lambda: None)
    monkeypatch.setattr(
        mdm,
        "request_fallback_refresh",
        lambda symbol, reason: rest_requests.append((symbol, reason)),
    )
    monkeypatch.setattr(
        mdm,
        "_trigger_zombie_ws_restart",
        lambda: pytest.fail("global restart should be suppressed"),
    )

    mdm._check_zombie_ticks()

    assert rest_requests == [(stale, "ws_symbol_stale_recovery")]


def test_spot_and_futures_both_stale_may_trigger_full_restart(monkeypatch):
    from nifty_scalper_bot.utils import market_hours

    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    now = time.monotonic()
    required = mdm._required_live_symbols()
    mdm._symbols_with_tick.update(required)
    for sym in required:
        mdm._last_valid_live_tick_mono[sym] = now
    mdm._last_valid_live_tick_mono["NSE:NIFTY"] = now - 120.0
    mdm._last_valid_live_tick_mono["NFO:NIFTY26JUNFUT"] = now - 120.0
    mdm._zombie_tick_threshold_sec = 60.0
    mdm._last_hb_mono = now
    restarted = []
    monkeypatch.setattr(market_hours, "is_market_open", lambda: True)
    monkeypatch.setattr(mdm, "_is_ws_healthy", lambda: True)
    monkeypatch.setattr(mdm, "_monitor_spot_ws_health", lambda: None)
    monkeypatch.setattr(
        mdm, "_trigger_zombie_ws_restart", lambda: restarted.append(True)
    )

    mdm._check_zombie_ticks()

    assert restarted == [True]


def test_rest_fallback_does_not_mark_ws_tick_freshness():
    mdm = MarketDataManager(kite=None)
    symbol = "NFO:NIFTY26JUN24000CE"
    mdm._ingest_normalized_tick(
        {"symbol": symbol, "ltp": 10.0, "timestamp": time.time(), "source": "rest"}
    )

    assert mdm.time_since_last_any_tick(symbol) is not None
    assert mdm.time_since_last_live_ws_tick(symbol) is None


def test_valid_ws_tick_updates_canonical_live_timestamp():
    mdm = MarketDataManager(kite=None)
    symbol = "NFO:NIFTY26JUN24000CE"
    mdm._ingest_normalized_tick(
        {"symbol": symbol, "ltp": 10.0, "timestamp": time.time(), "source": "ws"}
    )

    assert mdm.time_since_last_live_ws_tick(symbol) is not None
    assert mdm.time_since_last_live_ws_tick(symbol) < 1.0


def test_time_since_last_live_ws_tick_returns_none_without_ws_tick():
    mdm = MarketDataManager(kite=None)
    assert mdm.time_since_last_live_ws_tick("NFO:NIFTY26JUN24000CE") is None


@pytest.mark.asyncio
async def test_stale_scheduled_drain_with_done_task_is_repaired_once(monkeypatch):
    mdm = _make_mdm()
    loop = asyncio.get_running_loop()
    mdm.set_event_loop(loop)
    done_task = loop.create_task(asyncio.sleep(0))
    await done_task
    mdm._tick_drain_task = done_task
    mdm._tick_drain_scheduled = True
    mdm._tick_drain_callbacks_scheduled = 2
    mdm._tick_drain_callbacks_completed = 1
    mdm._pending_far_ticks["NFO:NIFTY26JUN24000CE"] = {
        "symbol": "NFO:NIFTY26JUN24000CE",
        "last_price": 1,
        "timestamp": 1,
    }
    mdm._schedule_tick_drain_locked(loop)
    mdm._schedule_tick_drain_locked(loop)
    await asyncio.sleep(0)

    assert mdm.get_tick_pressure_stats()["callbacks_scheduled"] == 3
    await _stop_mdm(mdm)


@pytest.mark.asyncio
async def test_cancelled_scheduled_drain_with_pending_is_repaired_once():
    mdm = _make_mdm()
    loop = asyncio.get_running_loop()
    mdm.set_event_loop(loop)
    cancelled_task = loop.create_task(asyncio.sleep(1))
    cancelled_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_task
    mdm._tick_drain_task = cancelled_task
    mdm._tick_drain_scheduled = True
    mdm._pending_far_ticks["NFO:NIFTY26JUN24000CE"] = {
        "symbol": "NFO:NIFTY26JUN24000CE",
        "last_price": 1,
        "timestamp": 1,
    }
    mdm._schedule_tick_drain_locked(loop)
    mdm._schedule_tick_drain_locked(loop)
    await asyncio.sleep(0)

    assert mdm.get_tick_pressure_stats()["callbacks_scheduled"] == 1
    await _stop_mdm(mdm)


@pytest.mark.asyncio
async def test_transient_scheduled_active_zero_with_live_task_not_rescheduled():
    mdm = _make_mdm()
    loop = asyncio.get_running_loop()
    mdm.set_event_loop(loop)
    live_task = loop.create_task(asyncio.sleep(0.05))
    mdm._tick_drain_task = live_task
    mdm._tick_drain_scheduled = True
    mdm._pending_far_ticks["NFO:NIFTY26JUN24000CE"] = {
        "symbol": "NFO:NIFTY26JUN24000CE",
        "last_price": 1,
        "timestamp": 1,
    }
    mdm._schedule_tick_drain_locked(loop)

    assert mdm.get_tick_pressure_stats()["callbacks_scheduled"] == 0
    live_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await live_task
    await _stop_mdm(mdm)


def test_required_symbol_without_first_ws_tick_gets_symbol_recovery(monkeypatch):
    from nifty_scalper_bot.utils import market_hours

    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    missing = "NFO:NIFTY26JUN24000CE"
    now = time.monotonic()
    required = mdm._required_live_symbols()
    for sym in required - {missing}:
        mdm._last_valid_live_tick_mono[sym] = now
    mdm._required_symbol_since_mono[missing] = now - 120.0
    mdm._required_symbol_missing_grace_sec = 1.0
    mdm._zombie_tick_threshold_sec = 60.0
    mdm._last_hb_mono = now
    rest_requests = []
    monkeypatch.setattr(market_hours, "is_market_open", lambda: True)
    monkeypatch.setattr(mdm, "_is_ws_healthy", lambda: True)
    monkeypatch.setattr(mdm, "_monitor_spot_ws_health", lambda: None)
    monkeypatch.setattr(
        mdm,
        "request_fallback_refresh",
        lambda symbol, reason: rest_requests.append((symbol, reason)),
    )
    monkeypatch.setattr(
        mdm,
        "_trigger_zombie_ws_restart",
        lambda: pytest.fail("missing one option must not restart globally"),
    )

    mdm._check_zombie_ticks()

    assert rest_requests == [(missing, "ws_symbol_stale_recovery")]


def test_two_stale_options_with_fresh_context_do_not_restart_full_ws(monkeypatch):
    from nifty_scalper_bot.utils import market_hours

    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    now = time.monotonic()
    required = mdm._required_live_symbols()
    for sym in required:
        mdm._last_valid_live_tick_mono[sym] = now
    mdm._last_valid_live_tick_mono["NFO:NIFTY26JUN24000CE"] = now - 120.0
    mdm._last_valid_live_tick_mono["NFO:NIFTY26JUN24000PE"] = now - 120.0
    mdm._zombie_tick_threshold_sec = 60.0
    mdm._last_hb_mono = now
    rest_requests = []
    monkeypatch.setattr(market_hours, "is_market_open", lambda: True)
    monkeypatch.setattr(mdm, "_is_ws_healthy", lambda: True)
    monkeypatch.setattr(mdm, "_monitor_spot_ws_health", lambda: None)
    monkeypatch.setattr(
        mdm,
        "request_fallback_refresh",
        lambda symbol, reason: rest_requests.append((symbol, reason)),
    )
    monkeypatch.setattr(
        mdm,
        "_trigger_zombie_ws_restart",
        lambda: pytest.fail("two stale options alone must not restart globally"),
    )

    mdm._check_zombie_ticks()

    assert sorted(symbol for symbol, _reason in rest_requests) == [
        "NFO:NIFTY26JUN24000CE",
        "NFO:NIFTY26JUN24000PE",
    ]


def test_record_ws_arrival_fast_does_not_mark_live_ws_freshness():
    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    symbol = "NFO:NIFTY26JUN24000CE"

    mdm._record_ws_arrival_fast(
        symbol=symbol,
        token=1,
        ltp=100.0,
        raw_tick={"instrument_token": 1, "last_price": 100.0, "timestamp": time.time()},
    )

    assert mdm.time_since_last_any_tick(symbol) is not None
    assert mdm.time_since_last_live_ws_tick(symbol) is None


def test_active_bracket_symbol_remains_required_before_position_reconcile():
    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    bracket_symbol = "NFO:NIFTY26JUN25000CE"

    mdm.set_active_bracket_symbols([bracket_symbol])

    assert bracket_symbol in mdm._required_live_symbols()


def test_unhealthy_ws_still_diagnoses_and_restarts_for_stale_context(monkeypatch):
    from nifty_scalper_bot.utils import market_hours

    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    now = time.monotonic()
    required = mdm._required_live_symbols()
    for sym in required:
        mdm._last_valid_live_tick_mono[sym] = now
    mdm._last_valid_live_tick_mono["NSE:NIFTY"] = now - 120.0
    mdm._last_valid_live_tick_mono["NFO:NIFTY26JUNFUT"] = now - 120.0
    mdm._zombie_tick_threshold_sec = 60.0
    mdm._last_hb_mono = now
    restarted = []
    monkeypatch.setattr(market_hours, "is_market_open", lambda: True)
    monkeypatch.setattr(mdm, "_is_ws_healthy", lambda: False)
    monkeypatch.setattr(mdm, "_monitor_spot_ws_health", lambda: None)
    monkeypatch.setattr(
        mdm, "_trigger_zombie_ws_restart", lambda: restarted.append(True)
    )

    mdm._check_zombie_ticks()

    assert restarted == [True]


def test_subscription_divergence_needs_grace_and_symbol_recovery(monkeypatch):
    from nifty_scalper_bot.utils import market_hours

    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    now = time.monotonic()
    required = mdm._required_live_symbols()
    for sym in required:
        mdm._last_valid_live_tick_mono[sym] = now
    stale = "NFO:NIFTY26JUN24000CE"
    mdm._last_valid_live_tick_mono[stale] = now - 120.0
    mdm._desired_tokens = {1, 2, 3, 4}
    mdm._confirmed_subscriptions = {2, 3, 4}
    mdm._subscription_divergence_since_mono = now - 120.0
    mdm._required_symbol_missing_grace_sec = 1.0
    mdm._zombie_tick_threshold_sec = 60.0
    mdm._last_hb_mono = now
    rest_requests = []
    restarted = []
    monkeypatch.setattr(market_hours, "is_market_open", lambda: True)
    monkeypatch.setattr(mdm, "_is_ws_healthy", lambda: True)
    monkeypatch.setattr(mdm, "_monitor_spot_ws_health", lambda: None)
    monkeypatch.setattr(
        mdm,
        "request_fallback_refresh",
        lambda symbol, reason: rest_requests.append((symbol, reason)),
    )
    monkeypatch.setattr(
        mdm, "_trigger_zombie_ws_restart", lambda: restarted.append(True)
    )

    mdm._check_zombie_ticks()

    assert rest_requests == [(stale, "ws_symbol_stale_recovery")]
    assert restarted == []

    mdm._last_valid_live_tick_mono[stale] = now - 120.0
    mdm._last_symbol_level_recovery_mono = now
    mdm._check_zombie_ticks()

    assert restarted == [True]
