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


def _set_pending_far(mdm, symbol="NFO:NIFTY26JUN24000CE", *, enqueued_mono=None):
    tick = {"symbol": symbol, "last_price": 1, "timestamp": 1}
    if enqueued_mono is not None:
        tick["_mdm_enqueued_mono"] = enqueued_mono
    mdm._pending_far_ticks[symbol] = tick
    mdm._pending_tick_count = max(0, int(getattr(mdm, "_pending_tick_count", 0))) + 1
    if enqueued_mono is not None:
        mdm._pending_heap_push_locked(tick, symbol)
    return tick


def _set_pending_queue(mdm, symbol, tick):
    mdm._pending_tick_queues[symbol].append(tick)
    mdm._pending_tick_count = max(0, int(getattr(mdm, "_pending_tick_count", 0))) + 1
    mdm._pending_heap_push_locked(tick, symbol)


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


def test_spot_and_futures_stale_use_symbol_recovery_when_transport_healthy(monkeypatch):
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
    rest_requests: list[tuple[str, str]] = []
    monkeypatch.setattr(market_hours, "is_market_open", lambda: True)
    monkeypatch.setattr(mdm, "_is_ws_healthy", lambda: True)
    monkeypatch.setattr(mdm, "_monitor_spot_ws_health", lambda: None)
    monkeypatch.setattr(
        mdm,
        "request_fallback_refresh",
        lambda symbol, reason: rest_requests.append((symbol, reason)) or True,
    )
    monkeypatch.setattr(
        mdm,
        "_trigger_zombie_ws_restart",
        lambda: pytest.fail("healthy transport must not restart globally"),
    )

    mdm._check_zombie_ticks()

    assert {symbol for symbol, _ in rest_requests} == {
        "NSE:NIFTY",
        "NFO:NIFTY26JUNFUT",
    }


def test_heartbeat_stale_triggers_single_global_restart(monkeypatch):
    from nifty_scalper_bot.utils import market_hours

    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    now = time.monotonic()
    required = mdm._required_live_symbols()
    mdm._symbols_with_tick.update(required)
    for sym in required:
        mdm._last_valid_live_tick_mono[sym] = now - 120.0
    mdm._zombie_tick_threshold_sec = 60.0
    mdm._last_hb_mono = now - 120.0
    restarted: list[bool] = []
    monkeypatch.setattr(market_hours, "is_market_open", lambda: True)
    monkeypatch.setattr(mdm, "_is_ws_healthy", lambda: False)
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
    token = 24000
    assert mdm.request_token_subscription(token, symbol=symbol)
    mdm._ingest_normalized_tick(
        {
            "symbol": symbol,
            "instrument_token": token,
            "ltp": 10.0,
            "timestamp": time.time(),
            "source": "ws",
        }
    )

    assert mdm.time_since_last_live_ws_tick(symbol) is not None
    assert mdm.time_since_last_live_ws_tick(symbol) < 1.0


def test_time_since_last_live_ws_tick_returns_none_without_ws_tick():
    mdm = MarketDataManager(kite=None)
    assert mdm.time_since_last_live_ws_tick("NFO:NIFTY26JUN24000CE") is None


def test_required_symbol_without_current_generation_tick_is_not_fresh(monkeypatch):
    from nifty_scalper_bot.utils import market_hours

    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    missing = "NFO:NIFTY26JUN24000CE"
    now = time.monotonic()
    required = mdm._required_live_symbols()
    for sym in required - {missing}:
        mdm._last_valid_live_tick_mono[sym] = now
    mdm._desired_tokens.add(1)
    mdm._dispatched_subscriptions.add(1)
    mdm._confirmed_subscriptions.add(1)
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
        lambda: pytest.fail("never-ticked one option must not restart globally"),
    )

    readiness = mdm.classify_live_tick_readiness(missing, 1, max_age_s=60.0)
    assert mdm.time_since_last_live_ws_tick(missing) is None
    assert readiness["tick_age_s"] is None
    assert readiness["current_generation_tick_received"] is False
    assert readiness["current_generation_tick_received"] is False
    assert readiness["reason"] == "current_generation_tick_pending"

    mdm._check_zombie_ticks()

    assert rest_requests == [(missing, "ws_symbol_stale_recovery")]


def test_basket_reentry_same_token_requires_new_current_generation_tick():
    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    symbol = "NFO:NIFTY26JUN24000CE"
    token = 1
    mdm._begin_subscription_generation_locked(
        symbol, token, reason="test_initial_entry"
    )
    mdm._desired_tokens.add(token)
    mdm._dispatched_subscriptions.add(token)
    mdm._confirmed_subscriptions.add(token)
    mdm._ingest_normalized_tick(
        {
            "symbol": symbol,
            "instrument_token": token,
            "ltp": 10.0,
            "timestamp": time.time(),
            "source": "ws",
        }
    )
    old_generation = mdm._symbol_subscription_generation[symbol]
    assert mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)["ready"]

    mdm.set_active_contract_basket(
        {
            "all_tokens": [2, 3, 4],
            "token_by_symbol": {
                "NFO:NIFTY26JUN24000PE": 2,
                "NSE:NIFTY": 3,
                "NFO:NIFTY26JUNFUT": 4,
            },
            "spot_symbol": "NSE:NIFTY",
            "spot_token": 3,
            "futures_symbol": "NFO:NIFTY26JUNFUT",
            "selected_pe": "NFO:NIFTY26JUN24000PE",
            "option_symbols": ["NFO:NIFTY26JUN24000PE"],
        }
    )
    mdm.reconcile_active_subscriptions({2, 3, 4})
    assert symbol not in mdm._required_live_symbols()
    assert mdm.time_since_last_live_ws_tick(symbol) is None

    assert mdm.request_token_subscription(token, symbol=symbol)
    mdm._dispatched_subscriptions.add(token)
    mdm._confirmed_subscriptions.add(token)
    reentered = mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)
    assert mdm._symbol_subscription_generation[symbol] > old_generation
    assert reentered["ready"] is False
    assert reentered["tick_age_s"] is None
    assert reentered["current_generation_tick_received"] is False
    assert reentered["reason"] == "current_generation_tick_pending"

    mdm._symbol_first_tick_generation[symbol] = old_generation
    mdm._last_valid_live_tick_mono[symbol] = time.monotonic()
    old_tick = mdm.classify_live_tick_readiness(symbol, token, max_age_s=60.0)
    assert old_tick["ready"] is False
    assert old_tick["reason"] == "subscription_generation_mismatch"


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
    _set_pending_far(mdm)
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
    _set_pending_far(mdm)
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
    _set_pending_far(mdm)
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
    mdm._desired_tokens.add(1)
    mdm._dispatched_subscriptions.add(1)
    mdm._confirmed_subscriptions.add(1)
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


def test_pipeline_overload_enters_recovers_with_hysteresis() -> None:
    """PR-1 item E: canonical overload state. Enters on pending backlog OR
    oldest-pending age; recovers only when BOTH are below exit thresholds
    (hysteresis). New entries consult MarketDataManager.pipeline_overloaded
    via the runner signal-prep guard; exits never pass through that guard."""
    import collections
    import logging
    import threading
    import time as time_mod

    from nifty_scalper_bot.data.market_data_manager import MarketDataManager

    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = logging.getLogger("test")
    mdm._pending_tick_lock = threading.Lock()
    mdm._pending_tick_queues = collections.defaultdict(collections.deque)
    mdm._pending_far_ticks = {}
    mdm._overload_enter_pending = 5
    mdm._overload_exit_pending = 2
    mdm._overload_enter_oldest_ms = 2000.0
    mdm._overload_exit_oldest_ms = 800.0
    mdm._pipeline_overloaded = False
    mdm._overload_since_mono = None
    mdm._pending_tick_count = 0
    mdm._pending_tick_oldest_heap = []
    mdm._pending_tick_heap_seq = 0
    mdm._pending_heap_compactions_total = 0

    now = time_mod.monotonic()
    for i in range(6):
        _set_pending_queue(mdm, f"s{i}", {"_mdm_enqueued_mono": now})
    with mdm._pending_tick_lock:
        mdm._update_pipeline_overload_locked()
    assert mdm.pipeline_overloaded is True

    for i in range(3):  # partially drained: still above exit bound
        mdm._pending_tick_queues.pop(f"s{i}")
        mdm._pending_decrement_locked(1)
    with mdm._pending_tick_lock:
        mdm._update_pipeline_overload_locked()
    assert mdm.pipeline_overloaded is True, "hysteresis must hold"

    for i in range(3, 5):
        mdm._pending_tick_queues.pop(f"s{i}")
        mdm._pending_decrement_locked(1)
    with mdm._pending_tick_lock:
        mdm._update_pipeline_overload_locked()
    assert mdm.pipeline_overloaded is False

    # Oldest-age evidence alone triggers even with tiny backlog.
    _set_pending_queue(mdm, "x", {"_mdm_enqueued_mono": time_mod.monotonic() - 3.0})
    with mdm._pending_tick_lock:
        mdm._update_pipeline_overload_locked()
    assert mdm.pipeline_overloaded is True


def test_runner_signal_prep_blocks_on_overload_before_unarmed_guard() -> None:
    """The overload guard sits at the same choke point as the unarmed guard,
    BEFORE _schedule_signal_preparation, and exits (bracket paths) never
    reference it."""
    import inspect

    from nifty_scalper_bot.strategies.runner import StrategyRunner
    from nifty_scalper_bot.execution import bracket_core

    src = inspect.getsource(StrategyRunner)
    i_over = src.index("RUNNER_SIGNAL_PREP_BLOCKED_OVERLOAD")
    i_unarmed = src.index("RUNNER_SIGNAL_PREP_BLOCKED_UNARMED")
    i_sched = src.index("scheduled, prepare_reason = self._schedule_signal_preparation")
    assert i_over < i_unarmed < i_sched
    assert "pipeline_overloaded" not in inspect.getsource(bracket_core)


def test_tick_pressure_stats_report_pruned_o1_pending_state() -> None:
    mdm = _make_mdm()
    loop = asyncio.new_event_loop()
    try:
        mdm._enqueue_latest_tick_for_drain(
            {"instrument_token": 1, "last_price": 100, "timestamp": 1}, loop
        )
        stats = mdm.get_tick_pressure_stats()
        assert stats["pending_tick_count"] == 1
        assert stats["active_queue_count"] == 1
        assert stats["retained_empty_queue_count"] == 0
        assert stats["oldest_pending_age_ms"] >= 0
        popped = mdm._pop_pending_tick_batch()
        assert len(popped) == 1
        stats = mdm.get_tick_pressure_stats()
        assert stats["pending_tick_count"] == 0
        assert stats["active_queue_count"] == 0
        assert stats["retained_empty_queue_count"] == 0
    finally:
        loop.close()


def test_transport_classifier_distinguishes_processing_backlog_from_silence() -> None:
    import time as time_mod

    mdm = _make_mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    now = time_mod.monotonic()
    mdm._zombie_tick_threshold_sec = 10.0
    mdm._last_raw_ws_receive_mono = now
    mdm._desired_tokens.add(1)
    mdm._token_by_symbol[symbol] = 1
    mdm._last_raw_ws_receive_by_token[1] = {
        "received_mono": now,
        "subscription_generation": mdm._subscription_generation,
    }
    mdm._last_valid_live_tick_mono[symbol] = now - 30.0
    mdm._pending_tick_count = 2
    mdm._pipeline_overloaded = True

    state = mdm.classify_transport_backlog(symbol)

    assert state["transport_classification"] == "processing_backlog"
    assert state["global_restart_eligible"] is False
    assert state["pipeline_overloaded"] is True

    mdm._last_raw_ws_receive_mono = now - 30.0
    mdm._last_raw_ws_receive_by_token[1] = {
        "received_mono": now - 30.0,
        "subscription_generation": mdm._subscription_generation,
    }
    state = mdm.classify_transport_backlog(symbol)
    assert state["transport_classification"] == "transport_silent"
    assert state["global_restart_eligible"] is True


class _NoValuesDict(dict):
    def values(self):  # pragma: no cover - failure path if production scans
        raise AssertionError("pending count must not scan queue values")


def test_pending_count_zero_one_and_many_keys_do_not_scan_values() -> None:
    mdm = _make_mdm()
    mdm._pending_tick_queues = _NoValuesDict({f"s{i}": [] for i in range(10_000)})
    mdm._pending_far_ticks = {}
    mdm._pending_tick_count = 0
    assert mdm._pending_count_locked() == 0
    mdm._pending_tick_count = 1
    assert mdm._pending_count_locked() == 1


def test_pending_counter_exact_across_enqueue_coalesce_pop_requeue_and_drop() -> None:
    mdm = _make_mdm()
    loop = asyncio.new_event_loop()
    try:
        far = "NFO:NIFTY26JUN26000CE"
        mdm._symbol_by_token[2] = far
        mdm._symbol_to_token[far] = 2
        mdm._enqueue_latest_tick_for_drain(
            {"instrument_token": 2, "last_price": 1, "timestamp": 1}, loop
        )
        assert mdm._pending_count_locked() == 1
        mdm._enqueue_latest_tick_for_drain(
            {"instrument_token": 2, "last_price": 2, "timestamp": 2}, loop
        )
        assert (
            mdm._pending_count_locked() == 1
        ), "far coalesce replaces without count growth"
        popped = mdm._pop_pending_tick_batch()
        assert len(popped) == 1
        assert mdm._pending_count_locked() == 0
        mdm._requeue_unprocessed_ticks(popped)
        assert mdm._pending_count_locked() == 1
        mdm._pending_decrement_locked(99)
        assert mdm._pending_count_locked() == 0
    finally:
        loop.close()


def test_pending_heap_compacts_after_repeated_far_replacements() -> None:
    mdm = _make_mdm()
    far = "NFO:NIFTY26JUN26000CE"
    now = time.monotonic()
    with mdm._pending_tick_lock:
        mdm._pending_far_ticks[far] = {"_mdm_enqueued_mono": now, "symbol": far}
        mdm._pending_tick_count = 1
        for i in range(2_000):
            mdm._pending_tick_heap_seq += 1
            mdm._pending_tick_oldest_heap.append(
                (now - i - 1, mdm._pending_tick_heap_seq, far)
            )
        mdm._maybe_compact_pending_heap_locked()
    stats = mdm.get_tick_pressure_stats()
    assert stats["pending_tick_count"] == 1
    assert stats["pending_heap_size"] == 1
    assert stats["pending_heap_compactions_total"] > 0
    assert stats["oldest_pending_age_ms"] < 1000


def test_global_classifier_uses_required_context_not_fresh_irrelevant_option() -> None:
    now = time.monotonic()
    mdm = _make_mdm()
    mdm._zombie_tick_threshold_sec = 10.0
    mdm._last_raw_ws_receive_mono = now
    mdm._symbol_to_token["NSE:NIFTY"] = 256265
    mdm._token_by_symbol["NSE:NIFTY"] = 256265
    mdm._symbol_to_token["NFO:NIFTY26JUNFUT"] = 2
    mdm._token_by_symbol["NFO:NIFTY26JUNFUT"] = 2
    mdm._desired_tokens.update({256265, 2})
    mdm._last_raw_ws_receive_by_symbol["NSE:NIFTY"] = {
        "received_mono": now,
        "subscription_generation": mdm._subscription_generation,
    }
    mdm._last_raw_ws_receive_by_symbol["NFO:NIFTY26JUNFUT"] = {
        "received_mono": now,
        "subscription_generation": mdm._subscription_generation,
    }
    mdm._required_live_symbols = lambda: {"NSE:NIFTY", "NFO:NIFTY26JUNFUT"}
    mdm._last_valid_live_tick_mono["NSE:NIFTY"] = now - 30
    mdm._last_valid_live_tick_mono["NFO:NIFTY26JUNFUT"] = now - 30
    mdm._last_valid_live_tick_mono["NFO:IRRELEVANT26JUN26000CE"] = now
    mdm._pending_tick_count = 10
    mdm._pipeline_overloaded = True

    state = mdm.classify_transport_backlog()

    assert state["transport_classification"] == "processing_backlog"
    assert state["global_restart_eligible"] is False
    assert set(state["required_symbols_stale"]) >= {"NSE:NIFTY", "NFO:NIFTY26JUNFUT"}
    assert state["raw_transport_fresh"] is True


def test_global_classifier_requires_current_generation_raw_evidence() -> None:
    now = time.monotonic()
    mdm = _make_mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    mdm._zombie_tick_threshold_sec = 10.0
    mdm._required_live_symbols = lambda: {symbol}
    mdm._desired_tokens.add(1)
    mdm._token_by_symbol[symbol] = 1
    mdm._symbol_to_token[symbol] = 1
    mdm._last_raw_ws_receive_mono = now
    mdm._last_raw_ws_receive_by_token[1] = {
        "received_mono": now,
        "subscription_generation": 1,
    }
    mdm._last_valid_live_tick_mono[symbol] = now - 30
    mdm._symbol_subscription_generation[symbol] = 2
    mdm._symbol_first_tick_generation[symbol] = 1
    mdm._subscription_generation = 2
    mdm._pending_tick_count = 1

    state = mdm.classify_transport_backlog()

    assert state["transport_classification"] == "symbol_feed_stale"
    assert state["required_current_generation_raw_receive_fresh"] is False


def test_global_classifier_recovers_to_transport_healthy() -> None:
    now = time.monotonic()
    mdm = _make_mdm()
    symbol = "NSE:NIFTY"
    mdm._zombie_tick_threshold_sec = 10.0
    mdm._required_live_symbols = lambda: {symbol}
    mdm._last_raw_ws_receive_mono = now
    mdm._last_valid_live_tick_mono[symbol] = now
    mdm._pending_tick_count = 0
    mdm._pipeline_overloaded = False

    state = mdm.classify_transport_backlog()

    assert state["transport_classification"] == "transport_healthy"
    assert state["global_restart_eligible"] is False


def test_raw_ws_ingress_malformed_symbol_cannot_break_diagnostics(monkeypatch) -> None:
    mdm = _make_mdm()
    original = mdm._canonical_symbol

    def raising_canonical(symbol: str) -> str:
        if symbol == "%%%BAD%%%":
            raise ValueError("malformed symbol")
        return original(symbol)

    monkeypatch.setattr(mdm, "_canonical_symbol", raising_canonical)
    now = time.monotonic()

    mdm._record_raw_ws_receive(
        {"symbol": "%%%BAD%%%", "instrument_token": 1, "last_price": 100}, now
    )
    key, _priority, _bucket, reason = mdm._resolve_tick_key_and_priority(
        {"symbol": "%%%BAD%%%", "instrument_token": 1, "last_price": 100}
    )

    assert key is None
    assert reason == "bad_symbol"
    assert mdm._last_raw_ws_receive_mono == now
    assert mdm._last_raw_ws_receive_by_token[1]["received_mono"] == now
    assert "%%%BAD%%%" not in mdm._last_raw_ws_receive_by_symbol


def test_current_generation_raw_receive_requires_matching_generation() -> None:
    mdm = _make_mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    now = time.monotonic()
    mdm._desired_tokens.add(1)
    mdm._token_by_symbol[symbol] = 1
    mdm._symbol_subscription_generation[symbol] = 2
    mdm._subscription_generation = 2

    assert mdm._required_current_generation_raw_fresh([symbol], now, 10.0) is False

    mdm._last_raw_ws_receive_by_token[1] = {
        "received_mono": now,
        "subscription_generation": 1,
    }
    assert mdm._required_current_generation_raw_fresh([symbol], now, 10.0) is False

    mdm._last_raw_ws_receive_by_token[1] = {
        "received_mono": now,
        "subscription_generation": 2,
    }
    assert mdm._required_current_generation_raw_fresh([symbol], now, 10.0) is True


def test_global_classifier_does_not_treat_irrelevant_socket_traffic_as_backlog() -> (
    None
):
    now = time.monotonic()
    mdm = _make_mdm()
    mdm._zombie_tick_threshold_sec = 10.0
    mdm._required_live_symbols = lambda: {"NSE:NIFTY"}
    mdm._token_by_symbol["NSE:NIFTY"] = 256265
    mdm._symbol_to_token["NSE:NIFTY"] = 256265
    mdm._desired_tokens.add(256265)
    mdm._symbol_subscription_generation["NSE:NIFTY"] = 3
    mdm._subscription_generation = 3
    mdm._last_raw_ws_receive_mono = now
    mdm._last_raw_ws_receive_by_token[1] = {
        "received_mono": now,
        "subscription_generation": 3,
    }
    mdm._last_valid_live_tick_mono["NSE:NIFTY"] = now - 30
    mdm._last_valid_live_tick_mono["NFO:IRRELEVANT26JUN26000CE"] = now
    mdm._pending_tick_count = 10
    mdm._pipeline_overloaded = True

    state = mdm.classify_transport_backlog()

    assert state["transport_classification"] == "symbol_feed_stale"
    assert state["global_restart_eligible"] is False
    assert state["raw_transport_fresh"] is True
    assert state["required_current_generation_raw_receive_fresh"] is False


def test_pending_current_required_work_can_explain_required_backlog() -> None:
    now = time.monotonic()
    mdm = _make_mdm()
    mdm._zombie_tick_threshold_sec = 10.0
    mdm._required_live_symbols = lambda: {"NSE:NIFTY"}
    mdm._symbol_subscription_generation["NSE:NIFTY"] = 4
    mdm._subscription_generation = 4
    mdm._last_raw_ws_receive_mono = now
    mdm._last_valid_live_tick_mono["NSE:NIFTY"] = now - 30
    with mdm._pending_tick_lock:
        _set_pending_queue(
            mdm,
            "NSE:NIFTY",
            {
                "symbol": "NSE:NIFTY",
                "last_price": 100,
                "_mdm_enqueued_mono": now,
                "_mdm_subscription_generation": 4,
            },
        )
    mdm._pipeline_overloaded = True

    state = mdm.classify_transport_backlog()

    assert state["transport_classification"] == "processing_backlog"
    assert state["pending_required_current_generation"] is True
    assert state["global_restart_eligible"] is False


def test_oldest_pending_age_repairs_empty_heap_once() -> None:
    mdm = _make_mdm()
    now = time.monotonic() - 0.25
    with mdm._pending_tick_lock:
        mdm._pending_tick_queues["NSE:NIFTY"].append(
            {"symbol": "NSE:NIFTY", "_mdm_enqueued_mono": now}
        )
        mdm._pending_tick_count = 1
        mdm._pending_tick_oldest_heap.clear()
        first = mdm._oldest_pending_age_ms_locked()
        second = mdm._oldest_pending_age_ms_locked()

    assert first >= 0
    assert second >= 0
    assert mdm._pending_heap_repairs_total == 1
    assert mdm._pending_tick_oldest_heap


@pytest.mark.asyncio
async def test_high_rate_burst_preserves_heartbeat_and_single_drain(monkeypatch):
    mdm = _make_mdm()
    processed: list[int] = []
    monkeypatch.setattr(
        mdm, "_process_queued_tick", lambda raw: processed.append(raw["last_price"])
    )
    loop = asyncio.get_running_loop()
    mdm.set_event_loop(loop)
    heartbeat = 0
    running = True

    async def beat():
        nonlocal heartbeat
        while running:
            heartbeat += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(beat())

    def submit():
        for i in range(20_000):
            mdm._enqueue_tick_threadsafe(
                {"instrument_token": 1, "last_price": i, "timestamp": i}
            )

    thread = threading.Thread(target=submit)
    thread.start()
    while thread.is_alive():
        await asyncio.sleep(0)
    thread.join()
    await mdm.drain_pending_ticks(timeout=3.0)
    running = False
    await task

    stats = mdm.get_tick_pressure_stats()
    assert heartbeat > 0
    assert stats["max_active_drains"] == 1
    assert processed[-1] == 19_999
    assert stats["unexplained_loss"] == 0
    await _stop_mdm(mdm)


def test_symbol_recovery_attempts_are_cooldown_bounded(monkeypatch):
    from nifty_scalper_bot.utils import market_hours

    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    stale = "NFO:NIFTY26JUN24000CE"
    now = time.monotonic()
    required = mdm._required_live_symbols()
    mdm._symbols_with_tick.update(required)
    for sym in required:
        mdm._last_valid_live_tick_mono[sym] = now
    mdm._last_valid_live_tick_mono[stale] = now - 120.0
    mdm._zombie_tick_threshold_sec = 60.0
    mdm._last_hb_mono = now
    mdm._symbol_recovery_cooldown_s = 30.0
    requests: list[str] = []
    monkeypatch.setattr(market_hours, "is_market_open", lambda: True)
    monkeypatch.setattr(mdm, "_is_ws_healthy", lambda: True)
    monkeypatch.setattr(mdm, "_monitor_spot_ws_health", lambda: None)
    monkeypatch.setattr(
        mdm,
        "request_fallback_refresh",
        lambda symbol, reason: requests.append(symbol) or True,
    )
    monkeypatch.setattr(
        mdm,
        "_trigger_zombie_ws_restart",
        lambda: pytest.fail("healthy transport must not restart globally"),
    )

    for _ in range(20):
        mdm._check_zombie_ticks()

    assert requests == [stale]


def test_trading_feed_health_exposes_required_symbol_recovery():
    mdm = MarketDataManager(kite=None)
    _wire_symbols(mdm)
    mdm.set_readiness_requirements(
        spot_symbol="NSE:NIFTY",
        futures_symbol="NFO:NIFTY26JUNFUT",
        atm_ce_symbol="NFO:NIFTY26JUN24000CE",
        atm_pe_symbol="NFO:NIFTY26JUN24000PE",
        option_symbols=[
            "NFO:NIFTY26JUN24000CE",
            "NFO:NIFTY26JUN24000PE",
        ],
    )

    health = mdm.trading_feed_health(max_age_ms=60_000)

    assert health["required_symbol_recovery_active"] is True
    assert set(health["stale_required_symbols"]) == mdm._required_live_symbols()
    assert health["trading_feed_healthy"] is False
