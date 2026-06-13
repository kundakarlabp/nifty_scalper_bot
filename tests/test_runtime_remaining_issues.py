import asyncio
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.core.active_basket import normalize_active_basket_schema
import nifty_scalper_bot.core.app as app
from nifty_scalper_bot.core.app import _commit_active_dynamic_basket, _next_eod_flatten_time_ist
from nifty_scalper_bot.core.option_universe import OptionUniverseManager
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.execution.readiness import evaluate_quote_readiness

CE = "NFO:NIFTY26JUN25000CE"
PE = "NFO:NIFTY26JUN25000PE"


def test_quote_readiness_separates_ltp_and_blocks_unknown_age():
    ltp_only = {"ltp": 100.0, "source": "ws", "tick_age_s": 1.0}
    status = evaluate_quote_readiness(CE, ltp_only, max_spread_pct=5.0, max_age_s=5.0)
    assert status.ltp_ready is True
    assert status.bid_ask_available is False
    assert status.tradable_quote_ready is False
    assert status.reason == "bid_ask_missing"

    crossed = {"ltp": 100.0, "bid": 101.0, "ask": 100.0, "tick_age_s": 1.0}
    assert evaluate_quote_readiness(CE, crossed, max_spread_pct=5.0).reason == "bid_ask_crossed"

    unknown_age = {"ltp": 100.0, "bid": 99.9, "ask": 100.1, "source": "ws"}
    status = evaluate_quote_readiness(CE, unknown_age, max_spread_pct=5.0, max_age_s=5.0)
    assert status.reason == "quote_age_unknown"
    assert status.tradable_quote_ready is False

    ready = {"ltp": 100.0, "bid": 99.9, "ask": 100.1, "tick_age_s": 1.0, "source": "ws"}
    status = evaluate_quote_readiness(CE, ready, max_spread_pct=5.0, max_age_s=5.0)
    assert status.tradable_quote_ready is True
    assert status.reason == "ready"


def test_quote_readiness_is_shared_by_mdm_startup_and_runtime(monkeypatch):
    mdm = MarketDataManager(broker=SimpleNamespace(), settings=SimpleNamespace(history_min_interval_sec=0))
    tick = {"ltp": 100.0, "bid": 99.9, "ask": 100.1, "tick_age_s": 1.0, "source": "ws", "bid_ask_source": "ws_full"}
    monkeypatch.setattr(mdm, "get_latest_tick", lambda _s: tick)
    monkeypatch.setattr(mdm, "get_quote", lambda _s: None)
    monkeypatch.setattr(mdm, "get_ohlc_bars", lambda _s, **_k: [1] * 10)
    mdm._latest_ticks[CE] = tick
    report = mdm.hydrate_active_contract_basket({"selected_ce": CE, "selected_pe": PE, "option_symbols": [CE], "token_by_symbol": {CE: 1}, "all_symbols": [CE]})
    assert report["symbols"][CE]["tradable_quote_ready"] is True
    assert mdm.has_ws_tradable_quote([CE]) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("first_bars,second_bars,expected_calls", [(300, 30, 1), (30, 300, 2), (300, 300, 1)])
async def test_hydration_request_coalescing_orders(monkeypatch, first_bars, second_bars, expected_calls):
    mdm = MarketDataManager(broker=SimpleNamespace(), settings=SimpleNamespace(history_min_interval_sec=0))
    calls = 0
    cached = []

    async def fake_fetch(symbol, interval, days):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return [{"timestamp": datetime(2026, 6, 12, 9, 15, tzinfo=ZoneInfo("UTC")), "open": i, "high": i, "low": i, "close": i} for i in range(first_bars if calls == 1 else second_bars)]

    def ingest(_symbol, rows):
        cached[:] = list(rows)
        return len(rows)

    monkeypatch.setattr(mdm, "fetch_history", fake_fetch)
    monkeypatch.setattr(mdm, "ingest_historical_ohlc", ingest)
    monkeypatch.setattr(mdm, "update_hydration_status", lambda *_a, **_k: None)
    monkeypatch.setattr(mdm, "get_ohlc_bars", lambda *_a, **_k: list(cached))

    first = asyncio.create_task(mdm.hydrate_symbol_history(CE, max_bars=first_bars))
    await asyncio.sleep(0)
    second = asyncio.create_task(mdm.hydrate_symbol_history(CE, max_bars=second_bars))
    await asyncio.gather(first, second)
    assert calls == expected_calls


@pytest.mark.asyncio
async def test_hydration_failed_task_cleanup_and_independent_symbols(monkeypatch):
    mdm = MarketDataManager(broker=SimpleNamespace(), settings=SimpleNamespace(history_min_interval_sec=0))
    calls = []

    async def failing_fetch(symbol, interval, days):
        calls.append(symbol)
        raise RuntimeError("boom")

    monkeypatch.setattr(mdm, "fetch_history", failing_fetch)
    with pytest.raises(RuntimeError):
        await mdm.hydrate_symbol_history(CE, max_bars=30)
    assert mdm._history_inflight == {}

    async def ok_fetch(symbol, interval, days):
        calls.append(symbol)
        await asyncio.sleep(0.02)
        return [{"timestamp": datetime(2026, 6, 12, 9, 15, tzinfo=ZoneInfo("UTC")), "open": 1, "high": 1, "low": 1, "close": 1}]

    monkeypatch.setattr(mdm, "fetch_history", ok_fetch)
    monkeypatch.setattr(mdm, "ingest_historical_ohlc", lambda *_a, **_k: 1)
    monkeypatch.setattr(mdm, "update_hydration_status", lambda *_a, **_k: None)
    monkeypatch.setattr(mdm, "get_ohlc_bars", lambda *_a, **_k: [])
    await asyncio.gather(mdm.hydrate_symbol_history(CE, max_bars=30), mdm.hydrate_symbol_history(PE, max_bars=30))
    assert CE in calls and PE in calls


def test_active_basket_normalizes_counts_and_preserves_partial_tokens():
    basket = normalize_active_basket_schema({
        "spot_symbol": "NSE:NIFTY",
        "futures_symbol": "NFO:NIFTY26JUNFUT",
        "selected_ce": CE,
        "selected_pe": PE,
        "option_symbols": ["NSE:NIFTY", "NFO:NIFTY26JUNFUT", CE, PE, CE, "NFO:NIFTY26JUN25050CE", "NFO:NIFTY26JUN25050PE", "NFO:NIFTY26JUN25100CE", "NFO:NIFTY26JUN25100PE", "NFO:NIFTY26JUN25150CE", "NFO:NIFTY26JUN25150PE", "NFO:NIFTY26JUN25200CE"],
        "all_tokens": [1, 2, 3, 4, 5],
        "option_tokens": [3, 4, 5],
        "token_by_symbol": {CE: 3},
    })
    assert all(str(sym).endswith(("CE", "PE")) for sym in basket["option_symbols"])
    assert "NSE:NIFTY" not in basket["option_symbols"]
    assert "NFO:NIFTY26JUNFUT" not in basket["option_symbols"]
    assert basket["futures_symbol"].endswith("FUT")
    assert len(basket["option_symbols"]) <= 8
    assert len(basket["symbols"]) <= 10
    assert len(basket["all_tokens"]) <= 10
    assert basket["all_tokens"] == [1, 2, 3, 4, 5]
    assert basket["option_tokens"] == [3, 4, 5]
    assert basket["option_symbols"].count(CE) == 1
    assert basket["option_symbols"].count(PE) == 1


def test_context_only_basket_cannot_overwrite_tradable_basket_and_universe_stays_available():
    class Dummy:
        pass
    ctx = Dummy()
    ctx.active_trading_universe = {"selected_ce": CE, "selected_pe": PE, "option_symbols": [CE, PE], "token_by_symbol": {CE: 1, PE: 2}}
    ctx.active_contract_basket = dict(ctx.active_trading_universe)
    ctx.selected_ce = CE
    ctx.selected_pe = PE
    ctx.atm_ce_symbol = CE
    ctx.atm_pe_symbol = PE
    ctx.active_symbol_tokens = {CE: 1, PE: 2}
    ctx.strategy_runner = SimpleNamespace(set_active_trading_universe=lambda basket: setattr(ctx, "runner_basket", basket))
    ctx.data_hub = SimpleNamespace(set_active_contract_basket=lambda basket: setattr(ctx, "datahub_basket", basket))
    ctx.option_universe = OptionUniverseManager(SimpleNamespace(strike_step=50, strikes_each_side=2))
    ctx.strategy_manager = None
    ctx.market_data_manager = None
    ctx.instrument_manager = None
    old_basket = ctx.active_contract_basket

    _commit_active_dynamic_basket(ctx, basket={"spot_symbol": "NSE:NIFTY"}, option_symbols=[], symbols=["NSE:NIFTY"], atm_strike=None)
    assert ctx.active_contract_basket is old_basket
    assert ctx.option_universe.get_current_universe() == [CE, PE]


def test_eod_flatten_skips_weekend_and_holiday(monkeypatch):
    ist = ZoneInfo("Asia/Kolkata")
    holidays = {datetime(2026, 10, 2, tzinfo=ist).date()}
    monkeypatch.setattr(app, "is_nse_trading_day", lambda day: day.weekday() < 5 and day not in holidays)
    assert _next_eod_flatten_time_ist(datetime(2026, 6, 12, 15, 0, tzinfo=ist)).date().isoformat() == "2026-06-12"
    assert _next_eod_flatten_time_ist(datetime(2026, 6, 12, 15, 25, tzinfo=ist)).date().isoformat() == "2026-06-15"
    assert _next_eod_flatten_time_ist(datetime(2026, 6, 13, 10, 0, tzinfo=ist)).date().isoformat() == "2026-06-15"
    assert _next_eod_flatten_time_ist(datetime(2026, 6, 14, 10, 0, tzinfo=ist)).date().isoformat() == "2026-06-15"
    assert _next_eod_flatten_time_ist(datetime(2026, 10, 2, 10, 0, tzinfo=ist)).date().isoformat() == "2026-10-05"
