import asyncio
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.core.active_basket import normalize_active_basket_schema
from nifty_scalper_bot.core.app import _next_eod_flatten_time_ist
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.execution.readiness import evaluate_quote_readiness


def test_quote_readiness_separates_ltp_and_tradable_quote():
    ltp_only = {"ltp": 100.0, "source": "ws", "tick_age_s": 1.0}
    status = evaluate_quote_readiness("NFO:NIFTY26JUN25000CE", ltp_only, max_spread_pct=5.0, max_age_s=5.0)
    assert status.ltp_ready is True
    assert status.bid_ask_available is False
    assert status.tradable_quote_ready is False
    assert status.reason == "bid_ask_missing"

    crossed = {"ltp": 100.0, "bid": 101.0, "ask": 100.0, "tick_age_s": 1.0}
    assert evaluate_quote_readiness("NFO:NIFTY26JUN25000CE", crossed, max_spread_pct=5.0).reason == "bid_ask_crossed"

    ready = {"ltp": 100.0, "bid": 99.9, "ask": 100.1, "tick_age_s": 1.0, "source": "ws"}
    status = evaluate_quote_readiness("NFO:NIFTY26JUN25000CE", ready, max_spread_pct=5.0, max_age_s=5.0)
    assert status.ltp_ready is True
    assert status.bid_ask_available is True
    assert status.tradable_quote_ready is True
    assert status.reason == "ready"


@pytest.mark.asyncio
async def test_hydration_request_coalesces_smaller_request(monkeypatch):
    mdm = MarketDataManager(broker=SimpleNamespace(), settings=SimpleNamespace(history_min_interval_sec=0))
    calls = 0

    async def fake_fetch(symbol, interval, days):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return [{"timestamp": datetime(2026, 6, 12, 9, 15, tzinfo=ZoneInfo("UTC")), "open": 1, "high": 1, "low": 1, "close": 1}]

    monkeypatch.setattr(mdm, "fetch_history", fake_fetch)
    monkeypatch.setattr(mdm, "ingest_historical_ohlc", lambda *_a, **_k: 1)
    monkeypatch.setattr(mdm, "update_hydration_status", lambda *_a, **_k: None)
    monkeypatch.setattr(mdm, "get_ohlc_bars", lambda *_a, **_k: [])

    big = asyncio.create_task(mdm.hydrate_symbol_history("NFO:NIFTY26JUN25000CE", max_bars=300))
    await asyncio.sleep(0)
    small = asyncio.create_task(mdm.hydrate_symbol_history("NFO:NIFTY26JUN25000CE", max_bars=30))
    await asyncio.gather(big, small)
    assert calls == 1


def test_active_basket_normalizes_option_counts_and_context_tokens():
    basket = normalize_active_basket_schema({
        "spot_symbol": "NSE:NIFTY",
        "futures_symbol": "NFO:NIFTY26JUNFUT",
        "selected_ce": "NFO:NIFTY26JUN25000CE",
        "selected_pe": "NFO:NIFTY26JUN25000PE",
        "option_symbols": [
            "NSE:NIFTY", "NFO:NIFTY26JUNFUT", "NFO:NIFTY26JUN25000CE", "NFO:NIFTY26JUN25000PE",
            "NFO:NIFTY26JUN25050CE", "NFO:NIFTY26JUN25050PE", "NFO:NIFTY26JUN25100CE", "NFO:NIFTY26JUN25100PE",
            "NFO:NIFTY26JUN25150CE", "NFO:NIFTY26JUN25150PE", "NFO:NIFTY26JUN25200CE",
        ],
        "token_by_symbol": {"NSE:NIFTY": 1, "NFO:NIFTY26JUNFUT": 2, "NFO:NIFTY26JUN25000CE": 3, "NFO:NIFTY26JUN25000PE": 4},
    })
    assert all(str(sym).endswith(("CE", "PE")) for sym in basket["option_symbols"])
    assert "NSE:NIFTY" not in basket["option_symbols"]
    assert "NFO:NIFTY26JUNFUT" not in basket["option_symbols"]
    assert len(basket["option_symbols"]) <= 8
    assert len(basket["symbols"]) <= 10
    assert len(basket["all_tokens"]) <= 10
    assert basket["option_symbols"].count("NFO:NIFTY26JUN25000CE") == 1
    assert basket["option_symbols"].count("NFO:NIFTY26JUN25000PE") == 1


def test_eod_flatten_skips_weekend_and_holiday_and_after_cutoff():
    ist = ZoneInfo("Asia/Kolkata")
    # normal weekday before cutoff schedules same day
    assert _next_eod_flatten_time_ist(datetime(2026, 6, 12, 15, 0, tzinfo=ist)).date().isoformat() == "2026-06-12"
    # Friday after cutoff rolls to Monday (2026-06-13/14 are Saturday/Sunday)
    assert _next_eod_flatten_time_ist(datetime(2026, 6, 12, 15, 25, tzinfo=ist)).date().isoformat() == "2026-06-15"
    assert _next_eod_flatten_time_ist(datetime(2026, 6, 13, 10, 0, tzinfo=ist)).date().isoformat() == "2026-06-15"
    assert _next_eod_flatten_time_ist(datetime(2026, 6, 14, 10, 0, tzinfo=ist)).date().isoformat() == "2026-06-15"
    # 2026-10-02 is in the project NSE_HOLIDAYS set; roll to next weekday.
    assert _next_eod_flatten_time_ist(datetime(2026, 10, 2, 10, 0, tzinfo=ist)).date().isoformat() == "2026-10-5"
