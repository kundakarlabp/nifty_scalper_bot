"""REST fallback cannot renew or downgrade an existing WS FULL quote."""

from __future__ import annotations

import time
from types import SimpleNamespace

from nifty_scalper_bot.data.market_data_manager import MarketDataManager

SYMBOL = "NFO:NIFTY26NOV22800CE"


def _manager() -> MarketDataManager:
    mdm = MarketDataManager(kite=None)
    mdm._tick_cache[SYMBOL] = {
        "symbol": SYMBOL,
        "ltp": 100.0,
        "last_price": 100.0,
        "bid": 99.95,
        "ask": 100.05,
        "depth_available": True,
        "tradable_quote": True,
        "source": "ws",
    }
    mdm._latest_ticks[SYMBOL] = dict(mdm._tick_cache[SYMBOL])
    mdm._last_quote_ts_ms[SYMBOL] = mdm._now_ms() - 30_000
    mdm._last_mid[SYMBOL] = (100.0, mdm._now_ms() - 30_000)
    return mdm


def test_quote_read_never_renews_receive_age_or_mid_age() -> None:
    mdm = _manager()
    initial_stamp = mdm._last_quote_ts_ms[SYMBOL]
    initial_mid = mdm._last_mid[SYMBOL]
    before_age = mdm.quote_age_ms(SYMBOL)

    for _ in range(10):
        quote = mdm.get_quote(SYMBOL)
        assert quote is not None
        assert quote["source"] == "ws"
        assert quote["bid"] == 99.95
        assert quote["ask"] == 100.05

    assert mdm._last_quote_ts_ms[SYMBOL] == initial_stamp
    assert mdm._last_mid[SYMBOL] == initial_mid
    assert mdm.quote_age_ms(SYMBOL) >= before_age
    assert mdm.quote_age_ms(SYMBOL) >= 29_000


def test_invalid_rest_quote_never_replaces_current_full_websocket_quote(
    monkeypatch,
) -> None:
    mdm = _manager()
    initial_quote = mdm.get_latest_tick(SYMBOL)
    initial_stamp = mdm._last_quote_ts_ms[SYMBOL]
    monkeypatch.setattr(mdm, "_candidate_quote_keys", lambda _symbol: [SYMBOL])
    monkeypatch.setattr(
        mdm, "_broker_quote_any", lambda _symbol: {"bid": 1.0, "ask": 2.0}
    )
    monkeypatch.setattr(mdm, "_normalize_tick", lambda *_args: None)

    result = mdm.refresh_quote_now(SYMBOL)

    assert result is None
    assert mdm.get_latest_tick(SYMBOL) == initial_quote
    assert mdm.get_quote(SYMBOL) == initial_quote
    assert mdm._last_quote_ts_ms[SYMBOL] == initial_stamp
    assert mdm.quote_age_ms(SYMBOL) >= 29_000


def test_direct_rest_ltp_failure_does_not_corrupt_canonical_quote(
    monkeypatch,
) -> None:
    mdm = _manager()
    original = mdm.get_latest_tick(SYMBOL)
    original_stamp = mdm._last_quote_ts_ms[SYMBOL]
    mdm._last_tick_time[SYMBOL] = time.time() - 3600
    mdm._last_tick_wallclock[SYMBOL] = time.time() - 3600
    mdm._broker = SimpleNamespace(get_quote=lambda _symbol: {"last_price": 201.0})
    mdm._md_policy = SimpleNamespace(
        should_suppress_rest_quote=lambda **_kwargs: False,
        to_broker_quote_symbol=lambda symbol: symbol,
    )
    monkeypatch.setattr(mdm, "_normalize_tick", lambda *_args: None)

    assert mdm.get_ltp(SYMBOL, allow_rest_fallback=True) == 201.0
    assert mdm.get_latest_tick(SYMBOL) == original
    assert mdm._last_quote_ts_ms[SYMBOL] == original_stamp


def test_accepted_ws_tick_updates_mid_without_a_quote_read() -> None:
    mdm = _manager()
    quote = {
        "symbol": SYMBOL,
        "ltp": 102.0,
        "bid": 101.9,
        "ask": 102.1,
        "timestamp": time.time(),
        "received_at": time.time(),
        "source": "ws",
    }

    assert mdm._store_tick(SYMBOL, quote) is True
    mid, age_ms = mdm.cached_mid(SYMBOL)

    assert mid == 102.0
    assert age_ms is not None and age_ms < 1000
    assert mdm.quote_age_ms(SYMBOL) < 1000
