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


def test_rest_reply_rejected_after_ws_race_cannot_replace_mid_or_age(
    monkeypatch,
) -> None:
    mdm = _manager()
    # Accepted WS FULL timestamp is more recent than the delayed REST reply.
    mdm._tick_cache[SYMBOL]["timestamp"] = time.time()
    mdm._tick_cache[SYMBOL]["received_at"] = time.time()
    mdm._latest_ticks[SYMBOL] = dict(mdm._tick_cache[SYMBOL])
    quote_before = mdm.get_latest_tick(SYMBOL)
    mid_before = mdm._last_mid[SYMBOL]
    stamp_before = mdm._last_quote_ts_ms[SYMBOL]
    monkeypatch.setattr(mdm, "_candidate_quote_keys", lambda _symbol: [SYMBOL])
    monkeypatch.setattr(
        mdm,
        "_broker_quote_any",
        lambda _symbol: {"last_price": 99.0, "bid": 50.0, "ask": 51.0},
    )
    monkeypatch.setattr(
        mdm,
        "_normalize_tick",
        lambda _symbol, _raw, _prev: {
            "symbol": SYMBOL,
            "ltp": 99.0,
            "timestamp": time.time() - 100,
            "source": "rest",
        },
    )

    mdm.refresh_quote_now(SYMBOL)

    assert mdm.get_latest_tick(SYMBOL) == quote_before
    assert mdm._last_mid[SYMBOL] == mid_before
    assert mdm._last_quote_ts_ms[SYMBOL] == stamp_before


def test_fresh_full_book_is_not_downgraded_by_newer_ltp_only_ws_tick() -> None:
    mdm = _manager()
    now = time.time()
    full = {
        "symbol": SYMBOL,
        "ltp": 100.0,
        "bid": 99.9,
        "ask": 100.1,
        "source": "ws",
        "timestamp": now,
        "received_at": now,
        "timestamp_quality": "exchange",
        "source_timestamp_valid": True,
        "depth": {
            "buy": [{"price": 99.9, "quantity": 65}],
            "sell": [{"price": 100.1, "quantity": 65}],
        },
    }
    assert mdm._store_tick(SYMBOL, full) is True
    saved = mdm.get_latest_tick(SYMBOL)
    assert saved is not None

    degraded = {
        "symbol": SYMBOL,
        "ltp": 102.0,
        "source": "ws",
        "timestamp": now + 0.01,
        "received_at": now + 0.01,
        "timestamp_quality": "exchange",
        "source_timestamp_valid": True,
    }
    assert mdm._store_tick(SYMBOL, degraded) is False
    assert mdm.get_latest_tick(SYMBOL) == saved


def test_verified_full_book_is_not_replaced_by_receipt_only_full_depth() -> None:
    mdm = _manager()
    now = time.time()
    full = {
        "symbol": SYMBOL,
        "ltp": 100.0,
        "bid": 99.9,
        "ask": 100.1,
        "source": "ws",
        "timestamp": now,
        "received_at": now,
        "timestamp_quality": "exchange",
        "depth": {
            "buy": [{"price": 99.9, "quantity": 65}],
            "sell": [{"price": 100.1, "quantity": 65}],
        },
    }
    assert mdm._store_tick(SYMBOL, full)
    attempted = {
        **full,
        "ltp": 101.0,
        "timestamp": now + 0.05,
        "received_at": now + 0.05,
        "timestamp_quality": "received_at",
    }

    assert mdm._store_tick(SYMBOL, attempted) is False
    assert mdm.get_latest_tick(SYMBOL)["ltp"] == 100.0


def test_strategy_snapshot_uses_market_event_age_not_new_receipt_age() -> None:
    mdm = _manager()
    now = time.time()
    mdm._tick_cache[SYMBOL] = {
        "symbol": SYMBOL,
        "ltp": 100.0,
        "bid": 99.9,
        "ask": 100.1,
        "timestamp": now - 90,
        "received_at": now,
        "source": "ws",
        "timestamp_quality": "exchange",
        "tradable_quote": True,
    }
    mdm._latest_ticks[SYMBOL] = dict(mdm._tick_cache[SYMBOL])

    snap = mdm.get_symbol_snapshot(SYMBOL)

    assert snap.tick_age_s is not None
    assert snap.tick_age_s >= 89.0


def test_strategy_snapshot_rejects_untrusted_book_and_one_sided_depth() -> None:
    mdm = _manager()
    now = time.time()
    mdm._tick_cache[SYMBOL] = {
        "symbol": SYMBOL,
        "ltp": 100.0,
        "bid": 99.9,
        "ask": 100.1,
        "timestamp": now,
        "received_at": now,
        "timestamp_quality": "received_at",
        "timestamp_source": "received_at",
        "source": "ws",
        "tradable_quote": False,
        "depth": {
            "buy": [{"price": 99.9, "quantity": 65}],
            "sell": [],
        },
    }
    mdm._latest_ticks[SYMBOL] = dict(mdm._tick_cache[SYMBOL])

    snap = mdm.get_symbol_snapshot(SYMBOL)

    assert snap.depth_available is False
    assert snap.tradable_quote is False
    assert snap.tick_age_s is None


def test_receipt_only_ws_event_cannot_confirm_current_generation() -> None:
    mdm = MarketDataManager(kite=None)
    token = 127654
    assert mdm.request_token_subscription(token, symbol=SYMBOL)
    now = time.time()
    tick = {
        "symbol": SYMBOL,
        "instrument_token": token,
        "ltp": 100.0,
        "timestamp": now,
        "received_at": now,
        "source": "ws",
        "timestamp_source": "received_at",
        "timestamp_quality": "received_at",
        "source_timestamp_valid": True,
    }

    mdm._ingest_normalized_tick(tick)

    assert mdm.time_since_last_live_ws_tick(SYMBOL) is None
    assert mdm._symbol_first_tick_generation.get(SYMBOL) is None

    valid = {
        **tick,
        "timestamp_source": "exchange_timestamp",
        "timestamp_quality": "exchange",
    }
    mdm._ingest_normalized_tick(valid)

    assert mdm.time_since_last_live_ws_tick(SYMBOL) is not None
