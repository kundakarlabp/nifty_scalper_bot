"""REST fallback cannot renew or downgrade an existing WS FULL quote."""

from __future__ import annotations

import time

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
