from __future__ import annotations

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_cached_mid_tracks_age(monkeypatch) -> None:
    mdm = MarketDataManager(broker_client=object())
    symbol = "NFO:NIFTY24OCT20000CE"

    monkeypatch.setattr(mdm, "_now_ms", lambda: 1_000.0)
    mdm._latest_ticks[symbol] = {"bid": 100.0, "ask": 102.0, "ltp": 101.0}

    quote = mdm.get_quote(symbol)
    assert quote["bid"] == 100.0
    mid, age = mdm.cached_mid(symbol)
    assert mid == pytest.approx(101.0)
    assert age == 0

    monkeypatch.setattr(mdm, "_now_ms", lambda: 1_600.0)
    mid_again, age_again = mdm.cached_mid(symbol)
    assert mid_again == pytest.approx(101.0)
    assert isinstance(age_again, int)
    assert age_again == 600

    quote_age = mdm.quote_age_ms(symbol)
    assert isinstance(quote_age, int)
    assert quote_age == age_again
