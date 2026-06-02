from __future__ import annotations

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_cumulative_volume_normalized_to_delta() -> None:
    mdm = MarketDataManager(websocket=None)
    first = mdm._normalise_tick_volume_delta("NFO:NIFTY26JUN25000CE", {"volume_traded_today": 1000})  # noqa: SLF001
    second = mdm._normalise_tick_volume_delta("NFO:NIFTY26JUN25000CE", {"volume_traded_today": 1300})  # noqa: SLF001

    assert first["volume"] == 0
    assert second["volume"] == 300
    assert second["volume_delta"] == 300
    assert second["volume_cumulative"] == 1300


def test_cumulative_volume_reset_is_safe_delta() -> None:
    mdm = MarketDataManager(websocket=None)
    mdm._normalise_tick_volume_delta("NFO:NIFTY26JUN25000PE", {"volume_traded_today": 5000})  # noqa: SLF001
    reset = mdm._normalise_tick_volume_delta("NFO:NIFTY26JUN25000PE", {"volume_traded_today": 100})  # noqa: SLF001

    assert reset["volume"] == 100


def test_huge_cumulative_first_tick_does_not_reach_volume() -> None:
    mdm = MarketDataManager(websocket=None)
    normalized = mdm._normalize_tick("NFO:NIFTY26JUN25000CE", {"last_price": 120.0, "volume_traded_today": 563803045})  # noqa: SLF001

    assert normalized is not None
    assert normalized["volume"] == 0
    assert normalized["volume_cumulative"] == 563803045
