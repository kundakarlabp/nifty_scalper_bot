from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_mdm_volume_delta_pipeline() -> None:
    mdm = MarketDataManager()
    symbol = 'NFO:NIFTY26MAY23500CE'
    cumulatives = [6000000, 6000065, 6000200]
    expected = [0, 65, 135]
    for cumulative, delta in zip(cumulatives, expected):
        raw = {'symbol': symbol, 'volume_traded_today': cumulative, 'ltp': 100, 'timestamp': '2026-05-13T08:27:25Z'}
        normalized = mdm._normalise_tick_volume_delta(symbol, raw)
        live = mdm.normalize_live_tick(normalized, source='ws')
        assert normalized['volume_delta'] == delta
        assert normalized['volume'] == delta
        assert live is not None
        assert live['volume_delta'] == delta
        assert live['volume'] == delta
        assert int(live['volume_cumulative']) == cumulative
