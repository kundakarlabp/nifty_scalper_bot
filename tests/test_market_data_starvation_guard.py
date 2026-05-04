from datetime import datetime, timezone

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_starvation_guard_skips_when_spot_fresh() -> None:
    mdm = MarketDataManager(broker=None, websocket=None, settings={})
    now = datetime.now(timezone.utc)
    mdm._last_tick_time['NSE:NIFTY'] = now.timestamp()  # noqa: SLF001
    assert mdm._is_true_market_data_starvation(now) is False  # noqa: SLF001
