from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _build_mdm() -> MarketDataManager:
    mdm = MarketDataManager(broker=None, websocket=None, settings={})
    mdm._is_market_open_now = lambda: True
    return mdm


def test_starvation_guard_skips_when_spot_fresh() -> None:
    mdm = _build_mdm()
    now = datetime.now(timezone.utc)
    mdm._last_tick_time['NSE:NIFTY'] = now.timestamp()
    assert mdm._is_true_market_data_starvation(now) is False


def test_starvation_guard_skips_when_option_fresh() -> None:
    mdm = _build_mdm()
    now = datetime.now(timezone.utc)
    mdm._last_tick_time['NFO:NIFTY26MAY22500CE'] = now.timestamp()
    assert mdm._is_true_market_data_starvation(now) is False


def test_starvation_guard_confirms_when_no_recent_ticks_and_market_open() -> None:
    mdm = _build_mdm()
    now = datetime.now(timezone.utc)
    mdm._last_tick_time['NSE:NIFTY'] = (now - timedelta(seconds=120)).timestamp()
    assert mdm._is_true_market_data_starvation(now) is True
