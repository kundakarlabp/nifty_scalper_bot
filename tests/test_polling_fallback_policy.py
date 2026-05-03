from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_fresh_ws_tick_should_not_poll() -> None:
    mdm = MarketDataManager()
    mdm._ws_connected = True
    mdm._active_subscribed_symbols = {'NSE:NIFTY'}
    mdm._last_tick_time['NSE:NIFTY'] = datetime.now(timezone.utc).timestamp()
    assert mdm._should_poll_symbol('NSE:NIFTY', datetime.now(timezone.utc)) is False


def test_stale_context_symbol_should_poll() -> None:
    mdm = MarketDataManager()
    mdm._ws_connected = True
    mdm._active_subscribed_symbols = {'NSE:NIFTY'}
    stale = datetime.now(timezone.utc) - timedelta(seconds=mdm._critical_symbol_stale_seconds + 1)
    mdm._last_tick_time['NSE:NIFTY'] = stale.timestamp()
    assert mdm._should_poll_symbol('NSE:NIFTY', datetime.now(timezone.utc)) is True


def test_stale_option_symbol_should_poll() -> None:
    mdm = MarketDataManager()
    mdm._ws_connected = True
    sym = 'NFO:NIFTY26MAY22500CE'
    mdm._active_subscribed_symbols = {sym}
    stale = datetime.now(timezone.utc) - timedelta(seconds=mdm._option_stale_seconds + 1)
    mdm._last_tick_time[sym] = stale.timestamp()
    assert mdm._should_poll_symbol(sym, datetime.now(timezone.utc)) is True


def test_inactive_symbol_should_not_poll() -> None:
    mdm = MarketDataManager()
    mdm._ws_connected = True
    mdm._active_subscribed_symbols = {'NSE:NIFTY'}
    assert mdm._should_poll_symbol('NFO:NIFTY26MAY22500CE', datetime.now(timezone.utc)) is False
