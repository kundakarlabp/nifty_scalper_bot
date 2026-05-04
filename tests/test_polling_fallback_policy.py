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


def test_poll_candidates_only_returns_stale_active_symbols() -> None:
    mdm = MarketDataManager()
    now = datetime.now(timezone.utc)
    fresh = 'NSE:NIFTY'
    stale = 'NFO:NIFTY26MAY22500CE'
    mdm._active_subscribed_symbols = {fresh, stale}
    mdm._last_tick_time[fresh] = now.timestamp()
    mdm._last_tick_time[stale] = (now - timedelta(seconds=mdm._poll_option_stale_seconds + 1)).timestamp()
    candidates = mdm._poll_candidates(now)
    assert stale in candidates
    assert fresh not in candidates


def test_log_poll_error_throttles_duplicate_symbol_logs() -> None:
    mdm = MarketDataManager()
    mdm._log_poll_error('NSE:NIFTY', RuntimeError('failure_one'))
    first = mdm._poll_error_last_log.get('NSE:NIFTY')
    assert isinstance(first, float)
    mdm._log_poll_error('NSE:NIFTY', RuntimeError('failure_two'))
    second = mdm._poll_error_last_log.get('NSE:NIFTY')
    assert second == first
