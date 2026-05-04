import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from nifty_scalper_bot.data.market_data_manager import MarketDataManager, PollingPolicy


def test_no_bare_policy_reference_in_market_data_manager_source() -> None:
    source = Path('src/nifty_scalper_bot/data/market_data_manager.py').read_text()
    masked = source.replace('self._polling_policy.', '').replace('PollingPolicy', '')
    assert re.search(r'\bpolicy\.', masked) is None


def test_should_poll_symbol_does_not_raise_name_error() -> None:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._polling_policy = PollingPolicy()
    mdm._poll_error_last_log = {}
    mdm._last_tick_time = {}
    mdm._logger = logging.getLogger('test')
    mdm._lock = __import__('threading').RLock()
    mdm._active_subscribed_symbols = {'NSE:NIFTY'}
    mdm._tracked_symbols = set()
    mdm._is_ws_connected = lambda: True
    mdm._canonical_symbol = lambda s: s
    assert mdm._should_poll_symbol('NSE:NIFTY', datetime.now(timezone.utc)) is True


def test_log_poll_error_throttle_does_not_raise() -> None:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._polling_policy = PollingPolicy()
    mdm._poll_error_last_log = {}
    mdm._logger = logging.getLogger('test')
    mdm._log_poll_error('NSE:NIFTY', 'x')
    mdm._log_poll_error('NSE:NIFTY', 'x')
