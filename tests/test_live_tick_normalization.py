from datetime import datetime, timezone

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_ws_raw_tick_normalizes_to_canonical_tick() -> None:
    mdm = MarketDataManager()
    mdm.register_symbol('NSE:NIFTY', 256265)
    tick = mdm.normalize_live_tick({'instrument_token': 256265, 'last_price': 22450.5}, source='ws')
    assert tick is not None
    assert tick['symbol'] == 'NSE:NIFTY'
    assert tick['token'] == 256265
    assert tick['ltp'] == 22450.5
    assert tick['source'] == 'ws'


def test_missing_bid_ask_stays_none() -> None:
    mdm = MarketDataManager()
    tick = mdm.normalize_live_tick({'symbol': 'NSE:NIFTY', 'ltp': 22450.0}, source='ws')
    assert tick is not None
    assert tick['bid'] is None
    assert tick['ask'] is None


def test_ltp_non_positive_returns_none() -> None:
    mdm = MarketDataManager()
    assert mdm.normalize_live_tick({'symbol': 'NSE:NIFTY', 'ltp': 0}, source='ws') is None


def test_timestamp_is_timezone_aware_utc() -> None:
    mdm = MarketDataManager()
    tick = mdm.normalize_live_tick({'symbol': 'NSE:NIFTY', 'ltp': 1.0, 'timestamp': '2026-01-01T10:00:00'}, source='poll')
    assert tick is not None
    ts = tick['timestamp']
    assert isinstance(ts, datetime)
    assert ts.tzinfo is not None
    assert ts.tzinfo == timezone.utc
