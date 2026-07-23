from __future__ import annotations

from typing import Any

import pytest
from src.nifty_scalper_bot.streaming.polling_streamer import PollingStreamer


class FakeResolver:
    """Simple resolver for mapping tokens to symbols."""

    def format_token_as_symbol(self, token: int) -> str:
        """Return symbol. Args: token. Returns: symbol. Raises: None."""
        return 'NSE:NIFTY'


def test_fetch_ticks_falls_back_to_ohlc_close_for_vwap() -> None:
    """Test VWAP fallback. Args: None. Returns: None. Raises: None."""

    class QuoteBroker:
        def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
            """Return quote payload. Args: symbols. Returns: quote map. Raises: None."""
            return {
                symbols[0]: {
                    'last_price': 123.45,
                    'instrument_token': 101,
                    'average_price': 0.0,
                    'volume': 0,
                    'ohlc': {'close': 122.0},
                }
            }

    broker = QuoteBroker()
    resolver = FakeResolver()
    streamer = PollingStreamer(
        broker_client=broker,
        on_tick=lambda t: None,
        instrument_resolver=resolver,
    )
    ticks = streamer._fetch_ticks([101])
    assert len(ticks) == 1
    assert ticks[0]['average_price'] == pytest.approx(122.0)


def test_fetch_ticks_preserves_valid_broker_datetime_and_labels_receipt() -> None:
    """Poll producer emits broker timestamp only when parseable."""
    from datetime import datetime

    broker_ts = datetime(2026, 7, 23, 9, 20)

    class QuoteBroker:
        def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
            return {
                symbols[0]: {
                    "last_price": 123.45,
                    "instrument_token": 101,
                    "timestamp": broker_ts,
                    "average_price": 120.0,
                    "volume": 1,
                }
            }

    streamer = PollingStreamer(
        broker_client=QuoteBroker(),
        on_tick=lambda t: None,
        instrument_resolver=FakeResolver(),
    )

    tick = streamer._fetch_ticks([101])[0]

    assert tick["timestamp"] is broker_ts
    assert isinstance(tick["received_at"], float)


def test_fetch_ticks_omits_invalid_broker_timestamp_but_emits_received_at() -> None:
    """Poll producer does not put invalid broker values in timestamp."""

    class QuoteBroker:
        def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
            return {
                symbols[0]: {
                    "last_price": 123.45,
                    "instrument_token": 101,
                    "timestamp": "bad",
                    "average_price": 120.0,
                    "volume": 1,
                }
            }

    streamer = PollingStreamer(
        broker_client=QuoteBroker(),
        on_tick=lambda t: None,
        instrument_resolver=FakeResolver(),
    )

    tick = streamer._fetch_ticks([101])[0]

    assert "timestamp" not in tick
    assert isinstance(tick["received_at"], float)


def test_fetch_ticks_uses_last_trade_time_when_timestamp_missing() -> None:
    """Poll producer accepts last_trade_time as broker event timestamp."""
    from datetime import datetime

    broker_ts = datetime(2026, 7, 23, 9, 21)

    class QuoteBroker:
        def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
            return {
                symbols[0]: {
                    "last_price": 123.45,
                    "instrument_token": 101,
                    "last_trade_time": broker_ts,
                    "average_price": 120.0,
                    "volume": 1,
                }
            }

    streamer = PollingStreamer(
        broker_client=QuoteBroker(),
        on_tick=lambda t: None,
        instrument_resolver=FakeResolver(),
    )

    tick = streamer._fetch_ticks([101])[0]

    assert tick["timestamp"] is broker_ts
    assert isinstance(tick["received_at"], float)


def test_fetch_ticks_continues_to_last_trade_time_after_bad_timestamp() -> None:
    """Malformed timestamp must not block valid last_trade_time."""
    from datetime import datetime

    valid_last_trade = datetime(2026, 7, 23, 9, 22)

    class QuoteBroker:
        def quote(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
            return {
                symbols[0]: {
                    "last_price": 123.45,
                    "instrument_token": 101,
                    "timestamp": "bad",
                    "last_trade_time": valid_last_trade,
                    "average_price": 120.0,
                    "volume": 1,
                }
            }

    streamer = PollingStreamer(
        broker_client=QuoteBroker(),
        on_tick=lambda t: None,
        instrument_resolver=FakeResolver(),
    )

    tick = streamer._fetch_ticks([101])[0]

    assert tick["timestamp"] is valid_last_trade
    assert isinstance(tick["received_at"], float)
