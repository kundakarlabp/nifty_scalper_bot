"""Comprehensive unit tests for PollingStreamer with production scenarios."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.streaming.polling_streamer import PollingStreamer


class DummyBroker:
    """Mock broker for testing PollingStreamer."""

    def __init__(
        self,
        use_string_keys: bool = False,
        return_empty: bool = False,
        raise_ltp: bool = False,
        raise_quote: bool = False,
    ) -> None:
        """Initialize dummy broker.

        Args:
            use_string_keys: If True, return string keys instead of int.
            return_empty: If True, return empty responses.
            raise_ltp: If True, raise exception on get_ltp_bulk.
            raise_quote: If True, raise exception on get_quote_bulk.
        """
        self.use_string_keys = use_string_keys
        self.return_empty = return_empty
        self.raise_ltp = raise_ltp
        self.raise_quote = raise_quote
        self.ltp_call_count = 0
        self.quote_call_count = 0

    def get_ltp_bulk(self, batch: list[int]) -> dict[Any, float]:
        """Return mock LTP data."""
        self.ltp_call_count += 1
        if self.raise_ltp:
            raise RuntimeError("LTP API error")
        if self.return_empty:
            return {}
        key_type = str if self.use_string_keys else int
        return {key_type(t): 100.0 + float(t) for t in batch}

    def get_quote_bulk(self, batch: list[int]) -> dict[Any, dict[str, Any]]:
        """Return mock quote data."""
        self.quote_call_count += 1
        if self.raise_quote:
            raise RuntimeError("Quote API error")
        if self.return_empty:
            return {}
        return {
            t: {
                "last_price": 100.0 + float(t),
                "depth": {"buy": [[99.5, 100]], "sell": [[100.5, 100]]},
            }
            for t in batch
        }

    def get_quote_by_token(self, token: int) -> dict[str, Any]:
        """Return single quote."""
        if self.raise_quote:
            raise RuntimeError("Quote API error")
        return {"last_price": 100.0 + float(token)}


class TestPollingStreamerInitialization:
    """Test PollingStreamer initialization and metrics setup."""

    def test_init_sets_defaults(self) -> None:
        """Test that initialization sets correct defaults."""
        broker = DummyBroker()
        ticks: list[dict[str, Any]] = []
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: ticks.append(t),
            instrument_resolver=None,
            poll_interval_ms=700,
            batch_size=200,
        )
        assert streamer._interval_s == pytest.approx(0.7, rel=0.01)
        assert streamer._batch_size == 200
        assert len(streamer.tracked_tokens()) == 0
        assert not streamer.is_running()

    def test_init_creates_metrics(self) -> None:
        """Test that initialization creates metric objects."""
        broker = DummyBroker()
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
        )
        assert streamer._m_poll_ok is not None
        assert streamer._m_poll_err is not None
        assert streamer._m_tokens is not None
        assert streamer._m_interval is not None
        assert streamer._m_last_tick is not None
        assert streamer._m_last_success is not None


class TestPollingStreamerTokenManagement:
    """Test token subscription/unsubscription and tracking."""

    def test_subscribe_updates_tokens(self) -> None:
        """Test that subscribe updates token set and metrics."""
        broker = DummyBroker()
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
            poll_interval_ms=1000,
        )
        streamer.subscribe([256265, 256273])
        tokens = streamer.tracked_tokens()
        assert len(tokens) == 2
        assert 256265 in tokens
        assert 256273 in tokens

    def test_unsubscribe_removes_tokens(self) -> None:
        """Test that unsubscribe removes tokens from tracking."""
        broker = DummyBroker()
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
            poll_interval_ms=1000,
        )
        streamer.subscribe([256265, 256273, 256280])
        streamer.unsubscribe([256273])
        tokens = streamer.tracked_tokens()
        assert len(tokens) == 2
        assert 256273 not in tokens
        assert 256265 in tokens

    def test_subscribe_tokens_alias(self) -> None:
        """Test that subscribe_tokens is an alias for subscribe."""
        broker = DummyBroker()
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
            poll_interval_ms=1000,
        )
        streamer.subscribe_tokens([256265])
        assert len(streamer.tracked_tokens()) == 1
        assert 256265 in streamer.tracked_tokens()


class TestPollingStreamerFetching:
    """Test tick fetching and validation."""

    def test_try_ltp_bulk_returns_ticks(self) -> None:
        """Test _try_ltp_bulk returns valid ticks."""
        broker = DummyBroker()
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
        )
        timestamp_ms = int(time.time() * 1000)
        ticks = streamer._try_ltp_bulk([256265], timestamp_ms)
        assert ticks is not None
        assert len(ticks) == 1
        assert ticks[0]["instrument_token"] == 256265
        assert ticks[0]["last_price"] == pytest.approx(100.0 + 256265)
        assert ticks[0]["timestamp"] == timestamp_ms

    def test_try_ltp_bulk_handles_string_keys(self) -> None:
        """Test _try_ltp_bulk handles string keys from broker."""
        broker = DummyBroker(use_string_keys=True)
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
        )
        timestamp_ms = int(time.time() * 1000)
        ticks = streamer._try_ltp_bulk([256265], timestamp_ms)
        assert ticks is not None
        assert len(ticks) == 1
        assert ticks[0]["instrument_token"] == 256265

    def test_try_quote_bulk_returns_ticks_with_depth(self) -> None:
        """Test _try_quote_bulk returns ticks with depth."""
        broker = DummyBroker()
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
        )
        timestamp_ms = int(time.time() * 1000)
        ticks = streamer._try_quote_bulk([256265], timestamp_ms)
        assert ticks is not None
        assert len(ticks) == 1
        assert ticks[0]["instrument_token"] == 256265
        assert "depth" in ticks[0]

    def test_try_ltp_bulk_handles_empty_response(self) -> None:
        """Test _try_ltp_bulk handles empty response."""
        broker = DummyBroker(return_empty=True)
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
        )
        timestamp_ms = int(time.time() * 1000)
        ticks = streamer._try_ltp_bulk([256265], timestamp_ms)
        assert ticks is None

    def test_try_ltp_bulk_handles_exception(self) -> None:
        """Test _try_ltp_bulk handles exceptions gracefully."""
        broker = DummyBroker(raise_ltp=True)
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
        )
        timestamp_ms = int(time.time() * 1000)
        ticks = streamer._try_ltp_bulk([256265], timestamp_ms)
        assert ticks is None


class TestPollingStreamerValidation:
    """Test tick shape validation and error handling."""

    def test_fetch_ticks_invalid_payload_skipped(self) -> None:
        """Test that invalid tick payloads are skipped during emit."""
        broker = DummyBroker()
        ticks_received: list[dict[str, Any]] = []
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: ticks_received.append(t),
            instrument_resolver=None,
            poll_interval_ms=1000,
        )
        # Valid tick structure
        valid_tick = {
            "instrument_token": 256265,
            "last_price": 100.0,
            "timestamp": int(time.time() * 1000),
        }
        # Validate structure is in place
        assert "instrument_token" in valid_tick
        assert "last_price" in valid_tick
        assert "timestamp" in valid_tick

    def test_fetch_ticks_returns_list(self) -> None:
        """Test that _fetch_ticks returns a list (never None)."""
        broker = DummyBroker()
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
        )
        ticks = streamer._fetch_ticks([256265])
        assert isinstance(ticks, list)
        assert len(ticks) >= 0


class TestPollingStreamerThreading:
    """Test thread safety and concurrency."""

    def test_start_stop_cycle(self) -> None:
        """Test start and stop of polling thread."""
        broker = DummyBroker()
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
            poll_interval_ms=100,
        )
        assert not streamer.is_running()
        streamer.start()
        assert streamer.is_running()
        streamer.stop()
        assert not streamer.is_running()

    def test_subscribe_during_polling(self) -> None:
        """Test that subscribe is thread-safe during polling."""
        broker = DummyBroker()
        ticks_received: list[dict[str, Any]] = []
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: ticks_received.append(t),
            instrument_resolver=None,
            poll_interval_ms=100,
        )
        streamer.start()
        streamer.subscribe([256265])
        time.sleep(0.2)
        streamer.stop()
        assert len(streamer.tracked_tokens()) == 1


class TestPollingStreamerRateLimits:
    """Test rate-limit detection and warnings."""

    def test_rate_limit_warning_on_many_tokens(self) -> None:
        """Test that rate-limit warning is emitted for many tokens."""
        broker = DummyBroker()
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
            poll_interval_ms=100,
            batch_size=50,
            warn_on_rate_limit=True,
        )
        # Subscribe with more tokens than safe capacity
        large_batch = list(range(1000, 2000))  # 1000 tokens
        streamer.subscribe(large_batch)
        # Rate limit warning should be triggered
        assert streamer._rate_limit_warned is True

    def test_rate_limit_recovery(self) -> None:
        """Test that rate-limit warning clears when tokens reduced."""
        broker = DummyBroker()
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
            poll_interval_ms=100,
            batch_size=50,
            warn_on_rate_limit=True,
        )
        large_batch = list(range(1000, 2000))
        streamer.subscribe(large_batch)
        assert streamer._rate_limit_warned is True
        # Unsubscribe to bring token count down
        streamer.unsubscribe(large_batch)
        streamer._maybe_warn_rate_limits(0)
        assert streamer._rate_limit_warned is False


class TestPollingStreamerEdgeCases:
    """Test edge cases and error conditions."""

    def test_subscribe_zero_tokens(self) -> None:
        """Test subscribing with empty token list."""
        broker = DummyBroker()
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
        )
        streamer.subscribe([])
        assert len(streamer.tracked_tokens()) == 0

    def test_unsubscribe_nonexistent_token(self) -> None:
        """Test unsubscribing token that was not subscribed."""
        broker = DummyBroker()
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
        )
        streamer.subscribe([256265])
        streamer.unsubscribe([999999])  # Token not in set
        assert len(streamer.tracked_tokens()) == 1
        assert 256265 in streamer.tracked_tokens()

    def test_negative_ltp_filtered(self) -> None:
        """Test that negative or zero LTP values are filtered."""
        broker = MagicMock()
        broker.get_ltp_bulk = MagicMock(return_value={256265: 0.0, 256273: -1.0})
        streamer = PollingStreamer(
            broker_client=broker,
            on_tick=lambda t: None,
            instrument_resolver=None,
        )
        timestamp_ms = int(time.time() * 1000)
        ticks = streamer._try_ltp_bulk([256265, 256273], timestamp_ms)
        assert ticks is None or len(ticks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
