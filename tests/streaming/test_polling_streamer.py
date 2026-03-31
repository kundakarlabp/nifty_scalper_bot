# tests/streaming/test_polling_streamer.py
import time

import pytest

from src.nifty_scalper_bot.streaming.polling_streamer import PollingStreamer


class FakeLtpBroker:
    """Fake broker returning LTP map with string keys."""

    def __init__(self, as_string_keys=True, value=123.45):
        self.as_string_keys = as_string_keys
        self.value = value
        self.called_with = None

    def get_ltp_bulk(self, batch):
        # return keys as strings or ints depending on configuration
        self.called_with = list(batch)
        if self.as_string_keys:
            return {str(int(t)): float(self.value) for t in batch}
        else:
            return {int(t): float(self.value) for t in batch}


class FakeQuoteBroker:
    """Fake broker providing quote map with string keys and nested depth."""

    def __init__(self, as_string_keys=True, price=200.5):
        self.as_string_keys = as_string_keys
        self.price = price
        self.called_with = None

    def get_quote_bulk(self, batch):
        self.called_with = list(batch)
        if self.as_string_keys:
            return {
                str(int(t)): {
                    'last_price': float(self.price),
                    'depth': {'bids': [], 'asks': []},
                }
                for t in batch
            }
        return {
            int(t): {
                'last_price': float(self.price),
                'depth': {'bids': [], 'asks': []},
            }
            for t in batch
        }


class FakeSingleQuoteBroker:
    def __init__(self, price=50.0):
        self.price = price
        self.called = []

    def get_quote_by_token(self, token):
        self.called.append(token)
        return {'last_price': float(self.price)}


def test_try_ltp_bulk_handles_string_keys():
    broker = FakeLtpBroker(as_string_keys=True, value=555.5)
    s = PollingStreamer(
        broker_client=broker,
        on_tick=lambda t: None,
        instrument_resolver=None,
    )
    timestamp = int(time.time() * 1000)
    ticks = s._try_ltp_bulk([101], timestamp)
    assert ticks is not None
    assert len(ticks) == 1
    assert ticks[0]['instrument_token'] == 101
    assert ticks[0]['last_price'] == pytest.approx(555.5)


def test_try_ltp_bulk_handles_int_keys():
    broker = FakeLtpBroker(as_string_keys=False, value=111.0)
    s = PollingStreamer(
        broker_client=broker,
        on_tick=lambda t: None,
        instrument_resolver=None,
    )
    timestamp = int(time.time() * 1000)
    ticks = s._try_ltp_bulk([202], timestamp)
    assert ticks is not None
    assert len(ticks) == 1
    assert ticks[0]['last_price'] == pytest.approx(111.0)


def test_try_quote_bulk_handles_string_keys():
    broker = FakeQuoteBroker(as_string_keys=True, price=210.75)
    s = PollingStreamer(
        broker_client=broker,
        on_tick=lambda t: None,
        instrument_resolver=None,
    )
    timestamp = int(time.time() * 1000)
    ticks = s._try_quote_bulk([303], timestamp)
    assert ticks is not None
    assert len(ticks) == 1
    assert ticks[0]['instrument_token'] == 303
    assert ticks[0]['last_price'] == pytest.approx(210.75)
    assert 'depth' in ticks[0]


def test_fetch_ticks_prefers_ltp_then_quote_then_single():
    # 1) Broker that provides LTP should satisfy _fetch_ticks early
    broker1 = FakeLtpBroker(as_string_keys=True, value=777.0)
    s1 = PollingStreamer(
        broker_client=broker1,
        on_tick=lambda t: None,
        instrument_resolver=None,
    )
    t = s1._fetch_ticks([404])
    assert len(t) == 1 and t[0]['last_price'] == pytest.approx(777.0)

    # 2) Broker without LTP but with quote_bulk
    class OnlyQuoteBroker:
        def get_quote_bulk(self, batch):
            return {str(batch[0]): {'last_price': 88.8}}

    broker2 = OnlyQuoteBroker()
    s2 = PollingStreamer(
        broker_client=broker2,
        on_tick=lambda t: None,
        instrument_resolver=None,
    )
    t2 = s2._fetch_ticks([505])
    assert len(t2) == 1 and t2[0]['last_price'] == pytest.approx(88.8)

    # 3) Broker with only single quote
    single = FakeSingleQuoteBroker(price=33.33)
    s3 = PollingStreamer(
        broker_client=single,
        on_tick=lambda t: None,
        instrument_resolver=None,
    )
    t3 = s3._fetch_ticks([606])
    assert len(t3) == 1 and t3[0]['last_price'] == pytest.approx(33.33)


def test_subscribe_unsubscribe_and_tracked_tokens():
    broker = FakeLtpBroker()
    collected = []
    s = PollingStreamer(
        broker_client=broker,
        on_tick=lambda t: collected.append(t),
        instrument_resolver=None,
    )
    s.subscribe([1, 2, 3])
    tokens = s.tracked_tokens()
    assert set(tokens) == {1, 2, 3}
    s.unsubscribe([2])
    tokens2 = s.tracked_tokens()
    assert set(tokens2) == {1, 3}


def test_try_ltp_skips_zero_or_negative_values():
    class ZeroBroker:
        def get_ltp_bulk(self, batch):
            return {
                str(batch[0]): 0.0,
                str(batch[1] if len(batch) > 1 else batch[0]): -5.0,
            }

    s = PollingStreamer(
        broker_client=ZeroBroker(),
        on_tick=lambda t: None,
        instrument_resolver=None,
    )
    out = s._try_ltp_bulk([707, 808], int(time.time() * 1000))
    # both values zero/negative -> returns None
    assert out is None


def test_maybe_warn_rate_limits_sets_flag_and_metric():
    # Use a tiny batch_size and interval so safe_capacity is small
    broker = FakeLtpBroker()
    s = PollingStreamer(
        broker_client=broker,
        on_tick=lambda t: None,
        instrument_resolver=None,
        poll_interval_ms=100,
        batch_size=1,
    )
    # Simulate large token count -> warn
    s._maybe_warn_rate_limits(5000)
    assert s._rate_limit_warned is True
