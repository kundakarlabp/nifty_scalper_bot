from __future__ import annotations

import time

import pytest

from nifty_scalper_bot.utils.errors import RateLimitError
from nifty_scalper_bot.utils.rate_limiter import RateLimiter


def test_rate_limiter_times_out_when_bucket_empty() -> None:
    limiter = RateLimiter()
    limiter.configure_bucket(name="orders", capacity=1, refill_rate_per_sec=0.1)

    limiter.acquire("orders", timeout=0.1)
    start = time.monotonic()
    with pytest.raises(RateLimitError):
        limiter.acquire("orders", timeout=0.2)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.1


def test_rate_limiter_refills_bucket_between_acquires() -> None:
    limiter = RateLimiter()
    limiter.configure_bucket(name="orders", capacity=1, refill_rate_per_sec=5.0)

    limiter.acquire("orders", timeout=0.1)
    after_first_acquire = limiter.snapshot()["orders"]
    assert 0.0 <= after_first_acquire["tokens"] < 1.0

    time.sleep(0.3)
    after_refill = limiter.snapshot()["orders"]
    assert after_refill["tokens"] == pytest.approx(1.0, rel=0.05)
    assert after_refill["tokens"] > after_first_acquire["tokens"]

    limiter.acquire("orders", timeout=0.5)
    after_second_acquire = limiter.snapshot()["orders"]
    assert after_second_acquire["tokens"] < after_refill["tokens"]
