from __future__ import annotations

import time

import pytest

from nifty_scalper_bot.quotes.freshness import FreshnessGuard
from nifty_scalper_bot.utils.errors import DataStaleError


def test_ensure_fresh_raises_on_stale_quote() -> None:
    guard = FreshnessGuard(stale_threshold_ms=100)
    stale_quote = {
        "symbol": "NIFTY",
        "ltp": 100.0,
        "ts_ms": int(time.time() * 1000) - 200,
    }
    with pytest.raises(DataStaleError):
        guard.ensure_fresh(stale_quote)
