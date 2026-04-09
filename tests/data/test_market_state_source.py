from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nifty_scalper_bot.data.market_state import MarketState
from nifty_scalper_bot.data.source import DataIntegrityError, MarketDataSource


class _StubKite:
    def __init__(self) -> None:
        self.ltp_payload = {"NFO:101": {"last_price": 100.5}}
        self.rows = [
            {
                "date": datetime.now(timezone.utc).isoformat(),
                "open": 1,
                "high": 2,
                "low": 1,
                "close": 2,
            }
            for _ in range(35)
        ]

    def ltp(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        return {key: self.ltp_payload.get(key, {}) for key in symbols}

    def historical_data(
        self,
        token: int,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> list[dict[str, object]]:
        return list(self.rows)


def test_market_state_snapshot() -> None:
    state = MarketState()
    state.set_spot_ltp(24001.0)
    state.update_tick(101, 123.4)
    snap = state.get_snapshot()
    assert snap.spot_ltp == 24001.0
    assert snap.option_ltps[101] == 123.4


def test_source_falls_back_to_poll_when_ws_stale() -> None:
    state = MarketState()
    kite = _StubKite()
    source = MarketDataSource(kite, state)
    prices = source.get_ltp([101], stale_after_seconds=0)
    assert prices[101] == pytest.approx(100.5)
    assert state.get_snapshot().option_ltps[101] == pytest.approx(100.5)


def test_get_ohlc_validates_minimum_rows() -> None:
    state = MarketState()
    kite = _StubKite()
    kite.rows = kite.rows[:5]
    source = MarketDataSource(kite, state)
    with pytest.raises(DataIntegrityError):
        source.get_ohlc(101, "minute")
