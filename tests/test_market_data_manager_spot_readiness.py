"""Tests for MarketDataManager fresh spot tick helpers."""

from __future__ import annotations

import asyncio
import time

import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _build_mdm() -> MarketDataManager:
    """Build a bare MarketDataManager without broker/WS for direct cache access."""
    return MarketDataManager(broker=None, websocket=None, settings={})


def _seed_tick(
    mdm: MarketDataManager,
    *,
    symbol: str = "NSE:NIFTY",
    price: float = 24500.0,
    source: str = "ws",
    age_s: float = 0.0,
) -> None:
    """Seed the MDM tick caches with a freshly-timestamped tick."""
    now = time.time() - max(0.0, age_s)
    tick = {
        "symbol": symbol,
        "ltp": price,
        "last_price": price,
        "source": source,
        "received_at": now,
        "timestamp": now,
    }
    with mdm._lock:
        mdm._latest_ticks[symbol] = tick
        mdm._tick_cache[symbol] = tick
        mdm._last_tick_time[symbol] = now
        mdm._last_tick_wallclock[symbol] = now
        mdm._last_tick_source[symbol] = source


class TestGetFreshSpotTick:
    def test_fresh_ws_tick_is_accepted(self) -> None:
        mdm = _build_mdm()
        _seed_tick(mdm, source="ws", price=24500.0)
        tick = mdm.get_fresh_spot_tick("NSE:NIFTY")
        assert tick is not None
        assert tick["last_price"] == pytest.approx(24500.0)

    def test_stale_ws_tick_is_rejected(self) -> None:
        mdm = _build_mdm()
        _seed_tick(mdm, source="ws", age_s=600.0)
        tick = mdm.get_fresh_spot_tick("NSE:NIFTY", max_age_seconds=60.0)
        assert tick is None

    def test_rest_tick_rejected_when_require_ws(self) -> None:
        mdm = _build_mdm()
        _seed_tick(mdm, source="rest", price=24500.0)
        tick = mdm.get_fresh_spot_tick(
            "NSE:NIFTY", max_age_seconds=300.0, require_ws=True
        )
        assert tick is None

    def test_rest_tick_accepted_when_require_ws_false(self) -> None:
        mdm = _build_mdm()
        _seed_tick(mdm, source="rest", price=24500.0)
        tick = mdm.get_fresh_spot_tick(
            "NSE:NIFTY", max_age_seconds=300.0, require_ws=False
        )
        assert tick is not None
        assert tick["last_price"] == pytest.approx(24500.0)

    def test_zero_price_rejected(self) -> None:
        mdm = _build_mdm()
        _seed_tick(mdm, source="ws", price=0.0)
        tick = mdm.get_fresh_spot_tick("NSE:NIFTY", max_age_seconds=120.0)
        assert tick is None

    def test_no_tick_returns_none(self) -> None:
        mdm = _build_mdm()
        tick = mdm.get_fresh_spot_tick("NSE:NIFTY", max_age_seconds=60.0)
        assert tick is None


class TestWaitForFreshSpotTick:
    def test_waits_until_tick_arrives(self) -> None:
        mdm = _build_mdm()

        async def runner() -> dict | None:
            async def deliver_tick() -> None:
                await asyncio.sleep(0.1)
                _seed_tick(mdm, source="ws", price=24700.0)

            asyncio.create_task(deliver_tick())
            return await mdm.wait_for_fresh_spot_tick(
                "NSE:NIFTY",
                timeout=2.0,
                max_age_seconds=60.0,
                poll_interval=0.05,
            )

        tick = asyncio.run(runner())
        assert tick is not None
        assert tick["last_price"] == pytest.approx(24700.0)

    def test_times_out_when_no_tick(self) -> None:
        mdm = _build_mdm()

        async def runner() -> dict | None:
            return await mdm.wait_for_fresh_spot_tick(
                "NSE:NIFTY",
                timeout=0.2,
                max_age_seconds=60.0,
                poll_interval=0.05,
            )

        tick = asyncio.run(runner())
        assert tick is None


def test_process_ticks_marks_nifty_spot_ready_immediately() -> None:
    mdm = _build_mdm()
    with mdm._lock:
        mdm._symbol_by_token[256265] = "NSE:NIFTY"
    mdm.process_ticks([{"instrument_token": 256265, "last_price": 24078.45}])
    assert mdm.symbol_has_tick("NSE:NIFTY") is True
    assert mdm.get_cached_ltp("NSE:NIFTY", require_ws=True) == pytest.approx(24078.45)
    assert mdm.get_fresh_spot_tick("NSE:NIFTY", require_ws=True) is not None
