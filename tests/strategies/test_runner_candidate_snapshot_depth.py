from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.strategies.runner import StrategyRunner


class _Logger:
    def debug(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_candidate_snapshot_preserves_normalized_depth_availability() -> None:
    option_symbol = "NFO:NIFTY26AUG24250CE"
    spot = SimpleNamespace(ltp=24242.0, canonical_symbol="NSE:NIFTY")
    option = SimpleNamespace(
        canonical_symbol=option_symbol,
        ltp=127.1,
        bid=127.05,
        ask=127.15,
        mid=127.1,
        tick_age_s=0.25,
        source="ws_full",
        real_ticks_last_60s=15,
        latest_candle_provisional=False,
        latest_candle_synthetic=False,
        latest_candle_volume=1300.0,
        ohlc_valid=True,
        depth_available=True,
        bid_missing=False,
        ask_missing=False,
        bid_ask_source="depth",
        tradable_quote=True,
    )
    market_data = SimpleNamespace(
        get_symbol_snapshot=lambda symbol: spot if symbol == "NIFTY" else option,
        request_symbol_subscription=lambda symbol: None,
    )
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._market_data = market_data
    runner._logger = _Logger()
    runner._resolve_candidate_contracts = lambda **kwargs: [(option_symbol, 24250)]

    snapshots, refresh_pending = await runner.build_candidate_snapshots_async(
        atm_strike=24250,
        window_each_side=1,
    )

    assert refresh_pending is False
    assert len(snapshots) == 1
    assert snapshots[0]["depth_available"] is True
    # Spread in this public snapshot contract is percent, not decimal ratio.
    assert snapshots[0]["spread_pct"] == pytest.approx(
        (127.15 - 127.05) / ((127.05 + 127.15) / 2) * 100.0
    )

    # Invalid/crossed prices cannot manufacture an executable spread.
    option.ask = 126.9
    crossed, refresh_pending = await runner.build_candidate_snapshots_async(
        atm_strike=24250,
        window_each_side=1,
    )
    assert refresh_pending is False
    assert crossed[0]["spread_pct"] is None


@pytest.mark.asyncio
async def test_missing_option_quote_does_not_erase_healthy_basket_peer() -> None:
    healthy_symbol = "NFO:NIFTY26AUG24250CE"
    missing_symbol = "NFO:NIFTY26AUG24300CE"
    spot = SimpleNamespace(ltp=24242.0, canonical_symbol="NSE:NIFTY")
    healthy = SimpleNamespace(
        canonical_symbol=healthy_symbol,
        ltp=127.1,
        bid=127.05,
        ask=127.15,
        mid=127.1,
        tick_age_s=0.25,
        source="ws_full",
        real_ticks_last_60s=15,
        latest_candle_provisional=False,
        latest_candle_synthetic=False,
        latest_candle_volume=1300.0,
        ohlc_valid=True,
        depth_available=True,
        bid_missing=False,
        ask_missing=False,
        bid_ask_source="depth",
        tradable_quote=True,
    )

    def get_symbol_snapshot(symbol):
        if symbol == "NIFTY":
            return spot
        if symbol == healthy_symbol:
            return healthy
        return None

    runner = StrategyRunner.__new__(StrategyRunner)
    runner._market_data = SimpleNamespace(
        get_symbol_snapshot=get_symbol_snapshot,
        request_symbol_subscription=lambda symbol: None,
    )
    runner._logger = _Logger()
    runner._resolve_candidate_contracts = lambda **kwargs: [
        (healthy_symbol, 24250),
        (missing_symbol, 24300),
    ]
    snapshots, pending = await runner.build_candidate_snapshots_async(
        atm_strike=24250, window_each_side=1
    )
    assert [candidate["symbol"] for candidate in snapshots] == [healthy_symbol]
    assert pending is False

    runner._resolve_candidate_contracts = lambda **kwargs: [(missing_symbol, 24300)]
    snapshots, pending = await runner.build_candidate_snapshots_async(
        atm_strike=24250, window_each_side=1
    )
    assert snapshots == []
    assert pending is True
