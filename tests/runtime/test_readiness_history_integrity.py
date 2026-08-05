from __future__ import annotations

import inspect

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_stale_spot_tick_is_not_promoted_by_history_bars() -> None:
    manager = MarketDataManager.__new__(MarketDataManager)
    manager._tick_stale_threshold_ms = 1_000
    manager._is_symbol_fresh = lambda _symbol, _threshold: False

    state = manager._readiness_state(
        {
            "NSE:NIFTY": 60,
            "NFO:NIFTY26AUG25000CE": 60,
            "NFO:NIFTY26AUG25000PE": 60,
        },
        20,
        {
            "spot": "NSE:NIFTY",
            "futures": "",
            "atm_ce": "NFO:NIFTY26AUG25000CE",
            "atm_pe": "NFO:NIFTY26AUG25000PE",
            "options": [
                "NFO:NIFTY26AUG25000CE",
                "NFO:NIFTY26AUG25000PE",
            ],
        },
    )

    assert state["spot_ready"] is False
    assert state["hard_ready"] is False


def test_history_reseed_does_not_activate_deferred_symbol() -> None:
    source = inspect.getsource(StrategyRunner.reseed_history_from_bars)

    assert "_active_symbols.add" not in source
    assert "_tracked_symbols.add" not in source


def test_fallback_backfill_uses_idempotent_reseed_not_append_ingestion() -> None:
    source = inspect.getsource(StrategyRunner._backfill_history)

    assert "reseed_history_from_bars" in source
    assert "ingest_historical_bar" not in source
