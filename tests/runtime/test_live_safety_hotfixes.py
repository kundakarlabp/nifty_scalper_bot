from __future__ import annotations

from datetime import datetime, timedelta
import logging
import sys
import types

import pandas as pd

from nifty_scalper_bot.runtime_live_safety_hotfixes import install_live_safety_hotfixes


def test_core_app_time_proxy_is_callable_and_has_time_api() -> None:
    module = types.ModuleType("nifty_scalper_bot.core.app")
    from datetime import time as datetime_time

    module.time = datetime_time
    sys.modules[module.__name__] = module
    try:
        install_live_safety_hotfixes()
        assert module.time(15, 24).hour == 15
        assert callable(module.time.time)
        assert callable(module.time.monotonic)
    finally:
        sys.modules.pop(module.__name__, None)


def test_candle_engine_drops_stale_finalized_candle_without_raising() -> None:
    install_live_safety_hotfixes()
    from nifty_scalper_bot.data.candle_engine import CandleEngine
    from nifty_scalper_bot.data.time_contract import IST

    engine = CandleEngine()
    engine.symbol = "NFO:NIFTY2671424100CE"
    base = pd.Timestamp(datetime.now(tz=IST).replace(second=0, microsecond=0)) - pd.Timedelta(minutes=10)
    engine.df = pd.DataFrame(
        [
            {
                "timestamp": base + pd.Timedelta(minutes=1),
                "open": 10.0,
                "high": 11.0,
                "low": 9.5,
                "close": 10.5,
                "volume": 100.0,
            }
        ]
    )
    engine.current_candle = {
        "timestamp": base,
        "open": 9.0,
        "high": 10.0,
        "low": 8.5,
        "close": 9.5,
        "volume": 50.0,
    }

    assert engine._finalize_current_candle() is None
    assert engine.current_candle is None
    assert engine.df["timestamp"].is_monotonic_increasing


def test_non_incremental_fill_filter_suppresses_duplicate_records() -> None:
    install_live_safety_hotfixes()
    logger = logging.getLogger("nifty_scalper_bot.execution.position_manager")
    record1 = logger.makeRecord(
        logger.name,
        logging.WARNING,
        __file__,
        1,
        "Ignoring non-incremental fill for order 2075432831139897346",
        args=(),
        exc_info=None,
    )
    record2 = logger.makeRecord(
        logger.name,
        logging.WARNING,
        __file__,
        1,
        "Ignoring non-incremental fill for order 2075432831139897346",
        args=(),
        exc_info=None,
    )
    assert all(filter_.filter(record1) for filter_ in logger.filters)
    assert not all(filter_.filter(record2) for filter_ in logger.filters)


def test_runner_signal_preparation_blocks_when_live_not_armed(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    install_live_safety_hotfixes()

    from nifty_scalper_bot.strategies.runner import StrategyRunner

    class DummySignal:
        symbol = "NFO:NIFTY2671424250PE"

    runner = object.__new__(StrategyRunner)
    runner._runtime_live_orders_armed = False
    runner._runtime_readiness_reason = "position_reconciliation_failed"
    runner._logger = logging.getLogger("test.runner")

    scheduled, reason = runner._schedule_signal_preparation(DummySignal(), 100.0, datetime.now(), "trace-1")
    assert scheduled is False
    assert str(reason).startswith("execution_not_armed")
