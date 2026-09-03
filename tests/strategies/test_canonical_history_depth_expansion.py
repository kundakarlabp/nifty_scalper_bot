from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.runner import StrategyRunner


SYMBOL = "NFO:NIFTY26SEPFUT"
OPTION = "NFO:NIFTY26SEP24000CE"


def _bars(count: int) -> list[dict[str, object]]:
    start = datetime(2026, 9, 3, 3, 45, tzinfo=timezone.utc)
    return [
        {
            "timestamp": start + timedelta(minutes=index),
            "open": 24_000.0 + index,
            "high": 24_001.0 + index,
            "low": 23_999.0 + index,
            "close": 24_000.5 + index,
            "volume": 1_000 + index,
        }
        for index in range(count)
    ]


def _seed_indicator(
    engine: IndicatorEngine,
    rows: list[dict[str, object]],
    *,
    symbol: str = SYMBOL,
) -> None:
    engine.replace_history(symbol, rows, source="test", min_bars=1)


def _install_reseed(runner: StrategyRunner) -> None:
    def _reseed(symbol, bars, *, source="", min_bars=1):
        materialized = list(bars)
        runner._symbol_history[symbol] = materialized
        runner._indicator_engine.replace_history(
            symbol, materialized, source=source, min_bars=min_bars
        )
        return len(materialized)

    runner.reseed_history_from_bars = _reseed


async def test_sync_reseeds_when_canonical_depth_expands_with_same_latest_timestamp() -> None:
    canonical = _bars(80)
    shallow = canonical[-50:]
    assert shallow[-1]["timestamp"] == canonical[-1]["timestamp"]

    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    runner._normalize_symbol = lambda value: str(value)
    runner._symbol_history = {SYMBOL: list(shallow)}
    runner._indicator_engine = IndicatorEngine()
    _seed_indicator(runner._indicator_engine, shallow)
    runner._get_mdm_bars = lambda _symbol, limit: list(canonical)[-limit:]
    runner._set_symbol_hydration_state = lambda *_a, **_k: None
    runner._schedule_runtime_history_ensure = lambda *_a, **_k: True
    runner._market_data = SimpleNamespace(history_capacity_for=lambda *_a, **_k: 80)
    runner._data_hub = None
    _install_reseed(runner)

    result = runner.sync_history_from_mdm(
        SYMBOL,
        required_bars=50,
        reason="depth_expansion",
        role="futures_context",
        request_if_short=False,
    )

    assert result.success is True
    assert result.mdm_bars == 80
    assert result.runner_bars == 80
    assert result.indicator_bars == 80


async def test_selected_option_sync_mirrors_canonical_depth_beyond_readiness_minimum() -> None:
    """A 20-bar readiness minimum must not truncate ADX/regime derived history."""
    canonical = _bars(50)
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    runner._normalize_symbol = lambda value: str(value)
    runner._symbol_history = {}
    runner._indicator_engine = IndicatorEngine()
    runner._history_count_for_symbol = runner._indicator_engine.history_count
    runner._get_mdm_bars = lambda _symbol, limit: list(canonical)[-limit:]
    runner._set_symbol_hydration_state = lambda *_a, **_k: None
    runner._schedule_runtime_history_ensure = lambda *_a, **_k: True
    runner._market_data = SimpleNamespace(history_capacity_for=lambda *_a, **_k: 500)
    runner._data_hub = None
    _install_reseed(runner)

    result = runner.sync_history_from_mdm(
        OPTION,
        required_bars=20,
        reason="selected_option_projection",
        role="selected_option",
        request_if_short=False,
    )

    assert result.success is True
    assert result.mdm_bars == 50
    assert result.runner_bars == 50
    assert result.indicator_bars == 50
    assert runner._indicator_engine.get_adx(OPTION) is not None
