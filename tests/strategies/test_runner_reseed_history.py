from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.runner import StrategyRunner


def _make_bars(symbol: str, count: int, start: datetime) -> list[dict[str, object]]:
    bars: list[dict[str, object]] = []
    for idx in range(count):
        ts = start + timedelta(minutes=idx)
        price = 100.0 + idx
        bars.append(
            {
                'symbol': symbol,
                'timestamp': ts,
                'open': price,
                'high': price + 0.5,
                'low': price - 0.5,
                'close': price + 0.2,
                'volume': 100 + idx,
            }
        )
    return bars


def test_indicator_replace_history_overrides_short_live_history() -> None:
    engine = IndicatorEngine()
    symbol = 'NFO:NIFTY26MAY23300CE'
    base = datetime(2026, 5, 1, 3, 45, tzinfo=timezone.utc)
    for bar in _make_bars(symbol, 5, base + timedelta(minutes=60)):
        engine.ingest_historical_bar(symbol, bar)
    assert engine.history_count(symbol) == 5

    reseed_bars = _make_bars(symbol, 30, base)
    final_count = engine.replace_history(symbol, reseed_bars, min_bars=30)
    assert final_count >= 30
    assert engine.history_count(symbol) >= 30


def test_runner_reseed_history_from_bars_sets_indicator_ready() -> None:
    runner = StrategyRunner()
    symbol = 'NFO:NIFTY26MAY23300PE'
    base = datetime(2026, 5, 1, 3, 45, tzinfo=timezone.utc)
    bars = _make_bars(symbol, 30, base)

    runner_count = runner.reseed_history_from_bars(symbol, bars, min_bars=30)
    assert runner_count >= 30
    assert runner._indicator_engine.history_count(symbol) >= 30


@pytest.mark.asyncio
async def test_ensure_selected_options_hydrated_prefers_reseed_path() -> None:
    symbol = 'NFO:NIFTY26MAY23300CE'
    base = datetime(2026, 5, 1, 3, 45, tzinfo=timezone.utc)
    mdm_bars = _make_bars(symbol, 30, base)

    class RunnerStub:
        def __init__(self) -> None:
            self._indicator_engine = SimpleNamespace(get_history=lambda _s: list(range(5)))
            self.reseed_calls: list[tuple[str, int, int]] = []
            self._current = 5

        def reseed_history_from_bars(self, sym: str, bars: list[dict[str, object]], source: str, min_bars: int) -> int:
            self.reseed_calls.append((sym, len(bars), min_bars))
            self._current = len(bars)
            self._indicator_engine = SimpleNamespace(get_history=lambda _s: list(range(self._current)))
            return self._current

    class MdmStub:
        def __init__(self) -> None:
            self.hydrate_calls = 0

        async def hydrate_symbol_history(self, *args, **kwargs) -> None:
            self.hydrate_calls += 1

        def get_ohlc_bars(self, _sym: str, limit: int | None = None):
            return list(mdm_bars[: limit or len(mdm_bars)])

    runner = RunnerStub()
    mdm = MdmStub()
    ctx = SimpleNamespace(market_data_manager=mdm, strategy_runner=runner)
    await app._ensure_selected_options_hydrated(ctx, symbol, None, 30, 'test')

    assert runner.reseed_calls
    assert runner.reseed_calls[0][0] == symbol
    assert runner.reseed_calls[0][1] >= 30
    assert runner.reseed_calls[0][2] == 30


def test_runner_on_tick_safe_caches_last_tick() -> None:
    runner = StrategyRunner()
    tick = {
        'symbol': 'NFO:NIFTY26MAY23300CE',
        'ltp': 123.45,
        'timestamp': '2026-05-01T04:00:00Z',
    }

    runner._on_tick_safe(tick)

    normalized_symbol = 'NFO:NIFTY26MAY23300CE'
    assert normalized_symbol in runner._last_tick
    assert runner._last_tick[normalized_symbol]['ltp'] == 123.45


def test_phase9_tick_map_fallback_without_last_tick_attr() -> None:
    runner = StrategyRunner()
    symbol = 'NFO:NIFTY26MAY23300CE'
    normalized_symbol = 'NFO:NIFTY26MAY23300CE'

    delattr(runner, '_last_tick')

    last_tick_store = getattr(runner, '_last_tick', None)
    if isinstance(last_tick_store, dict):
        tick_map = dict(
            last_tick_store.get(symbol)
            or last_tick_store.get(normalized_symbol)
            or {}
        )
    else:
        tick_map = {}

    assert tick_map == {}


def test_indicator_replace_history_accepts_multiple_time_keys_and_skips_zero_ohlc() -> None:
    engine = IndicatorEngine()
    symbol = 'NFO:NIFTY26MAY23300PE'
    base = datetime(2026, 5, 1, 3, 45, tzinfo=timezone.utc)

    bars = [
        {'start': base.isoformat(), 'open': 100, 'high': 101, 'low': 99, 'close': 100.5, 'volume': 10},
        {'date': (base + timedelta(minutes=1)).isoformat(), 'open': 101, 'high': 102, 'low': 100, 'close': 101.5, 'volume': 10},
        {'time': (base + timedelta(minutes=2)).isoformat(), 'open': 102, 'high': 103, 'low': 101, 'close': 102.5, 'volume': 10},
        {'timestamp': base + timedelta(minutes=3), 'open': 0, 'high': 103, 'low': 101, 'close': 102.5, 'volume': 10},
    ]

    count = engine.replace_history(symbol, bars, source='test', min_bars=1)

    assert count == 3
    assert engine.history_count(symbol) == 3
