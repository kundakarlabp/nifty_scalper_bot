from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.strategies.bar_builder import OneMinuteBar
from nifty_scalper_bot.strategies.runner import StrategyRunner


class _EngineProbe:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def update_price(self, symbol: str, _bar: dict[str, object], **kwargs: object) -> None:
        self.calls.append({'symbol': symbol, **kwargs})


class _Logger:
    def warning(self, *_a, **_k) -> None:
        return None


def test_runner_ingests_synthetic_gap_fill_as_non_provisional() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._last_bar_ts = {}
    runner._symbol_history = {}
    runner._candle_versions = {'NFO:NIFTY26MAYFUT': 0}
    runner._indicator_engine = _EngineProbe()
    runner._update_symbol_readiness = lambda _s: None
    runner._bracket_manager = None
    runner._strategy_manager = None
    runner._main_loop = None
    runner._logger = _Logger()

    ts = datetime(2026, 5, 1, 9, 15, tzinfo=timezone.utc)
    bar = OneMinuteBar(
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=100,
        start=ts,
        end=ts + timedelta(seconds=59),
        synthetic=True,
    )

    runner._ingest_bar('NFO:NIFTY26MAYFUT', bar, is_backfill=True)

    assert runner._indicator_engine.calls
    call = runner._indicator_engine.calls[-1]
    assert call['is_complete'] is True
    assert call['is_provisional'] is False
    assert call['volume'] == 0
