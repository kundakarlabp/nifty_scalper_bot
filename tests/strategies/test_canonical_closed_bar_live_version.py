"""MDM finalized WS bars advance LIVE provenance; REST hydration never does."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
import threading

import pytest

from nifty_scalper_bot.strategies.runner import StrategyRunner


SYMBOL = "NFO:NIFTY26N1725100CE"


class _Indicators:
    def __init__(self) -> None:
        self.bars: list[datetime] = []

    def update_bar(self, _symbol: str, bar: object) -> None:
        self.bars.append(bar.timestamp)

    def get_vwap(self, _symbol: str) -> float:
        return 110.0


def _runner() -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = logging.getLogger("test.runner.canonical_live_provenance")
    runner._lock = threading.RLock()
    runner._active_symbols = set()
    runner._data_phase = {}
    runner._symbol_state = {}
    runner._symbol_states = {}
    runner._symbol_history = {}
    runner._last_bar_ts = {}
    runner._candle_versions = defaultdict(int)
    runner._live_bar_seen = set()
    runner._vwap_state = {}
    runner._hydration_ready_streak = {}
    runner._bracket_manager = None
    runner._indicator_engine = _Indicators()
    runner._update_symbol_readiness = lambda _symbol: None
    runner._should_log_throttled = lambda *_args, **_kwargs: False
    return runner


def _bar(minute: datetime, *, source: str) -> dict[str, object]:
    return {
        "symbol": SYMBOL,
        "timestamp": minute,
        "open": 110.0,
        "high": 112.0,
        "low": 109.0,
        "close": 111.0,
        "volume": 65_000,
        "source": source,
    }


@pytest.mark.parametrize("canonical_source", ["ws_candle", "clock_flush_candle"])
def test_finalized_mdm_candle_marks_live_once(canonical_source: str) -> None:
    runner = _runner()
    minute = datetime.now(timezone.utc).replace(
        second=0, microsecond=0
    ) - timedelta(minutes=3)

    # REST hydration must not manufacture a completed LIVE bar/version.
    runner.ingest_historical_bar(_bar(minute, source="historical"))
    assert runner._candle_versions[SYMBOL] == 0
    assert runner._data_phase[SYMBOL] == "HYDRATION"
    assert SYMBOL not in runner._live_bar_seen

    closed = _bar(minute + timedelta(minutes=1), source=canonical_source)
    runner.ingest_historical_bar(closed)
    assert runner._candle_versions[SYMBOL] == 1
    assert runner._data_phase[SYMBOL] == "LIVE"
    assert SYMBOL in runner._live_bar_seen
    assert runner._symbol_state[SYMBOL].vwap == pytest.approx(111.0)
    assert len(runner._indicator_engine.bars) == 2

    # A duplicate publish cannot double-count OHLCV, versions or VWAP.
    runner.ingest_historical_bar(closed)
    assert runner._candle_versions[SYMBOL] == 1
    assert len(runner._indicator_engine.bars) == 2

    # A subsequent completed canonical minute progresses versions monotonically.
    runner.ingest_historical_bar(
        _bar(minute + timedelta(minutes=2), source=canonical_source)
    )
    assert runner._candle_versions[SYMBOL] == 2
    assert len(runner._indicator_engine.bars) == 3


def test_rested_historical_and_partial_bars_never_advance_live_version() -> None:
    runner = _runner()
    minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    runner.ingest_historical_bar(
        _bar(minute - timedelta(minutes=2), source="historical")
    )
    runner.ingest_historical_bar(
        _bar(minute - timedelta(minutes=1), source="rest_history")
    )
    runner.ingest_historical_bar(_bar(minute, source="ws_candle"))
    assert runner._candle_versions[SYMBOL] == 0
    assert runner._data_phase[SYMBOL] == "HYDRATION"
    assert len(runner._indicator_engine.bars) == 2
