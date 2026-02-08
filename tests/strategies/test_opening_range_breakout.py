"""Tests for :mod:`nifty_scalper_bot.strategies.opening_range_breakout`."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

from nifty_scalper_bot.strategies.opening_range_breakout import (
    OpeningRangeBreakoutStrategy,
    OpeningRangeHistoryProvider,
    OpeningRangeSnapshot,
)


class DummyHistoryProvider(OpeningRangeHistoryProvider):
    """Simple provider returning a fixed snapshot sequence."""

    def __init__(self, snapshots: Sequence[OpeningRangeSnapshot]) -> None:
        self._snapshots = snapshots

    def get_opening_range_history(
        self,
        symbol: str,
        *,
        interval_minutes: int,
        limit: int,
    ) -> Sequence[OpeningRangeSnapshot]:
        return tuple(self._snapshots)[:limit]


def _build_snapshots(base_range: float) -> list[OpeningRangeSnapshot]:
    start = date(2024, 1, 8)
    return [
        OpeningRangeSnapshot(
            session=start + timedelta(days=idx),
            high=base_range + idx,
            low=idx,
        )
        for idx in range(7)
    ]


def test_generate_signal_passes_nr7_filter() -> None:
    """Strategy should emit signal when current OR range is narrowest."""

    provider = DummyHistoryProvider(_build_snapshots(20.0))
    strategy = OpeningRangeBreakoutStrategy(history_provider=provider)
    indicators = {
        "opening_range_high": 25.0,
        "opening_range_low": 24.0,
        "orb_break_direction": "UP",
        "orb_entry_price": 125.0,
    }

    signal = strategy.generate_signal("NIFTY24JANFUT", indicators, 125.0, None)

    assert signal is not None
    assert signal.action == "BUY"
    assert signal.metadata.get("nr7") is True


def test_generate_signal_blocks_when_nr7_fails() -> None:
    """Strategy should block signals when range is not narrowest."""

    snapshots = _build_snapshots(20.0)
    snapshots[0] = OpeningRangeSnapshot(
        session=date(2024, 1, 1),
        high=15.0,
        low=10.0,
    )
    provider = DummyHistoryProvider(snapshots)
    strategy = OpeningRangeBreakoutStrategy(history_provider=provider)
    indicators = {
        "opening_range_high": 35.0,
        "opening_range_low": 25.0,
        "orb_break_direction": "UP",
        "orb_entry_price": 140.0,
    }

    signal = strategy.generate_signal("NIFTY24JANFUT", indicators, 140.0, None)

    assert signal is None


def test_generate_signal_handles_provider_errors() -> None:
    """Provider errors should cause NR7 check to fail safe."""

    class FailingProvider(OpeningRangeHistoryProvider):
        def get_opening_range_history(
            self,
            symbol: str,
            *,
            interval_minutes: int,
            limit: int,
        ) -> Sequence[OpeningRangeSnapshot]:
            raise RuntimeError("boom")

    strategy = OpeningRangeBreakoutStrategy(history_provider=FailingProvider())
    indicators = {
        "opening_range_high": 25.0,
        "opening_range_low": 23.0,
        "orb_break_direction": "UP",
        "orb_entry_price": 126.0,
    }

    signal = strategy.generate_signal("NIFTY24JANFUT", indicators, 126.0, None)

    assert signal is None


def test_generate_signal_blocks_on_breakout_buffer() -> None:
    """Strategy should skip signals that do not clear the breakout buffer."""

    strategy = OpeningRangeBreakoutStrategy()
    indicators = {
        "opening_range_high": 25.0,
        "opening_range_low": 24.0,
        "orb_break_direction": "UP",
        "orb_entry_price": 25.1,
    }

    signal = strategy.generate_signal("NIFTY24JANFUT", indicators, 25.1, None)

    assert signal is None
