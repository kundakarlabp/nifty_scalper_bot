from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from nifty_scalper_bot.backtest.replay import ReplayHarness
from nifty_scalper_bot.execution.paper_fill_engine import PaperFillEngine


class _DummyDataHub:
    def __init__(self) -> None:
        self.quotes: dict[str, dict[str, float]] = {}

    def get_quote(self, symbol: str, allow_pull: bool = False) -> dict[str, float]:
        return self.quotes.setdefault(symbol, {"ltp": 100.0})


class _DummyResolver:
    def lot_size_for_symbol(self, symbol: str) -> int:  # pragma: no cover - not used
        return 75


class _TradeRecord:
    def __init__(self, action: str) -> None:
        self.action = action

    def to_dict(self) -> dict[str, str]:
        return {"action": self.action}


class _DummyRunner:
    def __init__(self) -> None:
        self.ticks: list[tuple[str, dict[str, float]]] = []

    def _on_tick(self, symbol: str, tick: dict[str, float]) -> None:
        self.ticks.append((symbol, tick))

    def get_status(self) -> dict[str, dict[str, list[_TradeRecord]]]:
        return {"symbols": {"OPT": {"trade_history": [_TradeRecord("BUY")]}}}


def _sample_frame() -> pd.DataFrame:
    index = pd.date_range(datetime(2024, 1, 1, 9, 30), periods=3, freq="T")
    return pd.DataFrame(
        {
            "option_open": [100.0, 101.0, 102.0],
            "option_high": [101.0, 102.0, 103.0],
            "option_low": [99.5, 100.5, 101.5],
            "option_close": [100.5, 101.5, 102.5],
            "option_volume": [1000, 1100, 1200],
            "index_open": [19500.0, 19510.0, 19520.0],
            "index_high": [19520.0, 19530.0, 19540.0],
            "index_low": [19480.0, 19490.0, 19500.0],
            "index_close": [19510.0, 19520.0, 19530.0],
            "index_volume": [5000, 5100, 5200],
        },
        index=index,
    )


def test_replay_harness_dispatches_ticks(tmp_path: Path) -> None:
    hub = _DummyDataHub()
    resolver = _DummyResolver()
    paper = PaperFillEngine(hub, resolver)
    runner = _DummyRunner()
    harness = ReplayHarness(runner, paper, option_symbol="OPT", index_symbol="IDX")

    frame = _sample_frame()
    result = harness.run_dataframe(frame)

    assert result.bars_processed == len(frame)
    assert len(runner.ticks) == len(frame) * 2
    assert result.trades == [{"action": "BUY"}]
    assert result.orders == []

    csv_path = tmp_path / "20240101.csv"
    frame.to_csv(csv_path)
    result_from_file = harness.run_day(tmp_path, "20240101")
    assert result_from_file.bars_processed == len(frame)
