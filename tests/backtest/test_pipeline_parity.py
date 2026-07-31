from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from nifty_scalper_bot.backtest.parity import SinglePipelineParity
from nifty_scalper_bot.execution.paper_fill_engine import PaperFillEngine


class _StubDataHub:
    def __init__(self) -> None:
        self.quotes: dict[str, dict[str, float]] = {}

    def get_quote(self, symbol: str, allow_pull: bool = False) -> dict[str, float]:
        return self.quotes.setdefault(symbol, {"ltp": 100.0})


class _StubResolver:
    def lot_size_for_symbol(self, symbol: str) -> int:
        return 50


class _ParityRunner:
    def __init__(self) -> None:
        self.ticks: list[tuple[str, float]] = []
        self._trades: list[dict[str, Any]] = []

    def on_tick_event(self, tick: dict[str, Any]) -> None:
        symbol = str(tick["symbol"])
        self.ticks.append((symbol, float(tick.get("close", 0.0))))
        if symbol == "OPT" and float(tick.get("close", 0.0)) > 101.0:
            self._trades.append({"symbol": symbol, "close": float(tick["close"])})

    def get_status(self) -> dict[str, dict[str, Any]]:
        return {"symbols": {"OPT": {"trade_history": list(self._trades)}}}


def _sample_frame() -> pd.DataFrame:
    index = pd.date_range(datetime(2024, 1, 1, 9, 30), periods=4, freq="T")
    return pd.DataFrame(
        {
            "option_open": [100.0, 101.2, 102.4, 103.0],
            "option_high": [100.5, 101.5, 103.0, 103.2],
            "option_low": [99.8, 100.8, 101.8, 102.2],
            "option_close": [100.1, 101.4, 102.6, 103.1],
            "option_volume": [1000, 1100, 1200, 1300],
        },
        index=index,
    )


def _runner_factory() -> _ParityRunner:
    return _ParityRunner()


def _paper_factory() -> PaperFillEngine:
    return PaperFillEngine(_StubDataHub(), _StubResolver())


def test_single_pipeline_parity_matches() -> None:
    frame = _sample_frame()
    parity = SinglePipelineParity(
        runner_factory=_runner_factory,
        reference_runner_factory=_runner_factory,
        paper_factory=_paper_factory,
        option_symbol="OPT",
    )

    result = parity.run_frame(frame)

    assert result.diff == ""
    assert result.live.trades == result.backtest.trades
    assert result.live.orders == result.backtest.orders


def test_single_pipeline_parity_requires_independent_reference() -> None:
    parity = SinglePipelineParity(
        runner_factory=_runner_factory,
        paper_factory=_paper_factory,
        option_symbol="OPT",
    )

    try:
        parity.run_frame(_sample_frame())
    except ValueError as exc:
        assert "reference_runner_factory" in str(exc)
    else:  # pragma: no cover - explicit contract failure
        raise AssertionError("parity must not compare a pipeline with itself")
