from __future__ import annotations

import asyncio
import inspect
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from src.nifty_scalper_bot.backtesting.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
)

from nifty_scalper_bot.core.trading_switch import trading_switch


@pytest.fixture(autouse=True)
def reset_trading_switch() -> Generator[None, None, None]:
    switch = trading_switch()
    with suppress(Exception):
        switch.resume()
    yield
    with suppress(Exception):
        switch.resume()


@dataclass
class DeterministicStrategy:
    name: str = "deterministic"

    def generate_signals(
        self, market_data: pd.DataFrame
    ) -> pd.Series:  # pragma: no cover - test helper
        momentum = market_data["close"].pct_change().fillna(0.0)
        signals = pd.Series(0, index=market_data.index)
        signals.loc[momentum > 0] = 1
        signals.loc[momentum < 0] = -1
        return signals


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    index = pd.date_range("2024-01-01", periods=120, freq="1H")
    base = np.linspace(100, 105, len(index))
    noise = rng.normal(scale=0.3, size=len(index))
    price = base + noise
    data = pd.DataFrame(
        {
            "open": price,
            "high": price + rng.normal(scale=0.2, size=len(index)),
            "low": price - rng.normal(scale=0.2, size=len(index)),
            "close": price,
            "volume": rng.integers(1_000, 5_000, size=len(index)),
        },
        index=index,
    )
    return data


@pytest.fixture
def deterministic_strategy() -> DeterministicStrategy:
    return DeterministicStrategy()


@pytest.fixture
def default_backtest_config(tmp_path: Path) -> BacktestConfig:
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    return BacktestConfig(output_directory=output_dir)


@pytest.fixture
def backtest_engine(
    sample_price_data: pd.DataFrame,
    deterministic_strategy: DeterministicStrategy,
    default_backtest_config: BacktestConfig,
) -> BacktestEngine:
    return BacktestEngine(
        sample_price_data, deterministic_strategy, default_backtest_config
    )


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool:
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        loop = pyfuncitem.funcargs.get("event_loop")
        kwargs = {
            name: pyfuncitem.funcargs[name]
            for name in pyfuncitem._fixtureinfo.argnames  # type: ignore[attr-defined]
        }
        if loop is None or not isinstance(loop, asyncio.AbstractEventLoop):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(pyfuncitem.obj(**kwargs))
            finally:
                loop.close()
        else:
            loop.run_until_complete(pyfuncitem.obj(**kwargs))
        return True
    return False
