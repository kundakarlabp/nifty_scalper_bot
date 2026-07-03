from __future__ import annotations

import asyncio
import inspect
import os
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

# Set default mock credentials for tests to prevent ConfigurationError during
# Settings initialization.
os.environ.setdefault("BROKER_API_KEY", "mock_api_key")
os.environ.setdefault("BROKER_API_SECRET", "mock_api_secret")


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point DATA_DIR at a per-test directory.

    Bracket and order managers persist state (virtual_brackets.json,
    orders.json, fill ledgers) under DATA_DIR (default ``data/``). Without
    isolation, one test's saved bracket is silently restored by the next
    test's manager, causing order-dependent failures. Tests that set their
    own DATA_DIR still win: their monkeypatch runs after this fixture.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "state"))


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
    index = pd.date_range("2024-01-01", periods=120, freq="1h")
    base = np.linspace(100, 105, len(index))
    noise = rng.normal(scale=0.3, size=len(index))
    price = base + noise
    return pd.DataFrame(
        {
            "open": price,
            "high": price + rng.normal(scale=0.2, size=len(index)),
            "low": price - rng.normal(scale=0.2, size=len(index)),
            "close": price,
            "volume": rng.integers(1_000, 5_000, size=len(index)),
        },
        index=index,
    )


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


def _close_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel pending tasks and close a test-owned event loop cleanly."""
    if loop.is_closed():
        return

    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    with suppress(Exception):
        loop.run_until_complete(loop.shutdown_asyncgens())
    with suppress(Exception):
        loop.run_until_complete(loop.shutdown_default_executor())
    loop.close()


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        _close_event_loop(loop)


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    if not inspect.iscoroutinefunction(pyfuncitem.obj):
        return None

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
            _close_event_loop(loop)
    else:
        loop.run_until_complete(pyfuncitem.obj(**kwargs))
    return True
