from __future__ import annotations

from src.strategies.runner import StrategyRunner
from src.strategies.registry import StrategyRegistry, DataProviderRegistry
from src.data.source import DataSource


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - no behavior
        pass


class DummyStrategy:
    pass


class DummyProvider(DataSource):
    def connect(self) -> None:  # pragma: no cover - simple stub
        pass

    def fetch_ohlc(self, token, start, end, timeframe):  # type: ignore[override]
        return None

    def get_last_price(self, symbol_or_token):  # type: ignore[override]
        return None


def test_strategy_and_data_provider_swap(monkeypatch) -> None:
    """Active strategy and data provider can be selected via env vars."""
    StrategyRegistry.register("dummy", DummyStrategy)
    DataProviderRegistry.register("dummy", DummyProvider, lambda: 1.0)
    monkeypatch.setenv("ACTIVE_STRATEGY", "dummy")
    monkeypatch.setenv("ACTIVE_DATA_PROVIDER", "dummy")
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    snap = runner.get_status_snapshot()
    assert snap["strategy"] == "dummy"
    assert snap["data_provider"] == "dummy"
