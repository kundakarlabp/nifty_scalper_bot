import logging
import pandas as pd

from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:
        pass


class StubDataSource:
    def get_last_bars(self, n: int):
        idx = pd.date_range("2024-01-01 09:30", periods=3, freq="1min")
        data = {"open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}
        return pd.DataFrame(data, index=idx)

    def fetch_ohlc(self, token, start, end, timeframe):
        return self.get_last_bars(3)

    def get_last_price(self, symbol):
        return 1.0

    def api_health(self):
        return {}


def test_get_recent_bars_logs_when_insufficient(caplog):
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    runner.data_source = StubDataSource()

    with caplog.at_level(logging.WARNING):
        runner.get_recent_bars(5)

    assert "Fetched 3 bars (<5)" in caplog.text
