import pandas as pd

from src.strategies.runner import StrategyRunner


def test_runner_retries_empty_df(monkeypatch):
    class _Telegram:
        def send_message(self, msg: str) -> None:
            pass

    runner = StrategyRunner(telegram_controller=_Telegram())
    runner._start_time = runner._parse_hhmm("09:20")
    runner._end_time = runner._parse_hhmm("15:25")
    runner.data_source = object()  # placeholder to satisfy attribute

    empty = pd.DataFrame()
    calls: list[int] = []

    def fetch_mock():
        calls.append(1)
        return empty

    monkeypatch.setattr(runner, "_fetch_spot_ohlc", fetch_mock)

    called = False

    class _Strategy:
        def generate_signal(self, df, current_tick=None):
            nonlocal called
            called = True
            return {}

    runner.strategy = _Strategy()
    runner.process_tick(None)
    assert len(calls) == 2
    assert called is False
    assert runner._last_flow_debug["reason_block"] == "insufficient_data"
