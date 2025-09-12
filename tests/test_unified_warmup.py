from datetime import datetime, timedelta
from types import SimpleNamespace

from src.data.source import LiveKiteSource, MinuteBarBuilder
from src.strategies.runner import StrategyRunner
from src.config import settings


def test_ensure_warmup_builder() -> None:
    ds = LiveKiteSource(kite=None)
    ds.hist_mode = "live_warmup"
    ds.bar_builder = MinuteBarBuilder(max_bars=10)
    n = 3
    start = datetime(2024, 1, 1, 9, 15)
    ds.bar_builder.on_tick({"timestamp": start, "last_price": 100.0, "volume": 1})
    assert ds.ensure_warmup(n) is False
    ds.bar_builder.on_tick({"timestamp": start + timedelta(minutes=1), "last_price": 101.0, "volume": 1})
    ds.bar_builder.on_tick({"timestamp": start + timedelta(minutes=2), "last_price": 102.0, "volume": 1})
    assert ds.ensure_warmup(n) is True


def test_runner_waits_for_warmup(monkeypatch) -> None:
    runner = StrategyRunner(telegram_controller=SimpleNamespace())
    runner._start_time = runner._parse_hhmm("09:20")
    runner._end_time = runner._parse_hhmm("15:25")
    monkeypatch.setattr(settings, "WARMUP_MIN_BARS", 3, raising=False)
    monkeypatch.setattr("src.strategies.runner.is_market_open", lambda _now: True)
    runner.data_source = SimpleNamespace(
        have_min_bars=lambda n: False, ensure_warmup=lambda n: False
    )
    called = {"flag": False}
    runner.strategy = SimpleNamespace(
        generate_signal=lambda df, current_tick=None: called.update(flag=True)
    )
    runner.process_tick(None)
    assert called["flag"] is False


