from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_backfill_dedupe_logic() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._history_backfill_inflight = set()
    runner._last_history_backfill_request = {}
    ok = StrategyRunner._should_request_history_backfill(runner, 'NFO:NIFTY26MAY25000CE', min_interval=300.0)
    assert ok is True
    again = StrategyRunner._should_request_history_backfill(runner, 'NFO:NIFTY26MAY25000CE', min_interval=300.0)
    assert again is False
