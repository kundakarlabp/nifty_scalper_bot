from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_runner_has_backfill_guards() -> None:
    assert hasattr(StrategyRunner, '_should_request_history_backfill')
