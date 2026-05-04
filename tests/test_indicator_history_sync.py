from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_runner_exposes_history_ingestion() -> None:
    assert hasattr(StrategyRunner, 'ingest_historical_bar')
