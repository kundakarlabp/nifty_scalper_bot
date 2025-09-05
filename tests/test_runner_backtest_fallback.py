from types import SimpleNamespace

from src.strategies.runner import StrategyRunner


def test_run_backtest_missing_csv(tmp_path):
    runner = StrategyRunner(telegram_controller=SimpleNamespace())
    result = runner.run_backtest(str(tmp_path / "missing.csv"))
    assert result.startswith("Backtest")
