import logging
from types import SimpleNamespace

from src.strategies.runner import StrategyRunner


def test_run_backtest_no_traceback(tmp_path, caplog):
    """Run backtest with an invalid CSV path and ensure no traceback is logged."""
    invalid_dir = tmp_path / "dir"
    invalid_dir.mkdir()
    runner = StrategyRunner(telegram_controller=SimpleNamespace())
    with caplog.at_level(logging.ERROR, logger="StrategyRunner"):
        result = runner.run_backtest(str(invalid_dir))
    assert result.startswith("Backtest error")
    assert "Traceback" not in caplog.text
