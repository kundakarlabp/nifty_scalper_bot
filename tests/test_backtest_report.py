from tests.true_backtest_dynamic import BacktestRunner, project_root


def test_placeholder_report_created(tmp_path):
    """Backtest should write a placeholder report when no trades occur."""
    csv_path = project_root / "src/data/nifty_ohlc.csv"
    runner = BacktestRunner(csv_path)
    runner.run()
    report_path = project_root / "reports" / "backtest_report.txt"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "No trades were executed" in content
