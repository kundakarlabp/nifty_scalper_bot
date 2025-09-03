import shutil

from src.risk.session import Trade
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


def test_csv_report_creates_directory(tmp_path):
    """_generate_csv_report should create reports dir if missing."""
    reports_dir = project_root / "reports"
    if reports_dir.exists():
        shutil.rmtree(reports_dir)

    csv_path = project_root / "src/data/nifty_ohlc.csv"
    runner = BacktestRunner(csv_path)
    trade = Trade(
        symbol="X",
        direction="BUY",
        entry_price=100.0,
        quantity=1,
        order_id="1",
        atr_at_entry=0.0,
    )
    runner.session.add_trade(trade)
    runner.session.finalize_trade("1", exit_price=110.0)

    runner._generate_csv_report()

    assert (reports_dir / "trade_history.csv").exists()
