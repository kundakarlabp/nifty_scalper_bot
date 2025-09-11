from src.backtesting.backtest_engine import BacktestEngine


def test_summary_metrics_new_fields(tmp_path):
    engine = BacktestEngine(feed=None, cfg=None, risk=None, sim=None, outdir=str(tmp_path))
    trades = [
        {"pnl_R": 1.0, "pnl_rupees": 100.0, "slippage": 0.5, "mae": 2.0},
        {"pnl_R": -0.5, "pnl_rupees": -50.0, "slippage": 0.3, "mae": 1.5},
    ]
    summary = engine._summary_metrics(trades)
    assert summary["PnL"] == 50.0
    assert summary["HitRate"] == 50.0
    assert summary["AvgSlippage"] == 0.4
    assert summary["MaxAE"] == 2.0
