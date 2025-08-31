import pandas as pd
from zoneinfo import ZoneInfo
from src.backtesting.data_feed import SpotFeed
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.sim_connector import SimConnector
from src.strategies.strategy_config import StrategyConfig
from src.risk.limits import RiskEngine, LimitConfig


def test_backtest_writes_equity_curve(tmp_path):
    idx = pd.date_range("2024-01-01 09:30", periods=1, freq="1min")
    df = pd.DataFrame(
        {"open": [100], "high": [101], "low": [99], "close": [100], "volume": [1000]},
        index=idx,
    )
    feed = SpotFeed(df=df, tz=ZoneInfo("Asia/Kolkata"))
    cfg = StrategyConfig.load("config/strategy.yaml")
    risk = RiskEngine(LimitConfig())
    sim = SimConnector()
    engine = BacktestEngine(feed, cfg, risk, sim, outdir=str(tmp_path))
    summary = engine.run()
    curve = tmp_path / "equity_curve.csv"
    assert curve.exists()
    assert curve.read_text().splitlines()[0] == "ts,equity"
    assert "PF_trend" in summary
