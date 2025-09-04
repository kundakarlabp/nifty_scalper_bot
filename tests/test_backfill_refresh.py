import pandas as pd
from types import SimpleNamespace
from datetime import datetime, timedelta

from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.strategies.strategy_config import StrategyConfig


def _make_df(n: int, start: datetime | None = None) -> pd.DataFrame:
    start = start or datetime(2024, 1, 1, 9, 15)
    idx = pd.date_range(start, periods=n, freq="1min")
    data = {
        "open": [1.0] * n,
        "high": [1.0] * n,
        "low": [1.0] * n,
        "close": [1.0] * n,
        "volume": [1] * n,
    }
    return pd.DataFrame(data, index=idx)


class DummySource:
    def __init__(self) -> None:
        self.df = _make_df(10)

    def get_last_bars(self, n: int) -> pd.DataFrame:
        return self.df.tail(n)

    def ensure_backfill(self, *, required_bars: int, token: int = 0, timeframe: str = "minute") -> None:
        if len(self.df) >= required_bars:
            return
        start = self.df.index[-1] + timedelta(minutes=1)
        extra = _make_df(required_bars - len(self.df), start)
        self.df = pd.concat([self.df, extra])


def test_plan_refreshes_after_backfill() -> None:
    strat = EnhancedScalpingStrategy()
    strat.data_source = DummySource()
    cfg = StrategyConfig.load("config/strategy.yaml")
    cfg.min_bars_required = 20
    strat.runner = SimpleNamespace(strategy_cfg=cfg)
    df = strat.data_source.get_last_bars(10)
    plan = strat.generate_signal(df=df, current_tick={"ltp": 1.0}, current_price=None, spot_df=None)
    features = plan.get("features", {})
    assert features.get("have_bars", 0) >= 20
    assert plan.get("block") != "insufficient_bars"
