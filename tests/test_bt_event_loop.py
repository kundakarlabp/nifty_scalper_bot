from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtesting.data_feed import SpotFeed
from src.backtesting.sim_connector import SimConnector
from src.backtesting.backtest_engine import BacktestEngine
from src.risk.limits import LimitConfig, RiskEngine
from src.strategies.strategy_config import resolve_config_path, try_load
from src.strategies.scalping_strategy import ScalpingStrategy


def test_bt_event_loop(tmp_path, monkeypatch):
    ts = pd.date_range("2025-08-01 09:15", periods=30, freq="1min")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": range(100, 130),
            "high": range(101, 131),
            "low": range(99, 129),
            "close": range(100, 130),
            "volume": [100] * 30,
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    feed = SpotFeed.from_csv(str(csv_path))
    cfg = try_load(resolve_config_path(), None)
    risk = RiskEngine(LimitConfig(tz=cfg.tz))
    sim = SimConnector()
    bt = BacktestEngine(feed, cfg, risk, sim, outdir=str(tmp_path / "out"))

    def stub_eval(self, ts, o, h, l, c, v):
        return {
            "has_signal": True,
            "action": "BUY",
            "strike": "NIFTY25AUG18000CE",
            "qty_lots": 1,
            "sl": c - 5,
            "tp1_R": 1.0,
            "tp2_R": 1.6,
            "time_stop_min": 1,
        }

    monkeypatch.setattr(ScalpingStrategy, "evaluate_from_backtest", stub_eval)
    monkeypatch.setattr(RiskEngine, "pre_trade_check", lambda *a, **k: (True, "", {}))

    summary = bt.run()
    trades_csv = Path(bt.outdir) / "trades.csv"
    summary_json = Path(bt.outdir) / "summary.json"
    assert trades_csv.exists()
    assert summary_json.exists()
    assert summary["trades"] >= 1
