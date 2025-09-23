import logging
from types import SimpleNamespace

import pandas as pd

from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - stub
        pass


def _build_df(now):
    return pd.DataFrame(
        {
            "open": [1.0] * 60,
            "high": [1.0] * 60,
            "low": [1.0] * 60,
            "close": [1.0] * 60,
            "volume": [0] * 60,
        },
        index=pd.date_range(pd.Timestamp(now) - pd.Timedelta(minutes=59), periods=60, freq="1min"),
    )


def test_logs_when_scoring_skipped(monkeypatch, caplog):
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    monkeypatch.setattr(runner, "_within_trading_window", lambda *a, **k: True)
    now = pd.Timestamp("2024-01-01 01:00").to_pydatetime()
    monkeypatch.setattr(runner, "_now_ist", lambda: now)
    runner.event_guard_enabled = False
    runner.event_cal = None
    monkeypatch.setattr(runner, "_fetch_spot_ohlc", lambda: _build_df(now))
    monkeypatch.setattr(runner, "_ensure_day_state", lambda: None)
    monkeypatch.setattr(runner, "_refresh_equity_if_due", lambda: None)
    monkeypatch.setattr(runner, "_maybe_emit_minute_diag", lambda plan: None)
    runner.order_executor = SimpleNamespace(step_queue=lambda now: None, on_order_timeout_check=lambda: None)
    runner.executor = runner.order_executor
    runner.data_source = SimpleNamespace(cb_hist=None, cb_quote=None)
    runner.strategy_cfg = SimpleNamespace(raw={}, min_atr_pct_nifty=0, min_atr_pct_banknifty=0)

    def fake_signal(df, current_tick=None):
        return {
            "regime": "NO_TRADE",
            "score": None,
            "reason_block": "regime_no_trade",
            "rr": 0.0,
            "option_type": None,
            "strike": None,
            "token": 1,
            "option_token": 1,
            "reasons": [],
        }

    monkeypatch.setattr(runner.strategy, "generate_signal", fake_signal)
    with caplog.at_level(logging.DEBUG):
        runner.process_tick({})
    assert "Scoring skipped" in caplog.text
    flow = runner.get_last_flow_debug()
    assert flow["reason_block"] == "regime_no_trade"
