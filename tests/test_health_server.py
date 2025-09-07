from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from src.server.health import app
from src.strategies import runner as runner_module


def _runner_with_tick(ts: datetime) -> SimpleNamespace:
    ds = SimpleNamespace(
        last_tick_ts=lambda: ts,
        last_bar_open_ts=lambda: None,
        timeframe_seconds=60,
    )
    cfg = SimpleNamespace(max_tick_lag_s=10, max_bar_lag_s=75)
    kite = SimpleNamespace(is_connected=lambda: True)
    return SimpleNamespace(kite=kite, data_source=ds, strategy_cfg=cfg)


def test_ready_endpoint_ok(monkeypatch) -> None:
    fake_runner = _runner_with_tick(datetime.utcnow())
    monkeypatch.setattr(
        runner_module.StrategyRunner, "get_singleton", classmethod(lambda cls: fake_runner)
    )
    client = app.test_client()
    resp = client.get("/ready")
    assert resp.status_code == 200


def test_ready_endpoint_stale(monkeypatch) -> None:
    stale = datetime.utcnow() - timedelta(seconds=999)
    fake_runner = _runner_with_tick(stale)
    monkeypatch.setattr(
        runner_module.StrategyRunner, "get_singleton", classmethod(lambda cls: fake_runner)
    )
    client = app.test_client()
    resp = client.get("/ready")
    assert resp.status_code == 503
