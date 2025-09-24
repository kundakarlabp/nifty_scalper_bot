from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.config import settings
from src.diagnostics.healthkit import snapshot_pipeline
from src.strategies.runner import StrategyRunner


def test_snapshot_pipeline_no_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no runner exists the snapshot should return defaults."""

    monkeypatch.setattr(StrategyRunner, "_SINGLETON", None, raising=False)

    snap = snapshot_pipeline()

    assert snap["market_open"] is False
    assert snap["equity"] is None
    assert snap["risk"]["daily_dd"] is None
    assert snap["risk"]["cap_pct"] == pytest.approx(
        float(getattr(settings, "RISK__EXPOSURE_CAP_PCT", 0.0)) * 100.0
    )


def test_snapshot_pipeline_with_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot should derive live data from the active runner."""

    class DummyFrame:
        def __init__(self, ts: datetime) -> None:
            self.index = [ts]
            self.empty = False

    class DummySource:
        def __init__(self) -> None:
            self.last_tick_ts = time.time() - 5
            self.atm_tokens = (111, 222)

        def get_micro_state(self, token: int) -> dict[str, int]:
            return {"token": int(token)}

    class DummyExecutor:
        open_count = 2

        @staticmethod
        def get_active_orders() -> list[int]:
            return [1, 2]

        @staticmethod
        def get_positions_kite() -> dict[str, object]:
            return {"NIFTY": {}}

    now = datetime.now(timezone.utc)
    last_bar = now - timedelta(seconds=60)

    dummy_runner = SimpleNamespace(
        _within_trading_window=lambda _none: True,
        _equity_cached_value=123456.78,
        risk=SimpleNamespace(day_realized_loss=-123.45),
        last_plan={"regime": "bull", "atr_pct": 1.5, "score": 0.82},
        _last_signal_debug={"regime": "bull", "atr_pct": 1.5, "score": 0.82},
        data_source=DummySource(),
        executor=DummyExecutor(),
        _ohlc_cache=DummyFrame(last_bar),
        _now_ist=lambda: now,
    )

    monkeypatch.setattr(StrategyRunner, "_SINGLETON", dummy_runner, raising=False)

    snap = snapshot_pipeline()

    assert snap["market_open"] is True
    assert snap["equity"] == pytest.approx(123456.78, rel=1e-6)
    assert snap["risk"]["daily_dd"] == pytest.approx(-123.45, rel=1e-6)
    assert snap["signals"] == {"regime": "bull", "atr_pct": 1.5, "score": 0.82}
    assert snap["micro"]["ce"] == {"token": 111}
    assert snap["micro"]["pe"] == {"token": 222}
    assert snap["open_orders"] == 2
    assert snap["positions"] == 1
    assert snap["latency"]["tick_age"] is not None and snap["latency"]["tick_age"] >= 0
    assert snap["latency"]["bar_lag"] == pytest.approx(60.0, rel=0.2)
