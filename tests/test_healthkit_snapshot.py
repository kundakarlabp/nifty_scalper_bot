from types import SimpleNamespace

from src.diagnostics.healthkit import snapshot_pipeline


class DummyRunner:
    def __init__(self) -> None:
        self.ready = True
        self._paused = False
        self._status_calls: int = 0
        self._health_calls: int = 0
        self.data_source = SimpleNamespace(
            last_tick_ts=123.0,
            current_tokens=lambda: (111, 222),
        )
        self.order_executor = SimpleNamespace(
            _queues={"orders": [1, 2, 3]},
            cb_orders=SimpleNamespace(state="closed"),
        )

    def get_status_snapshot(self) -> dict[str, bool]:
        self._status_calls += 1
        return {"ok": True}

    def health_check(self) -> dict[str, bool]:
        self._health_calls += 1
        return {"ok": True}


def test_snapshot_pipeline_with_dummy_runner(monkeypatch) -> None:
    dummy = DummyRunner()
    monkeypatch.setattr(
        "src.strategies.runner.StrategyRunner._SINGLETON", dummy, raising=False
    )

    snap = snapshot_pipeline()

    assert snap["runner"]["ready"] is True
    assert snap["runner"]["paused"] is False
    assert snap["health"]["ok"] is True
    assert snap["data_source"]["current_tokens"] == (111, 222)
    assert snap["executor"]["queue_depth"] == 3


def test_snapshot_pipeline_without_runner(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.strategies.runner.StrategyRunner._SINGLETON", None, raising=False
    )

    snap = snapshot_pipeline()

    assert "loop" in snap and "runtime" in snap
    assert "runner" not in snap
