from types import SimpleNamespace

from src.logs import structured_log
from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - test stub
        pass


def _make_runner() -> StrategyRunner:
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    runner.data_source = SimpleNamespace()
    return runner


def test_quote_ready_barrier_allows_transient_when_last_plan_non_micro(monkeypatch):
    runner = _make_runner()

    runner.data_source.ensure_quote_ready = lambda *a, **k: SimpleNamespace(  # type: ignore[attr-defined]
        ok=False,
        reason="no_quote",
        last_tick_age_ms=2500,
        retries=1,
        bid=0.0,
        ask=0.0,
    )
    monkeypatch.setattr(runner, "_candidate_quote_tokens", lambda: [101])

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        structured_log,
        "event",
        lambda name, **payload: events.append((name, payload)),
    )

    runner.last_plan = {"reason_block": "equity_low", "micro": {"reason": "ok"}}
    flow: dict[str, object] = {}

    assert runner._quote_ready_barrier(flow) is True
    assert flow["quotes_ready"] is False
    assert flow["quotes_ready_transient"] is True
    assert events == []


def test_quote_ready_barrier_blocks_and_logs_when_micro_reason(monkeypatch):
    runner = _make_runner()

    runner.data_source.ensure_quote_ready = lambda *a, **k: SimpleNamespace(  # type: ignore[attr-defined]
        ok=False,
        reason="stale_quote",
        last_tick_age_ms=3000,
        retries=2,
        bid=0.0,
        ask=0.0,
    )
    monkeypatch.setattr(runner, "_candidate_quote_tokens", lambda: [202])

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        structured_log,
        "event",
        lambda name, **payload: events.append((name, payload)),
    )

    runner.last_plan = {"reason_block": "no_quote", "micro": {"reason": "no_quote"}}
    flow: dict[str, object] = {}

    assert runner._quote_ready_barrier(flow) is False
    assert flow["quotes_ready"] is False
    assert events and events[0][0] == "micro_wait"
    assert events[0][1]["token"] == 202
