import logging
from types import SimpleNamespace

import pytest

from src.strategies.runner import StrategyRunner


def _make_runner() -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner.log = logging.getLogger("StrategyRunnerTest")
    runner._log_signal_changes_only = False
    runner._gate = SimpleNamespace(should_emit=lambda *args, **kwargs: False)
    runner._last_has_signal = None
    runner._last_reason_block = None
    runner.eval_count = 0
    runner.last_eval_ts = None
    runner.hb_enabled = False
    runner._last_hb_ts = 0.0
    runner.emit_heartbeat = lambda: None
    runner.plan_probe_window = None
    runner._shadow_blockers = lambda plan: []
    return runner


def test_build_plan_snapshot_uses_age_sec() -> None:
    runner = _make_runner()
    plan = {"quote": {"age_sec": 1.234}}

    snapshot = runner._build_plan_snapshot(plan)

    assert snapshot["quote_age_s"] == pytest.approx(1.234)


def test_build_plan_snapshot_supports_age_s_alias() -> None:
    runner = _make_runner()
    plan = {"quote": {"age_s": "2.5"}}

    snapshot = runner._build_plan_snapshot(plan)

    assert snapshot["quote_age_s"] == pytest.approx(2.5)


def test_record_plan_populates_quote_age() -> None:
    runner = _make_runner()
    plan = {"micro": {}, "quote": {"age_sec": 0.75}}

    runner._record_plan(plan)

    assert plan["quote_age_s"] == pytest.approx(0.75)
    assert runner.last_plan["quote_age_s"] == pytest.approx(0.75)
