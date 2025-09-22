from __future__ import annotations

import logging
from types import SimpleNamespace

from src.strategies.runner import StrategyRunner


class _StubSource:
    def __init__(self, micro_payload: dict | None, quote_payload: dict | None) -> None:
        self._micro_payload = micro_payload or {}
        self._quote_payload = quote_payload
        self.micro_tokens: list[int | None] = []
        self.quote_tokens: list[int | None] = []

    def get_micro_state(self, token: int | None) -> dict:
        self.micro_tokens.append(token)
        return dict(self._micro_payload)

    def quote_snapshot(self, token: int | None) -> dict | None:
        self.quote_tokens.append(token)
        if self._quote_payload is None:
            return None
        return dict(self._quote_payload)


def _make_runner() -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner.log = logging.getLogger("StrategyRunnerTest")
    runner.last_plan = {}
    runner._last_signal_debug = {}
    runner.data_source = SimpleNamespace()
    runner.source = None
    return runner


def test_plan_snapshot_uses_atm_token_when_missing_plan_token(monkeypatch) -> None:
    """ATM tokens from the data source are used when plan tokens are absent."""

    monkeypatch.setenv("MAX_SPREAD_PCT", "0.5")
    source = _StubSource(
        micro_payload={"spread_pct": 0.9, "depth_ok": False},
        quote_payload={"age_sec": 7, "spread_pct": 0.9},
    )
    runner = _make_runner()
    runner.last_plan = {"option_type": "CE"}
    runner._last_signal_debug = {"option_type": "CE"}
    runner.data_source = SimpleNamespace(
        atm_tokens=(123, 456),
        _current_ce_token=123,
        _current_pe_token=456,
    )
    runner.source = source

    snapshot = runner._build_plan_snapshot()

    assert source.micro_tokens == [123]
    assert source.quote_tokens == [123]
    plan = snapshot["plan"]
    assert plan["quote_age_s"] == 7
    assert plan["spread_pct"] == 0.9
    assert plan["micro"]["reason"] == "spread_wide"
    assert plan["micro"]["depth_ok"] is False


def test_plan_snapshot_marks_no_quote_when_spread_missing() -> None:
    """A missing spread marks the micro reason as ``no_quote``."""

    source = _StubSource(
        micro_payload={"depth_ok": True, "spread_pct": None},
        quote_payload={"age_s": 4},
    )
    runner = _make_runner()
    runner.last_plan = {"option_type": "PE"}
    runner._last_signal_debug = {"option_type": "PE"}
    runner.data_source = SimpleNamespace(
        atm_tokens=(111, 222),
        _current_ce_token=111,
        _current_pe_token=222,
    )
    runner.source = source

    snapshot = runner._build_plan_snapshot()

    plan = snapshot["plan"]
    assert plan["quote_age_s"] == 4
    assert plan["spread_pct"] is None
    assert plan["micro"]["reason"] == "no_quote"
    assert plan["micro"]["depth_ok"] is True


def test_plan_snapshot_preserves_no_token_reason() -> None:
    """When no token can be resolved, the ``no_token`` reason is preserved."""

    source = _StubSource(micro_payload={}, quote_payload=None)
    runner = _make_runner()
    runner.last_plan = {"option_type": "FUT"}
    runner.source = source

    snapshot = runner._build_plan_snapshot()

    assert source.micro_tokens == []
    assert source.quote_tokens == []
    assert snapshot["plan"]["micro"]["reason"] == "no_token"
