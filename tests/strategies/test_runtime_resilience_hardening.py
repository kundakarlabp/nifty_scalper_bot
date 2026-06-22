from __future__ import annotations

from types import SimpleNamespace

import pytest

import nifty_scalper_bot.strategies.hardened_strategy_runner as hardened_runner_module
import nifty_scalper_bot.strategies.hardened_trade_selector as hardened_selector_module
from nifty_scalper_bot.strategies.hardened_strategy_runner import HardenedStrategyRunner
from nifty_scalper_bot.strategies.hardened_trade_selector import HardenedTradeCandidateSelector
from nifty_scalper_bot.strategies.runner import SignalExecutionResult, StrategyRunner
from nifty_scalper_bot.strategies.trade_selector import TradeCandidateSelector


class _Logger:
    def __init__(self) -> None:
        self.critical_events: list[dict] = []
        self.warning_events: list[dict] = []

    def critical(self, *args, **kwargs) -> None:
        self.critical_events.append(dict(kwargs.get("extra") or {}))

    def warning(self, *args, **kwargs) -> None:
        self.warning_events.append(dict(kwargs.get("extra") or {}))

    def info(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None


def test_canonical_strategy_imports_use_hardened_classes() -> None:
    assert StrategyRunner is HardenedStrategyRunner
    assert TradeCandidateSelector is HardenedTradeCandidateSelector


def test_selector_preserves_exact_midday_pause_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        hardened_selector_module._legacy,
        "expiry_theta_block",
        lambda: (False, ""),
    )
    monkeypatch.setattr(
        hardened_selector_module._legacy,
        "midday_pause_block",
        lambda: (True, "midday_pause_11:30-13:15_ist"),
    )
    selector = HardenedTradeCandidateSelector()

    ranked = selector.select_ranked_candidates(
        direction_bias="CE",
        atm_strike=24000,
        snapshots=[{"symbol": "NFO:NIFTY24000CE", "side": "CE"}],
    )

    assert ranked == []
    assert selector.last_entry_window_reason == "midday_pause_11:30-13:15_ist"
    assert selector._last_rejects["entry_window_blocked"] == 1
    assert (
        selector._last_rejects[
            "entry_window_blocked:midday_pause_11:30-13:15_ist"
        ]
        == 1
    )


def test_runner_maps_empty_candidate_result_to_exact_window_reason(monkeypatch) -> None:
    runner = HardenedStrategyRunner.__new__(HardenedStrategyRunner)
    runner._trade_candidate_selector = SimpleNamespace(
        last_entry_window_reason="midday_pause_11:30-13:15_ist"
    )
    captured: dict = {}

    def parent_reject(_self, *args, **kwargs):
        captured.update(kwargs)
        return SignalExecutionResult(
            False,
            kwargs["reason"],
            details=kwargs.get("details") or {},
        )

    monkeypatch.setattr(
        hardened_runner_module._LegacyStrategyRunner,
        "_reject_signal_execution",
        parent_reject,
    )

    result = runner._reject_signal_execution(
        symbol="NFO:NIFTY24000CE",
        trace_id="t-midday",
        reason="no_execution_ready_candidate",
        details={"candidate_total": 5},
    )

    assert result.reason == "midday_pause_11:30-13:15_ist"
    assert captured["details"]["original_reason"] == "no_execution_ready_candidate"
    assert captured["details"]["stage"] == "entry_window_gate"


def _runner_for_circuit() -> HardenedStrategyRunner:
    runner = HardenedStrategyRunner.__new__(HardenedStrategyRunner)
    runner._logger = _Logger()
    runner._reset_execution_state = lambda _symbol: None
    runner._record_trade_decision_snapshot = lambda **_kwargs: None
    runner._trade_candidate_selector = SimpleNamespace(
        last_entry_window_reason=None
    )
    return runner


def test_identical_candidate_programming_errors_open_circuit(monkeypatch) -> None:
    monkeypatch.setenv("CANDIDATE_EXCEPTION_CIRCUIT_THRESHOLD", "3")
    monkeypatch.setenv("CANDIDATE_EXCEPTION_CIRCUIT_WINDOW_SECONDS", "30")
    monkeypatch.setenv("CANDIDATE_EXCEPTION_CIRCUIT_COOLDOWN_SECONDS", "300")
    runner = _runner_for_circuit()
    calls = {"count": 0}

    def parent_handle(_self, *args, **kwargs):
        calls["count"] += 1
        return SignalExecutionResult(
            False,
            "candidate_selection_exception",
            details={
                "error_type": "TypeError",
                "error": "same deterministic failure",
            },
        )

    def parent_reject(_self, *args, **kwargs):
        return SignalExecutionResult(
            False,
            kwargs["reason"],
            details=kwargs.get("details") or {},
        )

    monkeypatch.setattr(
        hardened_runner_module._LegacyStrategyRunner,
        "_handle_entry_signal_inner",
        parent_handle,
    )
    monkeypatch.setattr(
        hardened_runner_module._LegacyStrategyRunner,
        "_reject_signal_execution",
        parent_reject,
    )

    common = (
        SimpleNamespace(),
        "NFO:NIFTY24000CE",
        "NFO:NIFTY24000CE",
        100.0,
        SimpleNamespace(),
    )
    first = runner._handle_entry_signal_inner(*common, trace_id="t1")
    second = runner._handle_entry_signal_inner(*common, trace_id="t2")
    third = runner._handle_entry_signal_inner(*common, trace_id="t3")
    fourth = runner._handle_entry_signal_inner(*common, trace_id="t4")

    assert first.reason == "candidate_selection_exception"
    assert second.reason == "candidate_selection_exception"
    assert third.reason == "candidate_selection_circuit_open"
    assert fourth.reason == "candidate_selection_circuit_open"
    assert calls["count"] == 3
    assert len(runner._logger.critical_events) == 1
    snapshot = runner.candidate_selection_circuit_snapshot()
    assert snapshot["open"] is True
    assert snapshot["count"] == 3


def test_normal_candidate_rejection_does_not_open_programming_error_circuit(
    monkeypatch,
) -> None:
    runner = _runner_for_circuit()

    monkeypatch.setattr(
        hardened_runner_module._LegacyStrategyRunner,
        "_handle_entry_signal_inner",
        lambda *_args, **_kwargs: SignalExecutionResult(
            False,
            "no_execution_ready_candidate",
            details={"candidate_total": 5},
        ),
    )

    result = runner._handle_entry_signal_inner(
        SimpleNamespace(),
        "NFO:NIFTY24000CE",
        "NFO:NIFTY24000CE",
        100.0,
        SimpleNamespace(),
        trace_id="normal-reject",
    )

    assert result.reason == "no_execution_ready_candidate"
    assert runner.candidate_selection_circuit_snapshot()["open"] is False


def test_distinct_programming_errors_do_not_accumulate_as_same_failure(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CANDIDATE_EXCEPTION_CIRCUIT_THRESHOLD", "3")
    runner = _runner_for_circuit()
    errors = iter(["failure-a", "failure-b", "failure-a"])

    def parent_handle(_self, *args, **kwargs):
        return SignalExecutionResult(
            False,
            "candidate_selection_exception",
            details={"error_type": "TypeError", "error": next(errors)},
        )

    monkeypatch.setattr(
        hardened_runner_module._LegacyStrategyRunner,
        "_handle_entry_signal_inner",
        parent_handle,
    )

    for idx in range(3):
        result = runner._handle_entry_signal_inner(
            SimpleNamespace(),
            "NFO:NIFTY24000CE",
            "NFO:NIFTY24000CE",
            100.0,
            SimpleNamespace(),
            trace_id=f"distinct-{idx}",
        )
        assert result.reason == "candidate_selection_exception"

    snapshot = runner.candidate_selection_circuit_snapshot()
    assert snapshot["open"] is False
    assert snapshot["count"] == 1
