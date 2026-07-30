from __future__ import annotations

import threading
from pathlib import Path

from nifty_scalper_bot.execution.order_state_machine import (
    ExecutionState,
    OrderStateMachine,
)
from nifty_scalper_bot.strategies.runner import StrategyRunner


def _pending_runner(
    *, has_position: bool
) -> tuple[StrategyRunner, OrderStateMachine, list]:
    symbol = "NFO:NIFTY2670724100PE"
    releases = []
    runner = object.__new__(StrategyRunner)
    runner._submitted_entry_order_context = {
        "OID1": {
            "symbol": symbol,
            "underlying": "NIFTY",
            "underlying_reason_key": "NIFTY:PE:OrderFlow",
        }
    }
    runner._underlying_last_signal_ts = {"NIFTY": 1000.0}
    runner._reason_last_signal_ts = {"NIFTY:PE:OrderFlow": 1000.0}
    runner._position_manager = type(
        "PM", (), {"has_open_position": lambda self, _symbol: has_position}
    )()
    runner._order_manager = type(
        "OM",
        (),
        {
            "release_entry_reservation": (
                lambda self, released_symbol, *, start_cooldown: releases.append(
                    (released_symbol, start_cooldown)
                )
            )
        },
    )()
    runner._execution_state_lock = threading.RLock()
    machine = OrderStateMachine()
    machine.transition(ExecutionState.SIGNAL_RECEIVED)
    machine.transition(ExecutionState.ORDER_PENDING, order_id="OID1")
    runner._execution_state_by_symbol = {symbol: machine}
    runner._orchestrator = None
    runner._logger = type(
        "Log",
        (),
        {"info": lambda *a, **k: None, "warning": lambda *a, **k: None},
    )()
    return runner, machine, releases


def test_option_reason_cooldown_key_is_side_aware() -> None:
    pe_key = StrategyRunner._reason_order_cooldown_key(
        underlying="NIFTY", option_side="PE", reason_key="OrderFlow"
    )
    ce_key = StrategyRunner._reason_order_cooldown_key(
        underlying="NIFTY", option_side="CE", reason_key="OrderFlow"
    )
    assert pe_key == "NIFTY:PE:OrderFlow"
    assert ce_key == "NIFTY:CE:OrderFlow"
    reason_cache = {pe_key: 1000.0}
    assert ce_key not in reason_cache
    assert pe_key in reason_cache


def test_same_option_reason_cooldown_key_still_blocks_same_side() -> None:
    first = StrategyRunner._reason_order_cooldown_key(
        underlying="NIFTY", option_side="PE", reason_key="OrderFlow"
    )
    second = StrategyRunner._reason_order_cooldown_key(
        underlying="NIFTY", option_side="PE", reason_key="OrderFlow"
    )
    reason_cache = {first: 1000.0}
    assert second in reason_cache


def test_unknown_side_reason_cooldown_key_preserves_legacy_behavior() -> None:
    assert (
        StrategyRunner._reason_order_cooldown_key(
            underlying="NIFTY", option_side="UNKNOWN", reason_key="OrderFlow"
        )
        == "NIFTY:OrderFlow"
    )


def test_live_scalping_cooldown_defaults_match_operator_comments(monkeypatch) -> None:
    for key in (
        "RUNNER_UNDERLYING_SIGNAL_COOLDOWN_SECONDS",
        "RUNNER_REASON_SIGNAL_COOLDOWN_SECONDS",
        "RUNNER_MAX_ORDER_ATTEMPTS_PER_MINUTE",
        "SIGNAL_REJECT_COOLDOWN_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text()
    assert 'RUNNER_UNDERLYING_SIGNAL_COOLDOWN_SECONDS", "20"' in source
    assert 'RUNNER_REASON_SIGNAL_COOLDOWN_SECONDS", "30"' in source
    assert 'RUNNER_MAX_ORDER_ATTEMPTS_PER_MINUTE", "5"' in source
    assert 'SIGNAL_REJECT_COOLDOWN_SECONDS", "15"' in source


def test_failed_entry_order_clears_submission_cooldowns_without_position() -> None:
    runner, machine, releases = _pending_runner(has_position=False)

    runner.notify_entry_order_failed(
        order_id="OID1",
        symbol="NFO:NIFTY2670724100PE",
        reason="rejected",
    )

    assert runner._underlying_last_signal_ts == {}
    assert runner._reason_last_signal_ts == {}
    assert machine.state == ExecutionState.IDLE
    assert releases == [("NFO:NIFTY2670724100PE", False)]


def test_failed_entry_order_keeps_cooldowns_when_position_exists() -> None:
    runner, machine, releases = _pending_runner(has_position=True)

    runner.notify_entry_order_failed(
        order_id="OID1",
        symbol="NFO:NIFTY2670724100PE",
        reason="rejected",
    )

    assert runner._underlying_last_signal_ts == {"NIFTY": 1000.0}
    assert runner._reason_last_signal_ts == {"NIFTY:PE:OrderFlow": 1000.0}
    assert machine.state == ExecutionState.ORDER_PENDING
    assert releases == []


def test_failed_entry_order_keeps_guards_when_position_state_is_unknown() -> None:
    runner, machine, releases = _pending_runner(has_position=False)
    runner._position_manager = type(
        "PM",
        (),
        {
            "has_open_position": lambda self, _symbol: (_ for _ in ()).throw(
                RuntimeError("position sync unavailable")
            )
        },
    )()

    runner.notify_entry_order_failed(
        order_id="OID1",
        symbol="NFO:NIFTY2670724100PE",
        reason="rejected",
    )

    assert runner._underlying_last_signal_ts == {"NIFTY": 1000.0}
    assert machine.state == ExecutionState.ORDER_PENDING
    assert releases == []


def test_confirmed_flat_symbol_converges_bracket_and_entry_guards() -> None:
    runner, machine, releases = _pending_runner(has_position=False)
    reconciled = []
    runner._bracket_manager = type(
        "BM",
        (),
        {"reconcile_symbol_flat": lambda self, symbol: reconciled.append(symbol)},
    )()

    runner._on_symbols_flat(["NFO:NIFTY2670724100PE"])

    assert reconciled == ["NFO:NIFTY2670724100PE"]
    assert machine.state == ExecutionState.IDLE
    assert releases == [("NFO:NIFTY2670724100PE", True)]


def test_bracket_completion_releases_entry_guards_with_cooldown() -> None:
    symbol = "NFO:NIFTY2670724100PE"
    releases = []
    runner = object.__new__(StrategyRunner)
    runner._logger = type(
        "Log",
        (),
        {"info": lambda *a, **k: None, "exception": lambda *a, **k: None},
    )()
    runner._normalize_symbol = lambda value: value
    runner._notify_orchestrator_exit = lambda _symbol: None
    runner._clear_order_in_flight = lambda _symbol: None
    runner._release_entry_guards = (
        lambda released_symbol, *, start_cooldown, reason: releases.append(
            (released_symbol, start_cooldown, reason)
        )
    )

    runner._on_bracket_exit_complete(symbol)

    assert releases == [(symbol, True, "bracket_exit_complete")]


def test_bracket_completion_records_strategy_net_outcome_before_release() -> None:
    symbol = "NFO:NIFTY2670724100PE"
    releases = []
    recorded = []
    runner = object.__new__(StrategyRunner)
    runner._logger = type(
        "Log",
        (),
        {
            "info": lambda *a, **k: None,
            "error": lambda *a, **k: None,
            "exception": lambda *a, **k: None,
        },
    )()
    runner._strategy_manager = type(
        "SM",
        (),
        {
            "record_trade_result": (
                lambda self, strategy, pnl, *, metadata: recorded.append(
                    (strategy, pnl, metadata)
                )
            )
        },
    )()
    runner._normalize_symbol = lambda value: value
    runner._notify_orchestrator_exit = lambda _symbol: None
    runner._clear_order_in_flight = lambda _symbol: None
    runner._release_entry_guards = (
        lambda released_symbol, *, start_cooldown, reason: releases.append(
            (released_symbol, start_cooldown, reason)
        )
    )
    outcome = {
        "strategy_name": "VWAPPro",
        "setup_name": "continuation_pullback",
        "regime": "TREND",
        "gross_pnl": 650.0,
        "net_pnl": 575.0,
        "exit_reason": "HARD_TP_BREACH",
    }
    runner._bracket_manager = type(
        "BM",
        (),
        {"get_completed_trade_outcome": lambda self, _symbol: outcome},
    )()

    runner._on_bracket_exit_complete(symbol)

    assert recorded == [
        (
            "VWAPPro",
            575.0,
            {
                "setup_name": "continuation_pullback",
                "regime": "TREND",
                "gross_pnl": 650.0,
                "net_pnl": 575.0,
                "exit_reason": "HARD_TP_BREACH",
            },
        )
    ]
    assert releases == [(symbol, True, "bracket_exit_complete")]


def test_strategy_feedback_failure_never_blocks_bracket_guard_release() -> None:
    symbol = "NFO:NIFTY2670724100PE"
    releases = []
    runner = object.__new__(StrategyRunner)
    runner._logger = type(
        "Log",
        (),
        {
            "info": lambda *a, **k: None,
            "error": lambda *a, **k: None,
            "exception": lambda *a, **k: None,
        },
    )()
    runner._strategy_manager = type(
        "SM",
        (),
        {
            "record_trade_result": (
                lambda *_a, **_k: (_ for _ in ()).throw(
                    RuntimeError("analytics down")
                )
            )
        },
    )()
    runner._normalize_symbol = lambda value: value
    runner._notify_orchestrator_exit = lambda _symbol: None
    runner._clear_order_in_flight = lambda _symbol: None
    runner._release_entry_guards = (
        lambda released_symbol, *, start_cooldown, reason: releases.append(
            (released_symbol, start_cooldown, reason)
        )
    )

    runner._on_bracket_exit_complete(
        symbol,
        {"strategy_name": "VWAPPro", "net_pnl": -100.0, "regime": "RANGE"},
    )

    assert releases == [(symbol, True, "bracket_exit_complete")]


def test_runner_risk_halt_clears_after_authoritative_breaker_reset() -> None:
    states = iter([(True, "daily loss"), (False, "")])
    events = []
    runner = object.__new__(StrategyRunner)
    runner._risk_manager = type(
        "Risk",
        (),
        {"is_circuit_breaker_tripped": lambda self: next(states)},
    )()
    runner._risk_halt_active = False
    runner._risk_halt_logged = False
    runner._logger = type(
        "Log",
        (),
        {
            "debug": lambda *a, **k: None,
            "error": lambda self, *a, **k: events.append(k["extra"]["event"]),
            "info": lambda self, *a, **k: events.append(k["extra"]["event"]),
        },
    )()

    assert runner._refresh_risk_halt_state("NFO:NIFTY2670724100PE") is True
    assert runner._refresh_risk_halt_state("NFO:NIFTY2670724100PE") is False
    assert events == ["risk_halt_latched", "RISK_HALT_CLEARED"]
    assert runner._risk_halt_logged is False


def test_runner_risk_halt_stays_fail_closed_when_recheck_fails() -> None:
    runner = object.__new__(StrategyRunner)
    runner._risk_manager = type(
        "Risk",
        (),
        {
            "is_circuit_breaker_tripped": lambda self: (_ for _ in ()).throw(
                RuntimeError("unavailable")
            )
        },
    )()
    runner._risk_halt_active = True
    runner._risk_halt_logged = True
    runner._logger = type("Log", (), {"debug": lambda *a, **k: None})()

    assert runner._refresh_risk_halt_state("NFO:NIFTY2670724100PE") is True


def test_order_failure_cooldown_rejection_uses_dedup_rollback_path() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text()
    assert 'reason="order_failure_cooldown_active")' in source
    assert 'return self._reject_signal_execution(symbol=base_symbol, trace_id=trace_id, reason="order_failure_cooldown_active")' not in source
