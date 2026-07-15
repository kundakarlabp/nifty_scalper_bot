from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.strategies.runner import (
    ExecutionModeSnapshot,
    OptionSideReadiness,
    SignalExecutionResult,
    StrategyRunner,
)
from nifty_scalper_bot.strategies.signal_generator import Signal


def _runner() -> StrategyRunner:
    runner = object.__new__(StrategyRunner)
    runner._active_selected_ce = "NFO:CE"
    runner._active_selected_pe = "NFO:PE"
    runner._runtime_live_orders_armed = True
    runner._logger = SimpleNamespace(info=lambda *a, **k: None)
    return runner


def _snap(ce_ok: bool, pe_ok: bool):
    return {
        "CE": OptionSideReadiness(
            "CE",
            "NFO:CE",
            1,
            ce_ok,
            ce_ok,
            True,
            50 if ce_ok else 49,
            50,
            ce_ok,
            ce_ok,
            () if ce_ok else ("history_cold",),
        ),
        "PE": OptionSideReadiness(
            "PE",
            "NFO:PE",
            2,
            pe_ok,
            pe_ok,
            True,
            30 if pe_ok else 29,
            30,
            pe_ok,
            pe_ok,
            () if pe_ok else ("quote_missing",),
        ),
    }


def test_readiness_for_candidate_symbol_returns_candidate_side(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(
        runner, "_option_side_readiness_snapshot", lambda **_: _snap(True, False)
    )
    assert runner._readiness_for_candidate_symbol("NFO:CE").executable is True
    pe = runner._readiness_for_candidate_symbol("NFO:PE")
    assert pe is not None
    assert pe.side == "PE"
    assert pe.executable is False
    assert pe.blockers == ("quote_missing",)


def test_independent_required_bars_are_preserved_in_snapshot(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(
        runner, "_option_side_readiness_snapshot", lambda **_: _snap(False, True)
    )
    ce = runner._readiness_for_candidate_symbol("NFO:CE")
    pe = runner._readiness_for_candidate_symbol("NFO:PE")
    assert ce.required_bars == 50 and ce.history_count == 49 and not ce.history_ready
    assert pe.required_bars == 30 and pe.history_count == 30 and pe.history_ready


def test_desired_only_subscription_is_not_ready(monkeypatch):
    runner = _runner()
    mdm = SimpleNamespace(
        _token_by_symbol={"NFO:CE": 1, "NFO:PE": 2},
        _desired_tokens={1},
        _subscribed_tokens=set(),
        _confirmed_subscriptions=set(),
        _active_subscribed_symbols=set(),
        _ws=SimpleNamespace(_tokens=set()),
    )
    runner._market_data = mdm
    runner._indicator_engine = SimpleNamespace(get_history=lambda sym: [1] * 50)
    runner._current_active_contract_selection = lambda: SimpleNamespace(
        selected_ce="NFO:CE", selected_pe="NFO:PE"
    )
    runner._get_cached_quote_for_live_entry = lambda sym: {
        "ltp": 100,
        "bid": 99,
        "ask": 101,
        "depth_available": True,
    }
    runner._is_option_symbol_tick_fresh = lambda sym, max_age_s=60.0: True
    runner._selected_option_has_real_depth = lambda sym: True
    runner._required_bars_for_symbol = lambda sym: 50
    snap = runner._option_side_readiness_snapshot()
    assert snap["CE"].subscription_requested is True
    assert snap["CE"].subscribed is False


def test_current_generation_fresh_tick_can_prove_subscription(monkeypatch):
    runner = _runner()
    mdm = SimpleNamespace(
        _token_by_symbol={"NFO:CE": 1, "NFO:PE": 2},
        _desired_tokens=set(),
        _subscribed_tokens=set(),
        _confirmed_subscriptions=set(),
        _active_subscribed_symbols=set(),
        _ws=SimpleNamespace(_tokens=set()),
        _symbol_subscription_generation={"NFO:CE": 7},
        _symbol_first_tick_generation={"NFO:CE": 7},
    )
    runner._market_data = mdm
    runner._indicator_engine = SimpleNamespace(get_history=lambda sym: [1] * 50)
    runner._current_active_contract_selection = lambda: SimpleNamespace(
        selected_ce="NFO:CE", selected_pe=None
    )
    runner._get_cached_quote_for_live_entry = lambda sym: {
        "ltp": 100,
        "bid": 99,
        "ask": 101,
        "depth_available": True,
    }
    runner._is_option_symbol_tick_fresh = lambda sym, max_age_s=60.0: True
    runner._selected_option_has_real_depth = lambda sym: True
    runner._required_bars_for_symbol = lambda sym: 50
    assert runner._option_side_readiness_snapshot()["CE"].subscribed is True
    mdm._symbol_first_tick_generation["NFO:CE"] = 6
    assert runner._option_side_readiness_snapshot()["CE"].subscribed is False


class _Logger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass


class _OrderManager:
    def __init__(self):
        self.plans = []

    def resolve_lot_size(self, symbol):
        return 1

    def _lot_size_for_symbol(self, symbol):
        return 1

    def submit_trade_plan_result(self, plan):
        self.plans.append(plan)
        return SimpleNamespace(
            accepted=True,
            order_id="OID-1",
            broker_attempted=True,
            reason="accepted",
            details={},
        )


def _execution_runner(monkeypatch, *, ce_ok=True, pe_ok=False):
    import nifty_scalper_bot.strategies.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "score_signal_quality",
        lambda **kwargs: SimpleNamespace(
            allowed=True,
            final_score=10.0,
            direction_score=10.0,
            strategy_score=10.0,
            option_score=10.0,
            data_score=10.0,
            rr_score=10.0,
            components={"threshold": 0.0},
            reasons=[],
        ),
    )
    runner = object.__new__(StrategyRunner)
    runner._logger = _Logger()
    runner._order_manager = _OrderManager()
    runner._runtime_live_orders_armed = True
    runner._order_attempt_window = deque()
    runner._max_order_attempts_per_minute = 999
    runner._signal_reject_cooldown_ts = {}
    runner._execution_reject_cooldown_ts = {}
    runner._premium_squeeze_last_signal_ts = {}
    runner._underlying_last_signal_ts = {}
    runner._reason_last_signal_ts = {}
    runner._order_failure_cooldown_until = {}
    runner._submitted_entry_order_context = {}
    runner._signal_attempt_debounce_state = {}
    runner._session_date = "2026-07-15"
    runner._last_regime_by_symbol = {}
    runner._last_regime_inputs_by_symbol = {}
    runner._reason_signal_cooldown_seconds = 0.0
    runner._underlying_signal_cooldown_seconds = 0.0
    runner._cooldown_log_throttle_seconds = 0.0
    runner._order_failure_cooldown_seconds = 0.0
    runner._trade_candidate_selector = SimpleNamespace(_last_rejects={})
    runner._position_manager = None
    runner._orchestrator = None
    runner._active_atm_strike = 25000
    runner._active_option_symbols = {"NFO:CE", "NFO:PE"}
    runner._active_basket_all_symbols = {"NFO:CE", "NFO:PE"}
    runner._active_basket_token_by_symbol = {"NFO:CE": 1, "NFO:PE": 2}
    runner._active_contract_basket = {
        "selected_ce": "NFO:CE",
        "selected_pe": "NFO:PE",
        "token_by_symbol": {"NFO:CE": 1, "NFO:PE": 2},
    }
    runner._market_data = SimpleNamespace(
        _token_by_symbol={
            "NFO:CE": 1,
            "CE": 1,
            "NFO:PE": 2,
            "PE": 2,
            "NFO:OTHERCE": 3,
            "OTHERCE": 3,
        }
    )
    runner._resolve_execution_mode_snapshot = lambda: ExecutionModeSnapshot(
        "paper", False, True, False, False, False
    )
    runner._is_tradable_symbol = lambda symbol: True
    runner._extract_underlying = lambda symbol: "NIFTY"
    runner._reason_order_cooldown_key = lambda **kwargs: "NIFTY:CE:test"
    runner._execution_reject_cooldown_result = lambda *args, **kwargs: None
    runner._reset_execution_state = lambda *args, **kwargs: None
    runner._mark_directional_dedup_failed = lambda **kwargs: None
    runner._apply_directional_signal_dedup = lambda **kwargs: None
    runner._ensure_symbol_execution_ready_result = (
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True, reason="ready", details={}
        )
    )
    runner._prepare_order_state_for_submission = lambda *args, **kwargs: (
        True,
        "ready",
        {},
    )
    runner._strategy_regime_decision = lambda *args, **kwargs: (True, "allowed")
    runner._record_trade_decision_snapshot = lambda **kwargs: setattr(
        runner, "_last_decision", kwargs
    )
    runner._record_trade = lambda *args, **kwargs: None
    runner._reject_signal_execution = (
        lambda *, symbol, trace_id, reason, details=None: SignalExecutionResult(
            False, reason, details=dict(details or {})
        )
    )
    runner._current_active_contract_selection = lambda: SimpleNamespace(
        selected_ce="NFO:CE", selected_pe="NFO:PE"
    )
    snap = _snap(ce_ok, pe_ok)
    runner._option_side_readiness_snapshot = lambda **kwargs: snap
    calls = {"gate": 0}
    real = StrategyRunner._readiness_for_candidate_symbol.__get__(
        runner, StrategyRunner
    )

    def wrapped(symbol, **kwargs):
        calls["gate"] += 1
        return real(symbol, **kwargs)

    runner._readiness_for_candidate_symbol = wrapped
    return runner, calls


def _signal(symbol: str) -> Signal:
    return Signal(
        "BUY",
        symbol,
        1,
        1.0,
        "test",
        90.0,
        120.0,
        metadata={"strategy_name": "test", "final_score": 10.0},
    )


@pytest.mark.parametrize("trade_symbol", ["NFO:CE"])
def test_entry_path_allows_ready_ce_candidate_and_submits_order(
    monkeypatch, trade_symbol
):
    runner, calls = _execution_runner(monkeypatch, ce_ok=True, pe_ok=False)
    result = runner._handle_entry_signal_inner(
        _signal(trade_symbol),
        "NSE:NIFTY",
        trade_symbol,
        100.0,
        datetime.now(timezone.utc),
        trace_id="t",
    )
    assert calls["gate"] == 1
    assert result.accepted is True
    assert len(runner._order_manager.plans) == 1
    assert runner._order_manager.plans[0].symbol == "NFO:CE"


def test_entry_path_rejects_stale_pe_candidate_before_order_manager(monkeypatch):
    runner, calls = _execution_runner(monkeypatch, ce_ok=True, pe_ok=False)
    result = runner._handle_entry_signal_inner(
        _signal("NFO:PE"),
        "NSE:NIFTY",
        "NFO:PE",
        100.0,
        datetime.now(timezone.utc),
        trace_id="t",
    )
    assert calls["gate"] == 1
    assert result.accepted is False
    assert result.reason == "selected_pe_unready"
    assert result.details["candidate_blockers"] == ["quote_missing"]
    assert runner._order_manager.plans == []


def test_entry_path_rejects_unmapped_nifty_option_before_order_manager(monkeypatch):
    runner, calls = _execution_runner(monkeypatch, ce_ok=True, pe_ok=True)
    result = runner._handle_entry_signal_inner(
        _signal("NFO:OTHERCE"),
        "NSE:NIFTY",
        "NFO:OTHERCE",
        100.0,
        datetime.now(timezone.utc),
        trace_id="t",
    )
    assert calls["gate"] == 0
    assert result.accepted is False
    assert result.reason == "selected_contract_mismatch"
    assert result.details["candidate_symbol"] == "NFO:OTHERCE"
    assert result.details["selected_ce"] == "NFO:CE"
    assert runner._order_manager.plans == []


def test_exit_path_does_not_apply_candidate_entry_gate(monkeypatch):
    runner, calls = _execution_runner(monkeypatch, ce_ok=False, pe_ok=False)
    submitted = []
    runner._order_manager.place_reduce_only_exit = (
        lambda intent: submitted.append(intent) or "EXIT-1"
    )
    signal = Signal(
        "CLOSE_LONG",
        "NFO:PE",
        1,
        1.0,
        "protective_stop",
        None,
        None,
        metadata={},
    )
    runner._handle_exit_signal(
        signal, "NFO:PE", "NFO:PE", 100.0, datetime.now(timezone.utc)
    )
    assert calls["gate"] == 0
    assert len(submitted) == 1
    assert submitted[0].symbol == "NFO:PE"


def test_readiness_snapshot_handles_none_mdm_mappings_fail_closed():
    runner = _runner()
    runner._market_data = SimpleNamespace(
        _token_by_symbol=None,
        _active_subscribed_symbols=None,
        _desired_tokens=None,
        _subscribed_tokens=None,
        _confirmed_subscriptions=None,
        _ws=SimpleNamespace(_tokens=None),
        _symbol_subscription_generation=None,
        _symbol_first_tick_generation=None,
    )
    runner._indicator_engine = SimpleNamespace(get_history=lambda sym: [1] * 50)
    runner._current_active_contract_selection = lambda: SimpleNamespace(
        selected_ce="NFO:CE", selected_pe="NFO:PE"
    )
    runner._get_cached_quote_for_live_entry = lambda sym: {
        "ltp": 100,
        "bid": 99,
        "ask": 101,
        "depth_available": True,
    }
    runner._is_option_symbol_tick_fresh = lambda sym, max_age_s=60.0: True
    runner._selected_option_has_real_depth = lambda sym: True
    runner._required_bars_for_symbol = lambda sym: 50
    snap = runner._option_side_readiness_snapshot()
    assert snap["CE"].token is None
    assert snap["CE"].subscribed is False
    assert snap["CE"].executable is False
    assert "subscription_pending" in snap["CE"].blockers
    assert snap["PE"].subscribed is False


def test_entry_path_uses_one_readiness_snapshot_for_candidate_decision(monkeypatch):
    runner, calls = _execution_runner(monkeypatch, ce_ok=True, pe_ok=False)
    snapshot = _snap(True, False)
    snapshot_calls = {"count": 0}

    def snapshot_once(**kwargs):
        snapshot_calls["count"] += 1
        return snapshot

    runner._option_side_readiness_snapshot = snapshot_once
    result = runner._handle_entry_signal_inner(
        _signal("NFO:CE"),
        "NSE:NIFTY",
        "NFO:CE",
        100.0,
        datetime.now(timezone.utc),
        trace_id="t",
    )
    assert result.accepted is True
    assert calls["gate"] == 1
    assert snapshot_calls["count"] == 1
    assert len(runner._order_manager.plans) == 1
