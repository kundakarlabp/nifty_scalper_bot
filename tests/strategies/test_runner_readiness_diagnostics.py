from __future__ import annotations

import logging
import time
from types import SimpleNamespace

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.strategies.runner import StrategyRunner, _safe_positive_float, safe_positive_int_env


class _DummyIndicator:
    def __init__(self, history_map: dict[str, list[object]] | None = None) -> None:
        self._history_map = history_map or {}

    def get_history(self, symbol: str) -> list[object]:
        return list(self._history_map.get(symbol, []))


def _make_runner() -> StrategyRunner:
    runner = object.__new__(StrategyRunner)
    runner._logger = logging.getLogger("test.runner.readiness")
    runner._symbol_state = {}
    runner._symbol_states = {}
    runner._active_symbols = set()
    runner._candle_versions = {}
    runner._last_strategy_versions = {}
    runner._last_bar_ts = {}
    runner._data_phase = {"NFO:CE": "LIVE"}
    runner._hydration_attempted_symbols = set()
    runner._last_hydration_reason_by_symbol = {}
    runner._required_bars_for_symbol = lambda _symbol: 1
    runner._quote_update_versions = {}
    runner._live_bar_seen = set()
    runner._runtime_execution_ready_by_symbol = {"NFO:CE": False, "NFO:PE": False}
    runner._runtime_evaluation_ready = False
    runner._runtime_live_orders_armed = False
    runner._runtime_readiness_reason = "execution_not_armed:ce_depth_not_tradable"
    runner._active_selected_ce = "NFO:CE"
    runner._active_selected_pe = "NFO:PE"
    runner._indicator_engine = _DummyIndicator({"NFO:CE": [1], "NFO:PE": [1], "NFO:SYM": [1], "NFO:NIFTY26JUN23800CE": [1], "NFO:NIFTY26JUN23800PE": [1]})
    runner._market_data = SimpleNamespace(_last_tick_time={"NFO:CE": time.time() - 2, "NFO:PE": time.time() - 3}, get_ohlc_bars=lambda _s: [1])
    runner._last_same_bar_eval_block_reason_by_symbol = {}
    runner._last_same_bar_eval_block_detail_by_symbol = {}
    runner._should_log_throttled = lambda *_a, **_kw: True
    runner._selected_option_symbol_for_side = lambda side, _metadata: "NFO:CE" if side == "CE" else "NFO:PE"
    runner._is_option_symbol_tick_fresh = lambda _sym, max_age_s=60.0: True
    runner._is_symbol_execution_ready = lambda _sym: False
    return runner


def test_live_trading_readiness_snapshot_emitted(caplog) -> None:
    runner = _make_runner()
    with caplog.at_level(logging.INFO):
        runner._emit_live_trading_readiness_snapshot(
            symbol="NFO:SYM",
            strategy_signal_present=False,
            order_path_entered=False,
        )
    rec = next(r for r in caplog.records if getattr(r, "event", "") == "LIVE_TRADING_READINESS_SNAPSHOT")
    assert rec.selected_ce_symbol == "NFO:CE"
    assert rec.selected_pe_symbol == "NFO:PE"
    assert rec.live_orders_armed is False
    assert rec.order_path_entered is False


def test_runner_eval_decision_contains_split_flags(caplog) -> None:
    runner = _make_runner()
    with caplog.at_level(logging.INFO):
        runner._emit_runner_eval_decision(
            symbol="NFO:SYM",
            stage="phase9",
            reason="strategy_gate_blocked",
            allowed=False,
            diagnostic_eval_allowed=True,
            trading_allowed=False,
            order_forwarding_allowed=False,
        )
    rec = next(r for r in caplog.records if getattr(r, "event", "") == "RUNNER_EVAL_DECISION")
    assert rec.diagnostic_eval_allowed is True
    assert rec.trading_allowed is False
    assert rec.order_forwarding_allowed is False
    assert rec.market_session_state in {"open", "closed"}
    assert (not rec.order_forwarding_allowed) or rec.trading_allowed


def test_market_data_stale_symbols_detail_emitted(caplog) -> None:
    mdm = object.__new__(MarketDataManager)
    mdm._logger = logging.getLogger("test.mdm.stale")
    mdm._lock = SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: False)
    # use real lock-like minimal replacement
    import threading
    mdm._lock = threading.RLock()
    mdm._symbols_with_tick = {"NFO:CE"}
    mdm._active_subscribed_symbols = {"NFO:CE", "NFO:PE"}
    mdm._last_tick_time = {"NFO:CE": time.time() - 120, "NFO:PE": time.time() - 1}
    mdm._option_stale_seconds = 60.0
    mdm._poll_fallback_count = 0
    mdm._last_stale_detail_emit_epoch = 0.0
    mdm._is_ws_connected = lambda: True
    mdm.bus = SimpleNamespace(running=True)
    mdm._selected_ce_symbol = "NFO:CE"
    mdm._selected_pe_symbol = "NFO:PE"
    mdm._live_orders_armed = False
    mdm._main_loop = None
    mdm._last_tick_log_time = time.monotonic()
    mdm._last_tick_stats_log = 0.0
    mdm._tick_cache = {}
    mdm._tick_counter = 0
    mdm._last_tick_rate_snapshot_at = time.monotonic()
    mdm._last_tick_rate_snapshot = {}
    mdm._ticks_received_per_symbol = {}
    mdm._history = {}
    mdm._last_async_drop_log = 0.0
    mdm._async_dispatch_drops = 0
    mdm._dispatch_awaitable_callback_result = lambda *_a, **_kw: None
    mdm.bump_heartbeat = lambda: None
    mdm._m_ticks = SimpleNamespace(inc=lambda: None)
    tick = {"symbol": "NFO:CE", "ltp": 100.0}
    with caplog.at_level(logging.INFO):
        mdm._broadcast_tick(tick, [])
    rec = next(r for r in caplog.records if getattr(r, "event", "") == "MARKET_DATA_STALE_SYMBOLS_DETAIL")
    assert rec.selected_ce_stale is True
    assert rec.impact_on_trading == "live_orders_disarmed"


def test_candidate_refresh_pending_emits_deferred_or_terminal_event(caplog) -> None:
    runner = _make_runner()
    with caplog.at_level(logging.INFO):
        runner._reject_signal_execution(
            symbol="NFO:CE",
            trace_id="sig-1",
            reason="candidate_refresh_pending",
            details={"reason": "candidate_snapshot_refresh_pending", "event_loop_active": False, "candidate_refresh_age_ms": 1234, "retry_allowed": False},
        )
    events = {getattr(r, "event", "") for r in caplog.records}
    assert "SIGNAL_DROPPED_CANDIDATE_REFRESH_PENDING" in events or "SIGNAL_DEFERRED_CANDIDATE_REFRESH_PENDING" in events
    dropped = [r for r in caplog.records if getattr(r, "event", "") == "SIGNAL_DROPPED_CANDIDATE_REFRESH_PENDING"]
    if dropped:
        assert dropped[0].signal_id == "sig-1"


def test_stale_selected_symbol_does_not_arm_live_orders(caplog) -> None:
    runner = _make_runner()
    runner._runtime_live_orders_armed = False
    with caplog.at_level(logging.INFO):
        runner._emit_runner_eval_decision(
            symbol="NFO:SYM",
            stage="signal_forward",
            reason="execution_not_armed:ce_depth_not_tradable,pe_exec_quote_or_history_not_ready",
            allowed=False,
            diagnostic_eval_allowed=True,
            trading_allowed=False,
            order_forwarding_allowed=False,
        )
    rec = next(r for r in caplog.records if getattr(r, "event", "") == "RUNNER_EVAL_DECISION")
    assert rec.trading_allowed is False
    assert rec.order_forwarding_allowed is False


def test_no_trade_decision_reports_history_cold_not_broker(caplog) -> None:
    runner = _make_runner()
    category, reason = runner._classify_no_trade_decision(  # type: ignore[attr-defined]
        symbol="NFO:CE",
        signal=None,
        indicators_ctx={},
        option_count=1,
        option_required=5,
        broker_attempted=False,
    )
    assert category == "data_history_cold"
    assert reason == "insufficient_indicator_bar_count"


def test_runner_no_trade_decision_reports_context_direction_unavailable() -> None:
    runner = _make_runner()
    category, _ = runner._classify_no_trade_decision(  # type: ignore[attr-defined]
        symbol="NFO:CE",
        signal=None,
        indicators_ctx={},
        option_count=10,
        option_required=5,
        broker_attempted=False,
    )
    assert category == "context_direction_unavailable"


def test_runner_no_trade_decision_reports_context_direction_conflict() -> None:
    runner = _make_runner()
    category, _ = runner._classify_no_trade_decision(  # type: ignore[attr-defined]
        symbol="NFO:CE",
        signal=None,
        indicators_ctx={"direction_conflict": True},
        option_count=10,
        option_required=5,
        broker_attempted=False,
    )
    assert category == "context_direction_conflict"


def test_runner_no_trade_decision_reports_strategy_no_trigger_when_data_ready_but_signal_none() -> None:
    runner = _make_runner()
    runner._runtime_live_orders_armed = False
    category, _ = runner._classify_no_trade_decision(  # type: ignore[attr-defined]
        symbol="NFO:CE",
        signal=None,
        indicators_ctx={"direction_bias": "CE"},
        option_count=10,
        option_required=5,
        broker_attempted=False,
    )
    assert category == "strategy_no_trigger"


def test_runner_no_trade_uses_strategy_decision_by_symbol_even_when_indicators_lacks_symbol() -> None:
    runner = _make_runner()
    runner._strategy_manager = SimpleNamespace(get_last_no_signal_decision=lambda sym: SimpleNamespace(category="strategy_single_vote_disabled", reason="single_vote_scalp_disabled") if sym == "NFO:CE" else None)
    category, reason = runner._classify_no_trade_decision(symbol="NFO:CE", signal=None, indicators_ctx={}, option_count=10, option_required=5, broker_attempted=False)  # type: ignore[attr-defined]
    assert category == "strategy_single_vote_disabled"
    assert reason == "single_vote_scalp_disabled"


def test_runner_history_cold_overrides_stale_strategy_no_signal_decision() -> None:
    runner = _make_runner()
    runner._strategy_manager = SimpleNamespace(get_last_no_signal_decision=lambda _sym: SimpleNamespace(category="strategy_single_vote_disabled", reason="single_vote_scalp_disabled"))
    category, reason = runner._classify_no_trade_decision(symbol="NFO:CE", signal=None, indicators_ctx={}, option_count=1, option_required=5, broker_attempted=False)  # type: ignore[attr-defined]
    assert category == "data_history_cold"
    assert reason == "insufficient_indicator_bar_count"


def test_order_failure_cooldown_invalid_env_does_not_crash() -> None:
    assert _safe_positive_float("abc", 120.0, minimum=1.0) == 120.0


def test_order_failure_cooldown_zero_env_uses_default() -> None:
    assert _safe_positive_float("0", 120.0, minimum=1.0) == 120.0


def test_order_failure_cooldown_negative_env_uses_default() -> None:
    assert _safe_positive_float("-5", 120.0, minimum=1.0) == 120.0


def test_runner_emits_no_trade_decision_on_history_cold_eval_block(caplog) -> None:
    runner = _make_runner()
    runner._active_symbols = {"NFO:CE"}
    runner._indicator_engine = _DummyIndicator({"NFO:CE": [1]})
    runner._required_bars_for_symbol = lambda _s: 5
    runner._is_tradable_symbol = lambda _s: True
    runner._is_context_symbol = lambda _s: False
    runner._is_option_symbol_tick_fresh = lambda _s, max_age_s=60.0: True
    with caplog.at_level(logging.INFO):
        allowed = runner._strategy_evaluation_allowed("NFO:CE", trace_id="t-cold")
    assert allowed is False
    rec = next(r for r in caplog.records if getattr(r, "event", "") == "RUNNER_NO_TRADE_DECISION")
    assert rec.category == "data_history_cold"
    assert rec.broker_attempted is False


def test_runner_no_trade_decision_reports_integrity_failure_when_count_sufficient_but_has_min_bars_false(caplog) -> None:
    runner = _make_runner()
    runner._active_symbols = {"NFO:CE"}
    runner._indicator_engine = _DummyIndicator({"NFO:CE": [1, 2, 3, 4, 5]})
    runner._required_bars_for_symbol = lambda _s: 5
    runner._is_tradable_symbol = lambda _s: True
    runner._is_context_symbol = lambda _s: False
    runner._is_option_symbol_tick_fresh = lambda _s, max_age_s=60.0: True
    runner._indicator_engine.has_min_bars = lambda _s, _r: False  # type: ignore[attr-defined]
    with caplog.at_level(logging.INFO):
        allowed = runner._strategy_evaluation_allowed("NFO:CE", trace_id="t-integrity")
    assert allowed is False
    rec = next(r for r in caplog.records if getattr(r, "event", "") == "RUNNER_NO_TRADE_DECISION")
    assert rec.category == "data_history_integrity_failed"
    assert rec.reason == "indicator_history_integrity_failed"


def test_emit_no_trade_decision_does_not_raise_when_kill_switch_status_raises() -> None:
    runner = _make_runner()
    runner._order_manager = SimpleNamespace(is_kill_switch_active=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    runner._emit_no_trade_decision(symbol="NFO:CE", trace_id="t", category="strategy_no_trigger", reason="evaluation_no_signal")


def test_emit_no_trade_decision_does_not_raise_with_missing_order_manager() -> None:
    runner = _make_runner()
    del runner._order_manager
    runner._emit_no_trade_decision(symbol="NFO:CE", trace_id="t", category="strategy_no_trigger", reason="evaluation_no_signal")


def test_invalid_option_execution_min_bars_env_does_not_crash(monkeypatch) -> None:
    monkeypatch.setenv("OPTION_EXECUTION_MIN_BARS", "bad")
    assert safe_positive_int_env("OPTION_EXECUTION_MIN_BARS", 5, minimum=1) == 5


def test_runner_no_trade_prefers_strategy_decision_before_context_unavailable() -> None:
    runner = _make_runner()
    runner._strategy_manager = SimpleNamespace(
        get_last_no_signal_decision=lambda _sym: SimpleNamespace(
            category="strategy_single_vote_disabled", reason="single_vote_scalp_disabled"
        )
    )
    category, reason = runner._classify_no_trade_decision(
        symbol="NFO:CE", signal=None, indicators_ctx={}, option_count=10, option_required=5, broker_attempted=False
    )
    assert category == "strategy_single_vote_disabled"
    assert reason == "single_vote_scalp_disabled"


def test_symbol_live_entry_ready_false_when_kill_switch_active() -> None:
    runner = _make_runner()
    runner._is_tradable_symbol = lambda _s: True
    runner._runtime_live_orders_armed = True
    runner._order_manager_kill_switch_status_for_entry = lambda: (True, {"active": True, "kill_reason": "unexpected_exception", "consecutive_failures": 6, "last_exception_type": "RuntimeError", "last_exception_message": "test failure"})
    ready, reason, details = runner._symbol_live_entry_ready("NFO:CE")
    assert ready is False
    assert reason == "order_manager_kill_switch_active"
    assert details["kill_switch_status"]["consecutive_failures"] == 6


def test_symbol_live_entry_ready_false_when_context_stale(monkeypatch) -> None:
    runner = _make_runner()
    runner._is_tradable_symbol = lambda _s: True
    runner._order_manager_kill_switch_status_for_entry = lambda: (False, {})
    runner._runtime_indicators = {"NFO:CE": {"context_age_seconds": 11.0}}
    monkeypatch.setenv("MAX_CONTEXT_AGE_SECONDS", "5")
    ready, reason, _details = runner._symbol_live_entry_ready("NFO:CE")
    assert ready is False
    assert reason == "context_stale"


def test_ce_signal_blocked_when_direction_bias_pe() -> None:
    runner = _make_runner()
    runner._is_tradable_symbol = lambda _s: True
    runner._order_manager_kill_switch_status_for_entry = lambda: (False, {})
    runner._runtime_indicators = {"NFO:NIFTY26JUN23800CE": {"direction_bias": "PE"}}
    ready, reason, _details = runner._symbol_live_entry_ready("NFO:NIFTY26JUN23800CE")
    assert ready is False
    assert reason == "context_direction_conflict"


def test_pe_signal_allowed_when_direction_bias_pe_and_other_gates_pass() -> None:
    runner = _make_runner()
    runner._is_tradable_symbol = lambda _s: True
    runner._order_manager_kill_switch_status_for_entry = lambda: (False, {})
    runner._runtime_indicators = {"NFO:NIFTY26JUN23800PE": {"direction_bias": "PE"}}
    runner._active_selected_pe = "NFO:NIFTY26JUN23800PE"
    runner._get_cached_quote_for_live_entry = lambda _s: {"tradable_quote": True, "depth_available": True, "spread_pct": 0.2}
    ready, reason, _details = runner._symbol_live_entry_ready("NFO:NIFTY26JUN23800PE")
    assert ready is True
    assert reason == "symbol_live_ready"


def test_runner_ignores_stale_strategy_decision() -> None:
    runner = _make_runner()
    stale = SimpleNamespace(symbol="NFO:CE", category="strategy_single_vote_disabled", reason="single_vote_scalp_disabled", created_ts=time.time() - 10)
    runner._strategy_manager = SimpleNamespace(get_last_no_signal_decision=lambda _sym: stale)
    category, _reason = runner._classify_no_trade_decision(symbol="NFO:CE", signal=None, indicators_ctx={}, option_count=10, option_required=5, broker_attempted=False, trace_id="t1")
    assert category == "context_direction_unavailable"


def test_symbol_live_entry_ready_does_not_crash_when_active_atm_strike_missing() -> None:
    runner = _make_runner()
    if hasattr(runner, "_active_atm_strike"):
        delattr(runner, "_active_atm_strike")
    runner._is_tradable_symbol = lambda _s: True
    runner._order_manager_kill_switch_status_for_entry = lambda: (False, {})
    runner._active_selected_pe = "NFO:NIFTY26JUN23800PE"
    runner._get_cached_quote_for_live_entry = lambda _s: {"tradable_quote": True, "depth_available": True, "spread_pct": 0.2}
    ready, reason, _details = runner._symbol_live_entry_ready("NFO:NIFTY26JUN23800PE")
    assert ready is True
    assert reason == "symbol_live_ready"


def test_symbol_live_entry_ready_false_when_quote_missing() -> None:
    runner = _make_runner()
    runner._is_tradable_symbol = lambda _s: True
    runner._order_manager_kill_switch_status_for_entry = lambda: (False, {})
    runner._active_selected_pe = "NFO:NIFTY26JUN23800PE"
    runner._get_cached_quote_for_live_entry = lambda _s: {}
    ready, reason, _details = runner._symbol_live_entry_ready("NFO:NIFTY26JUN23800PE")
    assert ready is False
    assert reason == "quote_missing"


def test_runner_uses_current_strategy_decision() -> None:
    runner = _make_runner()
    decision = SimpleNamespace(symbol="NFO:CE", category="strategy_single_vote_disabled", reason="single_vote_scalp_disabled", trace_id="t1", created_ts=time.time())
    runner._strategy_manager = SimpleNamespace(get_last_no_signal_decision=lambda _sym: decision)
    category, reason = runner._classify_no_trade_decision(symbol="NFO:CE", signal=None, indicators_ctx={}, option_count=10, option_required=5, broker_attempted=False, trace_id="t1")
    assert category == "strategy_single_vote_disabled"
    assert reason == "single_vote_scalp_disabled"


def test_runner_ignores_strategy_decision_for_wrong_symbol() -> None:
    runner = _make_runner()
    decision = SimpleNamespace(symbol="NFO:PE", category="strategy_single_vote_disabled", reason="single_vote_scalp_disabled", trace_id="t1", created_ts=time.time())
    runner._strategy_manager = SimpleNamespace(get_last_no_signal_decision=lambda _sym: decision)
    category, _reason = runner._classify_no_trade_decision(symbol="NFO:CE", signal=None, indicators_ctx={}, option_count=10, option_required=5, broker_attempted=False, trace_id="t1")
    assert category == "context_direction_unavailable"


def test_runner_ignores_strategy_decision_for_wrong_trace_id() -> None:
    runner = _make_runner()
    decision = SimpleNamespace(symbol="NFO:CE", category="strategy_single_vote_disabled", reason="single_vote_scalp_disabled", trace_id="tX", created_ts=time.time())
    runner._strategy_manager = SimpleNamespace(get_last_no_signal_decision=lambda _sym: decision)
    category, _reason = runner._classify_no_trade_decision(symbol="NFO:CE", signal=None, indicators_ctx={}, option_count=10, option_required=5, broker_attempted=False, trace_id="t1")
    assert category == "context_direction_unavailable"


def test_symbol_live_entry_ready_false_when_depth_missing() -> None:
    runner = _make_runner()
    runner._is_tradable_symbol = lambda _s: True
    runner._order_manager_kill_switch_status_for_entry = lambda: (False, {})
    runner._active_selected_pe = "NFO:NIFTY26JUN23800PE"
    runner._get_cached_quote_for_live_entry = lambda _s: {"tradable_quote": True, "depth_available": False, "spread_pct": 0.2}
    ready, reason, _details = runner._symbol_live_entry_ready("NFO:NIFTY26JUN23800PE")
    assert ready is False
    assert reason == "quote_depth_unavailable"


def test_symbol_live_entry_ready_false_when_spread_unknown_by_default() -> None:
    runner = _make_runner()
    runner._is_tradable_symbol = lambda _s: True
    runner._order_manager_kill_switch_status_for_entry = lambda: (False, {})
    runner._active_selected_pe = "NFO:NIFTY26JUN23800PE"
    runner._get_cached_quote_for_live_entry = lambda _s: {"tradable_quote": True, "depth_available": True}
    ready, reason, _details = runner._symbol_live_entry_ready("NFO:NIFTY26JUN23800PE")
    assert ready is False
    assert reason == "spread_unknown"


def test_runner_emits_no_trade_decision_on_signal_none_strategy_no_trigger(caplog) -> None:
    runner = _make_runner()
    with caplog.at_level(logging.INFO):
        runner._emit_no_trade_decision(
            symbol="NFO:CE",
            trace_id="t-no-signal",
            category="strategy_no_trigger",
            reason="evaluation_no_signal",
            broker_attempted=False,
            indicators_ctx={"direction_bias": "CE"},
            option_history_count=10,
            option_history_required=5,
        )
    rec = next(r for r in caplog.records if getattr(r, "event", "") == "RUNNER_NO_TRADE_DECISION")
    assert rec.category == "strategy_no_trigger"
    assert rec.reason == "evaluation_no_signal"
    assert rec.broker_attempted is False

def test_is_option_symbol_tick_fresh_uses_cache_only_no_pull_quote() -> None:
    runner = _make_runner()
    calls: list[bool] = []
    runner._is_tradable_symbol = lambda _s: True
    runner._get_cached_quote_for_live_entry = lambda _s: {"tick_age_s": 1.0}
    runner._data_hub = SimpleNamespace(get_quote=lambda *_a, **_k: calls.append(True))
    assert runner._is_option_symbol_tick_fresh("NFO:CE", max_age_s=5.0) is True
    assert calls == []


def test_tick_fresh_returns_false_when_cache_missing() -> None:
    runner = _make_runner()
    runner._is_tradable_symbol = lambda _s: True
    runner._get_cached_quote_for_live_entry = lambda _s: {}
    runner._market_data = SimpleNamespace(time_since_last_tick=lambda _s: None)
    runner._data_hub = SimpleNamespace(time_since_last_tick=lambda _s: None)
    assert runner._is_option_symbol_tick_fresh("NFO:CE", max_age_s=5.0) is False

def test_symbol_live_entry_ready_does_not_call_datahub_get_quote_with_allow_pull_true() -> None:
    runner = _make_runner()
    pulled = {"called": False}
    runner._is_tradable_symbol = lambda _s: True
    runner._get_cached_quote_for_live_entry = lambda _s: {"tick_age_s": 0.5}
    runner._data_hub = SimpleNamespace(get_quote=lambda *_a, **kwargs: pulled.__setitem__("called", kwargs.get("allow_pull", True)))
    assert runner._is_option_symbol_tick_fresh("NFO:CE", max_age_s=5.0) is True
    assert pulled["called"] is False
