from __future__ import annotations

from pathlib import Path

RUNNER = Path("src/nifty_scalper_bot/strategies/runner.py")
TESTS = Path("tests/strategies/test_runner_symbol_role_gate.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


runner = RUNNER.read_text()

runner = replace_once(
    runner,
    '''    def _reject_signal_execution(\n        self,\n        *,\n        symbol: str,\n        trace_id: str,\n        reason: str,\n        details: Mapping[str, Any] | None = None,\n    ) -> SignalExecutionResult:\n        """Log and return rejection. Args: symbol/trace_id/reason/details. Returns: result. Raises: none."""\n        payload = dict(details or {})\n        log_throttled_live(\n            self._logger,\n            logging.INFO,\n            "SIGNAL_EXECUTION_RESULT",\n            f"SIGNAL_EXECUTION_RESULT:{symbol}:{reason}",\n            float(os.getenv("LOG_THROTTLE_STRATEGY_REJECT_SECONDS", "120") or "120"),\n            "SIGNAL_EXECUTION_RESULT accepted=False reason=%s symbol=%s trace_id=%s",\n            reason,\n            symbol,\n            trace_id,\n            extra={\n                "event": "SIGNAL_EXECUTION_RESULT",\n                "accepted": False,\n                "reason": reason,\n                "symbol": symbol,\n                "trace_id": trace_id,\n                **payload,\n            },\n        )\n''',
    '''    def _reject_signal_execution(\n        self,\n        *,\n        symbol: str,\n        trace_id: str,\n        reason: str,\n        details: Mapping[str, Any] | None = None,\n        terminal: bool = False,\n    ) -> SignalExecutionResult:\n        """Log and return rejection. Args: symbol/trace_id/reason/details/terminal. Returns: result. Raises: none."""\n        payload = dict(details or {})\n        event_extra = {\n            "event": "SIGNAL_EXECUTION_RESULT",\n            "accepted": False,\n            "reason": reason,\n            "symbol": symbol,\n            "trace_id": trace_id,\n            **payload,\n        }\n        if terminal:\n            # A candidate that has entered phase 10 must always have one visible\n            # terminal outcome. Do not throttle this trace-scoped lifecycle event.\n            self._logger.info(\n                "SIGNAL_EXECUTION_RESULT accepted=False reason=%s symbol=%s trace_id=%s",\n                reason,\n                symbol,\n                trace_id,\n                extra=event_extra,\n            )\n        else:\n            log_throttled_live(\n                self._logger,\n                logging.INFO,\n                "SIGNAL_EXECUTION_RESULT",\n                f"SIGNAL_EXECUTION_RESULT:{symbol}:{reason}",\n                float(\n                    os.getenv("LOG_THROTTLE_STRATEGY_REJECT_SECONDS", "120") or "120"\n                ),\n                "SIGNAL_EXECUTION_RESULT accepted=False reason=%s symbol=%s trace_id=%s",\n                reason,\n                symbol,\n                trace_id,\n                extra=event_extra,\n            )\n''',
    "terminal rejection helper",
)

runner = replace_once(
    runner,
    '''                if signal.action in {"BUY", "SELL"} and is_nifty_option_symbol(\n                    entry_symbol\n                ):\n                    expiry_blocked, expiry_reason = expiry_theta_block()\n                    if expiry_blocked:\n                        self._emit_runner_eval_decision(\n                            symbol=entry_symbol,\n                            stage="phase10_entry_policy",\n                            reason=expiry_reason,\n                            allowed=False,\n                            trace_id=trace_id,\n                        )\n                        return\n''',
    '''                if signal.action == "BUY" and is_nifty_option_symbol(entry_symbol):\n                    expiry_blocked, expiry_reason = expiry_theta_block()\n                    if expiry_blocked:\n                        self._emit_runner_eval_decision(\n                            symbol=entry_symbol,\n                            stage="phase10_entry_policy",\n                            reason=expiry_reason,\n                            allowed=False,\n                            trace_id=trace_id,\n                        )\n                        self._reject_signal_execution(\n                            symbol=entry_symbol,\n                            trace_id=trace_id,\n                            reason=expiry_reason,\n                            details={\n                                "stage": "phase10_entry_policy",\n                                "broker_attempted": False,\n                                "entry_action": signal.action,\n                            },\n                            terminal=True,\n                        )\n                        return\n''',
    "expiry buy-only terminal gate",
)

runner = replace_once(
    runner,
    '''                            "trace_id": trace_id,\n                        },\n                    )\n                    return\n                self._update_symbol_execution_phase(\n                    symbol, "LIVE_READY", "symbol_live_ready"\n                )\n''',
    '''                            "trace_id": trace_id,\n                        },\n                    )\n                    self._reject_signal_execution(\n                        symbol=entry_symbol,\n                        trace_id=trace_id,\n                        reason=live_ready_reason,\n                        details={\n                            "stage": "phase10_live_readiness",\n                            "broker_attempted": False,\n                            "data_phase": signal_phase,\n                        },\n                        terminal=True,\n                    )\n                    return\n                self._update_symbol_execution_phase(\n                    symbol, "LIVE_READY", "symbol_live_ready"\n                )\n''',
    "live readiness terminal result",
)

runner = replace_once(
    runner,
    '''                                "confidence": float(signal.confidence or 0.0),\n                            },\n                        )\n                        return\n                if (\n                    signal.action in {"BUY", "SELL"}\n                    and not self._strategy_slots_available()\n                ):\n''',
    '''                                "confidence": float(signal.confidence or 0.0),\n                            },\n                        )\n                        self._reject_signal_execution(\n                            symbol=entry_symbol,\n                            trace_id=trace_id,\n                            reason="low_volatility_rejected",\n                            details={\n                                "stage": "phase10_regime",\n                                "broker_attempted": False,\n                            },\n                            terminal=True,\n                        )\n                        return\n                if (\n                    signal.action in {"BUY", "SELL"}\n                    and not self._strategy_slots_available()\n                ):\n''',
    "low volatility terminal result",
)

runner = replace_once(
    runner,
    '''                    self._warn_symbol_gate(\n                        "strategy_slots_full",\n                        symbol,\n                        "Strategy execution slots are full; rejecting new entry signal",\n                        reason="max_concurrent_strategies_reached",\n                        max_slots=self._strategy_slot_limit,\n                    )\n                    return\n''',
    '''                    self._warn_symbol_gate(\n                        "strategy_slots_full",\n                        symbol,\n                        "Strategy execution slots are full; rejecting new entry signal",\n                        reason="max_concurrent_strategies_reached",\n                        max_slots=self._strategy_slot_limit,\n                    )\n                    self._reject_signal_execution(\n                        symbol=entry_symbol,\n                        trace_id=trace_id,\n                        reason="max_concurrent_strategies_reached",\n                        details={\n                            "stage": "phase10_strategy_slots",\n                            "broker_attempted": False,\n                            "max_slots": self._strategy_slot_limit,\n                        },\n                        terminal=True,\n                    )\n                    return\n''',
    "strategy slot terminal result",
)

runner = replace_once(
    runner,
    '''                                        "allowed": False,\n                                    },\n                                )\n                            return\n\n                        state.strategy_data["last_signal"] = {\n''',
    '''                                        "allowed": False,\n                                    },\n                                )\n                            self._reject_signal_execution(\n                                symbol=entry_symbol,\n                                trace_id=trace_id,\n                                reason="signal_cooldown_active",\n                                details={\n                                    "stage": "phase10_signal_cooldown",\n                                    "broker_attempted": False,\n                                    "elapsed_s": round(elapsed_s, 3),\n                                    "required_s": float(\n                                        self._config.signal_cooldown_seconds\n                                    ),\n                                },\n                                terminal=True,\n                            )\n                            return\n\n                        state.strategy_data["last_signal"] = {\n''',
    "signal cooldown terminal result",
)

runner = replace_once(
    runner,
    '''                    self._emit_runner_eval_decision(\n                        symbol=symbol,\n                        stage="phase10_execute",\n                        reason="data_pipeline_overloaded",\n                        allowed=False,\n                    )\n                    return\n''',
    '''                    self._emit_runner_eval_decision(\n                        symbol=symbol,\n                        stage="phase10_execute",\n                        reason="data_pipeline_overloaded",\n                        allowed=False,\n                        trace_id=trace_id,\n                    )\n                    self._reject_signal_execution(\n                        symbol=entry_symbol,\n                        trace_id=trace_id,\n                        reason="data_pipeline_overloaded",\n                        details={\n                            "stage": "phase10_execute",\n                            "broker_attempted": False,\n                        },\n                        terminal=True,\n                    )\n                    return\n''',
    "pipeline overload terminal result",
)

runner = replace_once(
    runner,
    '''                    self._emit_runner_eval_decision(\n                        symbol=symbol,\n                        stage="phase10_execute",\n                        reason="live_orders_not_armed",\n                        allowed=False,\n                    )\n                    return\n''',
    '''                    self._emit_runner_eval_decision(\n                        symbol=symbol,\n                        stage="phase10_execute",\n                        reason="live_orders_not_armed",\n                        allowed=False,\n                        trace_id=trace_id,\n                    )\n                    self._reject_signal_execution(\n                        symbol=entry_symbol,\n                        trace_id=trace_id,\n                        reason="live_orders_not_armed",\n                        details={\n                            "stage": "phase10_execute",\n                            "broker_attempted": False,\n                        },\n                        terminal=True,\n                    )\n                    return\n''',
    "live orders unarmed terminal result",
)

RUNNER.write_text(runner)


tests = TESTS.read_text()
marker = "def _phase10_terminal_events(runner_obj):"
if marker not in tests:
    tests += r'''


def _phase10_terminal_events(runner_obj):
    return [
        call.kwargs["extra"]
        for call in runner_obj._logger.info.call_args_list
        if isinstance(call.kwargs.get("extra"), dict)
        and call.kwargs["extra"].get("event") == "SIGNAL_EXECUTION_RESULT"
    ]


def test_expiry_entry_policy_emits_unthrottled_terminal_result(monkeypatch):
    runner_obj, strategy_manager, risk_manager, order_manager, selected_ce = (
        _build_phase9_runner(monkeypatch)
    )
    runner_obj._trigger_candidate_symbols = {"NSE:NIFTY"}
    runner_obj._logger = Mock()
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner.expiry_theta_block",
        lambda: (True, "expiry_day_after_13:30_ist"),
    )

    runner_obj._on_tick(
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "last_price": 24000.0,
            "timestamp": time.time(),
            "trace_id": "expiry-terminal-result",
            "source": "ws",
        },
    )

    strategy_manager.generate_signal.assert_called_once()
    risk_manager.validate.assert_not_called()
    order_manager.submit.assert_not_called()
    terminal = _phase10_terminal_events(runner_obj)
    assert terminal
    assert terminal[-1]["symbol"] == selected_ce
    assert terminal[-1]["reason"] == "expiry_day_after_13:30_ist"
    assert terminal[-1]["accepted"] is False
    assert terminal[-1]["broker_attempted"] is False
    assert terminal[-1]["stage"] == "phase10_entry_policy"


def test_expiry_theta_gate_does_not_block_option_sell_entry(monkeypatch):
    runner_obj, strategy_manager, _risk_manager, order_manager, selected_ce = (
        _build_phase9_runner(monkeypatch)
    )
    runner_obj._trigger_candidate_symbols = {"NSE:NIFTY"}
    strategy_manager.generate_signal.return_value = Signal(
        "SELL",
        selected_ce,
        75,
        0.9,
        "test_option_sell",
        120.0,
        90.0,
        metadata={"timestamp": time.time()},
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner.expiry_theta_block",
        lambda: (True, "expiry_day_after_13:30_ist"),
    )

    runner_obj._on_tick(
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "last_price": 24000.0,
            "timestamp": time.time(),
            "trace_id": "expiry-sell-not-blocked",
            "source": "ws",
        },
    )

    order_manager.submit.assert_called_once()


def test_signal_cooldown_emits_terminal_result_even_when_branch_log_throttled(
    monkeypatch,
):
    runner_obj, _strategy_manager, _risk_manager, order_manager, selected_ce = (
        _build_phase9_runner(monkeypatch)
    )
    runner_obj._trigger_candidate_symbols = {"NSE:NIFTY"}
    runner_obj._logger = Mock()
    runner_obj._config.signal_cooldown_seconds = 30.0
    runner_obj._symbol_last_signal_ts["NSE:NIFTY"] = time.time()
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner.expiry_theta_block",
        lambda: (False, "before_cutoff"),
    )

    runner_obj._on_tick(
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "last_price": 24000.0,
            "timestamp": time.time(),
            "trace_id": "cooldown-terminal-result",
            "source": "ws",
        },
    )

    order_manager.submit.assert_not_called()
    terminal = _phase10_terminal_events(runner_obj)
    assert terminal
    assert terminal[-1]["symbol"] == selected_ce
    assert terminal[-1]["reason"] == "signal_cooldown_active"
    assert terminal[-1]["broker_attempted"] is False
    assert terminal[-1]["required_s"] == 30.0


def test_phase10_strategy_slot_rejection_emits_terminal_result(monkeypatch):
    runner_obj, _strategy_manager, _risk_manager, order_manager, _selected_ce = (
        _build_phase9_runner(monkeypatch)
    )
    runner_obj._trigger_candidate_symbols = {"NSE:NIFTY"}
    runner_obj._logger = Mock()
    runner_obj._strategy_slots_available = lambda: False
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner.expiry_theta_block",
        lambda: (False, "before_cutoff"),
    )

    runner_obj._on_tick(
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "last_price": 24000.0,
            "timestamp": time.time(),
            "trace_id": "slots-terminal-result",
            "source": "ws",
        },
    )

    order_manager.submit.assert_not_called()
    terminal = _phase10_terminal_events(runner_obj)
    assert terminal[-1]["reason"] == "max_concurrent_strategies_reached"
    assert terminal[-1]["broker_attempted"] is False


def test_phase10_low_volatility_rejection_emits_terminal_result(monkeypatch):
    runner_obj, _strategy_manager, _risk_manager, order_manager, _selected_ce = (
        _build_phase9_runner(monkeypatch)
    )
    runner_obj._trigger_candidate_symbols = {"NSE:NIFTY"}
    runner_obj._logger = Mock()
    runner_obj.detect_market_regime = lambda _symbol: "low_volatility"
    runner_obj._block_low_volatility = True
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner.expiry_theta_block",
        lambda: (False, "before_cutoff"),
    )

    runner_obj._on_tick(
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "last_price": 24000.0,
            "timestamp": time.time(),
            "trace_id": "low-vol-terminal-result",
            "source": "ws",
        },
    )

    order_manager.submit.assert_not_called()
    terminal = _phase10_terminal_events(runner_obj)
    assert terminal[-1]["reason"] == "low_volatility_rejected"
    assert terminal[-1]["broker_attempted"] is False


def test_phase10_live_readiness_rejection_emits_terminal_result(monkeypatch):
    runner_obj, _strategy_manager, _risk_manager, order_manager, _selected_ce = (
        _build_phase9_runner(monkeypatch)
    )
    runner_obj._trigger_candidate_symbols = {"NSE:NIFTY"}
    runner_obj._logger = Mock()
    runner_obj._symbol_live_entry_ready = lambda _symbol, signal=None, trace_id=None: (
        False,
        "candidate_quote_stale",
        {},
    )
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner.expiry_theta_block",
        lambda: (False, "before_cutoff"),
    )

    runner_obj._on_tick(
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "last_price": 24000.0,
            "timestamp": time.time(),
            "trace_id": "readiness-terminal-result",
            "source": "ws",
        },
    )

    order_manager.submit.assert_not_called()
    terminal = _phase10_terminal_events(runner_obj)
    assert terminal[-1]["reason"] == "candidate_quote_stale"
    assert terminal[-1]["broker_attempted"] is False


def test_phase10_pipeline_overload_emits_terminal_result(monkeypatch):
    runner_obj, _strategy_manager, _risk_manager, order_manager, _selected_ce = (
        _build_phase9_runner(monkeypatch)
    )
    runner_obj._trigger_candidate_symbols = {"NSE:NIFTY"}
    runner_obj._logger = Mock()
    runner_obj._market_data.pipeline_overloaded = True
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner.expiry_theta_block",
        lambda: (False, "before_cutoff"),
    )

    runner_obj._on_tick(
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "last_price": 24000.0,
            "timestamp": time.time(),
            "trace_id": "overload-terminal-result",
            "source": "ws",
        },
    )

    order_manager.submit.assert_not_called()
    terminal = _phase10_terminal_events(runner_obj)
    assert terminal[-1]["reason"] == "data_pipeline_overloaded"
    assert terminal[-1]["broker_attempted"] is False
'''

TESTS.write_text(tests)

print("phase10 terminal entry policy patch applied")
