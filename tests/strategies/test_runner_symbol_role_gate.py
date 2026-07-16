from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from nifty_scalper_bot.core.message_bus import MessageBus
from nifty_scalper_bot.strategies.runner import (
    EntryEvaluationRoute,
    MarketRegime,
    MarketState,
    RunnerState,
    StrategyRunner,
    StrategyRunnerConfig,
    SymbolRuntimeState,
    SymbolState,
)
from nifty_scalper_bot.strategies.signal_generator import Signal


def runner():
    r = StrategyRunner.__new__(StrategyRunner)
    r._active_selected_ce = "NFO:NIFTY26JUN24000CE"
    r._active_selected_pe = "NFO:NIFTY26JUN24000PE"
    r._selected_ce_symbol = None
    r._selected_pe_symbol = None
    r._pending_selected_ce = None
    r._pending_selected_pe = None
    r._active_contract_basket = None
    r._data_hub = None
    r._market_data = None
    r._position_manager = SimpleNamespace(has_open_position=lambda _s: False)
    return r


def test_futures_context_updates_context_but_does_not_enter_phase9_entry():
    r = runner()
    strategy_manager = Mock()
    order_manager = Mock()

    assert (
        r._entry_evaluation_route("NFO:NIFTY26JUNFUT")
        == EntryEvaluationRoute.CONTEXT_ONLY
    )
    strategy_manager.generate_signal.assert_not_called()
    order_manager.submit.assert_not_called()


def test_spot_context_routes_as_underlying_context():
    assert (
        runner()._entry_evaluation_route("NSE:NIFTY") == EntryEvaluationRoute.UNDERLYING
    )


def test_non_selected_option_context_cannot_trigger_entry():
    assert (
        runner()._entry_evaluation_route("NFO:NIFTY26JUN24100CE")
        == EntryEvaluationRoute.CONTEXT_ONLY
    )


def test_selected_option_can_trigger_entry():
    assert (
        runner()._entry_evaluation_route("NFO:NIFTY26JUN24000CE")
        == EntryEvaluationRoute.OPTION_CANDIDATE
    )


def test_open_position_role_can_trigger_management_path():
    r = runner()
    r._position_manager = SimpleNamespace(
        has_open_position=lambda s: s.endswith("24100CE")
    )
    assert (
        r._entry_evaluation_route("NFO:NIFTY26JUN24100CE")
        == EntryEvaluationRoute.POSITION_MANAGEMENT
    )


class _SilentLogger:
    def debug(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def exception(self, *_args, **_kwargs):
        return None


class _Indicator:
    def get_history(self, _symbol):
        return [{"timestamp": time.time()}] * 100

    def has_min_bars(self, _symbol, _required):
        return True

    def update_price(self, *_args, **_kwargs):
        return None

    def update_bar(self, *_args, **_kwargs):
        return None

    def set_runtime_context(self, *_args, **_kwargs):
        return None

    def get_indicators(self, *_args, **_kwargs):
        return {
            "direction_bias": "CE",
            "underlying_direction_bias": "CE",
            "spot_fresh": True,
            "fut_fresh": True,
            "context_fresh": True,
            "underlying_direction_confidence": 0.9,
        }


class _MDM:
    def __init__(self, *, current_generation_ready: bool = True) -> None:
        self.current_generation_ready = current_generation_ready

    def classify_live_tick_readiness(self, symbol, token, *, max_age_s):
        return {
            "symbol": symbol,
            "token": token,
            "tracked": True,
            "subscription_requested": True,
            "subscription_confirmed": True,
            "token_matches": True,
            "expected_generation": 1,
            "tick_generation": 1 if self.current_generation_ready else None,
            "current_generation_tick_received": self.current_generation_ready,
            "tick_age_s": 1.0 if self.current_generation_ready else None,
            "fresh": self.current_generation_ready,
            "ready": self.current_generation_ready,
            "reason": (
                "ready"
                if self.current_generation_ready
                else "current_generation_tick_pending"
            ),
        }

    def current_live_token(self, _symbol):
        return 1

    def subscribe(self, *_args, **_kwargs):
        return None

    def get_latest_tick(self, symbol):
        return {
            "symbol": symbol,
            "last_price": 24000.0,
            "ltp": 24000.0,
            "timestamp": time.time(),
        }


class _PositionManager:
    def get_position(self, _symbol):
        return None

    def has_open_position(self, _symbol):
        return False

    def is_flat(self, _symbol):
        return True


class _OrderManager:
    def __init__(self) -> None:
        self.submit = Mock(return_value="OID-1")

    def submit_trade_plan(self, plan):
        return self.submit(plan)

    def resolve_lot_size(self, _symbol):
        return 75

    def is_kill_switch_active(self):
        return False

    def consume_skip_reason(self):
        return None

    def get_order(self, _order_id):
        return None


class _DataHub:
    def __init__(self) -> None:
        self.subscribed: list[tuple[str, object]] = []

    def subscribe_ticks(self, symbol, callback):
        self.subscribed.append((symbol, callback))


def _build_phase9_runner(
    monkeypatch,
    *,
    current_generation_ready: bool = True,
    expire_boot_grace: bool = True,
):
    monkeypatch.setenv("BROKER_API_KEY", "test")
    monkeypatch.setenv("BROKER_API_SECRET", "test")
    monkeypatch.setenv("BROKER_ACCESS_TOKEN", "test")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setattr(
        "nifty_scalper_bot.strategies.runner.get_market_state",
        lambda: MarketState.OPEN,
    )
    selected_ce = "NFO:NIFTY26JUN24000CE"
    risk_manager = SimpleNamespace(
        available_balance=100000.0,
        validate=Mock(return_value=True),
        is_circuit_breaker_tripped=lambda: (False, None),
    )
    strategy_manager = SimpleNamespace(
        generate_signal=Mock(
            return_value=Signal(
                "BUY",
                selected_ce,
                75,
                0.9,
                "test_underlying_signal",
                90.0,
                120.0,
                metadata={"timestamp": time.time()},
            )
        )
    )
    order_manager = _OrderManager()
    runner_obj = StrategyRunner(
        market_data_manager=_MDM(current_generation_ready=current_generation_ready),
        indicator_engine=_Indicator(),
        strategy_manager=strategy_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=_PositionManager(),
        message_bus=MessageBus(),
        config=StrategyRunnerConfig(signal_cooldown_seconds=0.0),
    )
    runner_obj._logger = _SilentLogger()
    runner_obj._active_symbols.update({"NSE:NIFTY", selected_ce})
    runner_obj._tracked_symbols.update(runner_obj._active_symbols)
    runner_obj._mdm_callback_registered = True
    runner_obj._active_selected_ce = selected_ce
    runner_obj._active_selected_pe = "NFO:NIFTY26JUN24000PE"
    runner_obj._active_basket_token_by_symbol = {selected_ce: 1}
    runner_obj._history_ready_by_symbol["NSE:NIFTY"] = True
    runner_obj._history_ready_by_symbol[selected_ce] = True
    runner_obj._data_phase["NSE:NIFTY"] = "LIVE"
    runner_obj._symbol_history = {"NSE:NIFTY": [{"timestamp": time.time()}]}
    runner_obj._last_bar_ts = {"NSE:NIFTY": datetime.now(timezone.utc)}
    runner_obj._symbol_state["NSE:NIFTY"] = SymbolRuntimeState("NSE:NIFTY", 100)
    runner_obj._symbol_state["NSE:NIFTY"].active = True
    runner_obj._symbol_states["NSE:NIFTY"] = SymbolState.READY
    runner_obj._runtime_startup_ready = True
    runner_obj._runtime_data_hard_ready = True
    runner_obj._runtime_evaluation_ready = True
    runner_obj._runtime_live_orders_armed = True
    runner_obj._runtime_readiness_reason = "ready"
    runner_obj.ready = True
    runner_obj._runner_state = (
        RunnerState.EXECUTION_ENABLED if expire_boot_grace else RunnerState.STARTING
    )
    runner_obj._startup_timestamp = (
        time.time() - 16.0 if expire_boot_grace else time.time()
    )
    runner_obj._emit_runner_eval_decision = Mock()
    runner_obj._health_watchdog = lambda: None
    runner_obj._should_log_throttled = lambda *_args, **_kwargs: False
    runner_obj._strategy_evaluation_allowed = lambda _symbol, trace_id=None: True
    runner_obj._same_bar_eval_reason = lambda **_kwargs: "intrabar_allowed"
    runner_obj.validate_market_depth = lambda: True
    runner_obj._update_symbol_readiness = lambda _symbol: SymbolState.READY
    runner_obj._ensure_symbol_vwap_state = lambda _symbol, _now: {
        "cum_vol": 0.0,
        "cum_pv": 0.0,
    }
    runner_obj._underlying_context_from_strategy_manager = lambda: {
        "direction_bias": "CE",
        "underlying_direction_bias": "CE",
        "spot_fresh": True,
        "fut_fresh": True,
        "context_fresh": True,
        "underlying_direction_confidence": 0.9,
    }
    runner_obj._symbol_live_entry_ready = lambda _symbol, signal=None, trace_id=None: (
        bool(risk_manager.validate()),
        "ready",
        {},
    )
    runner_obj._compute_regime_snapshot = lambda _symbol: MarketRegime.RANGE
    runner_obj.detect_market_regime = lambda _symbol: "normal"
    runner_obj._strategy_slots_available = lambda: True
    runner_obj._order_manager_kill_switch_status_for_entry = lambda: (False, {})
    runner_obj._schedule_signal_preparation = lambda signal, _price, _now, _trace_id: (
        order_manager.submit(signal) or True,
        "scheduled",
    )
    return runner_obj, strategy_manager, risk_manager, order_manager, selected_ce


def test_nifty_underlying_reaches_strategy_manager(monkeypatch):
    runner_obj, strategy_manager, risk_manager, order_manager, selected_ce = (
        _build_phase9_runner(monkeypatch)
    )

    runner_obj._on_tick(
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "last_price": 24000.0,
            "timestamp": time.time(),
            "trace_id": "underlying-e2e",
            "source": "ws",
        },
    )

    assert (
        runner_obj._entry_evaluation_route("NSE:NIFTY")
        == EntryEvaluationRoute.UNDERLYING
    )
    strategy_manager.generate_signal.assert_called_once()
    risk_manager.validate.assert_called_once()
    order_manager.submit.assert_called_once()
    assert order_manager.submit.call_args.args[0].symbol == selected_ce


def test_underlying_generated_candidate_activation_failure_blocks_after_signal(
    monkeypatch,
):
    runner_obj, strategy_manager, risk_manager, order_manager, _selected_ce = (
        _build_phase9_runner(monkeypatch, current_generation_ready=False)
    )

    runner_obj._on_tick(
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "last_price": 24000.0,
            "timestamp": time.time(),
            "trace_id": "underlying-negative",
            "source": "ws",
        },
    )

    strategy_manager.generate_signal.assert_called_once()
    risk_manager.validate.assert_not_called()
    order_manager.submit.assert_not_called()
    assert any(
        call.kwargs.get("reason") == "candidate_activation_pending"
        for call in runner_obj._emit_runner_eval_decision.call_args_list
    )


def test_underlying_does_not_evaluate_during_boot_grace(monkeypatch):
    runner_obj, strategy_manager, risk_manager, order_manager, _ = (
        _build_phase9_runner(monkeypatch, expire_boot_grace=False)
    )

    runner_obj._on_tick(
        "NSE:NIFTY",
        {
            "symbol": "NSE:NIFTY",
            "last_price": 24000.0,
            "timestamp": time.time(),
            "trace_id": "boot-grace",
            "source": "ws",
        },
    )

    strategy_manager.generate_signal.assert_not_called()
    risk_manager.validate.assert_not_called()
    order_manager.submit.assert_not_called()


def test_underlying_does_not_require_option_subscription_activation():
    r = runner()
    r._live_symbol_activation = Mock(
        side_effect=AssertionError("underlying must not be activation gated")
    )

    assert r._entry_evaluation_route("NSE:NIFTY") == EntryEvaluationRoute.UNDERLYING
    r._live_symbol_activation.assert_not_called()


def test_non_selected_option_context_does_not_enter_phase9_entry():
    assert (
        runner()._entry_evaluation_route("NFO:NIFTY26JUN24100CE")
        == EntryEvaluationRoute.CONTEXT_ONLY
    )


def test_selected_option_enters_phase9_when_fully_active():
    assert (
        runner()._entry_evaluation_route("NFO:NIFTY26JUN24000CE")
        == EntryEvaluationRoute.OPTION_CANDIDATE
    )


def test_open_position_routes_to_management_not_new_entry():
    r = runner()
    r._position_manager = SimpleNamespace(
        has_open_position=lambda s: s.endswith("24100CE")
    )
    strategy_manager = Mock()
    order_manager = Mock()

    assert (
        r._entry_evaluation_route("NFO:NIFTY26JUN24100CE")
        == EntryEvaluationRoute.POSITION_MANAGEMENT
    )
    strategy_manager.generate_signal.assert_not_called()
    order_manager.submit.assert_not_called()


def test_dynamic_selected_option_datahub_delivery_uses_real_subscription_path():
    r = runner()
    datahub = _DataHub()
    selected_ce = "NFO:NIFTY26JUN24000CE"
    r._data_hub = datahub
    r._active_symbols = {selected_ce}
    # Intentionally do not pre-create _datahub_registered_symbols; _subscribe_symbol
    # must initialize and populate it through the real registration path.
    if hasattr(r, "_datahub_registered_symbols"):
        delattr(r, "_datahub_registered_symbols")

    r._subscribe_symbol(selected_ce)

    assert datahub.subscribed and datahub.subscribed[0][0] == selected_ce
    assert selected_ce in r._datahub_registered_symbols
    assert r._runner_delivery_ready_for_symbol(selected_ce) is True


def test_runner_delivery_requires_callback_registration():
    r = runner()
    selected_ce = "NFO:NIFTY26JUN24000CE"
    r._data_hub = _DataHub()
    r._active_symbols = {selected_ce}
    r._datahub_registered_symbols = set()

    assert r._runner_delivery_ready_for_symbol(selected_ce) is False
