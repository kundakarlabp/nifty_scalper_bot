"""Regression: ATM rotation must not strand a healthy WS pair's evaluator."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from nifty_scalper_bot.core.strategy_runner_dynamic_universe_safety import (
    _repair_selected_pair_evaluation,
)

CE = "NFO:NIFTY26NOV22800CE"
PE = "NFO:NIFTY26NOV22800PE"


class _Log:
    def __init__(self) -> None:
        self.events: list[str] = []

    def warning(self, message: str, *_args: object, **_kwargs: object) -> None:
        self.events.append(message)

    def critical(self, message: str, *_args: object, **_kwargs: object) -> None:
        self.events.append(message)

    def info(self, message: str, *_args: object, **_kwargs: object) -> None:
        self.events.append(message)


def _runner(*, ws_age: float = 0.1, drain_ok: bool = True) -> SimpleNamespace:
    callbacks: set[str] = set()
    subscriptions: list[str] = []

    def subscribe(symbol: str, callback: object, **_kwargs: object) -> None:
        assert callback is not None
        callbacks.add(symbol)
        subscriptions.append(symbol)

    runner = SimpleNamespace(
        _active_selected_ce=CE,
        _active_selected_pe=PE,
        _running=True,
        _runtime_evaluation_ready=True,
        _runtime_live_orders_armed=True,
        _entry_eval_shutdown=False,
        _entry_eval_active=False,
        _entry_eval_drain_scheduled=False,
        _entry_eval_dispatch_stall_s=120.0,
        _selected_entry_eval_epoch_pair=(CE, PE),
        _selected_entry_eval_epoch_started_at=100.0,
        _last_selected_candidate_eval_completed_ts=90.0,
        _market_data=SimpleNamespace(time_since_last_live_ws_tick=lambda _s: ws_age),
        _data_hub=SimpleNamespace(subscribe_ticks=subscribe),
        _active_basket_token_by_symbol={CE: 1, PE: 2},
        on_datahub_tick=lambda _tick: None,
        has_datahub_subscription=lambda symbol, _token=None: symbol in callbacks,
        _eval_gate_lock=threading.Lock(),
        _pending_entry_eval_symbols=set(),
        _entry_eval_generation_by_symbol={},
        _schedule_entry_eval_drain=lambda: drain_ok,
        _logger=_Log(),
        _entry_eval_stall_disarmed=False,
    )
    runner._test_subscriptions = subscriptions
    return runner


def test_live_pair_missing_callback_is_repaired_and_reevaluation_scheduled() -> None:
    runner = _runner()

    _repair_selected_pair_evaluation(runner, now=125.0)

    assert runner._test_subscriptions == [CE, PE]
    assert runner._pending_entry_eval_symbols == {CE, PE}
    assert runner._entry_eval_generation_by_symbol == {CE: 1, PE: 1}
    assert runner._runtime_live_orders_armed is True


def test_second_failed_reevaluation_attempt_disarms_new_entries() -> None:
    runner = _runner()
    _repair_selected_pair_evaluation(runner, now=125.0)

    _repair_selected_pair_evaluation(runner, now=147.0)

    assert runner._entry_eval_stall_disarmed is True
    assert runner._runtime_live_orders_armed is False
    assert runner._runtime_readiness_reason == "strategy_evaluation_stalled"
    assert runner._pending_entry_eval_symbols == {CE, PE}


def test_selected_candidate_completion_prevents_disarm() -> None:
    runner = _runner()
    _repair_selected_pair_evaluation(runner, now=125.0)
    runner._last_selected_candidate_eval_completed_ts = 127.0

    _repair_selected_pair_evaluation(runner, now=147.0)

    assert runner._runtime_live_orders_armed is True
    assert runner._entry_eval_stall_disarmed is False


def test_stale_ws_receipts_never_authorize_rescheduling() -> None:
    runner = _runner(ws_age=20.0)

    _repair_selected_pair_evaluation(runner, now=125.0)

    assert not runner._pending_entry_eval_symbols
    assert runner._test_subscriptions == []
    assert runner._runtime_live_orders_armed is True


def test_unattached_worker_fails_closed_without_synthetic_evaluations() -> None:
    runner = _runner(drain_ok=False)

    _repair_selected_pair_evaluation(runner, now=125.0)

    assert runner._entry_eval_stall_disarmed is True
    assert runner._runtime_live_orders_armed is False
