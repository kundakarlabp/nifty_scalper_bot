"""Broker-free regression of durable VWAP same-setup entry price discipline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution import runtime_order_manager as runtime


class PositionState:
    def __init__(self):
        self.state = {"trading_date": "2026-10-29"}
        self.fail_save = False

    def get_risk_circuit_state(self):
        return dict(self.state)

    def _trading_date_ist(self):
        return "2026-10-29"

    def persist_risk_circuit_state(self, **changes):
        if self.fail_save:
            raise OSError("state persistence unavailable")
        self.state.update(changes, trading_date=self._trading_date_ist())


def _plan(
    price: float,
    *,
    setup: str = "same",
    signal: str = "signal",
    intent: str = "ENTRY",
    side: str = "BUY",
    strategy: str = "VWAPPro",
):
    return SimpleNamespace(
        symbol="NFO:NIFTY26N0322750CE",
        entry_price=price,
        intent=intent,
        side=side,
        strategy_name=strategy,
        signal_id=signal,
        trade_provenance={
            "setup_name": "vwap_continuation",
            "setup_contract": {"setup_id": setup},
        },
    )


def _manager(state: PositionState):
    return SimpleNamespace(
        _positions=state,
        _logger=SimpleNamespace(warning=lambda *a, **k: None),
        is_live_mode=lambda: True,
    )


def _broker(monkeypatch, *, attempted: bool = True, accepted: bool = False):
    calls: list[float] = []

    def fake_submit(_core, _manager, plan):
        calls.append(plan.entry_price)
        return SimpleNamespace(accepted=accepted, broker_attempted=attempted)

    monkeypatch.setattr(runtime, "_recover_submit", fake_submit)
    return calls


def test_uncertain_first_attempt_blocks_53pct_reprice_even_after_restart(monkeypatch):
    store = PositionState()
    calls = _broker(monkeypatch, attempted=True)
    first = runtime._submit_with_setup_price_guard(_manager(store), _plan(155.45))
    assert first.broker_attempted
    assert len(store.state["entry_attempt_price_anchors"]) == 1
    # A completely new owner instance restores the persisted setup price.
    second = runtime._submit_with_setup_price_guard(_manager(store), _plan(163.70))
    assert not second.accepted
    assert not second.broker_attempted
    assert second.reason == "same_setup_adverse_reprice"
    assert second.details["drift_pct"] > 5
    assert calls == [155.45]


def test_prebroker_rejection_does_not_anchor_or_block_next_attempt(monkeypatch):
    store = PositionState()
    calls = _broker(monkeypatch, attempted=False)
    first = runtime._submit_with_setup_price_guard(_manager(store), _plan(155.45))
    assert not first.broker_attempted
    assert not store.state["entry_attempt_price_anchors"]
    next_attempt = runtime._submit_with_setup_price_guard(
        _manager(store), _plan(163.70)
    )
    assert not next_attempt.broker_attempted
    assert calls == [155.45, 163.70]


def test_new_setup_and_protective_exit_unchanged(monkeypatch):
    store = PositionState()
    calls = _broker(monkeypatch, attempted=True, accepted=True)
    owner = _manager(store)
    assert runtime._submit_with_setup_price_guard(owner, _plan(100.0)).accepted
    assert runtime._submit_with_setup_price_guard(
        owner, _plan(120.0, setup="new", signal="new")
    ).accepted
    assert runtime._submit_with_setup_price_guard(
        owner, _plan(150.0, intent="EXIT", side="SELL")
    ).accepted
    assert calls == [100.0, 120.0, 150.0]


def test_persist_failure_never_places_live_order(monkeypatch):
    store = PositionState()
    store.fail_save = True
    calls = _broker(monkeypatch, attempted=True)
    result = runtime._submit_with_setup_price_guard(_manager(store), _plan(100.0))
    assert result.reason == "entry_price_guard_persistence_unavailable"
    assert not result.broker_attempted
    assert calls == []


def test_anchor_expires_after_15_minutes(monkeypatch):
    store = PositionState()
    now = [1_800_000_000.0]
    monkeypatch.setattr(runtime.time, "time", lambda: now[0])
    calls = _broker(monkeypatch, attempted=True)
    owner = _manager(store)
    runtime._submit_with_setup_price_guard(owner, _plan(100.0))
    now[0] += 901
    assert runtime._submit_with_setup_price_guard(owner, _plan(110.0)).broker_attempted
    assert calls == [100.0, 110.0]


def test_exception_after_reservation_keeps_guard_for_uncertain_broker(monkeypatch):
    store = PositionState()

    def interrupted(*args):
        raise TimeoutError("broker response ambiguous")

    monkeypatch.setattr(runtime, "_recover_submit", interrupted)
    with pytest.raises(TimeoutError):
        runtime._submit_with_setup_price_guard(_manager(store), _plan(100.0))
    calls = _broker(monkeypatch, attempted=True)
    result = runtime._submit_with_setup_price_guard(_manager(store), _plan(106.0))
    assert result.reason == "same_setup_adverse_reprice"
    assert calls == []


def test_shadow_mode_does_not_change_replay_behavior(monkeypatch):
    store = PositionState()
    calls = _broker(monkeypatch, attempted=False)
    shadow = _manager(store)
    shadow.is_live_mode = lambda: False
    runtime._submit_with_setup_price_guard(shadow, _plan(100.0))
    runtime._submit_with_setup_price_guard(shadow, _plan(106.0))
    assert calls == [100.0, 106.0]
    assert "entry_attempt_price_anchors" not in store.state
