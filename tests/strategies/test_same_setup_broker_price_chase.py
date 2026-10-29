"""Regression for the October 29 same-setup price chase."""
from types import SimpleNamespace
from nifty_scalper_bot.execution import order_manager as mod


def plan(price, signal="same", intent="ENTRY"):
    return SimpleNamespace(
        signal_id=signal, symbol="NFO:NIFTYCE", side="BUY",
        entry_price=price, intent=intent, max_entry_drift_pct=2.0,
    )


def test_broker_attempted_timeout_blocks_later_chasing(monkeypatch):
    manager = SimpleNamespace(is_live_mode=lambda: True)
    calls = []
    def submit(_, order):
        calls.append(order.entry_price)
        return SimpleNamespace(accepted=False, broker_attempted=True)
    monkeypatch.setattr(mod, "_original_submit_trade_plan_result", submit)
    mod._submit_trade_plan_with_setup_price_guard(manager, plan(100.0))
    result = mod._submit_trade_plan_with_setup_price_guard(manager, plan(106.0))
    assert calls == [100.0]
    assert result.reason == "same_setup_adverse_reprice"
    assert not result.broker_attempted


def test_fresh_setup_and_exit_are_unaffected(monkeypatch):
    manager = SimpleNamespace(is_live_mode=lambda: True)
    calls = []
    def submit(_, order):
        calls.append(order.entry_price)
        return SimpleNamespace(accepted=True, broker_attempted=True)
    monkeypatch.setattr(mod, "_original_submit_trade_plan_result", submit)
    for order in (plan(100.0), plan(106.0, signal="new"), plan(110.0, intent="EXIT")):
        assert mod._submit_trade_plan_with_setup_price_guard(manager, order).accepted
    assert calls == [100.0, 106.0, 110.0]


def test_prebroker_reject_does_not_anchor(monkeypatch):
    manager = SimpleNamespace(is_live_mode=lambda: True)
    calls = []
    def submit(_, order):
        calls.append(order.entry_price)
        return SimpleNamespace(accepted=False, broker_attempted=False)
    monkeypatch.setattr(mod, "_original_submit_trade_plan_result", submit)
    mod._submit_trade_plan_with_setup_price_guard(manager, plan(100.0))
    mod._submit_trade_plan_with_setup_price_guard(manager, plan(106.0))
    assert calls == [100.0, 106.0]
