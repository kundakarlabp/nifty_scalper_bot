"""Same-setup broker-attempt repricing regression (29 October 2026)."""

from types import SimpleNamespace

from nifty_scalper_bot.execution.entry_recovery import _recover_submit


def _plan(*, signal_id="same", price=100.0, intent="ENTRY"):
    return SimpleNamespace(
        signal_id=signal_id, symbol="NFO:NIFTY26N0322750CE",
        side="BUY", intent=intent, entry_price=price,
        max_entry_drift_pct=2.0,
    )


def test_first_real_broker_attempt_anchors_price_and_later_chase_is_blocked():
    manager = SimpleNamespace(is_live_mode=lambda: True)
    calls = []

    def submit(_manager, plan):
        calls.append(plan.entry_price)
        return SimpleNamespace(
            accepted=True, broker_attempted=True,
            order_id="oid", reason="submitted", details={},
        )

    _recover_submit(submit, manager, _plan(price=100.0))
    blocked = _recover_submit(submit, manager, _plan(price=106.0))
    assert len(calls) == 1
    assert not blocked.accepted
    assert not blocked.broker_attempted
    assert blocked.reason == "same_setup_adverse_reprice"


def test_new_signal_does_not_inherit_previous_setup_price():
    manager = SimpleNamespace(is_live_mode=lambda: True)
    calls = []

    def submit(_manager, plan):
        calls.append(plan.entry_price)
        return SimpleNamespace(
            accepted=True, broker_attempted=True,
            order_id="oid", reason="submitted", details={},
        )

    _recover_submit(submit, manager, _plan(signal_id="first", price=100.0))
    result = _recover_submit(submit, manager, _plan(signal_id="new", price=106.0))
    assert result.accepted
    assert calls == [100.0, 106.0]


def test_prebroker_rejection_does_not_create_false_price_anchor():
    manager = SimpleNamespace(is_live_mode=lambda: True)
    calls = []

    def submit(_manager, plan):
        calls.append(plan.entry_price)
        return SimpleNamespace(
            accepted=False, broker_attempted=False,
            order_id=None, reason="quote_unavailable", details={},
        )

    _recover_submit(submit, manager, _plan(price=100.0))
    _recover_submit(submit, manager, _plan(price=106.0))
    assert calls == [100.0, 106.0]


def test_protective_exit_is_not_blocked():
    manager = SimpleNamespace(is_live_mode=lambda: True)
    calls = []

    def submit(_manager, plan):
        calls.append(plan.entry_price)
        return SimpleNamespace(
            accepted=True, broker_attempted=True,
            order_id="oid", reason="submitted", details={},
        )

    _recover_submit(submit, manager, _plan(price=100.0))
    result = _recover_submit(submit, manager, _plan(price=106.0, intent="EXIT"))
    assert result.accepted
    assert calls == [100.0, 106.0]
