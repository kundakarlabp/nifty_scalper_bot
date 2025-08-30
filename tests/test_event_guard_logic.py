from datetime import timedelta

import pytest

from src.utils.events import load_calendar


def _apply_guard(cal, now, plan):
    active = cal.active(now)
    if active:
        block = any(ev.block_trading for ev in active)
        widen = max((ev.post_widen_spread_pct for ev in active), default=0.0)
        plan.setdefault("reasons", []).append(
            f"event_guard:{','.join(ev.name for ev in active)}"
        )
        plan["event_guard"] = {
            "active": True,
            "names": [ev.name for ev in active],
            "post_widen_spread_pct": round(widen, 3),
            "block": block,
        }
        if block:
            plan["has_signal"] = False
            plan["reason_block"] = "event_guard"
        else:
            plan["_event_post_widen"] = float(widen)
    return plan


def test_blocking_guard():
    cal = load_calendar("config/events.yaml")
    ev = next(e for e in cal.events if e.name == "RBI Policy")
    now = ev.start - timedelta(minutes=5)
    active = cal.active(now)
    assert any(e.name == "RBI Policy" for e in active)
    plan = {"has_signal": True}
    _apply_guard(cal, now, plan)
    assert plan["has_signal"] is False
    assert plan["reason_block"] == "event_guard"


def test_widening_guard():
    cal = load_calendar("config/events.yaml")
    ev = next(e for e in cal.events if e.name == "US CPI")
    now = ev.end + timedelta(minutes=1)
    active = cal.active(now)
    assert any(e.name == "US CPI" for e in active)
    plan = {"has_signal": True}
    _apply_guard(cal, now, plan)
    assert plan["has_signal"] is True
    assert plan["_event_post_widen"] == pytest.approx(ev.post_widen_spread_pct)
