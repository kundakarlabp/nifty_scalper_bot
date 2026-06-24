"""Adaptive trailing SL ratchet (the production hardened path) must notify Telegram."""
from __future__ import annotations

from typing import Any

from nifty_scalper_bot.execution.bracket_manager import BracketManager


class _OM:
    def __init__(self) -> None:
        self._broker = None
    def place_order(self, **k: Any) -> str:
        return "x"
    def cancel_order(self, _o: str) -> bool:
        return True


def _mgr_with_bracket(captured: list):
    mgr = BracketManager(order_manager=_OM())
    mgr._notify_event = lambda name, meta=None: captured.append((name, meta))  # type: ignore
    mgr.register_virtual_bracket(
        order_id="e1", symbol="NFO:NIFTY26JUN23950CE", side="BUY",
        qty=65, price=216.35, sl=210.0, tp=230.0,
    )
    b = mgr._brackets["e1"]
    b.last_ltp = 220.0
    mgr._trail_notify_sl.clear()
    mgr._trail_notify_at.clear()
    captured.clear()
    return mgr, b


async def test_trail_ratchet_emits_telegram_event() -> None:
    cap: list = []
    mgr, b = _mgr_with_bracket(cap)
    ok = mgr._virtual_modify_sl("vsl_e1", 215.0)  # BUY: 210 < 215 < ltp 220
    assert ok is True
    assert b.sl_trigger_price == 215.0
    trail = [c for c in cap if c[0] == "TRAILING_SL_UPDATED"]
    assert trail, "trail ratchet must emit TRAILING_SL_UPDATED"
    assert trail[0][1]["reason"] == "adaptive_atr_trail"
    assert trail[0][1]["new_sl"] == 215.0


async def test_trail_ratchet_monotonic_rejects_backward_move() -> None:
    cap: list = []
    mgr, b = _mgr_with_bracket(cap)
    # BUY SL can only move UP; a lower proposed SL is rejected (no notify)
    ok = mgr._virtual_modify_sl("vsl_e1", 205.0)
    assert ok is False
    assert b.sl_trigger_price == 210.0
    assert not [c for c in cap if c[0] == "TRAILING_SL_UPDATED"]
