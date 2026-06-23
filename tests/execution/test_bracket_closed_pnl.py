"""BRACKET_CLOSED must carry realized P&L so terminals can show daily P&L."""
from __future__ import annotations

import logging
from typing import Any

from nifty_scalper_bot.execution.bracket_manager import BracketManager


class _OM:
    def __init__(self) -> None:
        self._broker = None
    def place_order(self, **k: Any) -> str:
        return "x"
    def cancel_order(self, _o: str) -> bool:
        return True


def _mgr_with_bracket():
    mgr = BracketManager(order_manager=_OM())
    mgr.register_virtual_bracket(
        order_id="e1", symbol="NFO:NIFTY2660923100CE", side="BUY",
        qty=65, price=137.25, sl=130.0, tp=150.0,
    )
    return mgr, next(iter(mgr._brackets.values()))


async def test_bracket_closed_logs_realized_pnl(caplog) -> None:
    mgr, b = _mgr_with_bracket()
    b.entry_fill_price = 137.25
    with caplog.at_level(logging.INFO):
        mgr._close_bracket(b, close_source="broker_fill", exit_price=140.0)
    line = next((r.getMessage() for r in caplog.records if "BRACKET_CLOSED" in r.getMessage()), "")
    # (140.0 - 137.25) * 65 = 178.75
    assert "pnl=178.75" in line
    assert "entry=137.25" in line and "exit=140.0" in line


async def test_bracket_closed_pnl_negative_on_loss(caplog) -> None:
    mgr, b = _mgr_with_bracket()
    b.entry_fill_price = 137.25
    with caplog.at_level(logging.INFO):
        mgr._close_bracket(b, close_source="reconciled_flat", exit_price=133.55)
    line = next((r.getMessage() for r in caplog.records if "BRACKET_CLOSED" in r.getMessage()), "")
    # (133.55 - 137.25) * 65 = -240.5
    assert "pnl=-240.5" in line
