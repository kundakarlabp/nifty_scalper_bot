"""Regression for the real 23750CE incident:
BUY 223.90 -> TP latched -> SELL LIMIT 224.35 (ltp*0.98) cancelled,
then filled 222.30 (-₹104).

Two defects: (1) TP exit priced at ltp*0.98 (a 2% giveaway that turns a win into a
loss); (2) a sub-1.0 reward:risk bracket (RR 0.42) activated silently.
"""
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


def _mgr_with_bracket(monkeypatch, tmp_path, order_id: str = "e1"):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    mgr = BracketManager(order_manager=_OM())
    mgr.register_virtual_bracket(
        order_id=order_id, symbol="NFO:NIFTY26JUN23750CE", side="BUY",
        qty=65, price=223.90, sl=214.61, tp=227.76,
    )
    return mgr, next(iter(mgr._brackets.values()))


def test_tp_exit_prices_off_bid_not_ltp_concession(monkeypatch, tmp_path) -> None:
    mgr, b = _mgr_with_bracket(monkeypatch, tmp_path, "e1-price")
    b.last_ltp = 228.90
    mgr._extract_exit_quote = lambda s: (227.80, 228.00, 228.90)  # bid, ask, ltp
    ot, price, meta = mgr._price_exit_order(
        bracket=b, symbol=b.symbol, side="SELL", reason="HARD_TP_BREACH",
        preferred_order_type="LIMIT", qty=65,
    )
    assert ot == "LIMIT"
    assert meta["mode"] == "PROFIT_LIMIT"
    # must be well above entry (locking profit), NOT ltp*0.98 = 224.32
    assert price > 223.90
    assert abs(price - 228.90 * 0.98) > 1.0


def test_protective_exit_still_market(monkeypatch, tmp_path) -> None:
    mgr, b = _mgr_with_bracket(monkeypatch, tmp_path, "e1-sl")
    b.last_ltp = 210.0
    ot, price, meta = mgr._price_exit_order(
        bracket=b, symbol=b.symbol, side="SELL", reason="HARD_SL_BREACH",
        preferred_order_type="MARKET", qty=65,
    )
    assert ot == "MARKET"  # protective exits unchanged (flatten fast)


def test_post_fill_rr_below_floor_logged(caplog, monkeypatch, tmp_path) -> None:
    mgr, b = _mgr_with_bracket(monkeypatch, tmp_path, "e1-rr")
    with caplog.at_level(logging.CRITICAL):
        mgr.confirm_entry_fill("e1-rr", 223.90)  # RR = 3.86/9.29 = 0.42
    assert any("BRACKET_RR_BELOW_FLOOR" in r.getMessage() for r in caplog.records)


def test_good_rr_not_flagged(caplog, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    mgr = BracketManager(order_manager=_OM())
    mgr.register_virtual_bracket(
        order_id="e2", symbol="NFO:NIFTY26JUN23750CE", side="BUY",
        qty=65, price=100.0, sl=95.0, tp=110.0,  # risk 5, reward 10, RR 2.0
    )
    with caplog.at_level(logging.CRITICAL):
        mgr.confirm_entry_fill("e2", 100.0)
    assert not any("BRACKET_RR_BELOW_FLOOR" in r.getMessage() for r in caplog.records)
