"""Final execution-snapshot freshness gates (P1)."""

from __future__ import annotations

import time
from types import MethodType, SimpleNamespace

from nifty_scalper_bot.execution.order_manager_core import OrderManager, TradePlan

SYMBOL = "NFO:NIFTY2680424400CE"


def _stub(quote: dict) -> SimpleNamespace:
    stub = SimpleNamespace(
        is_live_mode=lambda: False,
        _lot_size_for_symbol=lambda symbol: 75,
        _get_latest_quote_safe=lambda symbol: quote,
    )
    for name in ("_extract_quote_diagnostics", "_validate_trade_plan"):
        setattr(stub, name, MethodType(getattr(OrderManager, name), stub))
    return stub


def _plan(**overrides) -> TradePlan:
    payload = dict(
        symbol=SYMBOL,
        side="BUY",
        quantity=75,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        resolved_lot_size=75,
        requested_lots=1,
        signal_id="sig-1",
        trade_provenance={"decision_ts": time.time()},
    )
    payload.update(overrides)
    return TradePlan(**payload)


def _quote(bid: float = 99.5, ask: float = 100.5) -> dict:
    return {
        "bid": bid,
        "ask": ask,
        "last_price": (bid + ask) / 2,
        "timestamp": time.time(),
        "bid_qty": 5000,
        "ask_qty": 5000,
    }


def test_fresh_plan_is_allowed() -> None:
    stub = _stub(_quote())
    result = stub._validate_trade_plan(
        _plan(max_signal_age_seconds=15.0, max_entry_drift_pct=2.0)
    )
    assert result.allowed is True


def test_stale_signal_is_rejected() -> None:
    stub = _stub(_quote())
    result = stub._validate_trade_plan(
        _plan(
            max_signal_age_seconds=5.0,
            trade_provenance={"decision_ts": time.time() - 60.0},
        )
    )
    assert result.allowed is False
    assert result.reason == "signal_stale"


def test_adverse_entry_price_drift_is_rejected() -> None:
    stub = _stub(_quote(bid=104.0, ask=105.0))
    result = stub._validate_trade_plan(_plan(max_entry_drift_pct=2.0))
    assert result.allowed is False
    assert result.reason == "entry_price_drift"
    assert result.details["drift_pct"] > 2.0


def test_favourable_drift_is_allowed() -> None:
    stub = _stub(_quote(bid=94.0, ask=95.0))
    result = stub._validate_trade_plan(_plan(max_entry_drift_pct=2.0))
    assert result.allowed is True


def test_sell_side_drift_uses_bid() -> None:
    stub = _stub(_quote(bid=95.0, ask=96.0))
    result = stub._validate_trade_plan(
        _plan(side="SELL", intended_position_side="SHORT", max_entry_drift_pct=2.0)
    )
    assert result.allowed is False
    assert result.reason == "entry_price_drift"


def test_limits_disabled_by_default() -> None:
    stub = _stub(_quote(bid=200.0, ask=205.0))
    result = stub._validate_trade_plan(
        _plan(trade_provenance={"decision_ts": time.time() - 3600.0})
    )
    assert result.allowed is True
