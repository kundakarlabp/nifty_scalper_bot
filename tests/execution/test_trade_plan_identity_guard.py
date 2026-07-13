from __future__ import annotations

from typing import Any

import nifty_scalper_bot.execution  # noqa: F401 - applies runtime safety patches
from nifty_scalper_bot.execution import order_manager_core as core


GOOD_SYMBOL = "NFO:NIFTY2670724250PE"
BAD_SYMBOL = "NFO:NIFTY2670724300PE"


class _Manager:
    def __init__(self, quote: dict[str, Any]) -> None:
        self.quote = dict(quote)

    def _lot_size_for_symbol(self, _symbol: str) -> int:
        return 65

    def _get_latest_quote_safe(self, _symbol: str) -> dict[str, Any]:
        return dict(self.quote)

    def _extract_quote_diagnostics(self, quote: dict[str, Any]) -> dict[str, Any]:
        return {
            "age_ms": quote.get("age_ms", 10),
            "bid": quote.get("bid", 88.0),
            "ask": quote.get("ask", 88.5),
            "spread_pct": quote.get("spread_pct", 0.5),
            "bid_qty": quote.get("bid_qty", 500),
            "ask_qty": quote.get("ask_qty", 500),
            "depth_qty": quote.get("depth_qty", 500),
            "ltp": quote.get("ltp", 88.25),
        }

    def _order_live_execution_enabled(self) -> bool:
        return True


def _plan(symbol: str = GOOD_SYMBOL) -> core.TradePlan:
    return core.TradePlan(
        symbol=symbol,
        side="BUY",
        quantity=65,
        entry_price=88.25,
        stop_loss=83.0,
        take_profit=96.0,
        trace_id="trace-identity",
        signal_id="sig-identity",
        trade_lifecycle_id="life-identity",
        client_order_id="client-identity",
        instrument_token=12345,
        requested_lots=1,
        resolved_lot_size=65,
    )


def test_live_trade_plan_rejects_quote_symbol_mismatch() -> None:
    manager = _Manager({"symbol": BAD_SYMBOL})

    result = core.OrderManager._validate_trade_plan(manager, _plan())

    assert result.allowed is False
    assert result.reason == "quote_symbol_identity_mismatch"
    assert result.details["symbol"] == GOOD_SYMBOL
    assert result.details["quote_symbols"] == [BAD_SYMBOL]


def test_live_trade_plan_rejects_missing_quote_identity() -> None:
    manager = _Manager({"bid": 88.0, "ask": 88.5, "ltp": 88.25})

    result = core.OrderManager._validate_trade_plan(manager, _plan())

    assert result.allowed is False
    assert result.reason == "quote_symbol_identity_missing"


def test_live_trade_plan_allows_matching_quote_identity() -> None:
    manager = _Manager({"tradingsymbol": "NIFTY2670724250PE"})

    result = core.OrderManager._validate_trade_plan(manager, _plan())

    assert result.allowed is True

def test_live_trade_plan_requires_lifecycle_identity_before_broker_attempt(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    m = _Manager({"tradingsymbol": "NIFTY2670724250PE"})
    m.is_live_mode = lambda: True  # type: ignore[attr-defined]
    plan = core.TradePlan(
        symbol=GOOD_SYMBOL,
        side="BUY",
        quantity=65,
        entry_price=88.25,
        stop_loss=83.0,
        take_profit=96.0,
        signal_id="sig-1",
        instrument_token=123,
        requested_lots=1,
        resolved_lot_size=65,
    )

    result = core.OrderManager._validate_trade_plan(m, plan)

    assert result.allowed is False
    assert result.reason == "trade_lifecycle_id_missing"
    assert result.details["broker_attempted"] is False
    assert result.details["retryable"] is False


def test_live_trade_plan_rejects_invalid_requested_lot_quantity(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    m = _Manager({"tradingsymbol": "NIFTY2670724250PE"})
    m.is_live_mode = lambda: True  # type: ignore[attr-defined]
    plan = core.TradePlan(
        symbol=GOOD_SYMBOL,
        side="BUY",
        quantity=40,
        entry_price=88.25,
        stop_loss=83.0,
        take_profit=96.0,
        signal_id="sig-1",
        trade_lifecycle_id="life-1",
        client_order_id="client-1",
        instrument_token=123,
        requested_lots=1,
        resolved_lot_size=65,
    )

    result = core.OrderManager._validate_trade_plan(m, plan)

    assert result.allowed is False
    assert result.reason == "invalid_entry_lot_quantity"
    assert result.details["required_value"] == 65
    assert result.details["actual_value"] == 40


def test_order_details_persists_lifecycle_contract_identity() -> None:
    order = core.OrderDetails(
        order_id="OID1",
        symbol=GOOD_SYMBOL,
        side="BUY",
        quantity=65,
        order_type=core.OrderType.LIMIT,
        status=core.OrderStatus.SUBMITTED,
        client_order_id="client-1",
        intent="ENTRY",
        trade_lifecycle_id="life-1",
        linked_entry_order_id="entry-1",
        bracket_id="bracket-1",
        basket_version="7",
        instrument_token=123,
        contract_expiry="2026-07-30",
        exchange_order_id="ex-1",
        signal_id="sig-1",
    )

    payload = core.OrderManager._serialize(m := object.__new__(core.OrderManager), order)
    restored = core.OrderManager._order_from_dict(m, payload)

    assert restored.trade_lifecycle_id == "life-1"
    assert restored.client_order_id == "client-1"
    assert restored.bracket_id == "bracket-1"
    assert restored.instrument_token == 123
    assert restored.contract_expiry == "2026-07-30"
