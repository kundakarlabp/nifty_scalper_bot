from __future__ import annotations

from types import SimpleNamespace
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
