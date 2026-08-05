from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.execution.affordability import (
    evaluate_minimum_lot_affordability,
)


class _OrderManager:
    _margin_factor = 1.0
    _margin_buffer = 0.8

    def resolve_lot_size(self, symbol: str) -> int:
        return 65


class _Hub:
    def __init__(self, balance):
        self.balance = balance

    def get_available_balance(self, force=False):
        return self.balance


def test_minimum_lot_unaffordable_uses_ask_lot_and_cash_reserve() -> None:
    result = evaluate_minimum_lot_affordability(
        symbol="NFO:NIFTY26AUG25000CE",
        quote={"bid": 171.0, "ask": 172.0, "ltp": 171.5},
        order_manager=_OrderManager(),
        data_hub=_Hub(10_000.0),
        fallback_balance=50_000.0,
    )

    assert result.determinate is True
    assert result.affordable is False
    assert result.reason == "minimum_lot_unaffordable"
    assert result.required == 11_180.0
    assert result.executable_capacity == 8_000.0
    assert result.balance_source == "data_hub"


def test_minimum_lot_affordable_when_reserved_cash_covers_premium() -> None:
    result = evaluate_minimum_lot_affordability(
        symbol="NFO:NIFTY26AUG25000CE",
        quote=SimpleNamespace(ask=172.0),
        order_manager=_OrderManager(),
        data_hub=_Hub(15_000.0),
    )

    assert result.determinate is True
    assert result.affordable is True
    assert result.required == 11_180.0
    assert result.executable_capacity == 12_000.0


def test_zero_balance_is_known_and_not_replaced_by_fallback() -> None:
    result = evaluate_minimum_lot_affordability(
        symbol="NFO:NIFTY26AUG25000CE",
        quote={"ask": 100.0},
        order_manager=_OrderManager(),
        data_hub=_Hub(0.0),
        fallback_balance=99_999.0,
    )

    assert result.determinate is True
    assert result.affordable is False
    assert result.available == 0.0
    assert result.balance_source == "data_hub"


def test_missing_ask_fails_closed_without_ltp_substitution() -> None:
    result = evaluate_minimum_lot_affordability(
        symbol="NFO:NIFTY26AUG25000CE",
        quote={"ltp": 100.0},
        order_manager=_OrderManager(),
        data_hub=_Hub(50_000.0),
    )

    assert result.determinate is False
    assert result.affordable is False
    assert result.reason == "executable_quote_unavailable"
