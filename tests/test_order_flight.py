"""Tests for StrategyRunner order-flight safeguards."""

from __future__ import annotations

import threading

from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_order_in_flight_prevents_duplicate_symbol_submission() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._orders_in_flight = {}
    runner._orders_lock = threading.RLock()
    runner._order_timeout_sec = 10

    runner._mark_order_in_flight("NIFTY26MAR25000CE", "NIFTY")

    assert runner._is_order_in_flight("NIFTY26MAR25000CE", "NIFTY") is True
    assert runner._is_order_in_flight("NIFTY26MAR25050CE", "NIFTY") is True

    runner._clear_order_in_flight("NIFTY26MAR25000CE")

    assert runner._is_order_in_flight("NIFTY26MAR25000CE", "NIFTY") is False
