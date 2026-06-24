"""Root fix: a healthy broker returning a low/zero available balance must still
record margin freshness, otherwise balance_stale=True / margin_age_s=None blocks
live orders forever (observed: BROKER_HEALTH_LIVE_ORDERS_BLOCKED all session)."""
from __future__ import annotations

import math
import time
from contextlib import suppress
from typing import Any

from nifty_scalper_bot.execution.order_manager import OrderManager


def _bare_om() -> OrderManager:
    om = OrderManager.__new__(OrderManager)
    om._last_margin_success_ts = None
    om._last_margin_refresh_ts = None
    om._last_margin_available_balance = None
    om._last_margin_balance_source = None
    om._last_margin_error_type = None
    om._last_margin_error = None
    om._margin_circuit_open = False
    om._margin_circuit_until_ts = None
    om._data_hub = None
    om._market_data = None
    om._risk_manager = None
    return om


class _MDMZero:
    def refresh_margin_snapshot(self) -> None:
        return None
    def get_available_balance(self) -> float:
        return 0.0  # healthy broker, genuinely zero/low available


class _MDMPositive:
    def refresh_margin_snapshot(self) -> None:
        return None
    def get_available_balance(self) -> float:
        return 16248.60


async def test_zero_balance_records_freshness() -> None:
    om = _bare_om()
    om._data_hub = _MDMZero()
    available, source = om._resolve_available_margin_raw()
    assert source == "mdm"
    assert available == 0.0
    # crucial: freshness timestamp set -> margin_age_s will be a real number,
    # not None -> balance_stale resolves to False
    assert om._last_margin_success_ts is not None


async def test_positive_balance_still_records() -> None:
    om = _bare_om()
    om._data_hub = _MDMPositive()
    available, source = om._resolve_available_margin_raw()
    assert source == "mdm" and available == 16248.60
    assert om._last_margin_success_ts is not None
