"""Tests for microstructure-aware enhancements in the order manager."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest

import nifty_scalper_bot.config.settings as app_settings
from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.utils.rate_limiter import RateLimiter


@dataclass(slots=True)
class _StubBroker:
    """Minimal broker stub capturing placed orders for assertions."""

    counter: int = 0
    last_payload: Mapping[str, Any] | None = None

    def place_order(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Capture payload and return a mock order identifier."""

        self.counter += 1
        self.last_payload = dict(payload)
        return {"order_id": f"TEST-{self.counter}", "status": "open"}


class _StubMarketData:
    """Test double returning canned quotes with depth information."""

    def __init__(self) -> None:
        self._quote: Mapping[str, Any] = {}

    def set_quote(self, payload: Mapping[str, Any]) -> None:
        """Set the quote snapshot returned by :meth:`get_quote`."""

        self._quote = payload

    def get_quote(self, symbol: str) -> Mapping[str, Any]:  # noqa: D401 - stub
        """Return the configured quote snapshot."""

        del symbol
        return self._quote


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    """Provide a configured rate limiter for order placement."""

    limiter = RateLimiter()
    limiter.configure_bucket("orders", capacity=10, refill_rate_per_sec=10)
    return limiter


@pytest.fixture()
def position_manager(tmp_path: Path) -> PositionManager:
    """Create a temporary position manager for tests."""

    state_dir = tmp_path / "positions"
    state_dir.mkdir(parents=True, exist_ok=True)
    return PositionManager(state_file=str(state_dir / "positions.json"))


@pytest.fixture()
def stub_market_data() -> _StubMarketData:
    """Return the stubbed market data source."""

    return _StubMarketData()


@pytest.fixture()
def order_manager(
    tmp_path: Path,
    rate_limiter: RateLimiter,
    position_manager: PositionManager,
    stub_market_data: _StubMarketData,
) -> OrderManager:
    """Construct an order manager wired with stubbed dependencies."""

    broker = _StubBroker()
    history_parent = tmp_path / "history"
    history_parent.mkdir(parents=True, exist_ok=True)
    manager = OrderManager(
        broker,
        position_manager,
        rate_limiter,
        history_path=history_parent / "history.json",
    )
    manager.POLL_INTERVAL_SEC = 0.01

    class _Resolver:
        """Resolver returning a consistent lot size for tests."""

        def lot_size_for_symbol(self, symbol: str) -> int:  # noqa: D401 - stub
            del symbol
            return 75

    manager.set_instrument_resolver(_Resolver())
    manager.set_market_data_manager(stub_market_data)
    return manager


def test_get_best_bid_ask_depth_prefers_high_liquidity(
    order_manager: OrderManager, stub_market_data: _StubMarketData
) -> None:
    """Depth snapshot should prefer the level with higher visible quantity."""

    stub_market_data.set_quote(
        {
            "depth": {
                "buy": [
                    {"price": 199.5, "quantity": 10},
                    {"price": 199.0, "quantity": 75},
                    {"price": 198.5, "quantity": 20},
                ],
                "sell": [
                    {"price": 200.5, "quantity": 15},
                    {"price": 201.0, "quantity": 60},
                ],
            }
        }
    )
    snapshot = order_manager.get_best_bid_ask_depth("NFO:TEST")
    assert snapshot["bid"] == pytest.approx(199.0)
    assert snapshot["bid_size"] == pytest.approx(75.0)
    assert snapshot["ask"] == pytest.approx(201.0)
    assert snapshot["ask_size"] == pytest.approx(60.0)


def test_calculate_queue_position_accumulates_levels(
    order_manager: OrderManager, stub_market_data: _StubMarketData
) -> None:
    """Queue position should include better and equal-priced liquidity."""

    stub_market_data.set_quote(
        {
            "depth": {
                "sell": [
                    {"price": 150.5, "quantity": 40},
                    {"price": 150.0, "quantity": 25},
                    {"price": 149.5, "quantity": 10},
                ]
            }
        }
    )
    queue = order_manager.calculate_queue_position("NFO:ABC", "BUY", 150.0)
    assert queue == 65  # 40 (better) + 25 (same level)


def test_place_order_skips_when_queue_too_deep(
    order_manager: OrderManager,
    stub_market_data: _StubMarketData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order placement should skip when queue depth exceeds configured threshold."""

    monkeypatch.setattr(app_settings, "ORDER_MAX_QUEUE_POSITION", 50)
    monkeypatch.setattr(app_settings, "ORDER_QUEUE_POSITION_PER_LOT", 0.0)
    stub_market_data.set_quote(
        {
            "depth": {
                "sell": [
                    {"price": 100.5, "quantity": 80},
                    {"price": 100.0, "quantity": 60},
                    {"price": 99.5, "quantity": 40},
                ]
            }
        }
    )
    order_id = order_manager.place_order(
        symbol="NFO:QUEUE",
        side="BUY",
        quantity=75,
        order_type=OrderType.LIMIT,
        price=100.0,
    )
    assert order_id == ""
    assert order_manager.consume_skip_reason() == "queue_depth"


def test_place_order_halts_on_slippage_guard(
    order_manager: OrderManager,
    stub_market_data: _StubMarketData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order placement should halt when slippage guard is breached."""

    monkeypatch.setattr(app_settings, "ORDER_MAX_QUEUE_POSITION", 0)
    monkeypatch.setattr(app_settings, "ORDER_QUEUE_POSITION_PER_LOT", 0.0)
    stub_market_data.set_quote(
        {
            "bid": 101.0,
            "depth": {
                "sell": [
                    {"price": 101.0, "quantity": 30},
                    {"price": 102.0, "quantity": 120},
                ]
            },
        }
    )
    order_id = order_manager.place_order(
        symbol="NFO:SLIP",
        side="BUY",
        quantity=150,
        order_type=OrderType.LIMIT,
        price=101.0,
        max_slippage_per_lot=10.0,
    )
    assert order_id == ""
    assert order_manager.consume_skip_reason() == "slippage_guard"
