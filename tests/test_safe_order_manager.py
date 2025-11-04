from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping

import pytest

from nifty_scalper_bot.config.settings import OrderSettings
from nifty_scalper_bot.core.market_regime import RegimeSnapshot
from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager
from nifty_scalper_bot.data.market_regime import MarketRegimeDetector
from nifty_scalper_bot.execution.order_manager import ExitIntent, OrderType
from nifty_scalper_bot.execution.safe_order_manager import SafeOrderManager
from nifty_scalper_bot.utils.errors import OrderPlacementError


@dataclass
class DummyOrderManager:
    calls: int = 0

    def place_order(self, **kwargs) -> str:  # noqa: D401 - test stub
        self.calls += 1
        return f"order-{self.calls}"


@dataclass
class QueueAwareOrderManager:
    queue_position: int
    best_bid: float = 99.0
    best_ask: float = 100.0
    last_kwargs: dict[str, object] = field(default_factory=dict)

    def get_best_bid_ask_depth(
        self, _symbol: str
    ) -> dict[str, float]:  # noqa: D401 - test stub
        return {"bid": self.best_bid, "ask": self.best_ask}

    def calculate_queue_position(self, _symbol: str, _side: str, _price: float) -> int:
        return int(self.queue_position)

    def place_order(self, **kwargs: object) -> str:  # noqa: D401 - test stub
        self.last_kwargs = dict(kwargs)
        return "order-1"


def make_settings(**overrides: object) -> OrderSettings:
    return OrderSettings(
        enable_live=True,
        max_per_second=1,
        max_per_minute=1,
        max_per_day=1,
        default_product="MIS",
        default_order_type="LIMIT",
        limit_offset_pct=0.0,
        retry_attempts=1,
        retry_delay_seconds=0.0,
        **overrides,
    )


def test_safe_order_manager_throttles_after_limit() -> None:
    order_manager = DummyOrderManager()
    manager = SafeOrderManager(order_manager=order_manager, settings=make_settings())
    manager.place_order(
        symbol="TEST", side="BUY", quantity=1, price=100.0, order_type=OrderType.LIMIT
    )
    with pytest.raises(OrderPlacementError):
        manager.place_order(
            symbol="TEST",
            side="BUY",
            quantity=1,
            price=100.0,
            order_type=OrderType.LIMIT,
        )
    assert manager.throttled_count() == 1
    assert manager.rejection_count() == 1


def test_safe_order_manager_records_rejections() -> None:
    settings = make_settings()
    settings.enable_live = False
    order_manager = DummyOrderManager()
    manager = SafeOrderManager(order_manager=order_manager, settings=settings)
    with pytest.raises(OrderPlacementError):
        manager.place_order(symbol="ABC", side="BUY", quantity=1, price=10.0)
    assert manager.rejection_count() == 1


def test_queue_policy_cross_switches_to_market_order() -> None:
    order_manager = QueueAwareOrderManager(queue_position=120)
    settings = make_settings(
        enable_queue_policy=True,
        queue_cross_threshold=100,
        queue_cross_spread_bps=150.0,
        queue_cross_use_market=True,
    )
    manager = SafeOrderManager(order_manager=order_manager, settings=settings)
    order_id = manager.place_order(
        symbol="TEST",
        side="BUY",
        quantity=1,
        price=100.0,
        order_type=OrderType.LIMIT,
    )
    assert order_id == "order-1"
    assert order_manager.last_kwargs["order_type"] == OrderType.MARKET
    assert order_manager.last_kwargs["price"] is None


def test_queue_policy_wait_blocks_when_threshold_hit() -> None:
    order_manager = QueueAwareOrderManager(queue_position=15)
    settings = make_settings(
        enable_queue_policy=True,
        queue_wait_threshold=10,
        queue_cross_threshold=0,
    )
    manager = SafeOrderManager(order_manager=order_manager, settings=settings)
    with pytest.raises(OrderPlacementError) as exc:
        manager.place_order(
            symbol="TEST",
            side="BUY",
            quantity=1,
            price=100.0,
            order_type=OrderType.LIMIT,
        )
    assert "queue_wait" in str(exc.value)


def test_regime_gate_blocks_orders() -> None:
    detector = MarketRegimeDetector()
    regime_manager = MarketRegimeManager(detector)
    regime_manager.ingest_snapshot(
        RegimeSnapshot(
            symbol="TEST",
            regime="event",
            confidence=0.9,
            reason="unit-test",
            updated_at=time.time(),
            adjustments={},
        )
    )
    allowed = regime_manager.can_trade()
    assert not allowed
    reasons = regime_manager.get_filter_reasons()
    assert any("regime_block_event" in reason for reason in reasons)
    order_manager = DummyOrderManager()
    manager = SafeOrderManager(
        order_manager=order_manager,
        settings=make_settings(),
        regime_manager=regime_manager,
    )
    with pytest.raises(OrderPlacementError) as exc:
        manager.place_order(symbol="TEST", side="BUY", quantity=1)
    assert "Regime gate blocked" in str(exc.value)
    regime_manager.set_regime_filter_bypass(True)
    assert regime_manager.can_trade() is True
    order_id = manager.place_order(symbol="TEST", side="BUY", quantity=1)
    assert order_id.startswith("order-")


@dataclass
class ReduceOnlyStub:
    last_intent: ExitIntent | None = None

    def place_reduce_only_exit(
        self, intent: ExitIntent
    ) -> str:  # noqa: D401 - test stub
        self.last_intent = intent
        return "reduce-1"


@dataclass
class RecordingRegimeManager:
    captured_context: dict[str, object] | None = None

    def can_trade(
        self, *, context: Mapping[str, object] | None = None
    ) -> bool:  # noqa: D401 - test stub
        self.captured_context = dict(context or {})
        return True

    def get_filter_reasons(self) -> tuple[str, ...]:  # noqa: D401 - test stub
        return tuple()


def test_reduce_only_exit_infers_buy_for_short() -> None:
    regime_manager = RecordingRegimeManager()
    order_manager = ReduceOnlyStub()
    manager = SafeOrderManager(
        order_manager=order_manager,
        settings=make_settings(),
        regime_manager=regime_manager,
    )
    intent = ExitIntent(
        symbol="TEST",
        qty=-5,
        product="MIS",
        exchange="NFO",
        order_type="MARKET",
        tag="reduce_only",
    )

    order_id = manager.place_reduce_only_exit(intent)
    assert order_id == "reduce-1"
    assert regime_manager.captured_context is not None
    assert regime_manager.captured_context.get("side") == "BUY"
