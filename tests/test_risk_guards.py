from decimal import Decimal

from src.broker.interface import OrderRequest, Side, OrderType, TimeInForce
from src.risk.guards import GuardConfig, GuardState, risk_check
from src.strategies.runner import StrategyRunner


def _order(qty: int = 1, price: Decimal | None = None) -> OrderRequest:
    return OrderRequest(
        instrument_id=1,
        side=Side.BUY,
        qty=qty,
        order_type=OrderType.MARKET if price is None else OrderType.LIMIT,
        price=price,
        tif=TimeInForce.DAY,
    )


def test_rate_and_position_caps() -> None:
    cfg = GuardConfig(max_position=2, max_exposure=Decimal("100"), max_orders_per_minute=2)
    state = GuardState()
    assert risk_check(_order(), state, cfg)
    assert risk_check(_order(), state, cfg)
    # Third order exceeds rate cap
    assert not risk_check(_order(), state, cfg)


def test_exposure_cap_blocks() -> None:
    cfg = GuardConfig(max_position=5, max_exposure=Decimal("100"))
    state = GuardState()
    assert not risk_check(_order(qty=2, price=Decimal("60")), state, cfg)


class DummyTelegram:
    def send_message(self, _msg: str) -> None:  # pragma: no cover - stub
        pass


def test_max_loss_blocks_orders() -> None:
    runner = StrategyRunner(kite=None, telegram_controller=DummyTelegram())
    runner._on_trade_closed(-2.0)
    cfg = GuardConfig(max_loss=Decimal("1"))
    assert not risk_check(_order(), runner._risk_state, cfg)
