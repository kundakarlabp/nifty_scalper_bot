import pytest

from nifty_scalper_bot.config.settings import OrderSettings
from nifty_scalper_bot.execution.order_manager import OrderType
from nifty_scalper_bot.execution.safe_order_manager import SafeOrderManager
from nifty_scalper_bot.utils.errors import OrderPlacementError


class _StubOrderManager:
    def __init__(self, message: str) -> None:
        self._message = message
        self.last_reason: str | None = None

    def place_order(self, **_: object) -> str:  # noqa: D401 - stub
        raise OrderPlacementError(self._message)

    def set_last_skip_reason(self, reason: str) -> None:  # noqa: D401 - stub
        self.last_reason = reason


@pytest.mark.parametrize(
    "message,expected",
    [
        ("NO_REFERENCE_PRICE", "no_reference_price"),
        ("margin_no_qty", "margin_no_qty"),
    ],
)
def test_safe_order_manager_propagates_skip_reasons(
    message: str, expected: str
) -> None:
    settings = OrderSettings(
        enable_live=True,
        min_spacing_seconds=0.0,
        retry_attempts=1,
        retry_delay_seconds=0.0,
    )
    safe = SafeOrderManager(order_manager=_StubOrderManager(message), settings=settings)
    with pytest.raises(OrderPlacementError):
        safe.place_order(
            symbol="NIFTY24OCT", side="BUY", quantity=1, order_type=OrderType.MARKET
        )
    assert safe.order_manager.last_reason == expected
