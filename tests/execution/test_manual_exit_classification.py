from types import SimpleNamespace

from nifty_scalper_bot.execution.broker_exposure_quarantine_extension import _is_manual_reduction_order


class _BadQuantity:
    def __float__(self):
        raise TypeError("bad quantity")


def test_manual_sell_against_existing_long_is_classified_as_manual_reduce():
    manager = SimpleNamespace(
        _positions={
            "NFO:NIFTY24JUL24000CE": SimpleNamespace(
                symbol="NFO:NIFTY24JUL24000CE",
                side="LONG",
                quantity=75,
            )
        }
    )
    order = SimpleNamespace(
        symbol="NFO:NIFTY24JUL24000CE",
        side="SELL",
        filled_quantity=75,
        quantity=75,
    )

    assert _is_manual_reduction_order(manager, order) is True


def test_manual_buy_against_existing_short_is_classified_as_manual_reduce():
    manager = SimpleNamespace(
        _positions={
            "NFO:NIFTY24JUL24000PE": SimpleNamespace(
                symbol="NFO:NIFTY24JUL24000PE",
                side="SHORT",
                quantity=75,
            )
        }
    )
    order = SimpleNamespace(
        symbol="NFO:NIFTY24JUL24000PE",
        side="BUY",
        filled_quantity=75,
        quantity=75,
    )

    assert _is_manual_reduction_order(manager, order) is True


def test_oversized_manual_order_is_not_treated_as_managed_reduction():
    manager = SimpleNamespace(
        _positions={
            "NFO:NIFTY24JUL24000CE": SimpleNamespace(
                symbol="NFO:NIFTY24JUL24000CE",
                side="LONG",
                quantity=75,
            )
        }
    )
    order = SimpleNamespace(
        symbol="NFO:NIFTY24JUL24000CE",
        side="SELL",
        filled_quantity=150,
        quantity=150,
    )

    assert _is_manual_reduction_order(manager, order) is False


def test_bad_existing_position_quantity_is_not_reduction_and_does_not_raise():
    manager = SimpleNamespace(
        _positions={
            "NFO:NIFTY24JUL24000CE": SimpleNamespace(
                symbol="NFO:NIFTY24JUL24000CE",
                side="LONG",
                quantity=_BadQuantity(),
            )
        }
    )
    order = SimpleNamespace(
        symbol="NFO:NIFTY24JUL24000CE",
        side="SELL",
        filled_quantity=75,
        quantity=75,
    )

    assert _is_manual_reduction_order(manager, order) is False
