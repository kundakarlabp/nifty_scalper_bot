from types import SimpleNamespace

from nifty_scalper_bot.risk import OrderSignal, RiskManager


def test_check_order_enforces_max_trades_per_day_at_final_gate():
    manager = RiskManager.__new__(RiskManager)
    manager.settings = SimpleNamespace(max_trades_per_day=3, max_open_positions=0)
    manager.position_manager = SimpleNamespace(
        trades_today=lambda: 3,
        get_open_positions=lambda: [],
    )
    manager._last_rejection = None
    trips = []
    manager._trip_breaker = lambda reason: trips.append(reason)

    allowed, reason = manager.check_order(
        OrderSignal(
            symbol="NFO:NIFTY24JUL24000CE",
            side="BUY",
            quantity=75,
            price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        ),
        live_enabled=True,
    )

    assert allowed is False
    assert reason == "max_trades_per_day breached: 3/3"
    assert manager._last_rejection == "MAX_TRADES:3/3"
    assert trips == ["max_trades_per_day breached: 3/3"]


def test_check_order_enforces_max_open_positions_at_final_gate():
    manager = RiskManager.__new__(RiskManager)
    manager.settings = SimpleNamespace(max_trades_per_day=0, max_open_positions=1)
    manager.position_manager = SimpleNamespace(
        trades_today=lambda: 0,
        get_open_positions=lambda: [object(), object()],
    )
    manager._last_rejection = None
    trips = []
    manager._trip_breaker = lambda reason: trips.append(reason)

    allowed, reason = manager.check_order(
        OrderSignal(
            symbol="NFO:NIFTY24JUL24000CE",
            side="BUY",
            quantity=75,
            price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        ),
        live_enabled=True,
    )

    assert allowed is False
    assert reason == "max_open_positions breached: 2/1"
    assert manager._last_rejection == "MAX_OPEN:2/1"
    assert trips == ["max_open_positions breached: 2/1"]
