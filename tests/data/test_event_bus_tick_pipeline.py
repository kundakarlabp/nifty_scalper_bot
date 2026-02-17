from nifty_scalper_bot.data.data_hub import TickBus


def test_tick_bus_supports_event_bus_channels() -> None:
    bus = TickBus()
    seen: list[dict[str, float]] = []

    bus.subscribe_event("tick", lambda payload: seen.append(payload))
    bus.publish_event("tick", {"symbol": "NSE:NIFTY 50", "last_price": 1.0})

    assert seen == [{"symbol": "NSE:NIFTY 50", "last_price": 1.0}]


def test_tick_bus_legacy_subscribe_routes_tick_event() -> None:
    bus = TickBus()
    seen: list[dict[str, float]] = []

    bus.subscribe(lambda payload: seen.append(payload))
    bus.publish({"symbol": "NSE:NIFTY 50", "last_price": 2.0})

    assert seen == [{"symbol": "NSE:NIFTY 50", "last_price": 2.0}]
