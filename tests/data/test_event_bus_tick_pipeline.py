from nifty_scalper_bot.data.data_hub import TickBus


def test_tick_bus_supports_event_bus_channels() -> None:
    bus = TickBus()
    seen: list[dict[str, float]] = []

    bus.subscribe_event("tick", lambda payload: seen.append(payload))
    bus.publish_event("tick", {"symbol": "NSE:NIFTY", "last_price": 1.0})

    assert seen == [{"symbol": "NSE:NIFTY", "last_price": 1.0}]


def test_tick_bus_legacy_subscribe_routes_tick_event() -> None:
    bus = TickBus()
    seen: list[dict[str, float]] = []

    bus.subscribe(lambda payload: seen.append(payload))
    bus.publish({"symbol": "NSE:NIFTY", "last_price": 2.0})

    assert seen == [{"symbol": "NSE:NIFTY", "last_price": 2.0}]


def test_tick_bus_publish_routes_into_data_hub_quotes() -> None:
    class _StubMDM:
        def attach_tick_bus(self, _tick_bus):
            return None

    class _StubResolver:
        pass

    from nifty_scalper_bot.data.data_hub import DataHub

    hub = DataHub(_StubMDM(), _StubResolver())
    payload = {"symbol": "NSE:NIFTY", "last_price": 123.4}
    hub.tick_bus.publish(payload)

    quote = hub.get_quote("NSE:NIFTY")
    assert quote is not None
    assert quote.get("last_price") == 123.4
