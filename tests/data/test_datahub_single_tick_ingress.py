"""DataHub must have exactly one tick ingress route.

Post-#943 audit, P0: DataHub was registered BOTH as a direct
MarketDataManager subscriber (DataHub.subscribe_ticks ->
mdm.subscribe(symbol, self.ingest_tick_sync)) AND as a MessageBus TICK
consumer (app.py: bus.subscribe_once(MessageType.TICK,
data_hub.ingest_tick_from_bus)).

One accepted market tick therefore had two ingress routes: the quote version
could increment twice, every DataHub subscriber could run twice, and runner
ingress plus position-protection work could repeat. The equal-market-timestamp
check is not an event identity and does not reliably reject the second
delivery, because wall-clock arrival differs.

The direct path is kept (deterministic ordering for position protection);
MessageBus TICK publication continues for independent observers.
"""

from __future__ import annotations

import inspect

from nifty_scalper_bot.core import app as app_module


def _bus_start_source() -> str:
    """Source of the function that wires MessageBus TICK subscribers."""
    src = inspect.getsource(app_module)
    marker = "MESSAGE_BUS_TICK_OWNER"
    assert marker in src, "tick-owner wiring not found"
    start = src.index("if not ctx.message_bus_tick_subscribed:")
    return src[start : start + 2000]


def test_datahub_is_not_registered_as_a_messagebus_tick_consumer() -> None:
    """THE FIX: no second ingress route back into DataHub."""
    region = _bus_start_source()
    assert "data_hub.ingest_tick_from_bus" not in region, (
        "DataHub must not consume MessageBus TICK: it is already a direct "
        "MarketDataManager subscriber, so this creates duplicate ingress."
    )


def test_datahub_remains_the_declared_tick_owner() -> None:
    """Ownership telemetry is preserved and states the ingress route."""
    region = _bus_start_source()
    assert "MESSAGE_BUS_TICK_OWNER owner=data_hub" in region
    assert "ingress=direct_mdm_only" in region


def test_runner_fallback_subscription_is_retained() -> None:
    """When DataHub is absent the runner must still receive bus ticks."""
    region = _bus_start_source()
    assert "strategy_runner.on_data.fallback" in region


def test_direct_mdm_subscription_is_still_the_ingress_path() -> None:
    """DataHub keeps its direct, ordered MDM subscription."""
    from nifty_scalper_bot.data import data_hub as dh_module

    src = inspect.getsource(dh_module)
    assert "mdm_sub(symbol, self.ingest_tick_sync)" in src


def test_ingest_tick_from_bus_remains_available_for_other_wiring() -> None:
    """The method is not deleted -- only this duplicate registration is."""
    from nifty_scalper_bot.data.data_hub import DataHub

    assert callable(getattr(DataHub, "ingest_tick_from_bus", None))
