from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace

from nifty_scalper_bot.core.boot_log_safety import BootLogRateControl
from nifty_scalper_bot.core.boot_readiness_safety import (
    adapt_compute_live_readiness,
    adapt_indicator_get_history,
    adapt_mdm_pipeline_overload,
    adapt_option_indicator_direction_context,
    adapt_register_and_subscribe_live_symbol,
    adapt_replay_latest_mdm_ticks_to_bus,
    adapt_sync_history_from_mdm,
    adapt_wire_and_start_message_bus,
)
from nifty_scalper_bot.execution.readiness import normalize_readiness_blockers


def _record(event: str, **extra):
    record = logging.LogRecord(
        name="nifty_scalper_bot.core.instrument_manager",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=event,
        args=(),
        exc_info=None,
    )
    record.event = event
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_rate_control_allows_changed_basket_state() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    first = _record(
        "CONTRACT_SSOT_BASKET_SELECTED",
        selected_ce="CE1",
        selected_pe="PE1",
        atm_strike=24250,
    )
    same = _record(
        "CONTRACT_SSOT_BASKET_SELECTED",
        selected_ce="CE1",
        selected_pe="PE1",
        atm_strike=24250,
    )
    changed = _record(
        "CONTRACT_SSOT_BASKET_SELECTED",
        selected_ce="CE2",
        selected_pe="PE2",
        atm_strike=24300,
    )

    assert control.filter(first) is True
    assert control.filter(same) is False
    assert control.filter(changed) is True


def test_rate_control_covers_bootstrap_state() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    first = _record(
        "LIVE_UNIVERSE_BOOTSTRAP_STATUS",
        symbol="NSE:NIFTY",
        ready=False,
        reason="waiting",
    )
    same = _record(
        "LIVE_UNIVERSE_BOOTSTRAP_STATUS",
        symbol="NSE:NIFTY",
        ready=False,
        reason="waiting",
    )

    assert control.filter(first) is True
    assert control.filter(same) is False


def test_rate_control_covers_orderflow_direction_context_noise() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    first = _record(
        "ORDERFLOW_TRIGGER_DECISION",
        symbol="NFO:NIFTY26JUL24000CE",
        side="CE",
        trigger_conditions_met=False,
        trigger_block_reason="direction_context_missing_live",
    )
    same = _record(
        "ORDERFLOW_TRIGGER_DECISION",
        symbol="NFO:NIFTY26JUL24000CE",
        side="CE",
        trigger_conditions_met=False,
        trigger_block_reason="direction_context_missing_live",
    )
    changed = _record(
        "ORDERFLOW_TRIGGER_DECISION",
        symbol="NFO:NIFTY26JUL24000CE",
        side="CE",
        trigger_conditions_met=True,
        trigger_block_reason="",
    )

    assert control.filter(first) is True
    assert control.filter(same) is False
    assert control.filter(changed) is True


def test_rate_control_covers_live_validation_checklist() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    first = _record(
        "LIVE_VALIDATION_CHECKLIST",
        primary_blocker="selected_option_quote_missing",
        blockers=("selected_option_quote_missing",),
        evaluation_ready=True,
        execution_ready=False,
        live_orders_armed=False,
    )
    same = _record(
        "LIVE_VALIDATION_CHECKLIST",
        primary_blocker="selected_option_quote_missing",
        blockers=("selected_option_quote_missing",),
        evaluation_ready=True,
        execution_ready=False,
        live_orders_armed=False,
    )
    changed = _record(
        "LIVE_VALIDATION_CHECKLIST",
        primary_blocker=None,
        blockers=(),
        evaluation_ready=True,
        execution_ready=True,
        live_orders_armed=True,
    )

    assert control.filter(first) is True
    assert control.filter(same) is False
    assert control.filter(changed) is True


def test_rate_control_covers_indicator_history_missing_noise() -> None:
    control = BootLogRateControl(interval_seconds=30.0)
    first = _record(
        "indicator_engine_history_missing",
        symbol="NFO:NIFTY2681124600PE",
    )
    same = _record(
        "indicator_engine_history_missing",
        symbol="NFO:NIFTY2681124600PE",
    )
    other = _record(
        "indicator_engine_history_missing",
        symbol="NFO:NIFTY2681124600CE",
    )

    assert control.filter(first) is True
    assert control.filter(same) is False
    assert control.filter(other) is True


def test_live_order_arm_unknown_is_replaced_by_visible_orchestration_reason() -> None:
    record = logging.LogRecord(
        name="nifty_scalper_bot.core.app",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=(
            "LIVE_ORDER_ARM_BLOCKED reason=%s ce_bars=%s pe_bars=%s required=%s "
            "indicators_ready=%s quote_ready=%s"
        ),
        args=("unknown", 105, 105, 30, True, True),
        exc_info=None,
    )

    assert BootLogRateControl(interval_seconds=30.0).filter(record) is True
    assert record.reason == "startup_orchestration_guard"
    assert "reason=startup_orchestration_guard" in record.getMessage()
    assert "reason=unknown" not in record.getMessage()


def test_missing_indicator_history_short_circuits_duplicate_instrumentation() -> None:
    calls: list[str] = []
    symbol = "NFO:NIFTY2681124600PE"

    def original(_self, resolved_symbol: str, *args, **kwargs):
        del args, kwargs
        calls.append(resolved_symbol)
        return [1.0]

    state = SimpleNamespace(
        _histories={},
        _lock=threading.RLock(),
        _logger=logging.getLogger("nifty_scalper_bot.strategies.indicators"),
    )
    adapted = adapt_indicator_get_history(original)

    assert adapted(state, symbol) == []
    assert calls == []

    state._histories[symbol] = object()
    assert adapted(state, symbol) == [1.0]
    assert calls == [symbol]


def test_option_indicator_direction_is_rederived_from_underlying_context() -> None:
    def original(_self, _symbol: str, *args, **kwargs):
        del args, kwargs
        return {
            "close": 85.0,
            "direction_bias": "CE",
            "underlying_direction_bias": "CE",
            "underlying_direction_confidence": 0.95,
            "context_age_seconds": 0.01,
            "context_fresh": True,
            "direction_context_source": "stale_option_payload",
        }

    adapted = adapt_option_indicator_direction_context(original)
    option_payload = adapted(object(), "NFO:NIFTY2681124600PE")
    spot_payload = adapted(object(), "NSE:NIFTY")

    assert option_payload == {"close": 85.0}
    assert spot_payload["direction_bias"] == "CE"
    assert spot_payload["context_fresh"] is True


def test_live_validation_checklist_log_contains_primary_gate(caplog) -> None:
    caplog.set_level(logging.INFO, logger="nifty_scalper_bot.execution.readiness")

    decision = normalize_readiness_blockers(
        ["selected_ce_quote_missing"],
        "OPEN",
        broker_state={"broker_balance_valid": True},
        live_mode=True,
        evaluation_ready=True,
        execution_ready=False,
    )

    record = next(
        r
        for r in caplog.records
        if getattr(r, "event", "") == "LIVE_VALIDATION_CHECKLIST"
    )
    assert decision.primary_blocker == "selected_option_quote_missing"
    assert record.primary_blocker == "selected_option_quote_missing"
    assert record.evaluation_ready is True
    assert record.execution_ready is False
    assert record.live_orders_armed is False


def test_session_readiness_adapter_removes_option_details_outside_session() -> None:
    def original(**kwargs):
        reasons = []
        if not kwargs["market_open"]:
            reasons.append("market_closed")
        if not kwargs["ce_quote_ready"]:
            reasons.append("ce_quote")
        if not kwargs["pe_quote_ready"]:
            reasons.append("pe_quote")
        if kwargs["ce_bars"] < kwargs["option_exec_min_bars"]:
            reasons.append("ce_history")
        if kwargs["pe_bars"] < kwargs["option_exec_min_bars"]:
            reasons.append("pe_history")
        return False, reasons

    adapted = adapt_compute_live_readiness(original)
    _armed, reasons = adapted(
        live_mode=True,
        market_open=False,
        ce_quote_ready=False,
        pe_quote_ready=False,
        ce_bars=0,
        pe_bars=0,
        option_exec_min_bars=30,
    )

    assert reasons == ["market_closed"]


def test_live_readiness_adapter_never_returns_unknown_empty_blocker() -> None:
    adapted = adapt_compute_live_readiness(lambda **_kwargs: (False, []))

    armed, reasons = adapted(
        live_mode=True,
        hard_ready=True,
        quote_available=True,
        ws_quote_proof=True,
        market_open=True,
        runner_running=True,
        selected_ce="NFO:NIFTY2681124600CE",
        selected_pe="NFO:NIFTY2681124600PE",
        ce_bars=105,
        pe_bars=105,
        option_exec_min_bars=30,
        ce_quote_ready=True,
        pe_quote_ready=True,
    )

    assert armed is False
    assert reasons == ["readiness_inconsistent"]


def test_cached_tick_replay_skips_when_message_bus_is_inactive() -> None:
    calls: list[str] = []

    async def original(ctx, *, reason: str) -> int:
        calls.append(reason)
        return 10

    adapted = adapt_replay_latest_mdm_ticks_to_bus(original)
    ctx = SimpleNamespace(message_bus=SimpleNamespace(running=False))

    assert asyncio.run(adapted(ctx, reason="post_runner_start")) == 0
    assert calls == []


def test_cached_tick_replay_uses_direct_datahub_when_bus_is_inactive() -> None:
    replayed: list[dict] = []

    async def original(ctx, *, reason: str) -> int:
        raise AssertionError(f"unexpected bus replay: {reason}")

    adapted = adapt_replay_latest_mdm_ticks_to_bus(original)
    ctx = SimpleNamespace(
        message_bus=SimpleNamespace(running=False),
        market_data_manager=SimpleNamespace(
            _latest_ticks={
                "NSE:NIFTY": {"last_price": 24775.0},
                "NFO:NIFTY26AUGFUT": {"last_price": 24665.0},
            }
        ),
        data_hub=SimpleNamespace(
            ingest_tick_sync=lambda tick: replayed.append(dict(tick))
        ),
        data_observation_ready=False,
    )

    assert asyncio.run(adapted(ctx, reason="post_runner_start")) == 2
    assert [tick["symbol"] for tick in replayed] == [
        "NSE:NIFTY",
        "NFO:NIFTY26AUGFUT",
    ]
    assert all(tick["source"] == "mdm_replay" for tick in replayed)


def test_cached_tick_replay_preserved_when_message_bus_is_running() -> None:
    calls: list[str] = []

    async def original(ctx, *, reason: str) -> int:
        calls.append(reason)
        return 3

    adapted = adapt_replay_latest_mdm_ticks_to_bus(original)
    ctx = SimpleNamespace(message_bus=SimpleNamespace(running=True))

    assert asyncio.run(adapted(ctx, reason="post_runner_start")) == 3
    assert calls == ["post_runner_start"]


def test_startup_symbol_wiring_does_not_activate_runner_early() -> None:
    class Runner:
        def __init__(self) -> None:
            self.added: list[str] = []

        def add_symbol(self, symbol: str) -> None:
            self.added.append(symbol)

        def on_datahub_tick(self, tick) -> None:
            del tick

    class Hub:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def subscribe_ticks(self, symbol, callback, **kwargs):
            self.calls.append(
                {
                    "symbol": symbol,
                    "callback": callback,
                    **kwargs,
                }
            )
            return True

    def original(ctx, symbol, token, reason, role="tradable_option") -> bool:
        del reason, role
        if hasattr(ctx.strategy_runner, "add_symbol"):
            ctx.strategy_runner.add_symbol(symbol)
        ctx.data_hub.subscribe_ticks(
            symbol,
            ctx.strategy_runner.on_datahub_tick,
            token=token,
            force_live=True,
        )
        return True

    runner = Runner()
    hub = Hub()
    ctx = SimpleNamespace(strategy_runner=runner, data_hub=hub)
    adapted = adapt_register_and_subscribe_live_symbol(original)

    assert (
        adapted(
            ctx,
            "NFO:NIFTY2680424750CE",
            16868098,
            "basket_commit_live_startup",
        )
        is True
    )
    assert runner.added == []
    assert hub.calls[0]["force_live"] is False


def test_runtime_symbol_wiring_preserves_activation_and_live_subscription() -> None:
    class Runner:
        def __init__(self) -> None:
            self.added: list[str] = []

        def add_symbol(self, symbol: str) -> None:
            self.added.append(symbol)

        def on_datahub_tick(self, tick) -> None:
            del tick

    class Hub:
        def __init__(self) -> None:
            self.force_live: bool | None = None

        def subscribe_ticks(self, symbol, callback, **kwargs):
            del symbol, callback
            self.force_live = bool(kwargs["force_live"])
            return True

    def original(ctx, symbol, token, reason, role="tradable_option") -> bool:
        del reason, role
        if hasattr(ctx.strategy_runner, "add_symbol"):
            ctx.strategy_runner.add_symbol(symbol)
        ctx.data_hub.subscribe_ticks(
            symbol,
            ctx.strategy_runner.on_datahub_tick,
            token=token,
            force_live=True,
        )
        return True

    runner = Runner()
    hub = Hub()
    ctx = SimpleNamespace(strategy_runner=runner, data_hub=hub)
    adapted = adapt_register_and_subscribe_live_symbol(original)

    assert adapted(ctx, "NFO:NIFTY2680424800PE", 16869634, "runtime_rotation")
    assert runner.added == ["NFO:NIFTY2680424800PE"]
    assert hub.force_live is True


def test_inactive_subscriberless_bus_is_detached_from_direct_mdm() -> None:
    bus = SimpleNamespace(
        running=False,
        subscribers={"tick": [], "signal": []},
    )
    mdm = SimpleNamespace(bus=bus)
    ctx = SimpleNamespace(message_bus=bus, market_data_manager=mdm)

    def original(input_ctx) -> bool:
        assert input_ctx is ctx
        return False

    adapted = adapt_wire_and_start_message_bus(original)

    assert adapted(ctx) is False
    assert mdm.bus is None


def test_message_bus_attachment_is_preserved_for_real_subscribers() -> None:
    bus = SimpleNamespace(
        running=False,
        subscribers={"tick": [object()]},
    )
    mdm = SimpleNamespace(bus=bus)
    ctx = SimpleNamespace(message_bus=bus, market_data_manager=mdm)

    adapted = adapt_wire_and_start_message_bus(lambda _ctx: False)

    assert adapted(ctx) is False
    assert mdm.bus is bus


def test_history_sync_adapter_corrects_futures_role_only() -> None:
    calls: list[tuple[str, str]] = []

    def original(self, symbol: str, *args, **kwargs):
        del self, args
        calls.append((symbol, kwargs["role"]))
        return kwargs["role"]

    adapted = adapt_sync_history_from_mdm(original)

    assert (
        adapted(
            object(),
            "NFO:NIFTY26AUGFUT",
            required_bars=20,
            role="spot_context",
        )
        == "futures_context"
    )
    assert (
        adapted(
            object(),
            "NFO:NIFTY2680424750CE",
            required_bars=20,
            role="selected_option",
        )
        == "selected_option"
    )
    assert calls == [
        ("NFO:NIFTY26AUGFUT", "futures_context"),
        ("NFO:NIFTY2680424750CE", "selected_option"),
    ]


def test_active_tick_drain_keeps_pipeline_overload_fail_closed() -> None:
    calls: list[str] = []

    def original(state):
        calls.append("original")
        state._pipeline_overloaded = False

    adapted = adapt_mdm_pipeline_overload(original)
    state = SimpleNamespace(_pipeline_overloaded=True, _tick_active_drains=1)

    adapted(state)
    assert state._pipeline_overloaded is True
    assert calls == []

    state._tick_active_drains = 0
    adapted(state)
    assert state._pipeline_overloaded is False
    assert calls == ["original"]
