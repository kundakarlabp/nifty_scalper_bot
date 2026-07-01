from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from telegram.ext import CommandHandler

from nifty_scalper_bot.notifications.operator_telegram import (
    OPERATOR_COMMAND_NAMES,
    _make_bound_handler,
    register_operator_commands,
    registered_command_names,
)


class FakeApp:
    def __init__(self) -> None:
        self.handlers: dict[int, list[object]] = {0: []}

    def add_handler(self, handler: object, group: int = 0) -> None:
        self.handlers.setdefault(group, []).append(handler)


class DummyMessage:
    def __init__(self) -> None:
        self.replies: list[str] = []
        self.chat = SimpleNamespace(id=12345)

    async def reply_text(self, text: str, **_: Any) -> None:
        self.replies.append(text)


class DummyUpdate:
    def __init__(self, chat_id: int = 12345, text: str = "/ping") -> None:
        self.effective_chat = SimpleNamespace(id=chat_id)
        self.effective_message = DummyMessage()
        self.effective_message.chat = self.effective_chat
        self.effective_message.text = text
        self.message = self.effective_message


class DummyContext:
    pass


def _handler(app: FakeApp, command: str) -> CommandHandler:
    for item in app.handlers.get(0, []):
        if isinstance(item, CommandHandler) and command in item.commands:
            return item
    raise AssertionError(f"missing command {command}")


@pytest.mark.asyncio
async def test_operator_registry_keeps_only_production_commands(caplog: pytest.LogCaptureFixture) -> None:
    app = FakeApp()
    app.add_handler(CommandHandler("legacy", lambda *_: None))
    service = SimpleNamespace(chat_id=12345)

    with caplog.at_level("INFO"):
        commands = register_operator_commands(app, service)  # type: ignore[arg-type]

    assert commands == sorted(OPERATOR_COMMAND_NAMES)
    assert registered_command_names(app) == sorted(OPERATOR_COMMAND_NAMES)  # type: ignore[arg-type]
    assert "legacy" not in commands
    assert any("TELEGRAM_COMMAND_REGISTRY_FINAL" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_help_lists_exactly_kept_commands() -> None:
    app = FakeApp()
    service = SimpleNamespace(chat_id=12345)
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate()

    await _handler(app, "help").callback(update, DummyContext())  # type: ignore[arg-type]

    lines = update.effective_message.replies[-1].splitlines()
    assert [line.split(" - ", 1)[0] for line in lines] == list(OPERATOR_COMMAND_NAMES)
    assert "mode -" not in update.effective_message.replies[-1]
    assert "emergencystop -" not in update.effective_message.replies[-1]


@pytest.mark.asyncio
async def test_ping_replies_pong_with_missing_subsystems(caplog: pytest.LogCaptureFixture) -> None:
    app = FakeApp()
    service = SimpleNamespace(chat_id=12345)
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate(text="/ping")

    with caplog.at_level("INFO"):
        await _handler(app, "ping").callback(update, DummyContext())  # type: ignore[arg-type]

    assert update.effective_message.replies == ["pong"]
    assert "TELEGRAM_COMMAND_RECEIVED command=ping chat_id=12345" in caplog.text
    assert "TELEGRAM_COMMAND_AUTHORIZED command=ping" in caplog.text
    assert "TELEGRAM_COMMAND_HANDLER_STARTED command=ping" in caplog.text
    assert "TELEGRAM_COMMAND_HANDLER_DONE command=ping" in caplog.text


@pytest.mark.asyncio
async def test_unauthorized_chat_is_rejected_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    app = FakeApp()
    service = SimpleNamespace(chat_id=12345)
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate(chat_id=999)

    with caplog.at_level("WARNING"):
        await _handler(app, "ping").callback(update, DummyContext())  # type: ignore[arg-type]

    assert update.effective_message.replies == []
    assert "received_chat_id=999 expected_chat_id=12345" in caplog.text


@pytest.mark.asyncio
async def test_missing_subsystem_warns_not_silent() -> None:
    app = FakeApp()
    service = SimpleNamespace(chat_id=12345)
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate()

    await _handler(app, "check_core").callback(update, DummyContext())  # type: ignore[arg-type]

    assert "WARN: StrategyRunner not attached" in update.effective_message.replies[-1]


@pytest.mark.asyncio
async def test_selftest_is_read_only() -> None:
    app = FakeApp()
    broker = SimpleNamespace(place_order=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not trade")))
    order = SimpleNamespace(cancel_order=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not cancel")))
    service = SimpleNamespace(chat_id=12345, deps=SimpleNamespace(broker_client=broker, order_manager=order))
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate()

    await _handler(app, "selftest").callback(update, DummyContext())  # type: ignore[arg-type]

    assert "Selftest (read-only)" in update.effective_message.replies[-1]


@pytest.mark.asyncio
async def test_emergency_calls_kill_switch_only_when_available() -> None:
    app = FakeApp()
    called: list[str] = []

    def kill_switch(reason: str) -> str:
        called.append(reason)
        return "armed"

    service = SimpleNamespace(
        chat_id=12345,
        deps=SimpleNamespace(order_manager=SimpleNamespace(engage_kill_switch=kill_switch)),
    )
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate()

    await _handler(app, "emergency").callback(update, DummyContext())  # type: ignore[arg-type]

    assert called == ["telegram_emergency"]
    assert "Emergency triggered" in update.effective_message.replies[-1]


@pytest.mark.asyncio
async def test_emergency_reports_unwired_without_side_effect() -> None:
    app = FakeApp()
    service = SimpleNamespace(chat_id=12345)
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate()

    await _handler(app, "emergency").callback(update, DummyContext())  # type: ignore[arg-type]

    assert update.effective_message.replies == ["Emergency handler not wired."]


@pytest.mark.asyncio
async def test_handler_exception_logs_structured_error(caplog: pytest.LogCaptureFixture) -> None:
    async def failing_handler(update: Any, context: Any, service: Any) -> None:
        del update, context, service
        raise RuntimeError("boom")

    service = SimpleNamespace(chat_id=12345)
    update = DummyUpdate(text="/boom")
    bound = _make_bound_handler(service, failing_handler)

    with caplog.at_level("ERROR"):
        await bound(update, DummyContext())  # type: ignore[arg-type]

    assert update.effective_message.replies == ["ERROR: RuntimeError: boom"]
    assert "TELEGRAM_COMMAND_HANDLER_ERROR command=boom error_type=RuntimeError error=boom" in caplog.text


EXPECTED_OPERATOR_COMMANDS = (
    "start",
    "help",
    "ping",
    "status",
    "health",
    "diag",
    "why",
    "check",
    "check_connectivity",
    "check_market",
    "check_core",
    "check_execution",
    "errors",
    "stderror",
    "selftest",
    "emergency",
)


def test_registered_command_names_equal_expected_set() -> None:
    assert OPERATOR_COMMAND_NAMES == EXPECTED_OPERATOR_COMMANDS
    assert OPERATOR_COMMAND_NAMES.count("check_execution") == 1


@pytest.mark.asyncio
async def test_why_includes_diagnostic_sections() -> None:
    app = FakeApp()
    runner = SimpleNamespace(
        _trade_candidate_selector=SimpleNamespace(_last_rejects={"cost_edge_insufficient": 2}),
        last_signal_reason="adx_hard_gate",
    )
    risk = SimpleNamespace(_last_rejection="MAX_TRADES:3/3", settings=SimpleNamespace(max_trades_per_day=3))
    order = SimpleNamespace(_last_skip_reason="order_preflight_rejected")
    service = SimpleNamespace(
        chat_id=12345,
        deps=SimpleNamespace(
            bot_context=SimpleNamespace(
                market_open=True,
                live_orders_armed=False,
                live_block_reason="live_orders_not_armed",
                selected_ce="NIFTY24JUN24000CE",
                selected_pe="NIFTY24JUN24000PE",
                data_hard_ready=True,
                evaluation_ready=True,
                selected_ce_exec_ready=False,
                selected_pe_exec_ready=True,
            ),
            strategy_runner=runner,
            risk_manager=risk,
            order_manager=order,
            broker_client=SimpleNamespace(is_connected=True),
        ),
    )
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate(text="/why")

    await _handler(app, "why").callback(update, DummyContext())  # type: ignore[arg-type]

    reply = update.effective_message.replies[-1]
    assert "Why no trade?" in reply
    assert "readiness:" in reply
    assert "strategy:" in reply
    assert "discipline:" in reply
    assert "execution:" in reply
    assert "final_reason: live_orders_not_armed" in reply
    assert "candidate_rejects: cost_edge_insufficient=2" in reply


@pytest.mark.asyncio
async def test_check_execution_includes_readiness_blockers() -> None:
    app = FakeApp()
    service = SimpleNamespace(
        chat_id=12345,
        deps=SimpleNamespace(
            bot_context=SimpleNamespace(
                live_orders_armed=False,
                live_block_reason="hydration_quote_depth_not_ready",
                execution_block_reason="broker_unavailable",
                selected_ce_exec_ready=False,
                selected_pe_exec_ready=False,
                broker_ready=False,
            ),
            order_manager=SimpleNamespace(_last_skip_reason="selected_option_bid_ask_missing"),
            risk_manager=SimpleNamespace(_breaker_tripped=True, _last_rejection="BREAKER"),
        ),
    )
    register_operator_commands(app, service)  # type: ignore[arg-type]
    update = DummyUpdate(text="/check_execution")

    await _handler(app, "check_execution").callback(update, DummyContext())  # type: ignore[arg-type]

    reply = update.effective_message.replies[-1]
    assert "live_block_reason: hydration_quote_depth_not_ready" in reply
    assert "execution_block_reason: broker_unavailable" in reply
    assert "selected_ce_exec_ready: BLOCKED" in reply
    assert "selected_pe_exec_ready: BLOCKED" in reply


@pytest.mark.asyncio
async def test_stale_rejections_do_not_become_current_execution_reason() -> None:
    app = FakeApp()
    service = SimpleNamespace(
        chat_id=12345,
        deps=SimpleNamespace(
            bot_context=SimpleNamespace(
                live_orders_armed=True,
                live_block_reason=None,
                execution_block_reason=None,
            ),
            order_manager=SimpleNamespace(_last_skip_reason="unresolved_exit_position"),
            risk_manager=SimpleNamespace(
                _breaker_tripped=False,
                _last_rejection="DAILY_REALIZED_LIMIT:-17627.19/-352.34",
            ),
        ),
    )
    register_operator_commands(app, service)  # type: ignore[arg-type]

    why_update = DummyUpdate(text="/why")
    await _handler(app, "why").callback(why_update, DummyContext())  # type: ignore[arg-type]
    why_reply = why_update.effective_message.replies[-1]
    assert "final_reason: waiting_for_valid_signal" in why_reply
    assert "final_reason: unresolved_exit_position" not in why_reply
    assert "final_reason: DAILY_REALIZED_LIMIT" not in why_reply
    assert "last_order_rejection: unresolved_exit_position" in why_reply
    assert "last_risk_rejection: DAILY_REALIZED_LIMIT:-17627.19/-352.34" in why_reply

    execution_update = DummyUpdate(text="/check_execution")
    await _handler(app, "check_execution").callback(execution_update, DummyContext())  # type: ignore[arg-type]
    execution_reply = execution_update.effective_message.replies[-1]
    assert "current_execution_blocker: none" in execution_reply
    assert "current_risk_breaker_reason: none" in execution_reply
    assert "recent_last_order_rejection: unresolved_exit_position" in execution_reply
    assert "recent_last_risk_rejection: DAILY_REALIZED_LIMIT:-17627.19/-352.34" in execution_reply


@pytest.mark.asyncio
async def test_diagnostic_handlers_do_not_crash_with_missing_dependencies() -> None:
    app = FakeApp()
    service = SimpleNamespace(chat_id=12345)
    register_operator_commands(app, service)  # type: ignore[arg-type]

    for command in ("why", "check_execution", "check_market", "check_core"):
        update = DummyUpdate(text=f"/{command}")
        await _handler(app, command).callback(update, DummyContext())  # type: ignore[arg-type]
        assert update.effective_message.replies
        assert not update.effective_message.replies[-1].startswith("ERROR:")


@pytest.mark.asyncio
async def test_diagnostic_handlers_remain_read_only() -> None:
    app = FakeApp()

    def forbidden(*_: Any, **__: Any) -> None:
        raise AssertionError("mutating method must not be called")

    order = SimpleNamespace(
        consume_skip_reason=forbidden,
        place_order=forbidden,
        cancel_order=forbidden,
        _last_skip_reason="preflight_block",
    )
    service = SimpleNamespace(
        chat_id=12345,
        deps=SimpleNamespace(
            bot_context=SimpleNamespace(selected_ce="CE", selected_pe="PE"),
            order_manager=order,
            risk_manager=SimpleNamespace(_last_rejection="risk_block"),
        ),
    )
    register_operator_commands(app, service)  # type: ignore[arg-type]

    for command in ("why", "check_execution"):
        update = DummyUpdate(text=f"/{command}")
        await _handler(app, command).callback(update, DummyContext())  # type: ignore[arg-type]
        assert "preflight_block" in update.effective_message.replies[-1]
