"""
Runtime role:
- Owns the single active Telegram operator command registry.
- Reports runtime state from attached services using read-only probes.
- Must not place, cancel, flatten, or modify broker/order state except /emergency via an existing kill-switch hook.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from nifty_scalper_bot.infra.diagnostics import LOG_TAP

LOG = logging.getLogger(__name__)

Handler = Callable[[Update, ContextTypes.DEFAULT_TYPE, Any], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    description: str
    handler: Handler


async def safe_reply(update: Update, text: str) -> None:
    """Reply to a Telegram update when a message is present."""

    message = getattr(update, "effective_message", None) or getattr(update, "message", None)
    if message is None:
        LOG.warning("TELEGRAM_REPLY_SKIPPED reason=message_missing")
        return
    await message.reply_text(str(text))


def _chat_id(update: Update) -> int | None:
    chat = getattr(update, "effective_chat", None)
    if chat is None:
        message = getattr(update, "effective_message", None) or getattr(update, "message", None)
        chat = getattr(message, "chat", None)
    raw = getattr(chat, "id", None)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _expected_chat_id(service: Any) -> int | None:
    for source in (service, getattr(service, "deps", None)):
        raw = getattr(source, "chat_id", None)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    return None


async def require_authorized_chat(update: Update, service: Any, *, command: str = "unknown") -> bool:
    """Return whether the update is from the configured operator chat."""

    received = _chat_id(update)
    expected = _expected_chat_id(service)
    if expected is None or received == expected:
        LOG.info("TELEGRAM_COMMAND_AUTHORIZED command=%s", command)
        return True
    LOG.warning(
        "TELEGRAM_COMMAND_REJECTED_UNAUTHORIZED received_chat_id=%s expected_chat_id=%s",
        received,
        expected,
    )
    return False


def _bool(value: Any) -> str:
    if value is None:
        return "WARN"
    return "OK" if bool(value) else "BLOCKED"


def _value(value: Any, missing: str = "WARN: unavailable") -> str:
    if value is None or value == "":
        return missing
    return str(value)


def _first_attr(obj: Any, names: Sequence[str]) -> Any:
    for name in names:
        if obj is None:
            return None
        if hasattr(obj, name):
            value = getattr(obj, name)
            if callable(value) and not inspect.iscoroutinefunction(value):
                try:
                    value = value()
                except TypeError:
                    pass
                except Exception as exc:  # noqa: BLE001 - read-only diagnostic boundary
                    return f"WARN: {name} failed: {type(exc).__name__}"
            if value is not None:
                return value
    return None


def _dep(service: Any, *names: str) -> Any:
    deps = getattr(service, "deps", service)
    for name in names:
        value = getattr(deps, name, None)
        if value is not None:
            return value
    return None


def _context(service: Any) -> Any:
    return _dep(service, "bot_context") or service


def _runtime_snapshot(service: Any) -> dict[str, Any]:
    ctx = _context(service)
    mdm = _dep(service, "market_data_manager")
    data_hub = _dep(service, "data_hub")
    runner = _dep(service, "strategy_runner")
    risk = _dep(service, "risk_manager")
    order = _dep(service, "order_manager", "safe_order_manager")
    broker = _dep(service, "broker_client", "broker")

    get_status = getattr(service, "get_status", None)
    status_text = None
    if callable(get_status):
        try:
            status_text = get_status()
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            status_text = f"WARN: get_status failed: {type(exc).__name__}"

    selected_ce = _first_attr(ctx, ("selected_ce", "atm_ce", "ce_symbol", "call_symbol"))
    selected_pe = _first_attr(ctx, ("selected_pe", "atm_pe", "pe_symbol", "put_symbol"))
    basket = _first_attr(ctx, ("active_basket", "contract_basket", "basket"))
    if basket is not None and not isinstance(basket, str):
        selected_ce = selected_ce or _first_attr(basket, ("ce_symbol", "call_symbol", "selected_ce"))
        selected_pe = selected_pe or _first_attr(basket, ("pe_symbol", "put_symbol", "selected_pe"))

    live_orders_armed = _first_attr(ctx, ("live_orders_armed", "live_order_armed"))
    if live_orders_armed is None:
        live_orders_armed = _first_attr(order, ("live_orders_armed", "is_live_armed"))
    live_block_reason = _first_attr(ctx, ("live_block_reason", "block_reason"))

    return {
        "mode": _first_attr(ctx, ("mode", "trading_mode", "effective_mode")) or getattr(service, "mode", None),
        "effective_mode": _first_attr(ctx, ("effective_mode", "execution_mode")),
        "market_state": _first_attr(ctx, ("market_state", "market_open", "session_state")),
        "selected_ce": selected_ce,
        "selected_pe": selected_pe,
        "spot_price": _first_attr(mdm, ("spot_ltp", "nifty_spot", "spot_price")) or _first_attr(data_hub, ("spot_ltp", "spot_price")),
        "futures_symbol": _first_attr(ctx, ("futures_symbol", "future_symbol")),
        "data_hard_ready": _first_attr(ctx, ("data_hard_ready", "market_data_ready")),
        "evaluation_ready": _first_attr(ctx, ("evaluation_ready", "strategy_ready")),
        "live_orders_armed": live_orders_armed,
        "live_block_reason": live_block_reason,
        "mdm": mdm,
        "data_hub": data_hub,
        "runner": runner,
        "risk": risk,
        "order": order,
        "broker": broker,
        "websocket": _dep(service, "websocket_manager", "streamer", "stream_supervisor"),
        "status_text": status_text,
    }


def _lines(title: str, items: Mapping[str, Any]) -> str:
    body = [title]
    for key, value in items.items():
        body.append(f"{key}: {_value(value)}")
    return "\n".join(body)


def _command_name_from_update(update: Update) -> str:
    message = getattr(update, "effective_message", None) or getattr(update, "message", None)
    text = str(getattr(message, "text", "") or "").strip()
    if not text.startswith("/"):
        return "unknown"
    command = text.split()[0].lstrip("/").split("@", 1)[0].strip().lower()
    return command or "unknown"


def _make_bound_handler(service: Any, func: Handler) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def _bound(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        command = _command_name_from_update(update)
        LOG.info(
            "TELEGRAM_COMMAND_RECEIVED command=%s chat_id=%s",
            command,
            _chat_id(update),
        )
        if not await require_authorized_chat(update, service, command=command):
            return
        LOG.info("TELEGRAM_COMMAND_HANDLER_STARTED command=%s", command)
        try:
            await func(update, context, service)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - Telegram command boundary must log and reply
            LOG.error(
                "TELEGRAM_COMMAND_HANDLER_ERROR command=%s error_type=%s error=%s",
                command,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            await safe_reply(update, f"ERROR: {type(exc).__name__}: {exc}")
            return
        LOG.info("TELEGRAM_COMMAND_HANDLER_DONE command=%s", command)

    return _bound


async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    await safe_reply(
        update,
        "✅ Nifty Scalper Bot online\n"
        f"chat_id: {_value(_chat_id(update))}\n"
        f"mode: {_value(snap.get('effective_mode') or snap.get('mode'))}\n"
        f"market: {_value(snap.get('market_state'))}\n"
        f"live_orders_armed: {_bool(snap.get('live_orders_armed'))}\n"
        "Use /help",
    )


async def cmd_help(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    del service
    await safe_reply(update, "\n".join(f"{spec.name} - {spec.description}" for spec in OPERATOR_COMMANDS))


async def cmd_ping(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    del service
    await safe_reply(update, "pong")


async def cmd_status(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    await safe_reply(
        update,
        _lines(
            "Status",
            {
                "mode": snap.get("mode"),
                "effective_mode": snap.get("effective_mode"),
                "market": snap.get("market_state"),
                "selected_ce": snap.get("selected_ce"),
                "selected_pe": snap.get("selected_pe"),
                "data_hard_ready": _bool(snap.get("data_hard_ready")),
                "evaluation_ready": _bool(snap.get("evaluation_ready")),
                "live_orders_armed": _bool(snap.get("live_orders_armed")),
                "live_block_reason": snap.get("live_block_reason") or "none",
            },
        ),
    )


def _component_state(name: str, value: Any) -> str:
    return "OK" if value is not None else f"WARN: {name} not attached"


async def cmd_health(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    top = snap.get("live_block_reason") or _why_reason(snap) or "none"
    overall = "BLOCKED" if snap.get("live_block_reason") else "WARN" if any(snap.get(k) is None for k in ("mdm", "broker", "runner", "risk", "order")) else "OK"
    await safe_reply(
        update,
        _lines(
            f"Health: {overall}",
            {
                "market_data": _component_state("MarketDataManager", snap.get("mdm")),
                "broker": _component_state("Broker", snap.get("broker")),
                "telegram": "OK",
                "strategy_runner": _component_state("StrategyRunner", snap.get("runner")),
                "risk_execution": "OK" if snap.get("risk") is not None and snap.get("order") is not None else "WARN: risk/execution not fully attached",
                "top_blocker": top,
            },
        ),
    )


async def cmd_diag(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    await safe_reply(
        update,
        "Diagnostics\n"
        "Connectivity\n"
        f"broker: {_component_state('Broker', snap.get('broker'))}\n"
        f"websocket: {_component_state('WebSocket', snap.get('websocket'))}\n"
        "Market Data\n"
        f"mdm: {_component_state('MarketDataManager', snap.get('mdm'))}\n"
        f"selected_ce: {_value(snap.get('selected_ce'))}\nselected_pe: {_value(snap.get('selected_pe'))}\n"
        "Hydration\n"
        f"data_hard_ready: {_bool(snap.get('data_hard_ready'))}\nevaluation_ready: {_bool(snap.get('evaluation_ready'))}\n"
        "Strategy/Core\n"
        f"runner: {_component_state('StrategyRunner', snap.get('runner'))}\n"
        "Execution/Risk\n"
        f"live_orders_armed: {_bool(snap.get('live_orders_armed'))}\nrisk: {_component_state('RiskManager', snap.get('risk'))}\norder: {_component_state('OrderManager', snap.get('order'))}\n"
        "Last Errors\n"
        f"{_recent_errors_text()}"
    )


def _why_reason(snap: Mapping[str, Any]) -> str | None:
    for obj_key, attrs in (
        ("runner", ("last_gate_reason", "last_block_reason", "latest_rejection_reason")),
        ("order", ("last_preflight_reject_reason", "last_reject_reason", "last_rejection")),
        ("risk", ("last_rejection", "last_reason", "last_reject_reason")),
    ):
        reason = _first_attr(snap.get(obj_key), attrs)
        if reason:
            return str(reason)
    return None


async def cmd_why(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    reason = snap.get("live_block_reason") or _why_reason(snap) or "No current rejection; waiting for valid signal"
    await safe_reply(update, f"Why\nreason: {reason}")


async def cmd_check(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    await safe_reply(update, _lines("Check", {
        "connectivity": "OK" if snap.get("broker") or snap.get("websocket") else "WARN: broker/websocket not attached",
        "market": "OK" if snap.get("mdm") else "WARN: MarketDataManager not attached",
        "core": "OK" if snap.get("runner") else "WARN: StrategyRunner not attached",
        "execution": "OK" if snap.get("risk") and snap.get("order") else "WARN: risk/execution not fully attached",
        "errors": _recent_errors_text(short=True),
    }))


async def cmd_check_connectivity(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    websocket = snap.get("websocket")
    await safe_reply(update, _lines("Connectivity", {
        "broker_session": _first_attr(snap.get("broker"), ("is_session_valid", "session_valid", "is_connected")) or _component_state("Broker", snap.get("broker")),
        "websocket_connected": _first_attr(websocket, ("is_connected", "connected", "is_running")) or _component_state("WebSocket", websocket),
        "mdm_ready": _bool(_first_attr(snap.get("mdm"), ("is_ready", "ready"))),
        "datahub_live": _component_state("DataHub", snap.get("data_hub")),
        "telegram": "OK",
    }))


async def cmd_check_market(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    mdm = snap.get("mdm")
    await safe_reply(update, _lines("Market", {
        "selected_ce": snap.get("selected_ce"),
        "selected_pe": snap.get("selected_pe"),
        "spot_price": snap.get("spot_price"),
        "futures_symbol": snap.get("futures_symbol"),
        "subscriptions": _first_attr(mdm, ("subscribed_tokens", "subscriptions", "get_subscription_snapshot")),
        "tick_freshness": _first_attr(mdm, ("tick_freshness", "last_tick_age", "freshness")),
        "quote_depth": _first_attr(mdm, ("quote_quality", "depth_status", "quote_depth_status")),
        "hydration_bars": _first_attr(mdm, ("hydration_bars", "bar_count", "ohlc_status")),
    }))


async def cmd_check_core(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    ctx = _context(service)
    await safe_reply(update, _lines("Core", {
        "startup_phase": _first_attr(ctx, ("startup_phase", "phase")),
        "data_phase": _first_attr(ctx, ("data_phase",)),
        "runner": _component_state("StrategyRunner", snap.get("runner")),
        "regime": _first_attr(_dep(service, "regime_manager", "market_regime"), ("current_regime", "regime", "snapshot")),
        "evaluation_ready": _bool(snap.get("evaluation_ready")),
        "strategy_gate_reason": _why_reason(snap) or "none",
    }))


async def cmd_check_execution(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    order = snap.get("order")
    risk = snap.get("risk")
    await safe_reply(update, _lines("Execution", {
        "live_orders_armed": _bool(snap.get("live_orders_armed")),
        "broker_health": _component_state("Broker", snap.get("broker")),
        "risk_breaker": _first_attr(risk, ("breaker_tripped", "is_breaker_tripped")) or _component_state("RiskManager", risk),
        "open_positions": _first_attr(_dep(service, "position_manager"), ("open_positions", "positions", "get_open_positions")),
        "open_orders": _first_attr(order, ("open_orders", "get_open_orders", "pending_orders")),
        "bracket_manager": _component_state("BracketManager", _dep(service, "bracket_manager")),
        "emergency_stop": _first_attr(order, ("emergency_stopped", "kill_switch_engaged", "is_kill_switch_engaged")) or "WARN: state unavailable",
    }))


def _recent_errors_text(short: bool = False) -> str:
    try:
        recent = LOG_TAP.recent(5 if short else 10, level="ERROR")
        if recent:
            return " | ".join(str(item) for item in recent[:3]) if short else "\n".join(str(item) for item in recent)
    except Exception as exc:  # noqa: BLE001 - optional diagnostics
        LOG.debug("LOG_TAP unavailable: %s", exc)
    return "No recent errors available from runtime buffer."


async def cmd_errors(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    del service
    await safe_reply(update, _recent_errors_text())


async def cmd_stderror(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    err = _first_attr(_context(service), ("last_exception", "last_error", "runtime_exception"))
    await safe_reply(update, str(err) if err else "No runtime exception captured.")


async def cmd_selftest(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    await safe_reply(update, _lines("Selftest (read-only)", {
        "settings": "OK" if _expected_chat_id(service) is not None else "WARN: Telegram chat id missing",
        "telegram_authorized": "OK",
        "mdm": _component_state("MarketDataManager", snap.get("mdm")),
        "datahub": _component_state("DataHub", snap.get("data_hub")),
        "runner": _component_state("StrategyRunner", snap.get("runner")),
        "broker": _component_state("Broker", snap.get("broker")),
        "selected_ce": snap.get("selected_ce"),
        "selected_pe": snap.get("selected_pe"),
        "hydration": _bool(snap.get("data_hard_ready")),
        "readiness": _bool(snap.get("evaluation_ready")),
    }))


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _call_emergency_hook(func: Callable[..., Any], *, name: str) -> Any:
    for args, kwargs in (
        ((), {"reason": "telegram_emergency"}),
        (("telegram_emergency",), {}),
        ((), {}),
    ):
        try:
            return await _maybe_await(func(*args, **kwargs))
        except TypeError:
            continue
    raise TypeError(f"{name} emergency hook signature unsupported")


async def cmd_emergency(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    order = _dep(service, "order_manager", "safe_order_manager")
    close_all = getattr(service, "close_all_positions", None)
    if callable(close_all):
        result = await _maybe_await(close_all())
        await safe_reply(update, f"Emergency triggered: {result if result is not None else 'handler completed'}")
        return
    for name in ("emergency_stop", "engage_kill_switch", "kill_switch"):
        func = getattr(order, name, None)
        if callable(func):
            result = await _call_emergency_hook(func, name=name)
            await safe_reply(update, f"Emergency triggered: {result if result is not None else name}")
            return
    await safe_reply(update, "Emergency handler not wired.")


OPERATOR_COMMANDS: list[CommandSpec] = [
    CommandSpec("start", "show bot online and compact command menu", cmd_start),
    CommandSpec("help", "list available commands", cmd_help),
    CommandSpec("ping", "simple bot responsiveness check", cmd_ping),
    CommandSpec("status", "compact current runtime status", cmd_status),
    CommandSpec("health", "system health and important alerts", cmd_health),
    CommandSpec("diag", "full diagnostics summary", cmd_diag),
    CommandSpec("why", "reason no trade was taken / current trade rejection reason", cmd_why),
    CommandSpec("check", "subsystem readiness overview", cmd_check),
    CommandSpec("check_connectivity", "broker, websocket, data, Telegram connectivity", cmd_check_connectivity),
    CommandSpec("check_market", "market data, selected CE/PE, hydration, quote/depth status", cmd_check_market),
    CommandSpec("check_core", "strategy runner, regime, session, startup/readiness state", cmd_check_core),
    CommandSpec("check_execution", "live-order arming, risk, positions, open orders, bracket status", cmd_check_execution),
    CommandSpec("errors", "recent error log summary", cmd_errors),
    CommandSpec("stderror", "last runtime exception/error details", cmd_stderror),
    CommandSpec("selftest", "run non-invasive system self-test", cmd_selftest),
    CommandSpec("emergency", "emergency stop / disable live orders", cmd_emergency),
]

OPERATOR_COMMAND_NAMES: tuple[str, ...] = tuple(spec.name for spec in OPERATOR_COMMANDS)


def _remove_command_handlers(application: Application) -> None:
    handlers = getattr(application, "handlers", None)
    if not isinstance(handlers, dict):
        return
    for group, group_handlers in list(handlers.items()):
        handlers[group] = [h for h in group_handlers if not isinstance(h, CommandHandler)]


def registered_command_names(application: Application) -> list[str]:
    commands: set[str] = set()
    handlers = getattr(application, "handlers", {})
    for group_handlers in getattr(handlers, "values", lambda: [])():
        for handler in group_handlers:
            raw = getattr(handler, "commands", None)
            if raw:
                commands.update(str(cmd) for cmd in raw)
    return sorted(commands)


def register_operator_commands(application: Application, service: Any) -> list[str]:
    """Install the one active operator command set on a PTB application."""

    _remove_command_handlers(application)
    for spec in OPERATOR_COMMANDS:
        application.add_handler(CommandHandler(spec.name, _make_bound_handler(service, spec.handler)))
    commands = registered_command_names(application)
    LOG.info("TELEGRAM_COMMAND_REGISTRY_FINAL commands=%s", commands)
    return commands


__all__ = [
    "CommandSpec",
    "OPERATOR_COMMANDS",
    "OPERATOR_COMMAND_NAMES",
    "register_operator_commands",
    "registered_command_names",
    "safe_reply",
    "require_authorized_chat",
]
