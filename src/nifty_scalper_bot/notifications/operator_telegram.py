"""Single active Telegram operator command registry for the NIFTY scalper bot.

Runtime role:
- Owns the production Telegram command registry.
- Reports runtime state through read-only probes.
- Exposes guarded control commands for pausing/resuming/shadow mode and emergency handling.
- Never submits entry orders. Destructive controls require confirmation except the kill-switch emergency.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import inspect
import logging
import secrets
import time
from typing import Any

from telegram import InputFile, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from nifty_scalper_bot.config.env_utils import resolve_build_sha as _resolve_build_sha
from nifty_scalper_bot.infra.diagnostics import LOG_TAP
from nifty_scalper_bot.risk.expiry_gate import expiry_theta_block, midday_pause_block

LOG = logging.getLogger(__name__)
Handler = Callable[[Update, ContextTypes.DEFAULT_TYPE, Any], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    description: str
    handler: Handler
    category: str = "Core"
    safety: str = "read-only"


async def safe_reply(update: Update, text: str) -> None:
    message = getattr(update, "effective_message", None) or getattr(update, "message", None)
    if message is None:
        LOG.warning("TELEGRAM_REPLY_SKIPPED reason=message_missing")
        return
    await message.reply_text(str(text))


def _message_text(update: Update) -> str:
    message = getattr(update, "effective_message", None) or getattr(update, "message", None)
    return str(getattr(message, "text", "") or "").strip()


def _command_args(update: Update) -> list[str]:
    return _message_text(update).split()[1:]


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
    return missing if value is None or value == "" else str(value)


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
                except Exception as exc:  # noqa: BLE001 - diagnostic boundary
                    return f"WARN: {name} failed: {type(exc).__name__}"
            if value is not None:
                return value
    return None


def _safe_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    if not callable(func):
        return None
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - diagnostic/control boundary
        return f"WARN: {getattr(func, '__name__', 'call')} failed: {type(exc).__name__}"


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _obj_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return asdict(value)
        except Exception:  # noqa: BLE001
            return {}
    data: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(value, name)
        except Exception:  # noqa: BLE001
            continue
        if not callable(attr):
            data[name] = attr
    return data


def _compact(value: Any, *, limit: int = 180) -> str:
    if value is None or value == "":
        return "WARN: unavailable"
    if isinstance(value, Mapping):
        items = list(value.items())
        text = ", ".join(f"{k}={v}" for k, v in items[:8])
        if len(items) > 8:
            text += f", +{len(items) - 8} more"
    elif isinstance(value, (list, tuple, set, frozenset)):
        seq = list(value)
        text = ", ".join(str(v) for v in seq[:8])
        if len(seq) > 8:
            text += f", +{len(seq) - 8} more"
        text = f"[{text}]"
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _count(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return value


def _format_gate(value: Any) -> str:
    if isinstance(value, tuple) and len(value) == 2:
        return f"blocked={bool(value[0])} reason={value[1]}"
    if isinstance(value, Mapping):
        return _compact(value)
    return _value(value)


def _dep(service: Any, *names: str) -> Any:
    deps = getattr(service, "deps", service)
    for name in names:
        value = getattr(deps, name, None)
        if value is not None:
            return value
    return None


def _context(service: Any) -> Any:
    return _dep(service, "bot_context") or service


def _extract_quote_summary(source: Any, symbol: Any) -> str | None:
    if not source or not symbol:
        return None
    quote = None
    get_quote = getattr(source, "get_quote", None)
    if callable(get_quote):
        try:
            signature = inspect.signature(get_quote)
            quote = get_quote(str(symbol), allow_pull=False) if "allow_pull" in signature.parameters else get_quote(str(symbol))
        except Exception as exc:  # noqa: BLE001
            return f"WARN: quote failed: {type(exc).__name__}"
    if quote is None:
        return "missing"
    q = _obj_to_dict(quote)
    bid = q.get("bid") or q.get("best_bid")
    ask = q.get("ask") or q.get("best_ask")
    ltp = q.get("ltp") or q.get("last_price") or q.get("price")
    depth = q.get("depth") or q.get("market_depth")
    stale = q.get("stale") if "stale" in q else q.get("is_stale")
    parts = [f"ltp={ltp}", f"bid={bid}", f"ask={ask}", f"depth={'yes' if depth else 'no'}"]
    if stale is not None:
        parts.append(f"stale={stale}")
    return _compact(", ".join(parts), limit=180)


def _symbol_metric(source: Any, symbol: Any, names: Sequence[str]) -> Any:
    if not source or not symbol:
        return None
    for name in names:
        func = getattr(source, name, None)
        if callable(func):
            value = _safe_call(func, str(symbol))
            if value is not None:
                return value
        elif hasattr(source, name):
            value = getattr(source, name)
            if isinstance(value, Mapping):
                return value.get(str(symbol)) or value.get(str(symbol).upper())
            if value is not None:
                return value
    return None


def _bar_count(source: Any, symbol: Any) -> Any:
    bars = _symbol_metric(source, symbol, ("get_ohlc_bars", "get_ohlc", "bars_for_symbol"))
    return bars if isinstance(bars, str) else _count(bars)


def _runtime_snapshot(service: Any) -> dict[str, Any]:
    ctx = _context(service)
    mdm = _dep(service, "market_data_manager")
    data_hub = _dep(service, "data_hub")
    runner = _dep(service, "strategy_runner")
    strategy_manager = _dep(service, "strategy_manager") or _first_attr(runner, ("strategy_manager",))
    risk = _dep(service, "risk_manager")
    order = _dep(service, "order_manager", "safe_order_manager")
    broker = _dep(service, "broker_client", "broker")
    position_manager = _dep(service, "position_manager") or _first_attr(order, ("position_manager", "positions"))
    bracket_manager = _dep(service, "bracket_manager") or _first_attr(order, ("bracket_manager",))
    websocket = _dep(service, "websocket_manager", "streamer", "stream_supervisor")

    selected_ce = _first_attr(ctx, ("selected_ce", "atm_ce_symbol", "atm_ce", "ce_symbol", "call_symbol"))
    selected_pe = _first_attr(ctx, ("selected_pe", "atm_pe_symbol", "atm_pe", "pe_symbol", "put_symbol"))
    basket = _first_attr(ctx, ("active_contract_basket", "active_basket", "contract_basket", "basket"))
    if basket is not None and not isinstance(basket, str):
        selected_ce = selected_ce or _first_attr(basket, ("ce_symbol", "call_symbol", "selected_ce"))
        selected_pe = selected_pe or _first_attr(basket, ("pe_symbol", "put_symbol", "selected_pe"))

    live_orders_armed = _first_attr(ctx, ("live_orders_armed", "live_order_armed", "execution_armed"))
    if live_orders_armed is None:
        live_orders_armed = _first_attr(order, ("live_orders_armed", "is_live_armed"))
    live_block_reason = _first_attr(ctx, ("live_block_reason", "block_reason"))

    selector = _first_attr(runner, ("trade_candidate_selector",)) or _first_attr(runner, ("_trade_candidate_selector",))
    candidate_rejects = _first_attr(runner, ("latest_candidate_rejects", "last_candidate_rejects", "candidate_rejects")) or _first_attr(selector, ("last_rejects", "_last_rejects"))
    strategy_decision = None
    for symbol in (selected_ce, selected_pe):
        getter = getattr(strategy_manager, "get_last_no_signal_decision", None)
        if callable(getter) and symbol:
            strategy_decision = _safe_call(getter, str(symbol))
            if strategy_decision:
                break
    strategy_decision_dict = _obj_to_dict(strategy_decision)

    risk_snapshot = _safe_call(getattr(risk, "snapshot", None)) if risk is not None else None
    risk_snapshot_dict = _obj_to_dict(risk_snapshot)
    risk_settings = getattr(risk, "settings", None)
    trades_today = _first_attr(risk, ("trades_today", "daily_trade_count", "trade_count_today")) or _first_attr(position_manager, ("trades_today", "daily_trade_count", "trade_count_today"))
    max_trades = _first_attr(risk, ("max_trades_per_day", "daily_trade_limit"))
    if max_trades is None and risk_settings is not None:
        max_trades = _first_attr(risk_settings, ("max_trades_per_day", "daily_trade_limit"))

    midday_pause = _first_attr(ctx, ("midday_pause_active", "midday_pause_blocked"))
    if midday_pause is None:
        midday_pause = midday_pause_block()
    expiry_theta = _first_attr(ctx, ("expiry_theta_blocked", "expiry_theta_gate"))
    if expiry_theta is None:
        expiry_theta = expiry_theta_block()

    open_orders = _first_attr(order, ("open_orders", "get_open_orders", "pending_orders"))
    open_positions = _first_attr(position_manager, ("open_positions", "positions", "get_open_positions")) or _first_attr(order, ("open_positions", "positions", "get_open_positions"))
    current_execution_blocker = live_block_reason or _first_attr(ctx, ("execution_block_reason",)) or _first_attr(order, ("execution_block_reason", "current_execution_blocker"))
    current_risk_breaker = _first_attr(risk, ("breaker_tripped", "is_breaker_tripped", "_breaker_tripped")) or risk_snapshot_dict.get("breaker_tripped")
    current_risk_breaker_reason = _first_attr(risk, ("breaker_reason", "current_risk_breaker_reason")) or risk_snapshot_dict.get("breaker_reason")
    if not current_risk_breaker:
        current_risk_breaker_reason = None
    last_risk_rejection = _first_attr(risk, ("last_rejection", "last_reason", "last_reject_reason", "_last_rejection")) or risk_snapshot_dict.get("last_rejection")
    last_order_rejection = _first_attr(order, ("last_preflight_reject_reason", "last_reject_reason", "last_rejection", "last_skip_reason", "_last_skip_reason"))

    unresolved_summary: dict[str, Any] = {}
    summary_getter = getattr(position_manager, "unresolved_terminal_summary", None)
    if callable(summary_getter):
        summary = _safe_call(summary_getter)
        if isinstance(summary, Mapping):
            unresolved_summary = dict(summary)
    pnl_snapshot: dict[str, Any] = {}
    pnl_getter = getattr(position_manager, "pnl_reconciliation_snapshot", None)
    if callable(pnl_getter):
        raw = _safe_call(pnl_getter)
        if isinstance(raw, Mapping):
            pnl_snapshot = dict(raw)
    pnl_blocker = _safe_call(getattr(position_manager, "current_pnl_reconciliation_blocker", None))
    if pnl_blocker and not current_execution_blocker:
        current_execution_blocker = str(pnl_blocker)

    snap: dict[str, Any] = {
        "mode": _first_attr(ctx, ("mode", "trading_mode", "effective_mode")) or getattr(service, "mode", None),
        "effective_mode": _first_attr(ctx, ("effective_mode", "execution_mode")),
        "market_state": _first_attr(ctx, ("market_state", "market_open", "market_session_state", "session_state")),
        "market_open": _first_attr(ctx, ("market_open", "is_market_open")),
        "selected_ce": selected_ce,
        "selected_pe": selected_pe,
        "spot_price": _first_attr(mdm, ("spot_ltp", "nifty_spot", "spot_price")) or _first_attr(data_hub, ("spot_ltp", "spot_price")),
        "futures_symbol": _first_attr(ctx, ("futures_symbol", "future_symbol")),
        "data_hard_ready": _first_attr(ctx, ("data_hard_ready", "market_data_ready")),
        "evaluation_ready": _first_attr(ctx, ("evaluation_ready", "strategy_evaluation_ready", "strategy_ready")),
        "live_orders_armed": live_orders_armed,
        "live_block_reason": live_block_reason,
        "execution_block_reason": _first_attr(ctx, ("execution_block_reason",)) or _first_attr(order, ("execution_block_reason", "last_execution_block_reason")),
        "current_execution_blocker": current_execution_blocker,
        "current_risk_breaker": current_risk_breaker,
        "current_risk_breaker_reason": current_risk_breaker_reason,
        "current_pnl_reconciliation_blocker": pnl_blocker,
        "pnl_reconciliation": pnl_snapshot,
        "selected_ce_exec_ready": _first_attr(ctx, ("selected_ce_exec_ready", "ce_exec_ready")),
        "selected_pe_exec_ready": _first_attr(ctx, ("selected_pe_exec_ready", "pe_exec_ready")),
        "broker_ready": _first_attr(ctx, ("broker_ready",)) or _first_attr(broker, ("is_ready", "ready", "is_connected", "session_valid", "is_session_valid")),
        "candidate_rejects": candidate_rejects,
        "last_signal_reason": _first_attr(runner, ("last_signal_reason", "last_signal_reasons", "latest_signal_reason")) or strategy_decision_dict.get("reason") or strategy_decision_dict.get("final_block_reason"),
        "regime_gate": _first_attr(strategy_manager, ("last_regime_gate", "_last_regime_gate")) or _first_attr(_dep(service, "regime_manager", "market_regime"), ("last_gate_reason", "get_filter_reasons")),
        "regime_state": _first_attr(strategy_manager, ("regime_state", "_regime_state")) or _first_attr(_dep(service, "regime_manager", "market_regime"), ("current_regime", "regime", "snapshot")),
        "adx_gate": _first_attr(runner, ("adx_gate", "last_adx_gate", "adx_gate_reason", "last_adx_gate_reason")) or _first_attr(strategy_manager, ("adx_gate", "last_adx_gate_reason")),
        "orb_direction_block": _first_attr(runner, ("orb_direction_block", "orb_direction_used", "orb_direction_state", "_orb_direction_used")) or _first_attr(strategy_manager, ("orb_direction_block", "orb_direction_state")),
        "last_risk_rejection": last_risk_rejection,
        "risk_breaker": current_risk_breaker,
        "last_order_rejection": last_order_rejection,
        "daily_trades": trades_today,
        "max_trades_per_day": max_trades,
        "midday_pause": midday_pause,
        "expiry_theta_gate": expiry_theta,
        "open_orders": open_orders,
        "open_positions": open_positions,
        "bracket_manager": bracket_manager,
        "bracket_manager_attached": bracket_manager is not None,
        "unresolved_exit": _first_attr(bracket_manager, ("has_unresolved_exit", "unresolved_exit", "get_first_unresolved_exit_bracket_id")),
        "unresolved_terminal_count": unresolved_summary.get("count"),
        "oldest_unresolved_terminal_age_s": unresolved_summary.get("oldest_age_s"),
        "active_basket": basket,
        "active_symbol_tokens": _first_attr(ctx, ("active_symbol_tokens",)) or _first_attr(mdm, ("active_symbol_tokens", "symbol_tokens")),
        "websocket_subscribed_tokens": _first_attr(websocket, ("subscribed_tokens", "tokens", "get_subscribed_tokens")),
        "mdm_tracked": _first_attr(mdm, ("tracked_snapshot", "list_tracked", "tracked_symbols", "_tracked_symbols")),
        "mdm": mdm,
        "data_hub": data_hub,
        "runner": runner,
        "strategy_manager": strategy_manager,
        "risk": risk,
        "risk_snapshot": risk_snapshot_dict,
        "order": order,
        "broker": broker,
        "position_manager": position_manager,
        "websocket": websocket,
    }
    snap["selected_ce_quote"] = _extract_quote_summary(data_hub, selected_ce) or _extract_quote_summary(mdm, selected_ce)
    snap["selected_pe_quote"] = _extract_quote_summary(data_hub, selected_pe) or _extract_quote_summary(mdm, selected_pe)
    snap["selected_ce_tick_age"] = _symbol_metric(mdm, selected_ce, ("quote_age_ms", "tick_age_ms", "last_tick_age")) or _symbol_metric(data_hub, selected_ce, ("quote_age_ms", "quote_age_s", "last_tick_age"))
    snap["selected_pe_tick_age"] = _symbol_metric(mdm, selected_pe, ("quote_age_ms", "tick_age_ms", "last_tick_age")) or _symbol_metric(data_hub, selected_pe, ("quote_age_ms", "quote_age_s", "last_tick_age"))
    snap["selected_ce_bars"] = _bar_count(data_hub, selected_ce) or _bar_count(mdm, selected_ce)
    snap["selected_pe_bars"] = _bar_count(data_hub, selected_pe) or _bar_count(mdm, selected_pe)
    return snap


def _lines(title: str, items: Mapping[str, Any]) -> str:
    return "\n".join([title, *(f"{key}: {_value(value)}" for key, value in items.items())])


def _command_name_from_update(update: Update) -> str:
    text = _message_text(update)
    if not text.startswith("/"):
        return "unknown"
    return text.split()[0].lstrip("/").split("@", 1)[0].strip().lower() or "unknown"


def _make_bound_handler(service: Any, func: Handler) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]:
    async def _bound(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        command = _command_name_from_update(update)
        LOG.info("TELEGRAM_COMMAND_RECEIVED command=%s chat_id=%s", command, _chat_id(update))
        if not await require_authorized_chat(update, service, command=command):
            return
        LOG.info("TELEGRAM_COMMAND_HANDLER_STARTED command=%s", command)
        try:
            await func(update, context, service)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
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


def _why_reason(snap: Mapping[str, Any]) -> str | None:
    for key in ("live_block_reason", "current_execution_blocker", "current_pnl_reconciliation_blocker", "execution_block_reason", "current_risk_breaker_reason", "last_signal_reason"):
        reason = snap.get(key)
        if reason:
            return str(reason)
    return None


def _component_state(name: str, value: Any) -> str:
    return "OK" if value is not None else f"WARN: {name} not attached"


def _recent_errors_text(short: bool = False) -> str:
    try:
        recent = LOG_TAP.recent(5 if short else 10, level="ERROR")
        if recent:
            return " | ".join(str(item) for item in recent[:3]) if short else "\n".join(str(item) for item in recent)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("LOG_TAP unavailable: %s", exc)
    return "No recent errors available from runtime buffer."


def _parse_count(update: Update, default: int, lo: int, hi: int) -> int:
    try:
        parts = _message_text(update).split()
        if len(parts) > 1:
            return max(lo, min(int(parts[1]), hi))
    except Exception:
        pass
    return default


def _log_ring_tail(n: int) -> list[str]:
    from nifty_scalper_bot.notifications.telegram_controller import RING

    return RING.tail(n)


def _control_audit(command: str, service: Any, *, action: str, result: Any = None) -> None:
    LOG.warning(
        "TELEGRAM_CONTROL_COMMAND command=%s chat_id=%s action=%s result=%s",
        command,
        _expected_chat_id(service),
        action,
        result,
    )


def _confirmation_store(service: Any) -> dict[str, dict[str, Any]]:
    current = getattr(service, "_telegram_pending_confirmations", None)
    if not isinstance(current, dict):
        current = getattr(service, "_pending_confirmation", None)
    if not isinstance(current, dict):
        current = {}
    for name in ("_telegram_pending_confirmations", "_pending_confirmation"):
        try:
            setattr(service, name, current)
        except Exception:  # noqa: BLE001
            pass
    return current


async def _request_confirmation(update: Update, service: Any, *, action: str, description: str, callback: Callable[[], Any], ttl_seconds: int = 30) -> None:
    code = f"{secrets.randbelow(9000) + 1000}"
    _confirmation_store(service)[action] = {"code": code, "deadline": time.time() + ttl_seconds, "callback": callback, "description": description}
    _control_audit("confirm_request", service, action=action, result="pending")
    await safe_reply(update, f"CONFIRM REQUIRED: {description}\nType /confirm {action} {code} within {ttl_seconds} seconds.")


async def cmd_confirm(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    args = _command_args(update)
    if len(args) < 2:
        await safe_reply(update, "Usage: /confirm <action> <code>")
        return
    action, code = args[0].strip().lower(), args[1].strip()
    store = _confirmation_store(service)
    item = store.get(action)
    if not item:
        await safe_reply(update, f"No pending confirmation for {action}.")
        return
    if time.time() > float(item.get("deadline", 0.0)):
        store.pop(action, None)
        await safe_reply(update, f"Confirmation expired for {action}.")
        return
    if code != str(item.get("code")):
        await safe_reply(update, "Confirmation code mismatch.")
        return
    callback = item.get("callback")
    store.pop(action, None)
    if not callable(callback):
        await safe_reply(update, f"Confirmation callback missing for {action}.")
        return
    result = await _maybe_await(callback())
    _control_audit("confirm", service, action=action, result=result)
    await safe_reply(update, f"Confirmed {action}: {result if result is not None else 'done'}")


async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    await safe_reply(update, "✅ Nifty Scalper Bot online\n" f"chat_id: {_value(_chat_id(update))}\n" f"mode: {_value(snap.get('effective_mode') or snap.get('mode'))}\n" f"market: {_value(snap.get('market_state'))}\n" f"live_orders_armed: {_bool(snap.get('live_orders_armed'))}\nUse /help")


async def cmd_help(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    del service
    grouped: dict[str, list[CommandSpec]] = {}
    for spec in OPERATOR_COMMANDS:
        grouped.setdefault(spec.category, []).append(spec)
    lines = ["Operator commands"]
    for category, specs in grouped.items():
        lines.append(f"\n{category}")
        for spec in specs:
            safety = "" if spec.safety == "read-only" else f" [{spec.safety}]"
            lines.append(f"/{spec.name}{safety} - {spec.description}")
    await safe_reply(update, "\n".join(lines))


async def cmd_ping(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    del service
    await safe_reply(update, "pong")


async def cmd_status(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    top = snap.get("live_block_reason") or _why_reason(snap) or "none"
    await safe_reply(update, _lines(f"Status: {'BLOCKED' if top != 'none' else 'OK'}", {"top_blocker": top, "build_sha": _resolve_build_sha(), "mode": snap.get("mode"), "effective_mode": snap.get("effective_mode"), "market": snap.get("market_state"), "selected_ce": snap.get("selected_ce"), "selected_pe": snap.get("selected_pe"), "data_hard_ready": _bool(snap.get("data_hard_ready")), "evaluation_ready": _bool(snap.get("evaluation_ready")), "live_orders_armed": _bool(snap.get("live_orders_armed")), "live_block_reason": snap.get("live_block_reason") or "none"}))


async def cmd_health(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    top = snap.get("live_block_reason") or _why_reason(snap) or "none"
    overall = "BLOCKED" if snap.get("live_block_reason") else "WARN" if any(snap.get(k) is None for k in ("mdm", "broker", "runner", "risk", "order")) else "OK"
    await safe_reply(update, _lines(f"Health: {overall}", {"market_data": _component_state("MarketDataManager", snap.get("mdm")), "broker": _component_state("Broker", snap.get("broker")), "telegram": "OK", "strategy_runner": _component_state("StrategyRunner", snap.get("runner")), "risk_execution": "OK" if snap.get("risk") is not None and snap.get("order") is not None else "WARN: risk/execution not fully attached", "top_blocker": top}))


async def cmd_diag(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    await safe_reply(update, "Diagnostics\nConnectivity\n" f"broker: {_component_state('Broker', snap.get('broker'))}\n" f"websocket: {_component_state('WebSocket', snap.get('websocket'))}\n" "Market Data\n" f"mdm: {_component_state('MarketDataManager', snap.get('mdm'))}\n" f"selected_ce: {_value(snap.get('selected_ce'))}\nselected_pe: {_value(snap.get('selected_pe'))}\n" "Hydration\n" f"data_hard_ready: {_bool(snap.get('data_hard_ready'))}\nevaluation_ready: {_bool(snap.get('evaluation_ready'))}\n" "Strategy/Core\n" f"runner: {_component_state('StrategyRunner', snap.get('runner'))}\n" "Execution/Risk\n" f"live_orders_armed: {_bool(snap.get('live_orders_armed'))}\nrisk: {_component_state('RiskManager', snap.get('risk'))}\norder: {_component_state('OrderManager', snap.get('order'))}\n" "Last Errors\n" f"{_recent_errors_text()}")


async def cmd_why(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    reason = _why_reason(snap) or "waiting_for_valid_signal"
    await safe_reply(update, "Why no trade?\n" f"final_reason: {reason}\n" "readiness:\n" f"  market_open: {_bool(snap.get('market_open'))}\n" f"  live_orders_armed: {_bool(snap.get('live_orders_armed'))}\n" f"  live_block_reason: {_value(snap.get('live_block_reason'), 'none')}\n" f"  selected_ce_exec_ready: {_bool(snap.get('selected_ce_exec_ready'))}\n" f"  selected_pe_exec_ready: {_bool(snap.get('selected_pe_exec_ready'))}\n" "data:\n" f"  selected_ce: {_value(snap.get('selected_ce'))}\n" f"  selected_pe: {_value(snap.get('selected_pe'))}\n" f"  data_hard_ready: {_bool(snap.get('data_hard_ready'))}\n" f"  evaluation_ready: {_bool(snap.get('evaluation_ready'))}\n" "strategy:\n" f"  last_signal_reason: {_value(snap.get('last_signal_reason'), 'none')}\n" f"  candidate_rejects: {_compact(snap.get('candidate_rejects'))}\n" f"  regime_gate: {_compact(snap.get('regime_gate'))}\n" f"  adx_gate: {_value(snap.get('adx_gate'), 'none')}\n" "discipline:\n" f"  midday_pause: {_format_gate(snap.get('midday_pause'))}\n" f"  expiry_theta_gate: {_format_gate(snap.get('expiry_theta_gate'))}\n" f"  daily_trades: {_value(snap.get('daily_trades'))}\n" f"  max_trades_per_day: {_value(snap.get('max_trades_per_day'))}\n" f"  orb_direction_block: {_compact(snap.get('orb_direction_block'))}\n" "execution:\n" f"  broker_ready: {_bool(snap.get('broker_ready'))}\n" f"  current_execution_blocker: {_value(snap.get('current_execution_blocker'), 'none')}\n" f"  current_pnl_reconciliation_blocker: {_value(snap.get('current_pnl_reconciliation_blocker'), 'none')}\n" f"  current_risk_breaker_reason: {_value(snap.get('current_risk_breaker_reason'), 'none')}\n" f"  unresolved_terminal_count: {_value(snap.get('unresolved_terminal_count'), '0')}\n" f"  oldest_unresolved_terminal_age_s: {_value(snap.get('oldest_unresolved_terminal_age_s'), 'none')}\n" "recent_history:\n" f"  last_risk_rejection: {_value(snap.get('last_risk_rejection'), 'none')}\n" f"  last_order_rejection: {_value(snap.get('last_order_rejection'), 'none')}\n" f"  open_orders: {_compact(_count(snap.get('open_orders')))}\n" f"  open_positions: {_compact(_count(snap.get('open_positions')))}")


async def cmd_check(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    await safe_reply(update, _lines("Check", {"connectivity": "OK" if snap.get("broker") or snap.get("websocket") else "WARN: broker/websocket not attached", "market": "OK" if snap.get("mdm") else "WARN: MarketDataManager not attached", "core": "OK" if snap.get("runner") else "WARN: StrategyRunner not attached", "execution": "OK" if snap.get("risk") and snap.get("order") else "WARN: risk/execution not fully attached", "errors": _recent_errors_text(short=True)}))


async def cmd_check_connectivity(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    websocket = snap.get("websocket")
    await safe_reply(update, _lines("Connectivity", {"broker_session": _first_attr(snap.get("broker"), ("is_session_valid", "session_valid", "is_connected")) or _component_state("Broker", snap.get("broker")), "websocket_connected": _first_attr(websocket, ("is_connected", "connected", "is_running")) or _component_state("WebSocket", websocket), "mdm_ready": _bool(_first_attr(snap.get("mdm"), ("is_ready", "ready"))), "datahub_live": _component_state("DataHub", snap.get("data_hub")), "telegram": "OK"}))


def _market_items(snap: Mapping[str, Any]) -> dict[str, Any]:
    return {"spot_price": snap.get("spot_price"), "selected_ce": snap.get("selected_ce"), "selected_ce_quote": snap.get("selected_ce_quote"), "selected_ce_tick_age": snap.get("selected_ce_tick_age"), "selected_ce_bars": snap.get("selected_ce_bars"), "selected_pe": snap.get("selected_pe"), "selected_pe_quote": snap.get("selected_pe_quote"), "selected_pe_tick_age": snap.get("selected_pe_tick_age"), "selected_pe_bars": snap.get("selected_pe_bars"), "futures_symbol": snap.get("futures_symbol"), "active_basket_symbols": _count(snap.get("active_basket")), "active_symbol_tokens": _count(snap.get("active_symbol_tokens")), "websocket_subscribed_tokens": _count(snap.get("websocket_subscribed_tokens")), "mdm_tracked_symbols": _count(snap.get("mdm_tracked"))}


async def cmd_check_market(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    await safe_reply(update, _lines("Market", _market_items(_runtime_snapshot(service))))


async def cmd_market(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    items = _market_items(snap)
    items.update({"data_hard_ready": _bool(snap.get("data_hard_ready")), "evaluation_ready": _bool(snap.get("evaluation_ready"))})
    await safe_reply(update, _lines("Market: OK" if snap.get("data_hard_ready") and snap.get("evaluation_ready") else "Market: WAIT", items))


async def cmd_check_core(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    ctx = _context(service)
    runner = snap.get("runner")
    await safe_reply(update, _lines("Core", {"startup_phase": _first_attr(ctx, ("startup_phase", "phase")), "data_phase": _first_attr(ctx, ("data_phase",)), "runner": _component_state("StrategyRunner", runner), "runner_running": _bool(_first_attr(runner, ("running", "is_running", "started"))), "evaluation_ready": _bool(snap.get("evaluation_ready")), "last_strategy_gate_reason": _why_reason(snap) or "none", "last_candidate_rejects": _compact(snap.get("candidate_rejects")), "regime_current_state": _compact(snap.get("regime_state")), "regime_gate": _compact(snap.get("regime_gate")), "adx_gate": _value(snap.get("adx_gate"), "none"), "orb_direction_state": _compact(snap.get("orb_direction_block"))}))


def _execution_items(snap: Mapping[str, Any]) -> dict[str, Any]:
    order = snap.get("order")
    return {"live_orders_armed": _bool(snap.get("live_orders_armed")), "live_block_reason": snap.get("live_block_reason") or "none", "execution_block_reason": snap.get("execution_block_reason") or "none", "selected_ce_exec_ready": _bool(snap.get("selected_ce_exec_ready")), "selected_pe_exec_ready": _bool(snap.get("selected_pe_exec_ready")), "broker_ready": _bool(snap.get("broker_ready")), "risk_breaker": snap.get("risk_breaker") if snap.get("risk_breaker") is not None else _component_state("RiskManager", snap.get("risk")), "current_execution_blocker": snap.get("current_execution_blocker") or "none", "current_pnl_reconciliation_blocker": snap.get("current_pnl_reconciliation_blocker") or "none", "current_risk_breaker_reason": snap.get("current_risk_breaker_reason") or "none", "unresolved_terminal_count": snap.get("unresolved_terminal_count") or 0, "oldest_unresolved_terminal_age_s": snap.get("oldest_unresolved_terminal_age_s") or "none", "daily_trades": snap.get("daily_trades"), "max_trades_per_day": snap.get("max_trades_per_day"), "midday_pause": _format_gate(snap.get("midday_pause")), "expiry_theta_gate": _format_gate(snap.get("expiry_theta_gate")), "recent_last_order_rejection": snap.get("last_order_rejection") or "none", "recent_last_risk_rejection": snap.get("last_risk_rejection") or "none", "open_orders": _compact(_count(snap.get("open_orders"))), "open_positions": _compact(_count(snap.get("open_positions"))), "bracket_manager_attached": _bool(snap.get("bracket_manager_attached")), "unresolved_exit": _value(snap.get("unresolved_exit"), "none"), "emergency_stop": _first_attr(order, ("emergency_stopped", "kill_switch_engaged", "is_kill_switch_engaged")) or "WARN: state unavailable"}


async def cmd_check_execution(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    await safe_reply(update, _lines("Execution", _execution_items(_runtime_snapshot(service))))


async def cmd_exec(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    blocker = snap.get("current_execution_blocker") or "none"
    await safe_reply(update, _lines("Execution: BLOCKED" if blocker != "none" else "Execution: OK", _execution_items(snap)))


async def cmd_risk(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    risk = snap.get("risk")
    risk_data = dict(snap.get("risk_snapshot") or {})
    await safe_reply(update, _lines("Risk", {"breaker": snap.get("current_risk_breaker") if snap.get("current_risk_breaker") is not None else "WARN: unavailable", "breaker_reason": snap.get("current_risk_breaker_reason") or "none", "daily_pnl": risk_data.get("day_pnl") or risk_data.get("daily_pnl") or _first_attr(risk, ("day_pnl", "daily_pnl", "_daily_realized_pnl")), "daily_loss_limit": risk_data.get("daily_loss_limit") or _first_attr(risk, ("daily_loss_limit", "max_daily_loss", "max_daily_loss_absolute")), "trades_today": snap.get("daily_trades"), "max_trades_per_day": snap.get("max_trades_per_day"), "last_risk_rejection": snap.get("last_risk_rejection") or "none"}))


async def cmd_positions(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    broker_positions = _first_attr(snap.get("broker"), ("positions", "get_positions", "get_open_positions"))
    local_positions = snap.get("open_positions")
    await safe_reply(update, _lines("Positions", {"broker_positions": _compact(broker_positions), "broker_position_count": _count(broker_positions), "local_positions": _compact(local_positions), "local_position_count": _count(local_positions), "match_hint": "OK" if _count(broker_positions) == _count(local_positions) else "WARN: verify broker/local quantity"}))


def _active_bracket_summary(bracket_manager: Any) -> Any:
    if bracket_manager is None:
        return None
    for name in ("active_brackets", "brackets", "_active_brackets", "_brackets"):
        value = getattr(bracket_manager, name, None)
        if isinstance(value, Mapping):
            if not value:
                return "none"
            first_key = next(iter(value))
            data = _obj_to_dict(value[first_key])
            return {"bracket_id": first_key, "symbol": data.get("symbol"), "state": data.get("exit_state") or data.get("entry_status"), "qty": data.get("quantity"), "remaining": data.get("remaining_quantity"), "entry": data.get("entry_price"), "sl": data.get("sl_trigger_price") or data.get("current_sl"), "tp": data.get("tp_trigger_price") or data.get("current_target"), "exit_pending": data.get("exit_pending") or data.get("exit_in_progress")}
    getter = getattr(bracket_manager, "snapshot", None) or getattr(bracket_manager, "status", None)
    return _safe_call(getter) if callable(getter) else None


async def cmd_bracket(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    await safe_reply(update, _lines("Bracket", {"attached": _bool(snap.get("bracket_manager_attached")), "unresolved_exit": _value(snap.get("unresolved_exit"), "none"), "active_bracket": _compact(_active_bracket_summary(snap.get("bracket_manager")))}))


async def cmd_reconcile(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    pm = snap.get("position_manager")
    order = snap.get("order")
    report = None
    for owner in (pm, order):
        for name in ("reconciliation_snapshot", "reconcile_snapshot", "reconciliation_report", "diagnostic_reconciliation_snapshot"):
            func = getattr(owner, name, None)
            if callable(func):
                report = _safe_call(func)
                if report is not None:
                    break
        if report is not None:
            break
    if report is None:
        report = {"broker_positions": _count(_first_attr(snap.get("broker"), ("positions", "get_positions", "get_open_positions"))), "local_positions": _count(snap.get("open_positions")), "open_orders": _count(snap.get("open_orders")), "unresolved_exit": snap.get("unresolved_exit") or "none", "orphan_positions": _first_attr(pm, ("orphan_count", "orphan_positions_count")) or 0}
    await safe_reply(update, _lines("Reconciliation (read-only)", _obj_to_dict(report) or {"summary": _compact(report)}))


async def cmd_doctor(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    reason = _why_reason(snap) or "waiting_for_valid_signal"
    await safe_reply(update, _lines("Doctor", {"status": "BLOCKED" if reason != "waiting_for_valid_signal" else "OK/WAIT", "top_reason": reason, "data_hard_ready": _bool(snap.get("data_hard_ready")), "evaluation_ready": _bool(snap.get("evaluation_ready")), "live_orders_armed": _bool(snap.get("live_orders_armed")), "execution_blocker": snap.get("current_execution_blocker") or "none", "risk_breaker_reason": snap.get("current_risk_breaker_reason") or "none", "errors": _recent_errors_text(short=True)}))


async def cmd_today(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    risk_data = dict(snap.get("risk_snapshot") or {})
    pm = snap.get("position_manager")
    await safe_reply(update, _lines("Today", {"daily_pnl": risk_data.get("day_pnl") or risk_data.get("daily_pnl") or _first_attr(snap.get("risk"), ("day_pnl", "daily_pnl", "_daily_realized_pnl")), "trades_today": snap.get("daily_trades"), "max_trades_per_day": snap.get("max_trades_per_day"), "open_positions": _count(snap.get("open_positions")), "open_orders": _count(snap.get("open_orders")), "last_trade": _compact(_first_attr(pm, ("last_trade", "last_closed_trade", "latest_trade"))), "last_order_rejection": snap.get("last_order_rejection") or "none", "last_risk_rejection": snap.get("last_risk_rejection") or "none"}))


async def cmd_latency(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    await safe_reply(update, _lines("Latency", {"selected_ce_tick_age": snap.get("selected_ce_tick_age"), "selected_pe_tick_age": snap.get("selected_pe_tick_age"), "ws_heartbeat_age": _first_attr(snap.get("websocket"), ("heartbeat_age_ms", "last_heartbeat_age_ms", "heartbeat_delta_ms", "last_heartbeat_delta_ms")), "order_latency_ms": _first_attr(snap.get("order"), ("last_order_latency_ms", "order_latency_ms", "p95_order_latency_ms")), "rest_latency_ms": _first_attr(snap.get("broker"), ("last_latency_ms", "latency_ms", "p95_latency_ms"))}))


async def cmd_version(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    deps = getattr(service, "deps", service)
    await safe_reply(update, _lines("Version", {"app_version": getattr(deps, "app_version", None) or getattr(service, "app_version", None), "git_sha": getattr(deps, "git_sha", None) or getattr(service, "git_sha", None) or "unknown", "build": getattr(deps, "build", None) or getattr(service, "build", None) or "unknown", "mode": getattr(service, "mode", None)}))


async def cmd_errors(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    del service
    await safe_reply(update, _recent_errors_text())


async def cmd_logs(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    del service
    n = _parse_count(update, default=80, lo=10, hi=400)
    text = "\n".join(_log_ring_tail(n)) or "No logs buffered yet."
    if len(text) > 3500:
        text = "…(truncated)…\n" + text[-3500:]
    await safe_reply(update, text)


async def cmd_dumplogs(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    del service
    import io
    import time as _time

    n = _parse_count(update, default=1500, lo=50, hi=5000)
    text = "\n".join(_log_ring_tail(n)) or "No logs buffered yet."
    bio = io.BytesIO(text.encode("utf-8"))
    bio.name = f"niftybot-logs-{_time.strftime('%Y%m%d-%H%M%S')}.txt"
    chat = getattr(update, "effective_chat", None)
    sender = getattr(chat, "send_document", None) if chat is not None else None
    if sender is None:
        await safe_reply(update, "Cannot send a document to this chat.")
        return
    await sender(InputFile(bio))


async def cmd_stderror(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    err = _first_attr(_context(service), ("last_exception", "last_error", "runtime_exception"))
    await safe_reply(update, str(err) if err else "No runtime exception captured.")


async def cmd_selftest(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    await safe_reply(update, _lines("Selftest (read-only)", {"settings": "OK" if _expected_chat_id(service) is not None else "WARN: Telegram chat id missing", "telegram_authorized": "OK", "mdm": _component_state("MarketDataManager", snap.get("mdm")), "datahub": _component_state("DataHub", snap.get("data_hub")), "runner": _component_state("StrategyRunner", snap.get("runner")), "broker": _component_state("Broker", snap.get("broker")), "selected_ce": snap.get("selected_ce"), "selected_pe": snap.get("selected_pe"), "hydration": _bool(snap.get("data_hard_ready")), "readiness": _bool(snap.get("evaluation_ready"))}))


async def cmd_pause(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    ctx = _context(service)
    order = _dep(service, "order_manager", "safe_order_manager")
    for owner, names in ((service, ("pause_trading", "pause_entries")), (ctx, ("pause_trading", "pause_entries", "set_live_orders_armed")), (order, ("pause_trading", "pause_queue", "pause_entries"))):
        for name in names:
            func = getattr(owner, name, None)
            if callable(func):
                result = await _maybe_await(func(False) if name == "set_live_orders_armed" else func())
                _control_audit("pause", service, action="pause", result=result)
                await safe_reply(update, f"Paused new entries: {result if result is not None else name}")
                return
    _control_audit("pause", service, action="pause", result="not_wired")
    await safe_reply(update, "Pause handler not wired.")


async def cmd_resume(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    ctx = _context(service)
    order = _dep(service, "order_manager", "safe_order_manager")

    async def _resume() -> Any:
        for owner, names in ((service, ("resume_trading", "resume_entries")), (ctx, ("resume_trading", "resume_entries", "set_live_orders_armed")), (order, ("resume_trading", "resume_queue", "resume_entries"))):
            for name in names:
                func = getattr(owner, name, None)
                if callable(func):
                    return await _maybe_await(func(True) if name == "set_live_orders_armed" else func())
        return "resume handler not wired"

    await _request_confirmation(update, service, action="resume", description="resume new entries if runtime readiness allows it", callback=_resume)


async def cmd_shadow(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    args = [a.lower() for a in _command_args(update)]
    getter = getattr(service, "get_shadow_mode", None) or getattr(getattr(service, "deps", None), "get_shadow_mode", None)
    setter = getattr(service, "set_shadow_mode", None) or getattr(getattr(service, "deps", None), "set_shadow_mode", None)
    if not args:
        state = _safe_call(getter) if callable(getter) else getattr(service, "shadow_mode", None)
        await safe_reply(update, f"shadow_mode: {_value(state)}")
        return
    if args[0] not in {"on", "off"}:
        await safe_reply(update, "Usage: /shadow [on|off]")
        return
    enabled = args[0] == "on"

    async def _apply_shadow() -> Any:
        if callable(setter):
            return await _maybe_await(setter(enabled))
        try:
            setattr(service, "shadow_mode", enabled)
            return enabled
        except Exception:  # noqa: BLE001
            return "shadow setter not wired"

    if enabled:
        result = await _apply_shadow()
        _control_audit("shadow", service, action="shadow_on", result=result)
        await safe_reply(update, f"shadow_mode set to {enabled}: {result}")
        return
    await _request_confirmation(update, service, action="shadow_off", description="turn shadow mode OFF and allow live-routing policy to take effect", callback=_apply_shadow)


async def _call_emergency_hook(func: Callable[..., Any], *, name: str) -> Any:
    for args, kwargs in (((), {"reason": "telegram_emergency"}), (("telegram_emergency",), {}), ((), {})):
        try:
            return await _maybe_await(func(*args, **kwargs))
        except TypeError:
            continue
    raise TypeError(f"{name} emergency hook signature unsupported")


async def cmd_emergency(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    order = _dep(service, "order_manager", "safe_order_manager")
    for owner in (service, order):
        for name in ("emergency_stop", "engage_kill_switch", "kill_switch"):
            func = getattr(owner, name, None)
            if callable(func):
                result = await _call_emergency_hook(func, name=name)
                _control_audit("emergency", service, action=name, result=result)
                await safe_reply(update, f"Emergency kill switch triggered: {result if result is not None else name}")
                return
    _control_audit("emergency", service, action="emergency", result="not_wired")
    await safe_reply(update, "Emergency handler not wired.")


async def cmd_flatten(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    order = _dep(service, "order_manager", "safe_order_manager")

    async def _flatten() -> Any:
        for owner in (service, order):
            for name in ("close_all_positions", "flatten_all", "flatten_positions"):
                func = getattr(owner, name, None)
                if callable(func):
                    return await _maybe_await(func())
        return "flatten handler not wired"

    await _request_confirmation(update, service, action="flatten", description="flatten bot-owned open positions", callback=_flatten)


async def cmd_cancel_pending(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    order = _dep(service, "order_manager", "safe_order_manager")

    async def _cancel() -> Any:
        for name in ("cancel_pending_orders", "cancel_all_open_orders", "cancel_non_protective_orders"):
            func = getattr(order, name, None)
            if callable(func):
                return await _maybe_await(func())
        return "cancel-pending handler not wired"

    await _request_confirmation(update, service, action="cancel_pending", description="cancel pending non-protective/open orders", callback=_cancel)


OPERATOR_COMMANDS: list[CommandSpec] = [
    CommandSpec("start", "show bot online and compact command menu", cmd_start, "Core"),
    CommandSpec("help", "grouped operator command menu", cmd_help, "Core"),
    CommandSpec("ping", "simple bot responsiveness check", cmd_ping, "Core"),
    CommandSpec("status", "compact current runtime status", cmd_status, "Core"),
    CommandSpec("why", "reason no trade was taken / current trade rejection reason", cmd_why, "Core"),
    CommandSpec("doctor", "one-shot triage: status, blocker, execution and recent errors", cmd_doctor, "Core"),
    CommandSpec("health", "system health and important alerts", cmd_health, "Diagnostics"),
    CommandSpec("diag", "full diagnostics summary", cmd_diag, "Diagnostics"),
    CommandSpec("check", "subsystem readiness overview", cmd_check, "Diagnostics"),
    CommandSpec("check_connectivity", "broker, websocket, data, Telegram connectivity", cmd_check_connectivity, "Diagnostics"),
    CommandSpec("check_market", "market data, selected CE/PE, hydration, quote/depth status", cmd_check_market, "Diagnostics"),
    CommandSpec("check_core", "strategy runner, regime, session, startup/readiness state", cmd_check_core, "Diagnostics"),
    CommandSpec("check_execution", "live-order arming, risk, positions, open orders, bracket status", cmd_check_execution, "Diagnostics"),
    CommandSpec("market", "operator-grade selected CE/PE quote, depth, age and bar status", cmd_market, "Market"),
    CommandSpec("exec", "execution state, blockers, open orders, positions and bracket protection", cmd_exec, "Execution"),
    CommandSpec("risk", "daily risk, breaker, trades and latest rejection", cmd_risk, "Execution"),
    CommandSpec("positions", "broker/local position comparison", cmd_positions, "Execution"),
    CommandSpec("bracket", "active virtual bracket, SL/TP/trailing and unresolved-exit state", cmd_bracket, "Execution"),
    CommandSpec("reconcile", "read-only broker/local reconciliation report", cmd_reconcile, "Execution"),
    CommandSpec("today", "today's P&L/trade count/open exposure summary", cmd_today, "Execution"),
    CommandSpec("latency", "quote, WebSocket, REST and order latency snapshot", cmd_latency, "Diagnostics"),
    CommandSpec("version", "app version, build and git SHA", cmd_version, "Diagnostics"),
    CommandSpec("errors", "recent error log summary", cmd_errors, "Logs"),
    CommandSpec("logs", "recent log lines inline (/logs [N])", cmd_logs, "Logs"),
    CommandSpec("dumplogs", "download recent logs as a .txt file (/dumplogs [N])", cmd_dumplogs, "Logs"),
    CommandSpec("stderror", "last runtime exception/error details", cmd_stderror, "Logs"),
    CommandSpec("selftest", "run non-invasive system self-test", cmd_selftest, "Diagnostics"),
    CommandSpec("pause", "pause new entries while keeping protective exits alive", cmd_pause, "Control", "control"),
    CommandSpec("resume", "request confirmed resume of new entries", cmd_resume, "Control", "confirmed-control"),
    CommandSpec("shadow", "inspect or toggle shadow mode (/shadow [on|off])", cmd_shadow, "Control", "confirmed-control"),
    CommandSpec("emergency", "immediate kill switch / disable live damage path", cmd_emergency, "Control", "emergency"),
    CommandSpec("flatten", "confirmed flatten of bot-owned open positions", cmd_flatten, "Control", "confirmed-destructive"),
    CommandSpec("cancel_pending", "confirmed cancel of pending non-protective/open orders", cmd_cancel_pending, "Control", "confirmed-destructive"),
    CommandSpec("confirm", "confirm a pending sensitive control command", cmd_confirm, "Control", "confirmation"),
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
