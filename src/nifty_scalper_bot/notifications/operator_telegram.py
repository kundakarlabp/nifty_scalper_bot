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
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from nifty_scalper_bot.infra.diagnostics import LOG_TAP
from nifty_scalper_bot.risk.expiry_gate import expiry_theta_block, midday_pause_block

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


def _safe_call(func: Any, *args: Any, **kwargs: Any) -> Any:
    if not callable(func):
        return None
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - read-only diagnostic boundary
        return f"WARN: {getattr(func, '__name__', 'call')} failed: {type(exc).__name__}"


def _obj_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return asdict(value)
        except Exception:  # noqa: BLE001 - diagnostic formatting only
            return {}
    data: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(value, name)
        except Exception:  # noqa: BLE001 - diagnostic formatting only
            continue
        if callable(attr):
            continue
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
    if value is None:
        return None
    if isinstance(value, str):
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
            if "allow_pull" in signature.parameters:
                quote = get_quote(str(symbol), allow_pull=False)
            else:
                quote = get_quote(str(symbol))
        except Exception as exc:  # noqa: BLE001 - read-only diagnostic boundary
            return f"WARN: quote failed: {type(exc).__name__}"
    if quote is None:
        return "missing"
    q = _obj_to_dict(quote)
    bid = q.get("bid") or q.get("best_bid")
    ask = q.get("ask") or q.get("best_ask")
    ltp = q.get("ltp") or q.get("last_price") or q.get("price")
    depth = q.get("depth") or q.get("market_depth")
    source_name = q.get("source") or q.get("quote_source")
    stale = q.get("stale") if "stale" in q else q.get("is_stale")
    parts = [f"ltp={ltp}", f"bid={bid}", f"ask={ask}", f"depth={'yes' if depth else 'no'}"]
    if source_name:
        parts.append(f"source={source_name}")
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
    if isinstance(bars, str):
        return bars
    return _count(bars)


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

    get_status = getattr(service, "get_status", None)
    status_text = None
    if callable(get_status):
        try:
            status_text = get_status()
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            status_text = f"WARN: get_status failed: {type(exc).__name__}"

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
    candidate_rejects = (
        _first_attr(runner, ("latest_candidate_rejects", "last_candidate_rejects", "candidate_rejects"))
        or _first_attr(selector, ("last_rejects", "_last_rejects"))
    )
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
    trades_today = _first_attr(risk, ("trades_today", "daily_trade_count", "trade_count_today"))
    if trades_today is None:
        trades_today = _first_attr(position_manager, ("trades_today", "daily_trade_count", "trade_count_today"))
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
    open_positions = _first_attr(position_manager, ("open_positions", "positions", "get_open_positions"))
    if open_positions is None:
        open_positions = _first_attr(order, ("open_positions", "positions", "get_open_positions"))

    last_trade_decision = (
        _first_attr(ctx, ("last_trade_decision", "LAST_TRADE_DECISION_SNAPSHOT"))
        or _first_attr(runner, ("last_trade_decision", "LAST_TRADE_DECISION_SNAPSHOT"))
    )

    snap: dict[str, Any] = {
        "mode": _first_attr(ctx, ("mode", "trading_mode", "effective_mode")) or getattr(service, "mode", None),
        "effective_mode": _first_attr(ctx, ("effective_mode", "execution_mode")),
        "market_state": _first_attr(ctx, ("market_state", "market_open", "market_session_state", "session_state")),
        "market_open": _first_attr(ctx, ("market_open", "is_market_open")),
        "market_session_state": _first_attr(ctx, ("market_session_state", "session_state")),
        "selected_ce": selected_ce,
        "selected_pe": selected_pe,
        "spot_price": _first_attr(mdm, ("spot_ltp", "nifty_spot", "spot_price")) or _first_attr(data_hub, ("spot_ltp", "spot_price")),
        "futures_symbol": _first_attr(ctx, ("futures_symbol", "future_symbol")),
        "data_hard_ready": _first_attr(ctx, ("data_hard_ready", "market_data_ready")),
        "evaluation_ready": _first_attr(ctx, ("evaluation_ready", "strategy_evaluation_ready", "strategy_ready")),
        "live_orders_armed": live_orders_armed,
        "live_block_reason": live_block_reason,
        "execution_block_reason": _first_attr(ctx, ("execution_block_reason",)) or _first_attr(order, ("execution_block_reason", "last_execution_block_reason")),
        "execution_ready_by_symbol": _first_attr(ctx, ("execution_ready_by_symbol",)) or _first_attr(runner, ("runtime_execution_ready_by_symbol", "_runtime_execution_ready_by_symbol")),
        "selected_ce_exec_ready": _first_attr(ctx, ("selected_ce_exec_ready", "ce_exec_ready")),
        "selected_pe_exec_ready": _first_attr(ctx, ("selected_pe_exec_ready", "pe_exec_ready")),
        "context_exec_ready": _first_attr(ctx, ("context_exec_ready",)),
        "broker_ready": _first_attr(ctx, ("broker_ready",)) or _first_attr(broker, ("is_ready", "ready", "is_connected", "session_valid", "is_session_valid")),
        "candidate_rejects": candidate_rejects,
        "last_signal_reason": _first_attr(runner, ("last_signal_reason", "last_signal_reasons", "latest_signal_reason")) or strategy_decision_dict.get("reason") or strategy_decision_dict.get("final_block_reason"),
        "regime_gate": _first_attr(strategy_manager, ("last_regime_gate", "_last_regime_gate")) or _first_attr(_dep(service, "regime_manager", "market_regime"), ("last_gate_reason", "get_filter_reasons")),
        "regime_state": _first_attr(strategy_manager, ("regime_state", "_regime_state")) or _first_attr(_dep(service, "regime_manager", "market_regime"), ("current_regime", "regime", "snapshot")),
        "adx_gate": _first_attr(runner, ("adx_gate", "last_adx_gate", "adx_gate_reason", "last_adx_gate_reason")) or _first_attr(strategy_manager, ("adx_gate", "last_adx_gate_reason")),
        "orb_direction_block": _first_attr(runner, ("orb_direction_block", "orb_direction_used", "orb_direction_state", "_orb_direction_used")) or _first_attr(strategy_manager, ("orb_direction_block", "orb_direction_state")),
        "risk_reject": _first_attr(risk, ("last_rejection", "last_reason", "last_reject_reason", "_last_rejection")) or risk_snapshot_dict.get("last_rejection"),
        "risk_breaker": _first_attr(risk, ("breaker_tripped", "is_breaker_tripped", "_breaker_tripped")) or risk_snapshot_dict.get("breaker_tripped"),
        "order_reject": _first_attr(order, ("last_preflight_reject_reason", "last_reject_reason", "last_rejection", "last_skip_reason", "_last_skip_reason")),
        "daily_trades": trades_today,
        "max_trades_per_day": max_trades,
        "midday_pause": midday_pause,
        "expiry_theta_gate": expiry_theta,
        "open_orders": open_orders,
        "open_positions": open_positions,
        "bracket_manager": bracket_manager,
        "bracket_manager_attached": _first_attr(ctx, ("bracket_manager_attached",)) if _first_attr(ctx, ("bracket_manager_attached",)) is not None else bracket_manager is not None,
        "unresolved_exit": _first_attr(bracket_manager, ("has_unresolved_exit", "unresolved_exit", "get_first_unresolved_exit_bracket_id")),
        "active_basket": basket,
        "active_symbol_tokens": _first_attr(ctx, ("active_symbol_tokens",)) or _first_attr(mdm, ("active_symbol_tokens", "symbol_tokens")),
        "websocket_subscribed_tokens": _first_attr(websocket, ("subscribed_tokens", "tokens", "get_subscribed_tokens")),
        "mdm_tracked": _first_attr(mdm, ("tracked_snapshot", "list_tracked", "tracked_symbols", "_tracked_symbols")),
        "mdm_subscribers": _first_attr(mdm, ("subscribers", "_subscribers")),
        "last_trade_decision": last_trade_decision,
        "mdm": mdm,
        "data_hub": data_hub,
        "runner": runner,
        "strategy_manager": strategy_manager,
        "risk": risk,
        "order": order,
        "broker": broker,
        "position_manager": position_manager,
        "websocket": websocket,
        "status_text": status_text,
    }
    snap["selected_ce_quote"] = _extract_quote_summary(data_hub, selected_ce) or _extract_quote_summary(mdm, selected_ce)
    snap["selected_pe_quote"] = _extract_quote_summary(data_hub, selected_pe) or _extract_quote_summary(mdm, selected_pe)
    snap["selected_ce_tick_age"] = _symbol_metric(mdm, selected_ce, ("quote_age_ms", "tick_age_ms", "last_tick_age")) or _symbol_metric(data_hub, selected_ce, ("quote_age_ms", "quote_age_s", "last_tick_age"))
    snap["selected_pe_tick_age"] = _symbol_metric(mdm, selected_pe, ("quote_age_ms", "tick_age_ms", "last_tick_age")) or _symbol_metric(data_hub, selected_pe, ("quote_age_ms", "quote_age_s", "last_tick_age"))
    snap["selected_ce_bars"] = _bar_count(data_hub, selected_ce) or _bar_count(mdm, selected_ce)
    snap["selected_pe_bars"] = _bar_count(data_hub, selected_pe) or _bar_count(mdm, selected_pe)
    return snap

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
    for key in (
        "live_block_reason",
        "execution_block_reason",
        "order_reject",
        "risk_reject",
        "last_signal_reason",
    ):
        reason = snap.get(key)
        if reason:
            return str(reason)
    for obj_key, attrs in (
        ("runner", ("last_gate_reason", "last_block_reason", "latest_rejection_reason")),
        ("order", ("last_preflight_reject_reason", "last_reject_reason", "last_rejection", "_last_skip_reason")),
        ("risk", ("last_rejection", "last_reason", "last_reject_reason", "_last_rejection")),
    ):
        reason = _first_attr(snap.get(obj_key), attrs)
        if reason:
            return str(reason)
    return None


async def cmd_why(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    reason = _why_reason(snap) or "waiting_for_valid_signal"
    await safe_reply(
        update,
        "Why no trade?\n"
        f"final_reason: {reason}\n"
        "readiness:\n"
        f"  market_open: {_bool(snap.get('market_open'))}\n"
        f"  live_orders_armed: {_bool(snap.get('live_orders_armed'))}\n"
        f"  live_block_reason: {_value(snap.get('live_block_reason'), 'none')}\n"
        f"  selected_ce_exec_ready: {_bool(snap.get('selected_ce_exec_ready'))}\n"
        f"  selected_pe_exec_ready: {_bool(snap.get('selected_pe_exec_ready'))}\n"
        "data:\n"
        f"  selected_ce: {_value(snap.get('selected_ce'))}\n"
        f"  selected_pe: {_value(snap.get('selected_pe'))}\n"
        f"  data_hard_ready: {_bool(snap.get('data_hard_ready'))}\n"
        f"  evaluation_ready: {_bool(snap.get('evaluation_ready'))}\n"
        "strategy:\n"
        f"  last_signal_reason: {_value(snap.get('last_signal_reason'), 'none')}\n"
        f"  candidate_rejects: {_compact(snap.get('candidate_rejects'))}\n"
        f"  regime_gate: {_compact(snap.get('regime_gate'))}\n"
        f"  adx_gate: {_value(snap.get('adx_gate'), 'none')}\n"
        "discipline:\n"
        f"  midday_pause: {_format_gate(snap.get('midday_pause'))}\n"
        f"  expiry_theta_gate: {_format_gate(snap.get('expiry_theta_gate'))}\n"
        f"  daily_trades: {_value(snap.get('daily_trades'))}\n"
        f"  max_trades_per_day: {_value(snap.get('max_trades_per_day'))}\n"
        f"  orb_direction_block: {_compact(snap.get('orb_direction_block'))}\n"
        "execution:\n"
        f"  broker_ready: {_bool(snap.get('broker_ready'))}\n"
        f"  risk_reject: {_value(snap.get('risk_reject'), 'none')}\n"
        f"  order_reject: {_value(snap.get('order_reject'), 'none')}\n"
        f"  open_orders: {_compact(_count(snap.get('open_orders')))}\n"
        f"  open_positions: {_compact(_count(snap.get('open_positions')))}"
    )

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
    await safe_reply(update, _lines("Market", {
        "selected_ce": snap.get("selected_ce"),
        "selected_pe": snap.get("selected_pe"),
        "selected_ce_quote_depth": snap.get("selected_ce_quote"),
        "selected_pe_quote_depth": snap.get("selected_pe_quote"),
        "selected_ce_tick_age": snap.get("selected_ce_tick_age"),
        "selected_pe_tick_age": snap.get("selected_pe_tick_age"),
        "selected_ce_hydration_bars": snap.get("selected_ce_bars"),
        "selected_pe_hydration_bars": snap.get("selected_pe_bars"),
        "spot_price": snap.get("spot_price"),
        "futures_symbol": snap.get("futures_symbol"),
        "active_basket_symbols": _count(snap.get("active_basket")),
        "active_symbol_tokens": _count(snap.get("active_symbol_tokens")),
        "websocket_subscribed_tokens": _count(snap.get("websocket_subscribed_tokens")),
        "mdm_tracked_symbols": _count(snap.get("mdm_tracked")),
        "mdm_subscribers": _count(snap.get("mdm_subscribers")),
    }))


async def cmd_check_core(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    ctx = _context(service)
    runner = snap.get("runner")
    await safe_reply(update, _lines("Core", {
        "startup_phase": _first_attr(ctx, ("startup_phase", "phase")),
        "data_phase": _first_attr(ctx, ("data_phase",)),
        "runner": _component_state("StrategyRunner", runner),
        "runner_running": _bool(_first_attr(runner, ("running", "is_running", "started"))),
        "evaluation_ready": _bool(snap.get("evaluation_ready")),
        "last_strategy_gate_reason": _why_reason(snap) or "none",
        "last_candidate_rejects": _compact(snap.get("candidate_rejects")),
        "regime_current_state": _compact(snap.get("regime_state")),
        "regime_gate": _compact(snap.get("regime_gate")),
        "adx_gate": _value(snap.get("adx_gate"), "none"),
        "orb_direction_state": _compact(snap.get("orb_direction_block")),
    }))


async def cmd_check_execution(update: Update, _: ContextTypes.DEFAULT_TYPE, service: Any) -> None:
    snap = _runtime_snapshot(service)
    order = snap.get("order")
    await safe_reply(update, _lines("Execution", {
        "live_orders_armed": _bool(snap.get("live_orders_armed")),
        "live_block_reason": snap.get("live_block_reason") or "none",
        "execution_block_reason": snap.get("execution_block_reason") or "none",
        "selected_ce_exec_ready": _bool(snap.get("selected_ce_exec_ready")),
        "selected_pe_exec_ready": _bool(snap.get("selected_pe_exec_ready")),
        "broker_ready": _bool(snap.get("broker_ready")),
        "risk_breaker": snap.get("risk_breaker") if snap.get("risk_breaker") is not None else _component_state("RiskManager", snap.get("risk")),
        "daily_trades": snap.get("daily_trades"),
        "max_trades_per_day": snap.get("max_trades_per_day"),
        "midday_pause": _format_gate(snap.get("midday_pause")),
        "expiry_theta_gate": _format_gate(snap.get("expiry_theta_gate")),
        "orb_direction_flags": _compact(snap.get("orb_direction_block")),
        "last_order_rejection": snap.get("order_reject") or "none",
        "last_risk_rejection": snap.get("risk_reject") or "none",
        "open_orders": _compact(_count(snap.get("open_orders"))),
        "open_positions": _compact(_count(snap.get("open_positions"))),
        "bracket_manager_attached": _bool(snap.get("bracket_manager_attached")),
        "unresolved_exit": _value(snap.get("unresolved_exit"), "none"),
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
