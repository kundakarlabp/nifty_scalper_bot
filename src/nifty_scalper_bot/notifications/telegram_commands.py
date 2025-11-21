from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, cast

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from nifty_scalper_bot.notifications.telegram_utils import redact, reply_err, reply_ok
from nifty_scalper_bot.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass(slots=True)
class Services:
    """Bundle of trading services exposed to Telegram commands."""
    
    order_manager: Any | None
    risk_manager: Any | None
    market_data: Any | None
    strategy_runner: Any | None
    config: Any | None
    broker: Any | None
    journal: Any | None
    metrics: Any | None
    market_regime: Any | None
    allowed_chat_id: int  # Critical for security
    order_execution_hub: Any | None = None
    order_queue: Any | None = None
    state_tracker: Any | None = None
    preflight_validator: Any | None = None
    version_info: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.version_info is None:
            self.version_info = {"build": "unknown", "sha": "unknown"}


CommandFunc = Callable[[Update, ContextTypes.DEFAULT_TYPE, Services], str]


def _wrap_command(
    services: Services, command: CommandFunc, name: str
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Any]:
    """Convert a synchronous command into a secure, async Telegram handler.
    
    Handles:
    1. Authentication (Chat ID Guard)
    2. Logging (Entry/Exit)
    3. Error Handling (Try/Except)
    4. Response Dispatch
    """

    async def _inner(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        # 1. Security Guard
        chat = update.effective_chat
        if not chat or chat.id != services.allowed_chat_id:
            LOG.warning("Unauthorized command attempt: %s from %s", name, chat.id if chat else "unknown")
            return

        # 2. Logging
        LOG.debug("Executing %s", name)

        # 3. Execution
        try:
            message = command(update, context, services)
            if message:
                await reply_ok(update, message)
            else:
                await reply_ok(update, "Done (no output).")
        except Exception as exc:
            LOG.error("Failure in %s: %s", name, exc, exc_info=True)
            await reply_err(update, f"Error: {redact(str(exc))}")

    return _inner


def _extract_argument(update: Update) -> str:
    """Return the argument string supplied with a command."""
    message = update.effective_message
    if not message or not message.text:
        return ""
    parts = message.text.split(maxsplit=1)
    return parts[1].strip() if len(parts) >= 2 else ""


# --- COMMAND IMPLEMENTATIONS ---

def cmd_mode(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    """Summarise current trading mode."""
    cfg = services.config
    if cfg is None:
        return "config unavailable"
    
    live = bool(getattr(cfg, "live", getattr(cfg, "allow_live", False)))
    shadow = bool(getattr(cfg, "shadow_mode", getattr(cfg, "shadow", False)))
    return f"mode: {'LIVE' if live else 'PAPER'} | shadow={shadow}"


def cmd_live(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    """Toggle the live trading flag."""
    cfg = services.config
    if cfg is None:
        return "config unavailable"
    
    current = bool(getattr(cfg, "live", getattr(cfg, "allow_live", False)))
    try:
        setattr(cfg, "live", not current)
        return f"live set to: {not current}"
    except Exception as exc:
        return f"live toggle failed: {exc}"


def cmd_flat(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    """Request a full flatten of open positions."""
    manager = services.order_manager
    if manager is None:
        return "flatten: order manager unavailable"
    
    fn = getattr(manager, "flatten_all", None) or getattr(manager, "close_all_positions", None)
    if not callable(fn):
        return "flatten: not supported by manager"
    
    fn()
    return "Requested: close all positions."


def cmd_brk(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    """Return broker connectivity status."""
    broker = services.broker
    if broker is None:
        return "broker unavailable"
    
    ok = True
    try:
        ping = getattr(broker, "ping", None)
        if callable(ping):
            ok = bool(ping())
    except Exception:
        ok = False
        
    name = str(getattr(broker, "name", "broker"))
    return f'{name}: {"OK" if ok else "UNREACHABLE"}'


def cmd_entry(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    """Trigger strategy entry dry-run."""
    runner = services.strategy_runner
    if runner is None:
        return "entry test: runner unavailable"
    
    fn = getattr(runner, "test_entry", None)
    if not callable(fn):
        return "entry test: not supported"
    
    return f"Paper entry OK: {fn()}"


def cmd_exit(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    """Trigger strategy exit dry-run."""
    runner = services.strategy_runner
    if runner is None:
        return "exit test: runner unavailable"
    
    fn = getattr(runner, "test_exit", None)
    if not callable(fn):
        return "exit test: not supported"
    
    return f"Paper exit OK: {fn()}"


def cmd_dryrun(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    """Execute a single dry-run simulation."""
    runner = services.strategy_runner
    if runner is None:
        return "dryrun: runner unavailable"
    
    fn = getattr(runner, "simulate_once", None)
    if not callable(fn):
        return "dryrun: not supported"
    
    return f"dryrun complete: {fn()}"


def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    """Return consolidated diagnostic snapshot."""
    broker_status = cmd_brk(update, context, services)
    
    pos_count = 0
    if services.order_manager:
        try:
            # Try iterator first, then list
            pos_source = getattr(services.order_manager, "get_open_positions", lambda: [])()
            pos_count = len(list(pos_source))
        except Exception:
            pass
            
    mdm_connected = False
    if services.market_data:
        mdm_connected = bool(getattr(services.market_data, "ws_connected", False))
        
    return (
        f"diag: broker={broker_status} | "
        f"positions={pos_count} | "
        f"mdm.connected={mdm_connected}"
    )


def cmd_uptime(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    metrics = services.metrics
    if not metrics:
        return "uptime: metrics unavailable"
    
    fn = getattr(metrics, "uptime_hhmmss", None)
    if callable(fn):
        return str(fn())
    return "uptime: n/a"


def cmd_limits(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    cfg = services.config
    if not cfg:
        return "limits: n/a"
    return str(getattr(cfg, "limits", "n/a"))


def cmd_net(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    metrics = services.metrics
    if not metrics:
        return "net: n/a"
    
    fn = getattr(metrics, "net_latency_ms", None)
    if callable(fn):
        try:
            return f"api latency: {float(fn()):.0f}ms"
        except Exception:
            pass
    return "net: n/a"


def cmd_save(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    journal = services.journal
    if not journal:
        return "save: journal unavailable"
    
    fn = getattr(journal, "save_snapshot", None)
    if callable(fn):
        return f"saved: {fn()}"
    return "save: not supported"


def cmd_book(update: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    sym = _extract_argument(update).upper()
    if not sym:
        return "Usage: /book <SYMBOL>"
    
    mdm = services.market_data
    if not mdm:
        return "MDM unavailable"
        
    if hasattr(mdm, "get_orderbook"):
        snap = mdm.get_orderbook(sym)
        if snap:
            from nifty_scalper_bot.notifications.telegram_controller import _format_orderbook_snapshot
            return _format_orderbook_snapshot(sym, snap)
            
    return "Orderbook unavailable"


def cmd_ohlc(update: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    sym = _extract_argument(update).upper()
    if not sym:
        return "Usage: /ohlc <SYMBOL>"
    
    if services.market_data and hasattr(services.market_data, "get_ohlc"):
        res = services.market_data.get_ohlc(sym)
        return str(res) if res else "n/a"
    return "MDM unavailable"


def cmd_chain(update: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /chain <ROOT> (e.g. NIFTY)"
    
    if services.market_data and hasattr(services.market_data, "get_option_chain"):
        # Simplistic parsing, assuming telegram_controller handles complex formatting
        # Here we just dump the raw result or a summary
        try:
            chain = services.market_data.get_option_chain(arg)
            if chain:
                return f"Chain fetched: {len(chain)} contracts."
        except Exception as e:
            return f"Chain error: {e}"
    return "Chain unavailable"


def cmd_atm(update: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /atm <ROOT>"
    
    if services.market_data and hasattr(services.market_data, "get_atm_option"):
        return str(services.market_data.get_atm_option(arg))
    return "ATM unavailable"


def cmd_spot(update: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /spot <ROOT>"
    
    if services.market_data and hasattr(services.market_data, "get_spot"):
        return str(services.market_data.get_spot(arg))
    return "Spot unavailable"


def cmd_greeks(update: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /greeks <SYMBOL>"
    
    if services.market_data and hasattr(services.market_data, "get_greeks"):
        return str(services.market_data.get_greeks(arg))
    return "Greeks unavailable"


def cmd_iv(update: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /iv <ROOT>"
    
    if services.market_data and hasattr(services.market_data, "get_iv_trend"):
        return str(services.market_data.get_iv_trend(arg))
    return "IV unavailable"


def cmd_prevclose(update: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /prevclose <SYMBOL>"
    
    if services.market_data and hasattr(services.market_data, "get_prev_close"):
        return str(services.market_data.get_prev_close(arg))
    return "PrevClose unavailable"


def cmd_session(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    if services.market_data and hasattr(services.market_data, "get_market_session"):
        return str(services.market_data.get_market_session())
    return "Session info unavailable"


def cmd_holiday(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    if services.market_data and hasattr(services.market_data, "next_holiday"):
        return str(services.market_data.next_holiday())
    return "Holiday info unavailable"


def cmd_sig(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    if services.strategy_runner and hasattr(services.strategy_runner, "current_signal"):
        return str(services.strategy_runner.current_signal())
    return "Signal unavailable"


def cmd_state(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    if services.strategy_runner and hasattr(services.strategy_runner, "state_snapshot"):
        return str(services.strategy_runner.state_snapshot())
    return "State unavailable"


def cmd_score(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    if services.strategy_runner and hasattr(services.strategy_runner, "composite_score"):
        val = services.strategy_runner.composite_score()
        return f"score={val}" if val is not None else "score: n/a"
    return "Score unavailable"


def cmd_regime(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    detector = services.market_regime
    if not detector:
        return "Regime detector unavailable"
    
    try:
        snap = detector.latest(limit=1)
        if snap:
            s = snap[0]
            return f"{s.symbol}: {s.regime} ({s.confidence:.2f})"
    except Exception:
        pass
    return "Regime unknown"


def cmd_gate_why(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    if services.strategy_runner and hasattr(services.strategy_runner, "gate_reason"):
        return str(services.strategy_runner.gate_reason())
    return "Gate info unavailable"


def cmd_size(update: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /size ..."
    
    rm = services.risk_manager
    if rm and hasattr(rm, "suggest_position_size"):
        # Simplified passthrough; complex parsing usually belongs in controller
        return "See /size logic in controller"
    return "Sizing unavailable"


def cmd_trail(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    om = services.order_manager
    if om and hasattr(om, "trailing_snapshot"):
        return str(om.trailing_snapshot())
    return "Trail info unavailable"


def cmd_riskstate(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    rm = services.risk_manager
    if not rm:
        return "Risk manager unavailable"
    
    breaker = getattr(rm, "breaker_tripped", False)
    reason = getattr(rm, "last_reason", "none")
    return f"Risk: breaker={breaker}, last_reason={reason}"


def cmd_journal_read(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    j = services.journal
    if j and hasattr(j, "last_entries"):
        entries = j.last_entries(limit=5)
        return "\n".join(str(e) for e in entries) if entries else "No entries"
    return "Journal unavailable"


def cmd_execstate(update: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    sym = _extract_argument(update).upper()
    if not sym:
        return "Usage: /execstate <SYMBOL>"
    
    hub = services.order_execution_hub
    if hub and hasattr(hub, "get_position_state"):
        return str(hub.get_position_state(sym))
    return "Exec state unavailable"


def cmd_execqueue(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    q = services.order_queue
    if q and hasattr(q, "get_queue_snapshot"):
        snap = q.get_queue_snapshot()
        return f"Queue depth: {len(snap)}"
    return "Queue unavailable"


def cmd_execlast(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    hub = services.order_execution_hub
    if hub and hasattr(hub, "get_stats"):
        return str(hub.get_stats())
    return "Exec stats unavailable"


def cmd_execwhy(update: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    sym = _extract_argument(update).upper()
    if not sym:
        return "Usage: /execwhy <SYMBOL>"
        
    val = services.preflight_validator
    if val and hasattr(val, "validate"):
        res = val.validate(sym, context={})
        return f"{sym}: allowed={res.allowed} reasons={res.reasons}"
    return "Validator unavailable"


def cmd_emergencystop(_u: Update, _c: ContextTypes.DEFAULT_TYPE, services: Services) -> str:
    hub = services.order_execution_hub
    if hub and hasattr(hub, "emergency_stop"):
        res = hub.emergency_stop()
        return f"STOP triggered: {res}"
    return "Emergency stop unavailable"


# --- REGISTRATION ---

def _command_exists(application: Application, command: str) -> bool:
    """Check if a command handler is already registered."""
    for handlers in application.handlers.values():
        for handler in handlers:
            if isinstance(handler, CommandHandler) and command in handler.commands:
                return True
    return False


def register_telegram_commands(
    bot: Any, application: Application, services: Services
) -> None:
    """Register production commands with security wrapping."""

    # 1. Inject correct chat ID from the bot instance if not already set
    if hasattr(bot, "deps") and hasattr(bot.deps, "chat_id"):
        # We modify the services instance in-place to carry the auth ID
        object.__setattr__(services, "allowed_chat_id", int(bot.deps.chat_id))

    # 2. Register aliases from the main bot controller
    aliases: dict[str, Callable] = {}
    if hasattr(bot, "cmd_positions"): aliases["pos"] = bot.cmd_positions
    if hasattr(bot, "cmd_trades"): aliases["fills"] = bot.cmd_trades
    if hasattr(bot, "cmd_tail"): aliases["logs"] = bot.cmd_tail
    if hasattr(bot, "cmd_debug_on"): aliases["trace"] = bot.cmd_debug_on
    if hasattr(bot, "cmd_kpi"): aliases["perf"] = bot.cmd_kpi

    for name, handler in aliases.items():
        if not _command_exists(application, name):
            application.add_handler(CommandHandler(name, handler))

    # 3. Register local commands with security wrapper
    new_commands: dict[str, CommandFunc] = {
        "mode": cmd_mode,
        "live": cmd_live,
        "flat": cmd_flat,
        "brk": cmd_brk,
        "entry": cmd_entry,
        "exit": cmd_exit,
        "dryrun": cmd_dryrun,
        "diag": cmd_diag,
        "uptime": cmd_uptime,
        "limits": cmd_limits,
        "net": cmd_net,
        "save": cmd_save,
        "book": cmd_book,
        "ohlc": cmd_ohlc,
        "chain": cmd_chain,
        "atm": cmd_atm,
        "spot": cmd_spot,
        "greeks": cmd_greeks,
        "iv": cmd_iv,
        "prevclose": cmd_prevclose,
        "session": cmd_session,
        "holiday": cmd_holiday,
        "sig": cmd_sig,
        "state": cmd_state,
        "score": cmd_score,
        "regime": cmd_regime,
        "gate_why": cmd_gate_why,
        "size": cmd_size,
        "trail": cmd_trail,
        "riskstate": cmd_riskstate,
        "journal": cmd_journal_read,
        "execstate": cmd_execstate,
        "execqueue": cmd_execqueue,
        "execlast": cmd_execlast,
        "execwhy": cmd_execwhy,
        "emergencystop": cmd_emergencystop,
    }

    for name, func in new_commands.items():
        if not _command_exists(application, name):
            # All local commands are wrapped to enforce allowed_chat_id
            application.add_handler(
                CommandHandler(name, _wrap_command(services, func, name))
            )
