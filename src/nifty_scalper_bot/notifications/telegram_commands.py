from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, cast

if TYPE_CHECKING:
    from telegram import Update as TelegramUpdate
    from telegram.ext import Application as TelegramApplication
    from telegram.ext import ContextTypes as TelegramContextTypes
else:
    TelegramUpdate = Any
    TelegramApplication = Any

    class _ContextTypesShim:
        DEFAULT_TYPE = Any

    TelegramContextTypes = _ContextTypesShim

# Correct imports (utils and logging only)
from nifty_scalper_bot.notifications.telegram_utils import redact, reply_err, reply_ok
from nifty_scalper_bot.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass(slots=True)
class Services:
    """Bundle of trading services exposed to Telegram commands."""
    
    order_manager: Any | None
    risk_manager: Any | None
    market_data_manager: Any | None
    data_hub: Any | None
    bracket_manager: Any | None
    strategy_runner: Any | None
    config: Any | None
    broker: Any | None
    journal: Any | None
    metrics: Any | None
    market_regime: Any | None
    allowed_chat_id: int
    order_queue: Any | None = None
    state_tracker: Any | None = None
    version_info: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.version_info is None:
            self.version_info = {"build": "unknown", "sha": "unknown"}


CommandFunc = Callable[[TelegramUpdate, TelegramContextTypes.DEFAULT_TYPE, Services], str]


def _wrap_command(
    services: Services, command: CommandFunc, name: str
) -> Callable[[TelegramUpdate, TelegramContextTypes.DEFAULT_TYPE], Any]:
    """Convert a synchronous command into a secure, async Telegram handler."""

    async def _inner(update: TelegramUpdate, context: TelegramContextTypes.DEFAULT_TYPE) -> None:
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


def _extract_argument(update: TelegramUpdate) -> str:
    """Return the argument string supplied with a command."""
    message = update.effective_message
    if not message or not message.text:
        return ""
    parts = message.text.split(maxsplit=1)
    return parts[1].strip() if len(parts) >= 2 else ""


def _load_command_handler() -> Any | None:
    """Load telegram CommandHandler lazily. Args: none. Returns: class or None. Raises: never."""
    try:
        from telegram.ext import CommandHandler

        return CommandHandler
    except Exception as exc:
        LOG.warning("TELEGRAM_COMMANDS_UNAVAILABLE reason=%s", exc)
        return None


def _parse_chain_argument(argument: str) -> tuple[str, str | None]:
    """Parse chain argument. Args: argument. Returns: underlying and optional expiry. Raises: never."""

    LOG.debug('Entered _parse_chain_argument')
    try:
        raw = argument.strip()
        if not raw:
            LOG.info('Condition met: empty chain argument')
            return '', None
        if ':' not in raw:
            return raw.upper(), None
        underlying, expiry = raw.split(':', maxsplit=1)
        parsed_underlying = underlying.strip().upper()
        parsed_expiry = expiry.strip() or None
        LOG.info('Condition met: parsed chain argument for %s', parsed_underlying)
        return parsed_underlying, parsed_expiry
    except Exception as exc:
        LOG.error('Failure in _parse_chain_argument: %s', exc, exc_info=True)
        return '', None


def _format_orderbook_snapshot(symbol: str, snapshot: Mapping[str, Any]) -> str:
    """Format orderbook. Args: symbol,snapshot. Returns: compact text summary. Raises: never."""

    LOG.debug('Entered _format_orderbook_snapshot')
    try:
        best_bid = float(snapshot.get('best_bid', 0.0))
        best_ask = float(snapshot.get('best_ask', 0.0))
        spread = float(snapshot.get('spread', best_ask - best_bid))
        liquidity_score = float(snapshot.get('liquidity_score', 0.0))

        buy_levels = cast(Sequence[Mapping[str, Any]], snapshot.get('buy', []))
        sell_levels = cast(Sequence[Mapping[str, Any]], snapshot.get('sell', []))

        buy_repr = ', '.join(
            f"{idx + 1}:{float(level.get('price', 0.0)):.2f}@{int(float(level.get('quantity', 0.0)))}"
            for idx, level in enumerate(buy_levels[:3])
        )
        sell_repr = ', '.join(
            f"{idx + 1}:{float(level.get('price', 0.0)):.2f}@{int(float(level.get('quantity', 0.0)))}"
            for idx, level in enumerate(sell_levels[:3])
        )
        LOG.info('Condition met: formatted orderbook snapshot for %s', symbol)
        return (
            f"{symbol} order book | bid={best_bid:.2f} ask={best_ask:.2f} spread={spread:.2f} "
            f"liq={liquidity_score:.1f}\n"
            f"buy: {buy_repr or 'n/a'}\n"
            f"sell: {sell_repr or 'n/a'}"
        )
    except Exception as exc:
        LOG.error('Failure in _format_orderbook_snapshot: %s', exc, exc_info=True)
        return f'{symbol} order book unavailable'


def _format_chain_summary(
    chain: Sequence[Mapping[str, Any]], underlying: str, expiry: str | None
) -> str:
    """Format option chain. Args: chain,underlying,expiry. Returns: compact analytics summary. Raises: never."""

    LOG.debug('Entered _format_chain_summary')
    try:
        if not chain:
            LOG.info('Condition met: empty chain data for %s', underlying)
            return f'{underlying} chain unavailable'

        total_oi = 0.0
        total_trades = 0.0
        ce_ivs: list[float] = []
        pe_ivs: list[float] = []
        strike_volume: dict[float, float] = defaultdict(float)
        liquidity: dict[float, float] = defaultdict(float)

        for contract in chain:
            strike = float(contract.get('strike', 0.0))
            oi = float(contract.get('open_interest', 0.0))
            trades = float(contract.get('trades', 0.0))
            bid = float(contract.get('bid', 0.0))
            ask = float(contract.get('ask', 0.0))
            iv = float(contract.get('iv', 0.0))

            total_oi += oi
            total_trades += trades
            strike_volume[strike] += trades

            option_type = str(contract.get('option_type', '')).upper()
            if option_type == 'CE' and iv > 0:
                ce_ivs.append(iv)
            if option_type == 'PE' and iv > 0:
                pe_ivs.append(iv)

            spread = max(ask - bid, 0.0)
            if spread > 0:
                liquidity[strike] += oi / spread

            depth = cast(Mapping[str, Sequence[Mapping[str, Any]]], contract.get('depth', {}))
            for side in ('buy', 'sell'):
                for level in depth.get(side, []):
                    liquidity[strike] += float(level.get('quantity', 0.0))

        hot_strikes = sorted(strike_volume.items(), key=lambda item: item[1], reverse=True)[:3]
        liquid_strikes = sorted(liquidity.items(), key=lambda item: item[1], reverse=True)[:3]
        hot_text = ', '.join(f'{int(strike)}:{int(volume)}' for strike, volume in hot_strikes) or 'n/a'
        liquid_text = ', '.join(
            f'{int(strike)}:{score:.1f}' for strike, score in liquid_strikes
        ) or 'n/a'

        ce_iv = (sum(ce_ivs) / len(ce_ivs)) if ce_ivs else 0.0
        pe_iv = (sum(pe_ivs) / len(pe_ivs)) if pe_ivs else 0.0

        expiry_suffix = f' {expiry}' if expiry else ''
        LOG.info('Condition met: formatted chain summary for %s%s', underlying, expiry_suffix)
        return (
            f'{underlying}{expiry_suffix} chain | total_oi={int(total_oi)} total_trades={int(total_trades)}\n'
            f'Volume heatmap: {hot_text}\n'
            f'Liquidity focus: {liquid_text}\n'
            f'IV% CE={ce_iv:.2f} PE={pe_iv:.2f}'
        )
    except Exception as exc:
        LOG.error('Failure in _format_chain_summary: %s', exc, exc_info=True)
        return f'{underlying} chain unavailable'


# --- COMMAND IMPLEMENTATIONS ---

def cmd_mode(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    """Summarise current trading mode."""
    cfg = services.config
    if cfg is None:
        return "config unavailable"
    
    live = bool(getattr(cfg, "live", getattr(cfg, "allow_live", False)))
    shadow = bool(getattr(cfg, "shadow_mode", getattr(cfg, "shadow", False)))
    return f"mode: {'LIVE' if live else 'PAPER'} | shadow={shadow}"


def cmd_live(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
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


def cmd_flat(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    """Request a full flatten of open positions."""
    manager = services.order_manager
    if manager is None:
        return "flatten: order manager unavailable"
    
    fn = getattr(manager, "flatten_all", None) or getattr(manager, "close_all_positions", None)
    if not callable(fn):
        return "flatten: not supported by manager"
    
    fn()
    return "Requested: close all positions."


def cmd_brk(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
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


def cmd_entry(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    """Trigger strategy entry dry-run."""
    runner = services.strategy_runner
    if runner is None:
        return "entry test: runner unavailable"
    
    fn = getattr(runner, "test_entry", None)
    if not callable(fn):
        return "entry test: not supported"
    
    return f"Paper entry OK: {fn()}"


def cmd_exit(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    """Trigger strategy exit dry-run."""
    runner = services.strategy_runner
    if runner is None:
        return "exit test: runner unavailable"
    
    fn = getattr(runner, "test_exit", None)
    if not callable(fn):
        return "exit test: not supported"
    
    return f"Paper exit OK: {fn()}"


def cmd_dryrun(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
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
        except Exception as e:
            __import__("logging").getLogger(__name__).exception("[CRITICAL] unhandled exception", exc_info=True)
            raise
            
    mdm_connected = False
    if services.market_data_manager:
        mdm_connected = bool(getattr(services.market_data_manager, "ws_connected", False))
        
    return (
        f"diag: broker={broker_status} | "
        f"positions={pos_count} | "
        f"mdm.connected={mdm_connected}"
    )


def cmd_uptime(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    metrics = services.metrics
    if not metrics:
        return "uptime: metrics unavailable"
    
    fn = getattr(metrics, "uptime_hhmmss", None)
    if callable(fn):
        return str(fn())
    return "uptime: n/a"


def cmd_limits(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    cfg = services.config
    if not cfg:
        return "limits: n/a"
    return str(getattr(cfg, "limits", "n/a"))


def cmd_net(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    metrics = services.metrics
    if not metrics:
        return "net: n/a"
    
    fn = getattr(metrics, "net_latency_ms", None)
    if callable(fn):
        try:
            return f"api latency: {float(fn()):.0f}ms"
        except Exception as e:
            __import__("logging").getLogger(__name__).exception("[CRITICAL] unhandled exception", exc_info=True)
            raise
    return "net: n/a"


def cmd_save(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    journal = services.journal
    if not journal:
        return "save: journal unavailable"
    
    fn = getattr(journal, "save_snapshot", None)
    if callable(fn):
        return f"saved: {fn()}"
    return "save: not supported"


def cmd_book(update: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    sym = _extract_argument(update).upper()
    if not sym:
        return "Usage: /book <SYMBOL>"
    
    mdm = services.market_data_manager
    if not mdm:
        return "MDM unavailable"
        
    if hasattr(mdm, "get_orderbook"):
        snap = mdm.get_orderbook(sym)
        if snap:
            # Formatting usually belongs in controller but basic dump here
            return str(snap)
            
    return "Orderbook unavailable"


def cmd_ohlc(update: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    sym = _extract_argument(update).upper()
    if not sym:
        return "Usage: /ohlc <SYMBOL>"
    
    if services.market_data_manager and hasattr(services.market_data_manager, "get_ohlc"):
        res = services.market_data_manager.get_ohlc(sym)
        return str(res) if res else "n/a"
    return "MDM unavailable"


def cmd_chain(update: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /chain <ROOT> (e.g. NIFTY)"
    
    if services.market_data_manager and hasattr(services.market_data_manager, "get_option_chain"):
        try:
            chain = services.market_data_manager.get_option_chain(arg)
            if chain:
                return f"Chain fetched: {len(chain)} contracts."
        except Exception as e:
            return f"Chain error: {e}"
    return "Chain unavailable"


def cmd_atm(update: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /atm <ROOT>"
    
    if services.market_data_manager and hasattr(services.market_data_manager, "get_atm_option"):
        return str(services.market_data_manager.get_atm_option(arg))
    return "ATM unavailable"


def cmd_spot(update: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /spot <ROOT>"
    
    if services.market_data_manager and hasattr(services.market_data_manager, "get_spot"):
        return str(services.market_data_manager.get_spot(arg))
    return "Spot unavailable"


def cmd_greeks(update: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /greeks <SYMBOL>"
    
    if services.market_data_manager and hasattr(services.market_data_manager, "get_greeks"):
        return str(services.market_data_manager.get_greeks(arg))
    return "Greeks unavailable"


def cmd_iv(update: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /iv <ROOT>"
    
    if services.market_data_manager and hasattr(services.market_data_manager, "get_iv_trend"):
        return str(services.market_data_manager.get_iv_trend(arg))
    return "IV unavailable"


def cmd_prevclose(update: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /prevclose <SYMBOL>"
    
    if services.market_data_manager and hasattr(services.market_data_manager, "get_prev_close"):
        return str(services.market_data_manager.get_prev_close(arg))
    return "PrevClose unavailable"


def cmd_session(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    if services.market_data_manager and hasattr(services.market_data_manager, "get_market_session"):
        return str(services.market_data_manager.get_market_session())
    return "Session info unavailable"


def cmd_holiday(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    if services.market_data_manager and hasattr(services.market_data_manager, "next_holiday"):
        return str(services.market_data_manager.next_holiday())
    return "Holiday info unavailable"


def cmd_sig(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    if services.strategy_runner and hasattr(services.strategy_runner, "current_signal"):
        return str(services.strategy_runner.current_signal())
    return "Signal unavailable"


def cmd_state(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    if services.strategy_runner and hasattr(services.strategy_runner, "state_snapshot"):
        return str(services.strategy_runner.state_snapshot())
    return "State unavailable"


def cmd_score(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    if services.strategy_runner and hasattr(services.strategy_runner, "composite_score"):
        val = services.strategy_runner.composite_score()
        return f"score={val}" if val is not None else "score: n/a"
    return "Score unavailable"


def cmd_regime(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    detector = services.market_regime
    if not detector:
        return "Regime detector unavailable"
    
    try:
        snap = detector.latest(limit=1)
        if snap:
            s = snap[0]
            return f"{s.symbol}: {s.regime} ({s.confidence:.2f})"
    except Exception as e:
        __import__("logging").getLogger(__name__).exception("[CRITICAL] unhandled exception", exc_info=True)
        raise
    return "Regime unknown"


def cmd_gate_why(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    if services.strategy_runner and hasattr(services.strategy_runner, "gate_reason"):
        return str(services.strategy_runner.gate_reason())
    return "Gate info unavailable"


def cmd_size(update: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    arg = _extract_argument(update)
    if not arg:
        return "Usage: /size ..."
    
    rm = services.risk_manager
    if rm and hasattr(rm, "suggest_position_size"):
        return "See /size logic in controller"
    return "Sizing unavailable"


def cmd_trail(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    om = services.order_manager
    if om and hasattr(om, "trailing_snapshot"):
        return str(om.trailing_snapshot())
    return "Trail info unavailable"


def cmd_riskstate(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    rm = services.risk_manager
    if not rm:
        return "Risk manager unavailable"
    
    breaker = getattr(rm, "breaker_tripped", False)
    reason = getattr(rm, "last_reason", "none")
    return f"Risk: breaker={breaker}, last_reason={reason}"


def cmd_journal_read(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    j = services.journal
    if j and hasattr(j, "last_entries"):
        entries = j.last_entries(limit=5)
        return "\n".join(str(e) for e in entries) if entries else "No entries"
    return "Journal unavailable"


def cmd_execstate(update: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    sym = _extract_argument(update).upper()
    if not sym:
        return "Usage: /execstate <SYMBOL>"
    
    om = services.order_manager
    if om and hasattr(om, "get_execution_state"):
        return str(om.get_execution_state(sym))
    if om and hasattr(om, "recent_orders"):
        recent = om.recent_orders(limit=10)
        matching = [o for o in recent if str(o.get("symbol", "")).upper() == sym]
        return str(matching[:3]) if matching else f"No recent execution state for {sym}"
    return "OrderManager unavailable"


def cmd_execqueue(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    om = services.order_manager
    if om and hasattr(om, "get_pending_orders"):
        return str(om.get_pending_orders())
    if om and hasattr(om, "recent_orders"):
        recent = om.recent_orders(limit=10)
        pending = [o for o in recent if str(o.get("status", "")).lower() in {"pending", "submitted", "open"}]
        return f"Pending/open orders: {len(pending)}"
    return "Queue unavailable"


def cmd_execlast(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    om = services.order_manager
    if om and hasattr(om, "get_execution_stats"):
        return str(om.get_execution_stats())
    if om and hasattr(om, "recent_orders"):
        return str(om.recent_orders(limit=5))
    return "Execution stats unavailable"


def cmd_execwhy(update: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    sym = _extract_argument(update).upper()
    if not sym:
        return "Usage: /execwhy <SYMBOL>"
        
    om = services.order_manager
    if om and hasattr(om, "explain_preflight"):
        return str(om.explain_preflight(sym))
    return "OrderManager preflight diagnostics unavailable"


def cmd_emergencystop(_u: TelegramUpdate, _c: TelegramContextTypes.DEFAULT_TYPE, services: Services) -> str:
    om = services.order_manager
    if om and hasattr(om, "emergency_stop"):
        res = om.emergency_stop(reason="telegram")
        return f"STOP triggered: {res}"
    if om and hasattr(om, "engage_kill_switch"):
        om.engage_kill_switch("telegram_emergency_stop")
        return "STOP triggered: order kill switch engaged"
    return "Emergency stop unavailable"


# --- REGISTRATION ---

def _command_exists(application: TelegramApplication, command: str) -> bool:
    """Check if a command handler is already registered."""
    command_handler_cls = _load_command_handler()
    if command_handler_cls is None:
        return False
    for handlers in application.handlers.values():
        for handler in handlers:
            if isinstance(handler, command_handler_cls) and command in handler.commands:
                return True
    return False


def register_telegram_commands(bot: Any, application: TelegramApplication, services: Services) -> bool:
    """Register production commands with security wrapping."""
    command_handler_cls = _load_command_handler()
    if command_handler_cls is None:
        return False

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
            application.add_handler(command_handler_cls(name, handler))

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
            application.add_handler(
                command_handler_cls(name, _wrap_command(services, func, name))
            )
    return True
