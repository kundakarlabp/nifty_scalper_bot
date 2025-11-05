from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from nifty_scalper_bot.notifications.telegram_utils import redact, reply_err, reply_ok
from nifty_scalper_bot.utils.logging import get_logger

LOG = get_logger(__name__)


@dataclass(slots=True)
class Services:
    """Bundle of trading services exposed to Telegram commands.

    Args:
        order_manager: Order manager handling live and paper orders.
        risk_manager: Risk manager exposing guard state.
        market_data: Market data access layer.
        strategy_runner: Strategy runner managing signals and plans.
        config: Live configuration object.
        broker: Broker client integration.
        journal: Journal interface for persisting snapshots.
        metrics: Metrics collector for runtime statistics.
        market_regime: Market regime detector used for diagnostics.
        order_execution_hub: Execution hub providing stats and controls.
        order_queue: Order queue for diagnostics.
        state_tracker: State tracker exposing execution state.
        preflight_validator: Preflight validator used for diagnostics.
        version_info: Optional build metadata dictionary.

    Returns:
        None.

    Raises:
        None.
    """

    order_manager: Any | None
    risk_manager: Any | None
    market_data: Any | None
    strategy_runner: Any | None
    config: Any | None
    broker: Any | None
    journal: Any | None
    metrics: Any | None
    market_regime: Any | None
    order_execution_hub: Any | None = None
    order_queue: Any | None = None
    state_tracker: Any | None = None
    preflight_validator: Any | None = None
    version_info: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Populate default version metadata when missing.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        if self.version_info is None:
            self.version_info = {"build": "unknown", "sha": "unknown"}


CommandFunc = Callable[[Update, ContextTypes.DEFAULT_TYPE, Services], str]


def _wrap_command(
    services: Services, command: CommandFunc, name: str
) -> Callable[[Update, ContextTypes.DEFAULT_TYPE], Any]:
    """Convert a synchronous *command* into an async Telegram handler.

    Args:
        services: Shared services container.
        command: Command function producing a message payload.
        name: Logging-friendly command name.

    Returns:
        Async handler compatible with python-telegram-bot.

    Raises:
        None.
    """

    async def _inner(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        LOG.debug("Entered %s handler", name)
        try:
            message = command(update, context, services)
            if message:
                LOG.info("%s produced response length=%d", name, len(message))
                await reply_ok(update, message)
            else:
                LOG.info("%s produced empty response", name)
                await reply_ok(update, "Done.")
        except Exception as exc:  # noqa: BLE001 - defensive
            LOG.error("Failure in %s: %s", name, exc)
            await reply_err(update, f"Error: {redact(str(exc))}")

    return _inner


def _message_text(update: Update) -> str:
    """Return the raw text for *update* when available.

    Args:
        update: Telegram update containing the command message.

    Returns:
        Message text or an empty string when the payload is missing.

    Raises:
        None.
    """

    LOG.debug("Entered _message_text")
    message = update.effective_message
    if message is None or message.text is None:
        LOG.info("_message_text found no text payload")
        return ""
    return message.text.strip()


def _extract_argument(update: Update) -> str:
    """Return the argument string supplied with a command.

    Args:
        update: Telegram update containing the command payload.

    Returns:
        Argument portion of the message excluding the command keyword.

    Raises:
        None.
    """

    LOG.debug("Entered _extract_argument")
    text = _message_text(update)
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        LOG.info("_extract_argument found no arguments")
        return ""
    return parts[1].strip()


def cmd_mode(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Summarise current trading mode for quick inspection.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing configuration.

    Returns:
        Short human readable summary of live/shadow flags.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_mode")
    cfg = services.config
    if cfg is None:
        LOG.info("cmd_mode missing configuration reference")
        return "config unavailable"
    live = bool(getattr(cfg, "live", getattr(cfg, "allow_live", False)))
    shadow = bool(getattr(cfg, "shadow_mode", getattr(cfg, "shadow", False)))
    LOG.info("cmd_mode resolved live=%s shadow=%s", live, shadow)
    return f"mode: {'LIVE' if live else 'PAPER'} | shadow={shadow}"


def cmd_live(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Toggle the live trading flag when configuration allows.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing configuration.

    Returns:
        Confirmation of the attempted toggle or current state when blocked.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_live")
    cfg = services.config
    if cfg is None:
        LOG.info("cmd_live missing configuration reference")
        return "config unavailable"
    current = bool(getattr(cfg, "live", getattr(cfg, "allow_live", False)))
    try:
        setattr(cfg, "live", not current)
        LOG.info("cmd_live toggled live=%s", not current)
        return f"live set to: {not current}"
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_live: %s", exc)
        return f"live: {current} (toggle not permitted)"


def cmd_flat(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Request a full flatten of open positions via the order manager.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing order management.

    Returns:
        Status text describing whether the flatten request was dispatched.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_flat")
    manager = services.order_manager
    if manager is None:
        LOG.info("cmd_flat missing order manager")
        return "flatten: order manager unavailable"
    fn = getattr(manager, "flatten_all", None) or getattr(
        manager, "close_all_positions", None
    )
    if not callable(fn):
        LOG.info("cmd_flat flatten hook not available")
        return "flatten: not supported"
    try:
        fn()
        LOG.info("cmd_flat invoked flatten function")
        return "Requested: close all positions."
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_flat: %s", exc)
        return "flatten request failed"


def cmd_brk(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return broker connectivity status snapshot.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing broker client.

    Returns:
        Broker connectivity summary string.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_brk")
    broker = services.broker
    if broker is None:
        LOG.info("cmd_brk missing broker client")
        return "broker unavailable"
    name = str(getattr(broker, "name", "broker"))
    ok = True
    try:
        ping = getattr(broker, "ping", None)
        if callable(ping):
            ok = bool(ping())
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_brk ping: %s", exc)
        ok = False
    LOG.info("cmd_brk status ok=%s", ok)
    return f'{name}: {"OK" if ok else "UNREACHABLE"}'


def cmd_entry(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Trigger strategy entry dry-run if supported.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing strategy runner.

    Returns:
        Text describing the dry-run entry result or lack of support.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_entry")
    runner = services.strategy_runner
    if runner is None:
        LOG.info("cmd_entry missing strategy runner")
        return "entry test not supported"
    fn = getattr(runner, "test_entry", None)
    if not callable(fn):
        LOG.info("cmd_entry strategy runner lacks test_entry")
        return "entry test not supported"
    try:
        result = fn()
        LOG.info("cmd_entry executed test_entry")
        return f"Paper entry OK: {result}"
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_entry: %s", exc)
        return "entry failed"


def cmd_exit(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Trigger strategy exit dry-run if supported.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing strategy runner.

    Returns:
        Text describing the dry-run exit result or lack of support.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_exit")
    runner = services.strategy_runner
    if runner is None:
        LOG.info("cmd_exit missing strategy runner")
        return "exit test not supported"
    fn = getattr(runner, "test_exit", None)
    if not callable(fn):
        LOG.info("cmd_exit strategy runner lacks test_exit")
        return "exit test not supported"
    try:
        result = fn()
        LOG.info("cmd_exit executed test_exit")
        return f"Paper exit OK: {result}"
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_exit: %s", exc)
        return "exit failed"


def cmd_dryrun(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Execute a single dry-run simulation if supported by the runner.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing strategy runner.

    Returns:
        Simulation result string or unsupported notification.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_dryrun")
    runner = services.strategy_runner
    if runner is None:
        LOG.info("cmd_dryrun missing strategy runner")
        return "dryrun: not supported"
    fn = getattr(runner, "simulate_once", None)
    if not callable(fn):
        LOG.info("cmd_dryrun simulate_once unavailable")
        return "dryrun: not supported"
    try:
        result = fn()
        LOG.info("cmd_dryrun executed simulation")
        return f"dryrun complete: {result}"
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_dryrun: %s", exc)
        return "dryrun failed"


def cmd_diag(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return consolidated diagnostic snapshot across broker and risk.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing runtime components.

    Returns:
        Human readable diagnostics summary.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_diag")
    broker_status = cmd_brk(_update, _context, services)
    order_manager = services.order_manager
    positions_count = 0
    if order_manager is not None:
        try:
            getter = getattr(order_manager, "get_all_positions", None)
            if callable(getter):
                positions = getter() or []
            else:
                positions = getattr(order_manager, "get_open_positions", lambda: [])()
            positions_count = len(list(positions))
        except Exception as exc:  # noqa: BLE001 - defensive
            LOG.error("Failure in cmd_diag positions: %s", exc)
    mdm_connected = False
    if services.market_data is not None:
        mdm_connected = bool(getattr(services.market_data, "connected", False))
    LOG.info(
        "cmd_diag summary positions=%s mdm_connected=%s", positions_count, mdm_connected
    )
    return (
        "diag: "
        f"broker={broker_status} | "
        f"positions={positions_count} | "
        f"mdm.connected={mdm_connected}"
    )


def cmd_uptime(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return formatted uptime when metrics expose the helper.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing metrics.

    Returns:
        Uptime string or fallback when unavailable.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_uptime")
    metrics = services.metrics
    if metrics is None:
        LOG.info("cmd_uptime metrics unavailable")
        return "uptime: n/a"
    fn = getattr(metrics, "uptime_hhmmss", None)
    if not callable(fn):
        LOG.info("cmd_uptime helper missing")
        return "uptime: n/a"
    try:
        value = str(fn())
        LOG.info("cmd_uptime value=%s", value)
        return value
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_uptime: %s", exc)
        return "uptime: n/a"


def cmd_limits(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Show configured trading time limits when available.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing configuration.

    Returns:
        Limits representation or fallback string.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_limits")
    cfg = services.config
    if cfg is None:
        LOG.info("cmd_limits config unavailable")
        return "limits: n/a"
    limits = getattr(cfg, "limits", None)
    LOG.info("cmd_limits found limits=%s", limits)
    return str(limits) if limits is not None else "limits: n/a"


def cmd_net(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Report API network latency when metrics expose it.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing metrics.

    Returns:
        Network latency summary string.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_net")
    metrics = services.metrics
    if metrics is None:
        LOG.info("cmd_net metrics unavailable")
        return "net: n/a"
    fn = getattr(metrics, "net_latency_ms", None)
    if not callable(fn):
        LOG.info("cmd_net latency helper missing")
        return "net: n/a"
    try:
        latency = float(fn())
        LOG.info("cmd_net latency_ms=%.2f", latency)
        return f"api latency: {latency:.0f}ms"
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_net: %s", exc)
        return "net: n/a"


def cmd_save(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Persist a journal snapshot if the interface is present.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing journal functionality.

    Returns:
        Path or confirmation of the saved snapshot.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_save")
    journal = services.journal
    if journal is None:
        LOG.info("cmd_save journal unavailable")
        return "save: n/a"
    fn = getattr(journal, "save_snapshot", None)
    if not callable(fn):
        LOG.info("cmd_save journal lacks save_snapshot")
        return "save: n/a"
    try:
        path = fn()
        LOG.info("cmd_save snapshot saved path=%s", path)
        return f"saved: {path}"
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_save: %s", exc)
        return "save failed"


def _format_object(value: Any) -> str:
    """Return a trimmed string representation of *value*.

    Args:
        value: Arbitrary object needing textual representation.

    Returns:
        Human friendly string limited to a reasonable length.

    Raises:
        None.
    """

    LOG.debug("Entered _format_object type=%s", type(value).__name__)
    if value is None:
        return "n/a"
    if hasattr(value, "to_dict"):
        try:
            return str(value.to_dict())
        except Exception:  # noqa: BLE001 - defensive
            return str(value)
    if hasattr(value, "head"):
        try:
            return str(value.head(5))
        except Exception:  # noqa: BLE001 - defensive
            return str(value)
    text = str(value)
    return text[:4096]


def _parse_chain_argument(argument: str) -> tuple[str, str]:
    """Return underlying and expiry token parsed from *argument*.

    Args:
        argument: Raw command argument supplied by the user.

    Returns:
        Tuple of (underlying, expiry_token) normalised for lookups.

    Raises:
        None.
    """

    LOG.debug("Entered _parse_chain_argument argument=%s", argument)
    try:
        token = (argument or "").strip()
        if not token:
            return "NIFTY", "weekly"
        if ":" in token:
            underlying_raw, expiry_raw = token.split(":", maxsplit=1)
            underlying = underlying_raw.strip().upper() or "NIFTY"
            expiry = expiry_raw.strip() or "weekly"
            return underlying, expiry
        return "NIFTY", token
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in _parse_chain_argument: %s", exc)
        return "NIFTY", "weekly"


def _safe_float(value: Any) -> float | None:
    """Return floating-point representation of *value* when valid.

    Args:
        value: Input potentially representing a numeric quantity.

    Returns:
        Floating-point number when coercion succeeds, else ``None``.

    Raises:
        None.
    """

    try:
        if value is None:
            return None
        number = float(value)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.debug("_safe_float failed value=%s err=%s", value, exc)
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_positive(value: Any) -> float | None:
    """Return positive finite float derived from *value* when possible.

    Args:
        value: Input potentially representing a numeric quantity.

    Returns:
        Positive float greater than zero, otherwise ``None``.

    Raises:
        None.
    """

    number = _safe_float(value)
    if number is None or number <= 0:
        return None
    return number


def _depth_quantity(entry: Mapping[str, Any], side: str) -> float:
    """Return the top-of-book quantity for *side* from *entry*.

    Args:
        entry: Option chain entry containing depth information.
        side: Either ``"buy"`` or ``"sell"`` selecting the depth array.

    Returns:
        Quantity at the first depth level or ``0.0`` when unavailable.

    Raises:
        None.
    """

    LOG.debug(
        "Entered _depth_quantity side=%s has_depth=%s",
        side,
        isinstance(entry.get("depth"), Mapping),
    )
    try:
        depth = entry.get("depth")
        if not isinstance(depth, Mapping):
            return 0.0
        levels = depth.get(side)
        if not isinstance(levels, Sequence):
            return 0.0
        for level in levels:
            if not isinstance(level, Mapping):
                continue
            quantity = _safe_positive(
                level.get("quantity")
                or level.get("qty")
                or level.get("volume")
                or level.get("size")
            )
            if quantity is not None:
                return quantity
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.debug("_depth_quantity failed side=%s err=%s", side, exc)
        return 0.0
    return 0.0


def _extract_iv(entry: Mapping[str, Any]) -> float | None:
    """Return implied volatility value from *entry* when present.

    Args:
        entry: Option chain entry from the market data manager.

    Returns:
        Float IV value or ``None`` when missing or invalid.

    Raises:
        None.
    """

    LOG.debug(
        "Entered _extract_iv symbol=%s",
        entry.get("tradingsymbol") or entry.get("symbol"),
    )
    try:
        candidates: tuple[Any, ...] = (
            entry.get("iv"),
            entry.get("implied_volatility"),
        )
        for candidate in candidates:
            value = _safe_positive(candidate)
            if value is not None:
                return value
        raw = entry.get("raw")
        if isinstance(raw, Mapping):
            for key in ("iv", "implied_volatility", "iv_percentile"):
                value = _safe_positive(raw.get(key))
                if value is not None:
                    return value
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.debug("_extract_iv failed err=%s", exc)
        return None
    return None


def _percentile_rank(
    sorted_values: Sequence[float], value: float | None
) -> float | None:
    """Return percentile rank of *value* within *sorted_values*.

    Args:
        sorted_values: Sorted numeric sequence used for ranking.
        value: Target value whose percentile is required.

    Returns:
        Percentile value between 0 and 100, or ``None`` when undefined.

    Raises:
        None.
    """

    LOG.debug(
        "Entered _percentile_rank count=%d value=%s",
        len(sorted_values),
        value,
    )
    try:
        if value is None or not sorted_values:
            return None
        if not math.isfinite(value):
            return None
        index = bisect_right(sorted_values, value)
        percentile = (index / len(sorted_values)) * 100.0
        return round(percentile, 2)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in _percentile_rank: %s", exc)
        return None


def _format_chain_summary(
    chain: Sequence[Mapping[str, Any]], underlying: str, expiry_label: str
) -> str:
    """Return formatted option chain analytics summary.

    Args:
        chain: Normalised option chain entries for the expiry window.
        underlying: Underlying instrument symbol.
        expiry_label: Expiry identifier requested by the user.

    Returns:
        Human readable summary with OI, volume, IV, and liquidity metrics.

    Raises:
        None.
    """

    LOG.debug(
        "Entered _format_chain_summary underlying=%s expiry=%s size=%d",
        underlying,
        expiry_label,
        len(chain),
    )
    try:
        grouped: dict[float, dict[str, Any]] = defaultdict(
            lambda: {
                "ce": {
                    "oi": 0.0,
                    "volume": 0.0,
                    "iv": None,
                    "bid": None,
                    "ask": None,
                    "spread": None,
                    "qty": 0.0,
                },
                "pe": {
                    "oi": 0.0,
                    "volume": 0.0,
                    "iv": None,
                    "bid": None,
                    "ask": None,
                    "spread": None,
                    "qty": 0.0,
                },
                "expiry": None,
            }
        )
        ce_iv_values: list[float] = []
        pe_iv_values: list[float] = []
        for entry in chain:
            strike = _safe_float(entry.get("strike"))
            option_type = str(entry.get("option_type") or "").upper()
            if strike is None or option_type not in {"CE", "PE"}:
                continue
            bucket = grouped[strike]
            bucket["expiry"] = entry.get("expiry") or bucket.get("expiry")
            side = "ce" if option_type == "CE" else "pe"
            oi_value = _safe_float(entry.get("open_interest"))
            if oi_value is None:
                raw_entry = entry.get("raw")
                raw = raw_entry if isinstance(raw_entry, Mapping) else None
                if isinstance(raw, Mapping):
                    oi_value = _safe_float(raw.get("open_interest") or raw.get("oi"))
            if oi_value is not None:
                bucket[side]["oi"] = oi_value
            volume_value = _safe_float(entry.get("trades"))
            if volume_value is None:
                volume_value = _safe_float(entry.get("volume"))
            if volume_value is not None:
                bucket[side]["volume"] = volume_value
            bid_value = _safe_float(entry.get("bid"))
            ask_value = _safe_float(entry.get("ask"))
            bucket[side]["bid"] = bid_value
            bucket[side]["ask"] = ask_value
            if bid_value is not None and ask_value is not None:
                bucket[side]["spread"] = max(0.0, ask_value - bid_value)
            qty_value = _depth_quantity(entry, "buy" if side == "ce" else "sell")
            if qty_value > 0:
                bucket[side]["qty"] = qty_value
            iv_value = _extract_iv(entry)
            if iv_value is not None:
                bucket[side]["iv"] = iv_value
                if side == "ce":
                    ce_iv_values.append(iv_value)
                else:
                    pe_iv_values.append(iv_value)

        aggregated: list[dict[str, Any]] = []
        total_oi = 0.0
        total_volume = 0.0
        spreads: list[float] = []
        ce_iv_values.sort()
        pe_iv_values.sort()
        for strike, data in sorted(grouped.items()):
            ce_data = data["ce"]
            pe_data = data["pe"]
            total = (ce_data["oi"] or 0.0) + (pe_data["oi"] or 0.0)
            volume_total = (ce_data["volume"] or 0.0) + (pe_data["volume"] or 0.0)
            total_oi += total
            total_volume += volume_total
            if ce_data.get("spread") is not None:
                spreads.append(float(ce_data["spread"]))
            if pe_data.get("spread") is not None:
                spreads.append(float(pe_data["spread"]))
            ce_iv_pct = _percentile_rank(ce_iv_values, ce_data.get("iv"))
            pe_iv_pct = _percentile_rank(pe_iv_values, pe_data.get("iv"))
            aggregated.append(
                {
                    "strike": strike,
                    "ce": ce_data,
                    "pe": pe_data,
                    "total_oi": total,
                    "total_volume": volume_total,
                    "liquidity": (ce_data.get("qty", 0.0) + pe_data.get("qty", 0.0)),
                    "ce_iv_pct": ce_iv_pct,
                    "pe_iv_pct": pe_iv_pct,
                    "expiry": data.get("expiry"),
                }
            )

        if not aggregated:
            return "option chain unavailable"
        expiry_dt = next(
            (item["expiry"] for item in aggregated if item.get("expiry") is not None),
            None,
        )
        expiry_text = (
            str(expiry_dt) if expiry_dt is not None else expiry_label or "weekly"
        )
        header = f"{underlying} {expiry_text}"
        lines = [header, f"strikes={len(aggregated)} entries={len(chain)}"]
        lines.append(f"total_oi={total_oi:.0f} total_volume={total_volume:.0f}")
        avg_spread = sum(spreads) / len(spreads) if spreads else None
        if avg_spread is not None:
            lines.append(f"avg_spread={avg_spread:.2f}")

        top_oi = sorted(aggregated, key=lambda item: item["total_oi"], reverse=True)[:3]
        if top_oi:
            lines.append("Max OI:")
            for item in top_oi:
                ce_oi = item["ce"]["oi"] or 0.0
                pe_oi = item["pe"]["oi"] or 0.0
                lines.append(
                    f"  {item['strike']:.0f}: {item['total_oi']:.0f} "
                    f"(CE={ce_oi:.0f} PE={pe_oi:.0f})"
                )

        top_volume = sorted(
            aggregated, key=lambda item: item["total_volume"], reverse=True
        )[:5]
        if top_volume:
            lines.append("Volume heatmap:")
            for item in top_volume:
                ce_vol = item["ce"]["volume"] or 0.0
                pe_vol = item["pe"]["volume"] or 0.0
                ce_pct = item.get("ce_iv_pct")
                pe_pct = item.get("pe_iv_pct")
                iv_label = (
                    f"IV% CE={ce_pct if ce_pct is not None else '-'} "
                    f"PE={pe_pct if pe_pct is not None else '-'}"
                )
                volume_line = (
                    f"  {item['strike']:.0f}: vol CE={ce_vol:.0f} "
                    f"PE={pe_vol:.0f} {iv_label}"
                )
                lines.append(volume_line)

        top_liquidity = sorted(
            aggregated, key=lambda item: item["liquidity"], reverse=True
        )[:3]
        if top_liquidity:
            lines.append("Liquidity focus:")
            for item in top_liquidity:
                spreads_available = [
                    spread
                    for spread in (
                        item["ce"].get("spread"),
                        item["pe"].get("spread"),
                    )
                    if spread is not None
                ]
                best_spread = min(spreads_available) if spreads_available else None
                spread_label = (
                    f"{best_spread:.2f}" if best_spread is not None else "n/a"
                )
                lines.append(
                    f"  {item['strike']:.0f}: depth_qty={item['liquidity']:.0f} "
                    f"spread(min)={spread_label}"
                )

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in _format_chain_summary: %s", exc)
        return "option chain formatting failed"


def _format_orderbook_snapshot(symbol: str, snapshot: Mapping[str, Any]) -> str:
    """Return formatted order book summary for *symbol*.

    Args:
        symbol: Normalised trading symbol for the requested order book.
        snapshot: Aggregated order book metrics from the data manager.

    Returns:
        Human readable summary including spread and depth metrics.

    Raises:
        None.
    """

    LOG.debug("Entered _format_orderbook_snapshot symbol=%s", symbol)
    try:
        levels = int(snapshot.get("levels", 0) or 0)
        best_bid = snapshot.get("best_bid")
        best_ask = snapshot.get("best_ask")
        spread = snapshot.get("spread")
        mid = snapshot.get("mid")
        bid_value = _safe_float(best_bid)
        ask_value = _safe_float(best_ask)
        bid_label = f"{bid_value:.2f}" if bid_value is not None else "n/a"
        ask_label = f"{ask_value:.2f}" if ask_value is not None else "n/a"
        spread_value = _safe_float(spread)
        spread_label = f"{spread_value:.2f}" if spread_value is not None else "n/a"
        mid_value = _safe_float(mid)
        mid_label = f"{mid_value:.2f}" if mid_value is not None else "n/a"
        total_bid_qty = snapshot.get("total_bid_qty") or 0.0
        total_ask_qty = snapshot.get("total_ask_qty") or 0.0
        liquidity_score = snapshot.get("liquidity_score")
        raw_buy = snapshot.get("buy")
        raw_sell = snapshot.get("sell")
        buy_levels: Sequence[Any] = raw_buy if isinstance(raw_buy, Sequence) else ()
        sell_levels: Sequence[Any] = raw_sell if isinstance(raw_sell, Sequence) else ()

        def _format_levels(levels_payload: Sequence[Any]) -> str:
            rows: list[str] = []
            for index, level in enumerate(levels_payload, start=1):
                if not isinstance(level, Mapping):
                    continue
                price = _safe_float(level.get("price"))
                quantity = _safe_float(level.get("quantity"))
                if price is None:
                    continue
                qty_label = f"{quantity:.0f}" if quantity is not None else "0"
                rows.append(f"{index}:{price:.2f}@{qty_label}")
            return " ".join(rows) if rows else "n/a"

        liquidity_label = liquidity_score if liquidity_score is not None else "n/a"
        lines = [
            f"{symbol} order book (levels={levels})",
            f"bid={bid_label} ask={ask_label} spread={spread_label} mid={mid_label}",
            (
                f"depth_qty bid={total_bid_qty:.0f} "
                f"ask={total_ask_qty:.0f} "
                f"liquidity_score={liquidity_label}"
            ),
            f"buy: {_format_levels(buy_levels)}",
            f"sell: {_format_levels(sell_levels)}",
        ]
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in _format_orderbook_snapshot: %s", exc)
        return "order book formatting failed"


def _market_data_call(services: Services, method: str, argument: str | None) -> str:
    """Invoke market-data *method* safely with optional *argument*.

    Args:
        services: Service container exposing market data.
        method: Attribute name to call on the market data object.
        argument: Optional string argument for the call.

    Returns:
        String representation of the fetched payload or fallback text.

    Raises:
        None.
    """

    LOG.debug("Entered _market_data_call method=%s argument=%s", method, argument)
    mdm = services.market_data
    if mdm is None:
        LOG.info("_market_data_call market data unavailable")
        return "market data unavailable"
    fn = getattr(mdm, method, None)
    if not callable(fn):
        LOG.info("_market_data_call method missing=%s", method)
        return f"{method}: not supported"
    try:
        result = fn(argument) if argument is not None else fn()
        LOG.info("_market_data_call success method=%s", method)
        return _format_object(result) if result is not None else "n/a"
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in _market_data_call method=%s err=%s", method, exc)
        return f"{method} failed"


def cmd_book(
    update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return current order book for the supplied symbol.

    Args:
        update: Telegram update containing the symbol argument.
        _context: Handler context (unused).
        services: Service container exposing market data.

    Returns:
        String describing the book snapshot.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_book")
    argument = _extract_argument(update)
    if not argument:
        LOG.info("cmd_book missing symbol argument")
        return "Usage: /book <SYMBOL>"
    symbol = argument.strip().upper()
    market_data = services.market_data
    if market_data is None:
        LOG.info("cmd_book market data unavailable symbol=%s", symbol)
        return "order book unavailable"
    try:
        snapshot = market_data.get_orderbook(symbol)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_book fetch symbol=%s err=%s", symbol, exc)
        return "order book unavailable"
    if not snapshot:
        LOG.info("cmd_book snapshot missing symbol=%s", symbol)
        return "order book unavailable"
    summary = _format_orderbook_snapshot(symbol, snapshot)
    LOG.info("cmd_book summary_ready symbol=%s", symbol)
    return summary


def cmd_ohlc(
    update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return recent OHLC data for the supplied symbol.

    Args:
        update: Telegram update containing the symbol argument.
        _context: Handler context (unused).
        services: Service container exposing market data.

    Returns:
        String describing the OHLC snapshot.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_ohlc")
    argument = _extract_argument(update)
    if not argument:
        LOG.info("cmd_ohlc missing symbol argument")
        return "Usage: /ohlc <SYMBOL>"
    return _market_data_call(services, "get_ohlc", argument)


def cmd_chain(
    update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return option chain data for the supplied root symbol.

    Args:
        update: Telegram update containing the root symbol.
        _context: Handler context (unused).
        services: Service container exposing market data.

    Returns:
        String describing the option chain snapshot.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_chain")
    argument = _extract_argument(update)
    if not argument:
        LOG.info("cmd_chain missing root argument")
        return "Usage: /chain <ROOT>"
    underlying, expiry_token = _parse_chain_argument(argument)
    market_data = services.market_data
    if market_data is None:
        LOG.info(
            "cmd_chain market data unavailable underlying=%s expiry=%s",
            underlying,
            expiry_token,
        )
        return "option chain unavailable"
    try:
        chain = market_data.get_option_chain(expiry_token, underlying=underlying)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error(
            "Failure in cmd_chain fetch underlying=%s expiry=%s err=%s",
            underlying,
            expiry_token,
            exc,
        )
        return "option chain unavailable"
    if not chain:
        LOG.info(
            "cmd_chain chain missing underlying=%s expiry=%s",
            underlying,
            expiry_token,
        )
        return "option chain unavailable"
    summary = _format_chain_summary(chain, underlying, expiry_token)
    LOG.info(
        "cmd_chain summary_ready underlying=%s expiry=%s size=%d",
        underlying,
        expiry_token,
        len(chain),
    )
    return summary


def cmd_atm(
    update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return at-the-money option detail for supplied root symbol.

    Args:
        update: Telegram update containing the root symbol.
        _context: Handler context (unused).
        services: Service container exposing market data.

    Returns:
        String describing the ATM option snapshot.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_atm")
    argument = _extract_argument(update)
    if not argument:
        LOG.info("cmd_atm missing root argument")
        return "Usage: /atm <ROOT>"
    return _market_data_call(services, "get_atm_option", argument)


def cmd_spot(
    update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return spot price for the supplied root symbol.

    Args:
        update: Telegram update containing the root symbol.
        _context: Handler context (unused).
        services: Service container exposing market data.

    Returns:
        Spot price string or fallback text.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_spot")
    argument = _extract_argument(update)
    if not argument:
        LOG.info("cmd_spot missing root argument")
        return "Usage: /spot <ROOT>"
    return _market_data_call(services, "get_spot", argument)


def cmd_greeks(
    update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return option Greeks for the supplied symbol.

    Args:
        update: Telegram update containing the option symbol.
        _context: Handler context (unused).
        services: Service container exposing market data.

    Returns:
        Greeks snapshot string or fallback text.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_greeks")
    argument = _extract_argument(update)
    if not argument:
        LOG.info("cmd_greeks missing symbol argument")
        return "Usage: /greeks <SYMBOL>"
    return _market_data_call(services, "get_greeks", argument)


def cmd_iv(
    update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return implied-volatility trend for the supplied root symbol.

    Args:
        update: Telegram update containing the root symbol.
        _context: Handler context (unused).
        services: Service container exposing market data.

    Returns:
        IV trend string or fallback text.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_iv")
    argument = _extract_argument(update)
    if not argument:
        LOG.info("cmd_iv missing root argument")
        return "Usage: /iv <ROOT>"
    return _market_data_call(services, "get_iv_trend", argument)


def cmd_prevclose(
    update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return previous close price for supplied symbol.

    Args:
        update: Telegram update containing the symbol.
        _context: Handler context (unused).
        services: Service container exposing market data.

    Returns:
        Previous close price string or fallback text.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_prevclose")
    argument = _extract_argument(update)
    if not argument:
        LOG.info("cmd_prevclose missing symbol argument")
        return "Usage: /prevclose <SYMBOL>"
    return _market_data_call(services, "get_prev_close", argument)


def cmd_session(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return current market session information if available.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing market data.

    Returns:
        Market session summary string or fallback text.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_session")
    mdm = services.market_data
    if mdm is None:
        LOG.info("cmd_session market data unavailable")
        return "session: n/a"
    fn = getattr(mdm, "get_market_session", None)
    if not callable(fn):
        LOG.info("cmd_session helper missing")
        return "session: n/a"
    try:
        result = fn()
        LOG.info("cmd_session obtained result")
        return _format_object(result)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_session: %s", exc)
        return "session: n/a"


def cmd_holiday(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return next trading holiday when supported.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing market data.

    Returns:
        Next holiday string or fallback text.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_holiday")
    mdm = services.market_data
    if mdm is None:
        LOG.info("cmd_holiday market data unavailable")
        return "holiday: n/a"
    fn = getattr(mdm, "next_holiday", None)
    if not callable(fn):
        LOG.info("cmd_holiday helper missing")
        return "holiday: n/a"
    try:
        result = fn()
        LOG.info("cmd_holiday obtained result")
        return _format_object(result)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_holiday: %s", exc)
        return "holiday: n/a"


def cmd_sig(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return current trading signal from the runner when exposed.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing strategy runner.

    Returns:
        Signal description string or fallback text.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_sig")
    runner = services.strategy_runner
    if runner is None:
        LOG.info("cmd_sig strategy runner unavailable")
        return "signal: n/a"
    fn = getattr(runner, "current_signal", None)
    if not callable(fn):
        LOG.info("cmd_sig current_signal missing")
        return "signal: n/a"
    try:
        result = fn()
        LOG.info("cmd_sig obtained signal=%s", result)
        return _format_object(result)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_sig: %s", exc)
        return "signal: n/a"


def cmd_state(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return strategy state snapshot when available.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing strategy runner.

    Returns:
        State snapshot string or fallback text.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_state")
    runner = services.strategy_runner
    if runner is None:
        LOG.info("cmd_state strategy runner unavailable")
        return "state: n/a"
    fn = getattr(runner, "state_snapshot", None)
    if not callable(fn):
        LOG.info("cmd_state state_snapshot missing")
        return "state: n/a"
    try:
        result = fn()
        LOG.info("cmd_state obtained snapshot")
        return _format_object(result)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_state: %s", exc)
        return "state: n/a"


def cmd_score(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return composite strategy score when exposed.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing strategy runner.

    Returns:
        Score string or fallback text.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_score")
    runner = services.strategy_runner
    if runner is None:
        LOG.info("cmd_score strategy runner unavailable")
        return "score: n/a"
    fn = getattr(runner, "composite_score", None)
    if not callable(fn):
        LOG.info("cmd_score composite_score missing")
        return "score: n/a"
    try:
        value = fn()
        LOG.info("cmd_score obtained score=%s", value)
        return f"score={value}" if value is not None else "score: n/a"
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_score: %s", exc)
        return "score: n/a"


def cmd_regime(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Summarise latest market regime snapshots for diagnostics.

    Args:
        _update: Telegram update initiating the command.
        _context: Handler context (unused).
        services: Service container exposing the regime detector.

    Returns:
        Textual summary describing the recent regime stream.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_regime")
    detector = services.market_regime
    if detector is None:
        LOG.info("cmd_regime detector unavailable")
        return "regime: n/a"
    try:
        snapshots = detector.latest(limit=5)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_regime: %s", exc)
        return "regime: n/a"
    if not snapshots:
        LOG.info("cmd_regime no snapshots available")
        return "No regime data yet."
    lines: list[str] = []
    for snapshot in reversed(snapshots):
        adjustments = snapshot.adjustments or {}
        blocklist = adjustments.get("blocklist")
        if isinstance(blocklist, list) and blocklist:
            block_display = ", ".join(str(item) for item in blocklist)
        else:
            block_display = "none"
        bias = adjustments.get("bias", "n/a")
        sizing = adjustments.get("sizing_multiplier")
        sizing_display = (
            f"{float(sizing):.2f}" if isinstance(sizing, (int, float)) else "n/a"
        )
        line = (
            f"{snapshot.symbol}: {snapshot.regime} ({snapshot.confidence:.2f}) "
            f"reason={snapshot.reason} bias={bias} block={block_display} "
            f"size={sizing_display}"
        )
        lines.append(line)
    LOG.info("cmd_regime compiled entries=%d", len(lines))
    return "\n".join(lines)


def cmd_execstate(
    update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return execution state for the supplied symbol."""

    LOG.debug("Entered cmd_execstate")
    symbol = _extract_argument(update).upper()
    if not symbol:
        LOG.info("cmd_execstate missing symbol argument")
        return "usage: /execstate SYMBOL"
    hub = services.order_execution_hub
    tracker = services.state_tracker
    state: dict[str, Any] | None = None
    if hub is not None:
        try:
            state = hub.get_position_state(symbol)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - defensive lookup
            LOG.error("Failure in cmd_execstate hub lookup: %s", exc)
            state = None
    if state is None and tracker is not None:
        try:
            state = tracker.get_position_state(symbol)
        except Exception as exc:  # noqa: BLE001 - defensive lookup
            LOG.error("Failure in cmd_execstate tracker lookup: %s", exc)
            state = None
    if not state:
        LOG.info("cmd_execstate position flat", extra={"symbol": symbol})
        return f"{symbol}: flat"
    quantity = int(state.get("quantity", 0))
    entry_price = float(state.get("entry_price", 0.0) or 0.0)
    pnl = float(state.get("unrealized_pnl", 0.0) or 0.0)
    stop_loss = state.get("stop_loss")
    tp1 = state.get("tp1")
    tp2 = state.get("tp2")
    lifecycle = state.get("lifecycle_stage", "UNKNOWN")
    LOG.info(
        "cmd_execstate resolved state",
        extra={"symbol": symbol, "quantity": quantity, "stage": lifecycle},
    )
    lines = [
        f"{symbol}",
        f"qty={quantity}",
        f"entry={entry_price:.2f}",
        f"pnl={pnl:.2f}",
    ]
    if stop_loss is not None:
        lines.append(f"sl={float(stop_loss):.2f}")
    if tp1 is not None:
        lines.append(f"tp1={float(tp1):.2f}")
    if tp2 is not None:
        lines.append(f"tp2={float(tp2):.2f}")
    lines.append(f"stage={lifecycle}")
    return "\n".join(lines)


def cmd_execqueue(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return snapshot of pending execution requests."""

    LOG.debug("Entered cmd_execqueue")
    queue = services.order_queue
    if queue is None:
        LOG.info("cmd_execqueue queue unavailable")
        return "queue: unavailable"
    try:
        snapshot = queue.get_queue_snapshot()
    except Exception as exc:  # noqa: BLE001 - defensive snapshot
        LOG.error("Failure in cmd_execqueue snapshot: %s", exc)
        return "queue: error"
    if not snapshot:
        LOG.info("cmd_execqueue queue empty")
        return "queue: empty"
    lines = ["queue:"]
    for item in snapshot[:5]:
        symbol = item.get("symbol", "?")
        intent = item.get("intent", "?")
        priority = item.get("priority", "?")
        age = float(item.get("age_sec", 0.0) or 0.0)
        lines.append(f"- {symbol} {intent} p={priority} age={age:.1f}s")
    LOG.info("cmd_execqueue reported %d items", len(snapshot))
    return "\n".join(lines)


def cmd_execlast(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return execution statistics from the hub."""

    LOG.debug("Entered cmd_execlast")
    hub = services.order_execution_hub
    if hub is None:
        LOG.info("cmd_execlast hub unavailable")
        return "exec: unavailable"
    try:
        stats = hub.get_stats()  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - defensive stats fetch
        LOG.error("Failure in cmd_execlast stats: %s", exc)
        return "exec: error"
    LOG.info("cmd_execlast stats computed")
    return (
        "executions:\n"
        f"submitted={stats.get('submitted', 0)}\n"
        f"validated={stats.get('validated', 0)}\n"
        f"rejected={stats.get('rejected', 0)}\n"
        f"executed={stats.get('executed', 0)}\n"
        f"failed={stats.get('failed', 0)}\n"
        f"queue_depth={stats.get('queue_depth', 0)}"
    )


def cmd_execwhy(
    update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Explain preflight validation outcome for a symbol."""

    LOG.debug("Entered cmd_execwhy")
    validator = services.preflight_validator
    if validator is None:
        LOG.info("cmd_execwhy validator unavailable")
        return "validator: unavailable"
    symbol = _extract_argument(update).upper()
    if not symbol:
        LOG.info("cmd_execwhy missing symbol argument")
        return "usage: /execwhy SYMBOL"
    try:
        result = validator.validate(symbol, context={})
    except Exception as exc:  # noqa: BLE001 - defensive validation
        LOG.error("Failure in cmd_execwhy: %s", exc)
        return f"{symbol}: validation error"
    if result.allowed:
        LOG.info("cmd_execwhy symbol allowed", extra={"symbol": symbol})
        return f"{symbol}: allowed"
    LOG.info(
        "cmd_execwhy blocked symbol",
        extra={"symbol": symbol, "reasons": result.reasons},
    )
    lines = [f"{symbol}: blocked ({result.blocking_level})"]
    for reason in result.reasons:
        gate = reason.get("gate")
        detail = reason.get("detail") or ""
        lines.append(f"- {gate}: {detail}")
    return "\n".join(lines)


def cmd_emergencystop(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Trigger emergency stop via the execution hub."""

    LOG.debug("Entered cmd_emergencystop")
    hub = services.order_execution_hub
    if hub is None:
        LOG.info("cmd_emergencystop hub unavailable")
        return "emergency stop unavailable"
    try:
        result = hub.emergency_stop()  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - defensive invocation
        LOG.error("Failure in cmd_emergencystop: %s", exc)
        return "emergency stop failed"
    closed = result.get("positions_closed", [])
    queue_paused = bool(result.get("queue_paused"))
    LOG.info(
        "cmd_emergencystop executed",
        extra={"closed": closed, "queue_paused": queue_paused},
    )
    closed_text = ", ".join(closed) if closed else "none"
    return (
        "emergency stop:\n"
        f"queue_paused={queue_paused}\n"
        f"positions_closed={closed_text}"
    )


def cmd_gate_why(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return the most recent gate reason from the strategy runner.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing strategy runner.

    Returns:
        Gate reason text or fallback when unavailable.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_gate_why")
    runner = services.strategy_runner
    if runner is None:
        LOG.info("cmd_gate_why strategy runner unavailable")
        return "gate why: n/a"
    fn = getattr(runner, "gate_reason", None)
    if not callable(fn):
        LOG.info("cmd_gate_why gate_reason missing")
        return "gate why: n/a"
    try:
        result = fn()
        LOG.info("cmd_gate_why obtained reason=%s", result)
        return _format_object(result)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_gate_why: %s", exc)
        return "gate why: n/a"


def cmd_size(
    update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Suggest a position size based on risk inputs.

    Args:
        update: Telegram update containing size parameters.
        _context: Handler context (unused).
        services: Service container exposing risk manager and config.

    Returns:
        Suggested size output or usage guidance.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_size")
    argument = _extract_argument(update)
    parts = argument.split()
    if len(parts) < 5:
        LOG.info("cmd_size missing arguments count=%d", len(parts))
        return "Usage: /size BUY|SELL price stop atr qty [symbol]"
    side_raw, price_raw, stop_raw, atr_raw, qty_raw, *symbol_rest = parts
    symbol = (
        symbol_rest[0]
        if symbol_rest
        else getattr(services.config, "symbol_root", "NIFTY")
    )
    risk_manager = services.risk_manager
    if risk_manager is None:
        LOG.info("cmd_size risk manager unavailable")
        return "sizing: n/a"
    fn = getattr(risk_manager, "suggest_position_size", None)
    if not callable(fn):
        LOG.info("cmd_size suggest_position_size missing")
        return "sizing: n/a"
    try:
        side = "BUY" if side_raw.upper().startswith("B") else "SELL"
        price = float(price_raw)
        stop_loss = float(stop_raw)
        atr = float(atr_raw)
        quantity = int(float(qty_raw))
        LOG.info(
            "cmd_size inputs side=%s price=%.2f stop=%.2f atr=%.2f qty=%d symbol=%s",
            side,
            price,
            stop_loss,
            atr,
            quantity,
            symbol,
        )
        result = fn(
            side=side,
            price=price,
            stop_loss=stop_loss,
            atr=atr,
            requested_quantity=quantity,
            symbol=symbol,
            confidence=1.0,
        )
        return f"size={result}"
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_size: %s", exc)
        return "sizing failed"


def cmd_trail(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return trailing-stop snapshot from the order manager when present.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing order manager.

    Returns:
        Trailing stop summary or fallback text.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_trail")
    manager = services.order_manager
    if manager is None:
        LOG.info("cmd_trail order manager unavailable")
        return "trailing: n/a"
    fn = getattr(manager, "trailing_snapshot", None)
    if not callable(fn):
        LOG.info("cmd_trail trailing_snapshot missing")
        return "trailing: n/a"
    try:
        snapshot = fn()
        LOG.info("cmd_trail obtained snapshot")
        return _format_object(snapshot)
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_trail: %s", exc)
        return "trailing: n/a"


def cmd_riskstate(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return key runtime fields from the risk manager.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing risk manager.

    Returns:
        Multi-line summary of breaker status and cooldown.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_riskstate")
    manager = services.risk_manager
    if manager is None:
        LOG.info("cmd_riskstate risk manager unavailable")
        return "risk: n/a"
    breaker = bool(
        getattr(manager, "_breaker_tripped", getattr(manager, "breaker_tripped", False))
    )
    try:
        cooldown = float(getattr(manager, "cooldown_remaining", lambda: 0.0)())
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_riskstate cooldown: %s", exc)
        cooldown = 0.0
    last_rejection = getattr(
        manager, "_last_rejection", getattr(manager, "last_reason", None)
    )
    LOG.info("cmd_riskstate breaker=%s cooldown=%.2f", breaker, cooldown)
    parts = [
        f"breaker={breaker}",
        f"cooldown={cooldown:.0f}s",
        f"last_rejection={last_rejection}",
    ]
    return "\n".join(parts)


def cmd_journal(
    _update: Update, _context: ContextTypes.DEFAULT_TYPE, services: Services
) -> str:
    """Return recent journal entries when available.

    Args:
        _update: Telegram update (unused).
        _context: Handler context (unused).
        services: Service container exposing journal access.

    Returns:
        Joined string of journal entries or fallback text.

    Raises:
        None.
    """

    LOG.debug("Entered cmd_journal")
    journal = services.journal
    if journal is None:
        LOG.info("cmd_journal journal unavailable")
        return "journal: n/a"
    fn = getattr(journal, "last_entries", None)
    if not callable(fn):
        LOG.info("cmd_journal last_entries missing")
        return "journal: n/a"
    try:
        entries = fn(limit=10) or []
        LOG.info("cmd_journal returned entries=%d", len(entries))
        return "\n".join(str(entry) for entry in entries) or "No journal entries."
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in cmd_journal: %s", exc)
        return "journal: n/a"


def _command_exists(application: Application, command: str) -> bool:
    """Return ``True`` if *command* already has a handler on *application*.

    Args:
        application: Telegram application containing registered handlers.
        command: Command name to inspect.

    Returns:
        Boolean indicating whether a handler is already present.

    Raises:
        None.
    """

    LOG.debug("Entered _command_exists command=%s", command)
    for handlers in application.handlers.values():
        for handler in handlers:
            if isinstance(handler, CommandHandler) and command in handler.commands:
                LOG.info("_command_exists command=%s already registered", command)
                return True
    return False


def register_telegram_commands(
    bot: Any, application: Application, services: Services
) -> None:
    """Register additional production commands on *application*.

    Args:
        bot: Telegram controller instance exposing existing handlers.
        application: Telegram application receiving extra handlers.
        services: Shared services bundle used by new commands.

    Returns:
        None.

    Raises:
        None.
    """

    LOG.debug("Entered register_telegram_commands")
    alias_commands: dict[str, Callable[[Update, ContextTypes.DEFAULT_TYPE], Any]] = {}
    if hasattr(bot, "cmd_positions"):
        alias_commands["pos"] = bot.cmd_positions  # type: ignore[assignment]
    if hasattr(bot, "cmd_trades"):
        alias_commands["fills"] = bot.cmd_trades  # type: ignore[assignment]
    if hasattr(bot, "cmd_tail"):
        alias_commands["logs"] = bot.cmd_tail  # type: ignore[assignment]
    if hasattr(bot, "cmd_debug_on"):
        alias_commands["trace"] = bot.cmd_debug_on  # type: ignore[assignment]
    if hasattr(bot, "cmd_kpi"):
        alias_commands["perf"] = bot.cmd_kpi  # type: ignore[assignment]

    for name, handler in alias_commands.items():
        if _command_exists(application, name):
            continue
        LOG.info("Registering alias command=%s handler=%s", name, handler)
        application.add_handler(CommandHandler(name, handler))

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
        "journal": cmd_journal,
        "execstate": cmd_execstate,
        "execqueue": cmd_execqueue,
        "execlast": cmd_execlast,
        "execwhy": cmd_execwhy,
        "emergencystop": cmd_emergencystop,
    }

    for name, func in new_commands.items():
        if _command_exists(application, name):
            continue
        LOG.info("Registering command=%s", name)
        application.add_handler(
            CommandHandler(name, _wrap_command(services, func, name))
        )
