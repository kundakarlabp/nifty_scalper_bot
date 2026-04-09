"""Lifecycle management for bracket-style trades."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, cast

from nifty_scalper_bot.data.option_chain import _round_to_tick

import yaml  # type: ignore[import]

from nifty_scalper_bot.execution.metrics import LIFECYCLE_EXITS

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class LifecycleConfig:
    """Resolved lifecycle configuration for target calculations."""

    tp1_r: float
    tp1_partial: float
    tp2_r_trend: float
    tp2_r_range: float
    trail_atr_mult: float
    time_stop_min: int
    gamma_enabled: bool
    gamma_tue_after: dtime | None
    gamma_tp2_r_cap: float
    gamma_trail_atr_mult: float
    gamma_time_stop_min: int


@dataclass(slots=True)
class LifecyclePosition:
    """Runtime metadata for an active trade lifecycle."""

    symbol: str
    entry_price: float
    quantity: int
    atr: float
    regime: str
    sl: float
    tp1: float
    tp2: float
    trail_stop: float | None = None
    trail_active: bool = False
    lifecycle_stage: str = "ENTRY"
    tp1_taken: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    highest_price: float = 0.0
    entry_iv: float | None = None
    expiry: datetime | None = None
    tick_size: float = 0.05


class StateTrackerProtocol(Protocol):
    """Protocol representing the lifecycle state tracker dependency."""

    def record_lifecycle_event(
        self, symbol: str, event_type: str, payload: Mapping[str, Any] | None = None
    ) -> None:
        """Record a lifecycle state transition.

        Args:
            symbol: Canonical position identifier.
            event_type: Name of the lifecycle event.
            payload: Optional JSON-serialisable payload for diagnostics.

        Returns:
            None.

        Raises:
            None.
        """

    def get_open_positions(self) -> Iterable[Mapping[str, Any]]:
        """Return iterable describing all open positions.

        Args:
            None.

        Returns:
            Iterable[Mapping[str, Any]]: Open positions snapshot.

        Raises:
            None.
        """


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Return YAML configuration payload from ``path``.

    Args:
        path: Location of the YAML configuration file.

    Returns:
        dict[str, Any]: Parsed configuration payload or an empty dictionary.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered _load_yaml_config",
        extra={"event": "lifecycle_load_yaml_enter", "path": str(path)},
    )
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        LOGGER.error(
            "Failure in _load_yaml_config: missing file",
            extra={"event": "lifecycle_config_missing", "path": str(path)},
        )
        return {}
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _load_yaml_config: %s",
            exc,
            extra={"event": "lifecycle_config_error", "path": str(path)},
            exc_info=exc,
        )
        return {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def _parse_time(value: str | None) -> dtime | None:
    """Return :class:`datetime.time` parsed from ``value``.

    Args:
        value: String in ``HH:MM`` format.

    Returns:
        datetime.time | None: Parsed time when valid, otherwise ``None``.

    Raises:
        None.
    """

    if not value:
        return None
    match = re.fullmatch(r"(\d{2}):(\d{2})", value.strip())
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    return dtime(hour=hour, minute=minute)


def _safe_float(value: Any, default: float) -> float:
    """Return ``value`` as ``float`` with ``default`` fallback.

    Args:
        value: Candidate numeric value.
        default: Fallback when conversion fails.

    Returns:
        float: Converted value or ``default``.

    Raises:
        None.
    """

    try:
        return float(value)
    except (TypeError, ValueError):  # noqa: BLE001
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    """Return ``value`` as ``int`` with ``default`` fallback.

    Args:
        value: Candidate numeric value.
        default: Fallback when conversion fails.

    Returns:
        int: Converted value or ``default``.

    Raises:
        None.
    """

    try:
        return int(float(value))
    except (TypeError, ValueError):  # noqa: BLE001
        return int(default)


def _resolve_lifecycle_config(config_path: Path) -> LifecycleConfig:
    """Load lifecycle configuration from YAML and environment overrides.

    Args:
        config_path: Strategy configuration path.

    Returns:
        LifecycleConfig: Resolved lifecycle configuration.

    Raises:
        None.
    """

    payload = _load_yaml_config(config_path)
    lifecycle = payload.get("lifecycle", {}) if isinstance(payload, dict) else {}
    gamma = lifecycle.get("gamma_mode", {}) if isinstance(lifecycle, dict) else {}

    tp1_r_env = os.getenv("LIFECYCLE_TP1_R")
    tp1_partial_env = os.getenv("LIFECYCLE_TP1_PARTIAL")
    tp2_trend_env = os.getenv("LIFECYCLE_TP2_R_TREND")
    tp2_range_env = os.getenv("LIFECYCLE_TP2_R_RANGE")
    trail_env = os.getenv("LIFECYCLE_TRAIL_ATR_MULT")
    time_stop_env = os.getenv("LIFECYCLE_TIME_STOP_MIN")

    tp1_r = _safe_float(tp1_r_env, lifecycle.get("tp1_R_min", 1.0))
    tp1_partial = _safe_float(tp1_partial_env, lifecycle.get("tp1_partial", 0.6))
    tp2_trend = _safe_float(tp2_trend_env, lifecycle.get("tp2_R_trend", 1.8))
    tp2_range = _safe_float(tp2_range_env, lifecycle.get("tp2_R_range", 1.4))
    trail_mult = _safe_float(trail_env, lifecycle.get("trail_atr_mult", 0.8))
    time_stop = _safe_int(time_stop_env, lifecycle.get("time_stop_min", 12))

    gamma_enabled = bool(gamma.get("enabled", False))
    gamma_tue_after = _parse_time(gamma.get("tue_after"))
    gamma_tp2_cap = _safe_float(gamma.get("tp2_R_cap", tp2_trend), tp2_trend)
    gamma_trail_mult = _safe_float(gamma.get("trail_atr_mult", trail_mult), trail_mult)
    gamma_time_stop = _safe_int(gamma.get("time_stop_min", time_stop), time_stop)

    return LifecycleConfig(
        tp1_r=tp1_r,
        tp1_partial=tp1_partial,
        tp2_r_trend=tp2_trend,
        tp2_r_range=tp2_range,
        trail_atr_mult=trail_mult,
        time_stop_min=time_stop,
        gamma_enabled=gamma_enabled,
        gamma_tue_after=gamma_tue_after,
        gamma_tp2_r_cap=gamma_tp2_cap,
        gamma_trail_atr_mult=gamma_trail_mult,
        gamma_time_stop_min=gamma_time_stop,
    )


class LifecycleManager:
    """Manage lifecycle transitions from entry through exits."""

    def __init__(
        self,
        *,
        data_hub: Any | None,
        state_tracker: StateTrackerProtocol | None = None,
        config_path: Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialise manager with market data and execution dependencies.

        Args:
            data_hub: Data hub instance for market access.
            state_tracker: Optional state tracker for persistence.
            config_path: Optional path to strategy configuration.
            clock: Optional callable returning current UTC time.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered LifecycleManager.__init__",
            extra={"event": "lifecycle_manager_init"},
        )
        self._data_hub = data_hub
        self._state_tracker = state_tracker
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._config = _resolve_lifecycle_config(
            config_path or Path("config/strategy.yaml")
        )
        self._positions: dict[str, LifecyclePosition] = {}
        self._tick_handlers: dict[str, Callable[[Mapping[str, Any]], None]] = {}
        self._monitor_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start background monitoring for lifecycle adjustments.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered LifecycleManager.start",
            extra={"event": "lifecycle_manager_start"},
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleManager.start: %s",
                exc,
                extra={"event": "lifecycle_manager_loop_missing"},
                exc_info=exc,
            )
            return
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        self._stop_event.clear()
        self._monitor_task = loop.create_task(self._monitor_loop())

    def _should_apply_gamma_mode(self, position: LifecyclePosition) -> bool:
        """Return ``True`` when gamma adjustments should be applied.

        Args:
            position: Lifecycle position metadata.

        Returns:
            bool: ``True`` when gamma rules should adjust the position.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered LifecycleManager._should_apply_gamma_mode",
            extra={
                "event": "lifecycle_gamma_check",
                "symbol": position.symbol,
            },
        )
        try:
            if not self._config.gamma_enabled:
                return False
            expiry = position.expiry
            if expiry is None:
                return False
            now = self._clock()
            if now.date() != expiry.date():
                return False
            if now.weekday() != 1:
                return False
            cutoff = self._config.gamma_tue_after
            if cutoff is None:
                return False
            should_apply = now.time() >= cutoff
            if should_apply:
                LOGGER.info(
                    "Condition met: gamma_mode_applicable",
                    extra={
                        "event": "gamma_mode_applicable",
                        "symbol": position.symbol,
                        "cutoff": cutoff.isoformat(),
                    },
                )
            return should_apply
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleManager._should_apply_gamma_mode: %s",
                exc,
                extra={
                    "event": "lifecycle_gamma_check_error",
                    "symbol": position.symbol,
                },
                exc_info=exc,
            )
            return False

    async def shutdown(self) -> None:
        """Stop background monitoring and unsubscribe from ticks.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered LifecycleManager.shutdown",
            extra={"event": "lifecycle_manager_shutdown"},
        )
        self._stop_event.set()
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        for symbol, handler in list(self._tick_handlers.items()):
            if self._data_hub is None:
                continue
            try:
                self._data_hub.unsubscribe_ticks(symbol, handler)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure in LifecycleManager.shutdown: %s",
                    exc,
                    extra={
                        "event": "lifecycle_manager_unsub_error",
                        "symbol": symbol,
                    },
                    exc_info=exc,
                )
        self._tick_handlers.clear()

    def on_fill(
        self,
        *,
        symbol: str,
        entry_price: float,
        quantity: int,
        atr: float,
        regime: str,
        iv: float | None = None,
    ) -> None:
        """Register a newly filled position for lifecycle monitoring.

        Args:
            symbol: Trading symbol.
            entry_price: Executed entry price.
            quantity: Position quantity.
            atr: Average true range at entry.
            regime: Market regime descriptor.
            iv: Optional implied volatility snapshot at entry.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered LifecycleManager.on_fill",
            extra={"event": "lifecycle_on_fill", "symbol": symbol},
        )
        try:
            tick_size = self._resolve_tick_size(symbol)
            normalised_entry = self._normalise_price(entry_price, tick_size)
            stop_loss = self._normalise_price(
                normalised_entry - atr * self._config.trail_atr_mult,
                tick_size,
            )
            tp1 = self._normalise_price(
                normalised_entry + atr * self._config.tp1_r,
                tick_size,
            )
            tp2_r = (
                self._config.tp2_r_trend
                if regime.upper() == "TREND"
                else self._config.tp2_r_range
            )
            tp2 = self._normalise_price(normalised_entry + atr * tp2_r, tick_size)
            expiry = self._infer_expiry(symbol)
            position = LifecyclePosition(
                symbol=symbol,
                entry_price=normalised_entry,
                quantity=quantity,
                atr=atr,
                regime=regime.upper(),
                sl=stop_loss,
                tp1=tp1,
                tp2=tp2,
                trail_stop=None,
                trail_active=False,
                lifecycle_stage="ENTRY",
                highest_price=entry_price,
                entry_iv=iv,
                expiry=expiry,
                tick_size=tick_size,
            )
            if self._should_apply_gamma_mode(position):
                LOGGER.warning(
                    "Gamma mode activated for %s",
                    symbol,
                    extra={
                        "event": "gamma_mode_applied",
                        "symbol": symbol,
                    },
                )
                position.tp2 = self._normalise_price(
                    min(
                        position.tp2,
                        position.entry_price
                        + position.atr * self._config.gamma_tp2_r_cap,
                    ),
                    position.tick_size,
                )
                position.sl = self._normalise_price(
                    position.entry_price
                    - (position.atr * self._config.gamma_trail_atr_mult),
                    position.tick_size,
                )
                self._record_event(
                    symbol,
                    "GAMMA_ADJUSTED",
                    {
                        "event_type": "GAMMA_ADJUSTED",
                        "new_tp2": position.tp2,
                        "new_sl": position.sl,
                    },
                )
            self._positions[symbol] = position
            self._subscribe_ticks(symbol)
            entry_snapshot = asdict(position)
            self._record_event(symbol, "ENTRY_REGISTERED", entry_snapshot)
            LOGGER.info(
                "Condition met: lifecycle_position_registered",
                extra={
                    "event": "lifecycle_position_registered",
                    "symbol": symbol,
                    "sl": position.sl,
                    "tp1": position.tp1,
                    "tp2": position.tp2,
                },
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleManager.on_fill: %s",
                exc,
                extra={"event": "lifecycle_on_fill_error", "symbol": symbol},
                exc_info=exc,
            )

    def update_all_positions(self, current_time: datetime) -> None:
        """Recalculate trailing stops and adjust targets dynamically.

        Args:
            current_time: Current timestamp used for calculations.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered LifecycleManager.update_all_positions",
            extra={"event": "lifecycle_update_positions"},
        )
        for symbol, position in list(self._positions.items()):
            try:
                candles = self._fetch_recent_candles(symbol)
                atr = self._calculate_atr(candles) if candles else position.atr
                if atr > 0:
                    position.atr = atr
                highest = max(position.highest_price, position.entry_price)
                trail_mult = self._config.trail_atr_mult
                if self._is_gamma_active(current_time):
                    trail_mult = self._config.gamma_trail_atr_mult
                trail_candidate = self._normalise_price(highest - atr * trail_mult, position.tick_size)
                if position.trail_stop is None:
                    position.trail_stop = trail_candidate
                else:
                    position.trail_stop = max(position.trail_stop, trail_candidate)
                greeks_iv = self._resolve_iv(symbol)
                if (
                    greeks_iv is not None
                    and position.entry_iv is not None
                    and greeks_iv < position.entry_iv * 0.8
                ):
                    adjustment = position.tp2 - position.entry_price
                    position.tp2 = self._normalise_price(position.entry_price + adjustment * 0.9, position.tick_size)
                    self._record_event(
                        symbol,
                        "TP2_ADJUSTED_IV",
                        {"event_type": "TP2_ADJUSTED_IV", "new_tp2": position.tp2},
                    )
                if position.expiry is not None:
                    if position.expiry - current_time <= timedelta(days=1):
                        cap = self._config.gamma_tp2_r_cap
                        position.tp2 = self._normalise_price(position.entry_price + position.atr * cap, position.tick_size)
                        self._record_event(
                            symbol,
                            "TP2_ADJUSTED_EXPIRY",
                            {
                                "event_type": "TP2_ADJUSTED_EXPIRY",
                                "new_tp2": position.tp2,
                            },
                        )
                if position.trail_stop is not None:
                    position.sl = max(position.sl, position.trail_stop)
                self._record_event(
                    symbol,
                    "TRAIL_UPDATED",
                    {
                        "event_type": "TRAIL_UPDATED",
                        "new_sl": position.sl,
                        "atr": position.atr,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure in LifecycleManager.update_all_positions: %s",
                    exc,
                    extra={"event": "lifecycle_update_error", "symbol": symbol},
                    exc_info=exc,
                )

    async def _monitor_loop(self) -> None:
        """Background loop applying periodic lifecycle checks.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered LifecycleManager._monitor_loop",
            extra={"event": "lifecycle_monitor_enter"},
        )
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(5.0)
                now = self._clock()
                self.update_all_positions(now)
                self._check_time_based_exits(now)
        except asyncio.CancelledError:
            LOGGER.debug(
                "LifecycleManager monitor cancelled",
                extra={"event": "lifecycle_monitor_cancelled"},
            )
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleManager._monitor_loop: %s",
                exc,
                extra={"event": "lifecycle_monitor_error"},
                exc_info=exc,
            )

    def _check_time_based_exits(self, now: datetime) -> None:
        """Evaluate time-based exit conditions for open positions.

        Args:
            now: Current timestamp used for cut-off calculations.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered LifecycleManager._check_time_based_exits",
            extra={"event": "lifecycle_time_exit_enter"},
        )
        market_close = now.replace(hour=15, minute=15, second=0, microsecond=0)
        for symbol, position in list(self._positions.items()):
            try:
                age = now - position.created_at
                limit_min = self._config.time_stop_min
                if self._is_gamma_active(now):
                    limit_min = self._config.gamma_time_stop_min
                if age >= timedelta(minutes=limit_min):
                    self.exit_at_market(symbol, "TIME_STOP")
                    continue
                if market_close - now <= timedelta(minutes=15):
                    self.exit_at_market(symbol, "MARKET_CLOSE")
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure in LifecycleManager._check_time_based_exits: %s",
                    exc,
                    extra={"event": "lifecycle_time_exit_error", "symbol": symbol},
                    exc_info=exc,
                )

    def _subscribe_ticks(self, symbol: str) -> None:
        """Subscribe to tick updates for ``symbol`` via the data hub.

        Args:
            symbol: Instrument identifier to subscribe.

        Returns:
            None.

        Raises:
            None.
        """

        if self._data_hub is None:
            return
        if symbol in self._tick_handlers:
            return

        def _handler(tick: Mapping[str, Any]) -> None:
            self._handle_tick(symbol, tick)

        try:
            self._data_hub.subscribe_ticks(symbol, _handler)
            self._tick_handlers[symbol] = _handler
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleManager._subscribe_ticks: %s",
                exc,
                extra={"event": "lifecycle_subscribe_error", "symbol": symbol},
                exc_info=exc,
            )

    def _handle_tick(self, symbol: str, tick: Mapping[str, Any]) -> None:
        """Process tick update and trigger lifecycle transitions.

        Args:
            symbol: Instrument identifier for the tick.
            tick: Market data payload.

        Returns:
            None.

        Raises:
            None.
        """

        position = self._positions.get(symbol)
        if position is None:
            return
        price = self._extract_price(tick)
        if price is None:
            return
        normalised_price = self._normalise_price(price, position.tick_size)
        epsilon = position.tick_size / 2.0
        position.highest_price = max(position.highest_price, normalised_price)
        if normalised_price <= position.sl + epsilon:
            self.exit_at_market(symbol, "STOP_LOSS")
            return
        if normalised_price >= position.tp2 - epsilon:
            self.exit_at_market(symbol, "TAKE_PROFIT_2")
            return
        if normalised_price >= position.tp1 - epsilon and not position.tp1_taken:
            self.partial_exit(symbol, position)
        if position.trail_stop is not None and normalised_price <= position.trail_stop + epsilon:
            self.exit_at_market(symbol, "TRAIL_STOP")

    def partial_exit(self, symbol: str, position: LifecyclePosition) -> None:
        """Execute TP1 partial exit and adjust stop to breakeven.

        Args:
            symbol: Instrument identifier.
            position: Lifecycle position metadata.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered LifecycleManager.partial_exit",
            extra={"event": "lifecycle_partial_exit", "symbol": symbol},
        )
        try:
            partial_qty = max(int(position.quantity * self._config.tp1_partial), 1)
            request = OrderRequest(
                symbol=symbol,
                side="SELL",
                quantity=partial_qty,
                intent="EXIT_TP1",
                parent_id=None,
                source="LifecycleManager",
            )
            position.tp1_taken = True
            position.quantity = max(position.quantity - partial_qty, 0)
            position.sl = position.entry_price
            position.lifecycle_stage = "TP1_TAKEN_RUNNING_TO_TP2"
            self._record_event(
                symbol,
                "TP1_TAKEN",
                {"event_type": "TP1_TAKEN", "quantity": partial_qty},
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleManager.partial_exit: %s",
                exc,
                extra={"event": "lifecycle_partial_exit_error", "symbol": symbol},
                exc_info=exc,
            )

    def exit_at_market(self, symbol: str, reason: str) -> None:
        """Submit a market exit request for ``symbol`` with ``reason``.

        Args:
            symbol: Instrument identifier.
            reason: Lifecycle transition reason.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered LifecycleManager.exit_at_market",
            extra={
                "event": "lifecycle_exit_market",
                "symbol": symbol,
                "reason": reason,
            },
        )
        position = self._positions.pop(symbol, None)
        if position is None:
            return
        LIFECYCLE_EXITS.labels(symbol=symbol, reason=reason).inc()
        try:
            intent: OrderIntent = cast(
                OrderIntent,
                "EXIT_TP2" if reason.startswith("TAKE_PROFIT") else "EXIT_SL",
            )
            request = OrderRequest(
                symbol=symbol,
                side="SELL",
                quantity=position.quantity,
                intent=intent,
                parent_id=None,
                source="LifecycleManager",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleManager.exit_at_market: %s",
                exc,
                extra={"event": "lifecycle_exit_error", "symbol": symbol},
                exc_info=exc,
            )
        finally:
            handler = self._tick_handlers.pop(symbol, None)
            if handler is not None and self._data_hub is not None:
                try:
                    self._data_hub.unsubscribe_ticks(symbol, handler)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        "Failure in LifecycleManager.exit_at_market unsubscribe: %s",
                        exc,
                        extra={
                            "event": "lifecycle_unsubscribe_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
            self._record_event(symbol, reason, {"event_type": reason})

    def _record_event(
        self, symbol: str, event_type: str, payload: Mapping[str, Any] | None
    ) -> None:
        """Record lifecycle event via state tracker when available.

        Args:
            symbol: Instrument identifier.
            event_type: Name of the lifecycle event.
            payload: Optional payload to persist.

        Returns:
            None.

        Raises:
            None.
        """

        if self._state_tracker is None:
            return
        try:
            self._state_tracker.record_lifecycle_event(symbol, event_type, payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleManager._record_event: %s",
                exc,
                extra={"event": "lifecycle_state_record_error", "symbol": symbol},
                exc_info=exc,
            )

    def _normalise_price(self, price: float, tick_size: float) -> float:
        """Return price rounded to instrument tick-size. Args: price, tick_size. Returns: float. Raises: None."""

        return float(_round_to_tick(price, tick_size=tick_size))

    def _resolve_tick_size(self, symbol: str) -> float:
        """Resolve symbol tick-size from data hub metadata. Args: symbol. Returns: float. Raises: None."""

        try:
            if self._data_hub is not None:
                resolver = getattr(self._data_hub, "resolve_symbol", None)
                if callable(resolver):
                    payload = resolver(symbol)
                    if isinstance(payload, Mapping):
                        tick = _safe_float(payload.get("tick_size"), 0.05)
                        if tick > 0:
                            return tick
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleManager._resolve_tick_size: %s",
                exc,
                extra={"event": "lifecycle_tick_size_error", "symbol": symbol},
                exc_info=exc,
            )
        return 0.05

    def _fetch_recent_candles(self, symbol: str) -> Iterable[Mapping[str, Any]] | None:
        """Return last 15 candles when the data hub exposes the API.

        Args:
            symbol: Instrument identifier.

        Returns:
            Iterable[Mapping[str, Any]] | None: Candle data when available.

        Raises:
            None.
        """

        if self._data_hub is None:
            return None
        fetcher = getattr(self._data_hub, "get_recent_candles", None)
        if not callable(fetcher):
            return None
        try:
            return fetcher(symbol, limit=15)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleManager._fetch_recent_candles: %s",
                exc,
                extra={"event": "lifecycle_fetch_candles_error", "symbol": symbol},
                exc_info=exc,
            )
            return None

    def _calculate_atr(self, candles: Iterable[Mapping[str, Any]]) -> float:
        """Compute ATR from ``candles``.

        Args:
            candles: Iterable with high, low, and close fields.

        Returns:
            float: Average true range value.

        Raises:
            None.
        """

        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        for candle in candles:
            highs.append(_safe_float(candle.get("high"), 0.0))
            lows.append(_safe_float(candle.get("low"), 0.0))
            closes.append(_safe_float(candle.get("close"), 0.0))
        if len(highs) < 2:
            return 0.0
        trs: list[float] = []
        prev_close = closes[0]
        for idx in range(1, len(highs)):
            high = highs[idx]
            low = lows[idx]
            tr = max(high - low, abs(high - prev_close), abs(prev_close - low))
            trs.append(tr)
            prev_close = closes[idx]
        if not trs:
            return 0.0
        return sum(trs) / len(trs)

    def _resolve_iv(self, symbol: str) -> float | None:
        """Return implied volatility using data hub caches when available.

        Args:
            symbol: Instrument identifier.

        Returns:
            float | None: Implied volatility if available.

        Raises:
            None.
        """

        if self._data_hub is None:
            return None
        try:
            iv = getattr(self._data_hub, "get_iv", None)
            if callable(iv):
                return iv(symbol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleManager._resolve_iv: %s",
                exc,
                extra={"event": "lifecycle_resolve_iv_error", "symbol": symbol},
                exc_info=exc,
            )
        return None

    def _extract_price(self, tick: Mapping[str, Any]) -> float | None:
        """Return representative price from ``tick`` payload.

        Args:
            tick: Market data payload.

        Returns:
            float | None: Extracted price when available.

        Raises:
            None.
        """

        for key in ("last_price", "ltp", "close", "price"):
            value = tick.get(key)
            if value is None:
                continue
            try:
                price = float(value)
            except (TypeError, ValueError):  # noqa: BLE001
                continue
            if price > 0:
                return price
        return None

    def _is_gamma_active(self, current_time: datetime) -> bool:
        """Return ``True`` when gamma mode overrides should be active.

        Args:
            current_time: Timestamp used for evaluation.

        Returns:
            bool: ``True`` if gamma overrides are active.

        Raises:
            None.
        """

        if not self._config.gamma_enabled:
            return False
        if current_time.weekday() != 1:
            return False
        if self._config.gamma_tue_after is None:
            return True
        after = self._config.gamma_tue_after
        return (current_time.hour, current_time.minute) >= (after.hour, after.minute)

    def _infer_expiry(self, symbol: str) -> datetime | None:
        """Infer expiry datetime from derivative ``symbol`` when possible.

        Args:
            symbol: Instrument identifier.

        Returns:
            datetime | None: Estimated expiry timestamp.

        Raises:
            None.
        """

        pattern = re.compile(
            r"^(?P<base>[A-Z]+)(?P<year>\d{2})(?P<month>[A-Z]{3})(?P<strike>\d+)(?P<opt>CE|PE)$"
        )
        match = pattern.match(symbol.upper())
        if not match:
            return None
        months = {
            "JAN": 1,
            "FEB": 2,
            "MAR": 3,
            "APR": 4,
            "MAY": 5,
            "JUN": 6,
            "JUL": 7,
            "AUG": 8,
            "SEP": 9,
            "OCT": 10,
            "NOV": 11,
            "DEC": 12,
        }
        month = months.get(match.group("month"))
        if month is None:
            return None
        year = int(match.group("year")) + 2000
        expiry = datetime(year, month, 28, 15, 30, tzinfo=timezone.utc)
        return expiry


__all__ = ["LifecycleManager", "LifecyclePosition", "LifecycleConfig"]
