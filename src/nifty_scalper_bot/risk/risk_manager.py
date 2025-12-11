"""Risk management guardrails and telemetry helpers."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, Mapping, Protocol

from nifty_scalper_bot.execution.position_manager import Position
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.risk.limits import RiskSwitches
from nifty_scalper_bot.utils.env import get_float
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.metrics import Counter, Gauge
from nifty_scalper_bot.utils.reasons import SOFT, canonical

if TYPE_CHECKING:
    from nifty_scalper_bot.config.settings import RiskSettings
    from nifty_scalper_bot.data.data_hub import DataHub
else:  # pragma: no cover - runtime fallback
    RiskSettings = Any
    DataHub = Any

OrderSide = Literal["BUY", "SELL"]


COOLDOWN_SLOP_MS = max(int(os.getenv("RISK_COOLDOWN_SLOP_MS", "250")), 0)


LOGGER = get_logger(__name__)


class PositionManagerProtocol(Protocol):
    """Subset of :class:`PositionManager` required by :class:`RiskManager`."""

    def get_all_positions(
        self,
    ) -> list[Position]:  # pragma: no cover - protocol signature
        ...

    def get_realized_pnl(self) -> float:  # pragma: no cover - protocol signature
        ...

    def get_total_exposure(self) -> float:  # pragma: no cover - protocol signature
        ...


@dataclass(slots=True)
class OrderSignal:
    """Normalized order intent evaluated by :class:`RiskManager`."""

    symbol: str
    side: OrderSide
    quantity: int
    price: float
    stop_loss: float | None
    take_profit: float | None

    def notional(self) -> float:
        return abs(self.price * self.quantity)


@dataclass(frozen=True, slots=True)
class RiskSnapshot:
    """Point-in-time snapshot exposed via Telegram and health endpoints."""

    daily_realized: float
    daily_loss_limit: float
    day_loss: float
    max_day_loss: float
    losses_in_row: int
    cooldown_remaining: float
    breaker_tripped: bool
    breaker_reason: str | None
    shadow_forced: bool
    per_trade_risk_pct: float
    last_rejection: str | None
    timestamp: datetime


@dataclass(slots=True)
class RiskManager:
    """Evaluate orders against configured guardrails."""

    settings: RiskSettings
    position_manager: PositionManagerProtocol
    account_balance: float
    alert_callback: Callable[[str, RiskSnapshot], None] | None = None
    risk_state: "RiskState | None" = field(default=None, repr=False)
    _risk_state: "RiskState | None" = field(init=False, repr=False, default=None)
    _logger: Any = field(init=False, repr=False)
    _cooldown_until: datetime | None = field(init=False, repr=False, default=None)
    _last_pnl_snapshot: float = field(init=False, repr=False, default=0.0)
    _trading_day_start: datetime = field(init=False, repr=False)
    _breaker_tripped: bool = field(init=False, repr=False, default=False)
    _breaker_reason: str | None = field(init=False, repr=False, default=None)
    _shadow_forced: bool = field(init=False, repr=False, default=False)
    _last_rejection: str | None = field(init=False, repr=False, default=None)
    _switches: RiskSwitches = field(init=False, repr=False)
    _m_blocks: Any = field(init=False, repr=False)
    _m_cooldown: Any = field(init=False, repr=False)
    _soft_override: bool = field(init=False, repr=False, default=False)
    _lot_size_lookup: Callable[[str], int] | None = field(
        init=False, repr=False, default=None
    )
    _lot_size_symbol: str | None = field(init=False, repr=False, default=None)
    breaker_alert_sender: Callable[[str], Awaitable[None]] | None = field(
        default=None, repr=False
    )
    _breaker_alerted: bool = field(init=False, repr=False, default=False)
    _last_trip_time: datetime | None = field(init=False, repr=False, default=None)
    _broker_client: Any | None = field(init=False, repr=False, default=None)
    _data_hub: "DataHub | None" = field(init=False, repr=False, default=None)
    _cached_balance: float = field(init=False, repr=False, default=0.0)
    _last_balance_refresh: float = field(init=False, repr=False, default=0.0)
    _balance_cache_ttl: float = field(init=False, repr=False, default=60.0)
    _balance_force_refresh: float = field(init=False, repr=False, default=0.0)
    _balance_source: str = field(init=False, repr=False, default="env")
    _market_data_manager: Any | None = field(init=False, repr=False, default=None)
    _balance_refresher: threading.Thread | None = field(
        init=False, repr=False, default=None
    )
    _unified_manager: Any | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        # [MODIFIED] Do not crash on 0 balance; allow API to fetch it later
        if self.account_balance <= 0:
            self._logger = get_logger(__name__) # Ensure logger is init first
            self._logger.warning("Initial account_balance is 0 or negative. Waiting for DataHub refresh.")
            # Default to a safe fallback (e.g. 1.0) or keep 0, but don't crash
            if self.account_balance < 0: 
                self.account_balance = 0.0
        
        # Ensure logger is initialized if not done above
        if not hasattr(self, "_logger"):
             self._logger = get_logger(__name__)

        self._cooldown_until = None
        # ... rest of the function remains same ...
        self._trading_day_start = self._current_trading_day()
        self._last_pnl_snapshot = float(self.position_manager.get_realized_pnl())
        self._breaker_tripped = False
        self._breaker_reason = None
        self._shadow_forced = False
        self._last_rejection = None
        self._breaker_alerted = False
        self._last_trip_time = None
        pct_cap = max(self.account_balance, 0.0) * (
            self._daily_loss_limit_pct() / 100.0
        )
        abs_cap = max(
            float(getattr(self.settings, "max_daily_loss_absolute", 0.0)), 0.0
        )
        if pct_cap > 0 and abs_cap > 0:
            day_loss_cap = min(pct_cap, abs_cap)
        else:
            day_loss_cap = max(pct_cap, abs_cap)
        self._switches = RiskSwitches(
            max_day_loss=day_loss_cap,
            max_consecutive_losses=self.settings.max_consecutive_losses,
            cooldown_minutes=self.settings.loss_cooldown_seconds / 60.0,
            reset_hour_utc=self.settings.trading_day_reset_hour_utc,
        )
        if abs(self._last_pnl_snapshot) > 1e-6:
            self._switches.record_pnl(self._last_pnl_snapshot)
        self._m_blocks = Counter("risk_blocks_total", "Orders blocked by risk manager")
        self._m_cooldown = Gauge(
            "risk_cooldown_seconds", "Seconds remaining on enforced cooldown"
        )
        self._risk_state = self.risk_state
        self._lot_size_lookup = None
        self._lot_size_symbol = None
        self._cached_balance = float(self.account_balance)
        self._last_balance_refresh = 0.0
        refresh_interval = max(
            5.0,
            float(get_float("RISK__BALANCE_REFRESH_SEC", default=60.0)),
        )
        self._balance_cache_ttl = refresh_interval
        self._balance_force_refresh = max(
            0.0,
            float(get_float("RISK__BALANCE_JIT_REFRESH", default=1.0)),
        )
        self._balance_refresher = None

    @property
    def risk_config(self) -> RiskSettings:
        """Expose the bound risk settings for telemetry consumers."""

        return self.settings

    @property
    def current_balance(self) -> float:
        """Return the account balance used for risk calculations."""

        try:
            return self.refresh_account_balance()
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.error(
                "Failure in RiskManager.current_balance: %s",
                exc,
                extra={"event": "risk_balance_refresh_error"},
                exc_info=exc,
            )
            return self.account_balance

    def balance_source_label(self) -> str:
        """Return label describing the origin of the cached balance.

        Args:
            None.

        Returns:
            str: Human-readable label for the last balance refresh.

        Raises:
            None.
        """

        return self._balance_source

    def set_broker_client(self, broker_client: Any | None) -> None:
        """Attach the broker client used for ancillary integrations.

        Args:
            broker_client: Broker API client capable of returning margin data.

        Returns:
            None.

        Raises:
            None.

        Notes:
            Balance refreshes are delegated to :class:`DataHub`; the broker client
            is retained for compatibility with legacy workflows.
        """

        self._logger.debug(
            "Entered RiskManager.set_broker_client",
            extra={"event": "risk_set_broker_client_start"},
        )
        self._broker_client = broker_client
        self._logger.info(
            "risk_manager_broker_attached",
            extra={
                "event": "risk_manager_broker_attached",
                "has_client": broker_client is not None,
            },
        )

    def set_market_data_manager(self, manager: Any | None) -> None:
        """Attach the market data manager used for balance hydration.

        Args:
            manager: Market data manager reference supplying broker data.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RiskManager.set_market_data_manager",
            extra={"event": "risk_set_mdm_start"},
        )
        self._market_data_manager = manager
        self._logger.info(
            "risk_manager_market_data_manager_attached",
            extra={
                "event": "risk_manager_market_data_manager_attached",
                "attached": manager is not None,
            },
        )
        
        # FIX APPLIED: Removed _start_balance_refresher() call from here.
        # The balance refresher requires DataHub to be attached to work correctly.
        # It will be started by attach_data_hub() instead.

    def set_unified_manager(self, manager: Any | None) -> None:
        """Attach unified manager callbacks for balance updates.

        Args:
            manager: Unified manager instance receiving balance snapshots.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RiskManager.set_unified_manager",
            extra={"event": "risk_set_unified_manager_start"},
        )
        self._unified_manager = manager

    def _notify_unified_manager(self, balance: float, *, source: str) -> None:
        """Forward balance updates to the unified manager when configured.

        Args:
            balance: Account balance value to publish.
            source: Identifier describing the balance origin.

        Returns:
            None.

        Raises:
            None.
        """

        manager = self._unified_manager
        if manager is None:
            return
        callback = getattr(manager, "on_balance_update", None)
        if not callable(callback):
            return
        try:
            callback(float(balance), source=source)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in RiskManager._notify_unified_manager: %s",
                exc,
                extra={
                    "event": "risk_unified_manager_notify_error",
                    "source": source,
                },
                exc_info=exc,
            )

    def attach_data_hub(self, hub: "DataHub | None") -> None:
        """Attach the shared data hub for broker balance hydration.

        Args:
            hub: Data hub instance providing broker account information.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RiskManager.attach_data_hub",
            extra={"event": "risk_attach_data_hub_start"},
        )
        self._data_hub = hub
        self._logger.info(
            "risk_manager_data_hub_attached",
            extra={
                "event": "risk_manager_data_hub_attached",
                "attached": hub is not None,
            },
        )
        if hub is not None:
            try:
                self._start_balance_refresher()
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in RiskManager.attach_data_hub refresher: %s",
                    exc,
                    extra={
                        "event": "risk_balance_refresher_start_error",
                        "source": "data_hub",
                    },
                    exc_info=exc,
                )

    def refresh_account_balance(self, *, force: bool = False) -> float:
        """Refresh account balance using the shared data hub cache.

        Args:
            force: Bypass cache and force a fresh data hub fetch when ``True``.

        Returns:
            float: Current balance derived from the central market data hub.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RiskManager.refresh_account_balance",
            extra={"event": "risk_balance_refresh_start", "force": force},
        )

        now = time.time()
        if not force and now - self._last_balance_refresh < self._balance_cache_ttl:
            self._logger.debug(
                "Condition met: risk_balance_cache_hit",
                extra={"event": "risk_balance_cache_hit"},
            )
            try:
                self._balance_source = "cache"
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "RiskManager balance cache tag failed: %s",
                    exc,
                    extra={"event": "risk_balance_cache_source_tag_error"},
                )
            self._notify_unified_manager(
                self.account_balance, source=self._balance_source
            )
            return self.account_balance

        hub = self._data_hub
        if hub is None:
            self._logger.error(
                "RiskManager data hub not attached",
                extra={"event": "risk_balance_no_data_hub"},
            )
            fallback_balance = self._cached_balance
            if fallback_balance <= 0:
                fallback_balance = self.account_balance
            if fallback_balance <= 0:
                fallback_balance = self._resolve_env_balance_fallback()
                self._logger.warning(
                    "Condition met: risk_balance_env_fallback_no_hub",
                    extra={
                        "event": "risk_balance_env_fallback_no_hub",
                        "fallback": round(fallback_balance, 2),
                    },
                )
            else:
                self._logger.info(
                    "Condition met: risk_balance_cache_fallback",
                    extra={
                        "event": "risk_balance_cache_fallback",
                        "balance": fallback_balance,
                    },
                )
            self.account_balance = float(fallback_balance)
            self._cached_balance = self.account_balance
            self._last_balance_refresh = now
            self._balance_source = "cache" if fallback_balance > 0 else "env"
            self._notify_unified_manager(
                self.account_balance, source=self._balance_source
            )
            return self.account_balance

        balance: float | None = None
        cached_balance_source: str | None = None
        cached_balance_value: float | None = None
        try:
            balance = hub.get_available_balance(force=force)
            if balance is None:
                snapshot = hub.get_account_snapshot(force=force)
                balance = self._extract_balance_from_payload(snapshot)
            if balance is not None and balance > 0:
                self.account_balance = float(balance)
                self._cached_balance = self.account_balance
                self._last_balance_refresh = now
                self._balance_source = "live"
                self._logger.info(
                    "Condition met: balance_updated_from_data_hub",
                    extra={
                        "event": "balance_updated",
                        "balance": round(self.account_balance, 2),
                    },
                )
                self._notify_unified_manager(self.account_balance, source="live")
                return self.account_balance

            if self._cached_balance > 0:
                cached_balance_source = "cached_balance"
                cached_balance_value = float(self._cached_balance)
            elif self.account_balance > 0:
                cached_balance_source = "account_balance"
                cached_balance_value = float(self.account_balance)

            if cached_balance_value is not None and cached_balance_value > 0:
                self._logger.debug(
                    "Condition met: risk_balance_cached_reuse",
                    extra={
                        "event": "risk_balance_cached_reuse",
                        "source": cached_balance_source,
                        "balance": round(cached_balance_value, 2),
                    },
                )
                self.account_balance = cached_balance_value
                self._cached_balance = cached_balance_value
                self._last_balance_refresh = now
                self._balance_source = "cache"
                self._notify_unified_manager(
                    self.account_balance, source=self._balance_source
                )
                return cached_balance_value

            self._logger.warning(
                "No valid balance supplied by data hub",
                extra={"event": "risk_balance_data_hub_empty"},
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in RiskManager.refresh_account_balance: %s",
                exc,
                extra={"event": "balance_fetch_error"},
                exc_info=exc,
            )
            fallback_value = cached_balance_value
            if fallback_value is None or fallback_value <= 0:
                for candidate in (self._cached_balance, self.account_balance):
                    if candidate and candidate > 0:
                        fallback_value = float(candidate)
                        break
            if fallback_value is None:
                fallback_value = self._resolve_env_balance_fallback()
                self._logger.warning(
                    "Condition met: risk_balance_env_exception_fallback",
                    extra={
                        "event": "risk_balance_env_exception_fallback",
                        "fallback": round(fallback_value, 2),
                    },
                )
            self.account_balance = float(fallback_value)
            self._cached_balance = self.account_balance
            self._last_balance_refresh = now
            self._balance_source = "env"
            self._notify_unified_manager(self.account_balance, source="env")
            return self.account_balance

        fallback_balance = self._cached_balance
        if fallback_balance <= 0:
            fallback_balance = self._resolve_env_balance_fallback()
            self._cached_balance = float(fallback_balance)
            self.account_balance = float(fallback_balance)
            self._logger.warning(
                "Condition met: risk_balance_env_cache_fallback",
                extra={
                    "event": "risk_balance_env_cache_fallback",
                    "fallback": round(fallback_balance, 2),
                },
            )
            self._balance_source = "env"
        else:
            self._logger.info(
                "Condition met: balance_cache_fallback",
                extra={
                    "event": "balance_fallback",
                    "balance": fallback_balance,
                },
            )
            self._balance_source = "cache"
        self._last_balance_refresh = now
        self._notify_unified_manager(self._cached_balance, source=self._balance_source)
        return self._cached_balance

    def _start_balance_refresher(self) -> None:
        """Start background thread that periodically refreshes balance.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RiskManager._start_balance_refresher",
            extra={"event": "risk_balance_refresher_enter"},
        )
        if self._balance_cache_ttl <= 0:
            self._logger.debug(
                "Condition met: risk_balance_refresher_disabled",
                extra={"event": "risk_balance_refresher_disabled"},
            )
            return
        if self._balance_refresher is not None and self._balance_refresher.is_alive():
            self._logger.debug(
                "Condition met: risk_balance_refresher_alive",
                extra={"event": "risk_balance_refresher_alive"},
            )
            return

        interval = max(float(self._balance_cache_ttl), 5.0)

        def _run() -> None:
            self._logger.debug(
                "Entered RiskManager._start_balance_refresher loop",
                extra={"event": "risk_balance_refresher_loop_enter"},
            )
            while True:
                try:
                    self.refresh_account_balance(force=True)
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in RiskManager balance refresher: %s",
                        exc,
                        extra={"event": "risk_balance_refresher_loop_error"},
                        exc_info=exc,
                    )
                time.sleep(interval)

        thread = threading.Thread(
            target=_run,
            name="risk-balance-refresher",
            daemon=True,
        )
        thread.start()
        self._balance_refresher = thread
        self._logger.info(
            "risk_balance_refresher_started",
            extra={"event": "risk_balance_refresher_started", "interval": interval},
        )

    def _resolve_env_balance_fallback(self) -> float:
        """Return fallback balance sourced from environment configuration.

        Args:
            None.

        Returns:
            float: Non-negative fallback balance value.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RiskManager._resolve_env_balance_fallback",
            extra={"event": "risk_balance_env_fallback_enter"},
        )
        fallback_value = 1_000_000.0
        try:
            candidates = (
                os.getenv("RISK__CAPITAL"),
                os.getenv("RISK_CAPITAL"),
                os.getenv("BACKTEST__CAPITAL"),
            )
            for candidate in candidates:
                if candidate is None:
                    continue
                token = candidate.strip()
                if not token:
                    continue
                fallback_value = max(float(token), 0.0)
                break
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in RiskManager._resolve_env_balance_fallback: %s",
                exc,
                extra={"event": "risk_balance_env_fallback_error"},
                exc_info=exc,
            )
            fallback_value = 1_000_000.0
        self._logger.info(
            "Condition met: risk_balance_env_fallback_resolved",
            extra={
                "event": "risk_balance_env_fallback_resolved",
                "fallback": round(fallback_value, 2),
            },
        )
        return fallback_value

    def _extract_balance_from_payload(self, payload: Any) -> float | None:
        """Extract an available balance figure from broker margin payloads.

        Args:
            payload: Response returned from the broker margin endpoint.

        Returns:
            float | None: Positive balance when detected, otherwise ``None``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RiskManager._extract_balance_from_payload",
            extra={"event": "risk_balance_extract_start"},
        )
        if isinstance(payload, Mapping):
            candidates: list[tuple[str, ...]] = [
                ("equity", "available", "live_balance"),
                ("equity", "available", "available_cash"),
                ("equity", "available", "cash"),
                ("available", "live_balance"),
                ("available", "available_cash"),
                ("available", "cash"),
                ("available", "opening_balance"),
                ("net", "available"),
                ("available_cash",),
                ("available",),
                ("balance",),
            ]
            for path in candidates:
                value = self._resolve_mapping_path(payload, path)
                if value is not None and value > 0:
                    return value
        if isinstance(payload, (int, float)) and payload > 0:
            return float(payload)
        numeric: float | None = None
        with suppress(Exception):
            numeric = float(payload)
        if numeric is not None and numeric > 0:
            return numeric
        fallback_balance = getattr(self, "_cached_balance", 0.0)
        if fallback_balance and fallback_balance > 0:
            self._logger.warning(
                "Condition met: risk_balance_extract_fallback",
                extra={
                    "event": "risk_balance_extract_fallback",
                    "fallback_balance": fallback_balance,
                },
            )
            return float(fallback_balance)
        self._logger.debug(
            "risk_balance_extract_empty",
            extra={"event": "risk_balance_extract_empty"},
        )
        return None

    def _resolve_mapping_path(
        self, source: Mapping[str, Any], path: tuple[str, ...]
    ) -> float | None:
        """Resolve a nested mapping path into a positive float when present.

        Args:
            source: Mapping containing nested broker payload data.
            path: Sequence of keys describing the lookup path.

        Returns:
            float | None: Positive float when lookup succeeds.

        Raises:
            None.
        """

        current: Any = source
        for key in path:
            if not isinstance(current, Mapping):
                return None
            current = current.get(key)
            if current is None:
                return None
        numeric: float | None
        if isinstance(current, (int, float)):
            numeric = float(current)
        else:
            numeric = None
            with suppress(Exception):
                numeric = float(current)
            if numeric is None:
                return None
        if numeric > 0:
            return numeric
        return None

    # ------------------------------------------------------------------
    def _should_force_balance_refresh(self) -> bool:
        """Return ``True`` when a just-in-time balance refresh is required.

        Args:
            None.

        Returns:
            bool: ``True`` when the cached balance should be refreshed.

        Raises:
            None.
        """

        threshold = float(self._balance_force_refresh)
        if threshold <= 0.0:
            return False
        age = max(time.time() - self._last_balance_refresh, 0.0)
        return age >= threshold

    # ------------------------------------------------------------------
    def check_order(
        self, signal: OrderSignal, live_enabled: bool
    ) -> tuple[bool, str]:  # pragma: no cover
        """Return ``(allowed, reason)`` for the provided order signal."""

        try:
            force_refresh = self._should_force_balance_refresh()
            if force_refresh:
                # [MODIFIED] Log optimization: remove heavy extra dict construction in hot path unless debug
                # age calculation moved inside debug check if needed, or simplified
                self.refresh_account_balance(force=True)
        except Exception as exc:  # noqa: BLE001
            # [MODIFIED] Non-blocking error handling. 
            # If refresh fails, we continue using the cached balance.
            self._logger.warning(
                "RiskManager balance refresh failed during check_order (using cache)",
                extra={"error": str(exc)}
            )
        
        self._reset_daily_if_needed()
        self._refresh_realized_pnl()

        if self._breaker_tripped:
            self._last_rejection = canonical(self._breaker_reason or "BREAKER")
            return False, self._breaker_reason or "BREAKER"

        if not live_enabled:
            self._last_rejection = "TRADING_DISABLED"
            return False, "Live trading disabled"

        override_active = self._soft_override

        reason = self._switches.breach_reason()
        if reason:
            formatted = self._format_switch_reason(reason)
            code = canonical(formatted)
            self._last_rejection = code if code != "OK" else formatted
            if code in SOFT:
                return False, code
            self._trip_breaker(formatted)
            return False, formatted

        cooldown = self.cooldown_remaining()
        self._set_cooldown_metric(cooldown)
        if cooldown > 0 and not override_active:
            self._last_rejection = "COOLDOWN"
            return False, "COOLDOWN"

        if signal.quantity <= 0:
            return False, "Quantity must be positive"
        if signal.price <= 0:
            return False, "Price must be positive"
        if signal.stop_loss is None:
            return False, "Stop loss required"

        if not self._check_per_trade_risk(signal):
            return False, "Per-trade risk limit breached"

        if not self._check_daily_loss_limit():
            self._last_rejection = canonical(self._breaker_reason or "BREAKER")
            return False, self._breaker_reason or "Daily loss limit reached"

        return True, ""

    def is_circuit_breaker_tripped(self) -> tuple[bool, str]:
        """Return the circuit breaker flag and reason.

        Args:
            None.

        Returns:
            tuple[bool, str]: Breaker status flag and descriptive reason text.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RiskManager.is_circuit_breaker_tripped",
            extra={"event": "risk_breaker_tripped_status"},
        )
        try:
            tripped = bool(self._breaker_tripped)
            reason = self._breaker_reason or ""
            if tripped:
                self._logger.info(
                    "Condition met: circuit breaker tripped",
                    extra={
                        "event": "risk_breaker_tripped",
                        "reason": reason or "unknown",
                    },
                )
            return tripped, reason
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in RiskManager.is_circuit_breaker_tripped: %s",
                exc,
                extra={"event": "risk_breaker_tripped_error"},
                exc_info=exc,
            )
            return False, ""

    def is_circuit_breaker_active(self) -> bool:
        """Return ``True`` when the circuit breaker is active.

        Args:
            None.

        Returns:
            bool: ``True`` when breaker lockout should halt trading.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RiskManager.is_circuit_breaker_active",
            extra={"event": "risk_breaker_active_status"},
        )
        try:
            active = bool(self._breaker_tripped)
            if active:
                self._logger.info(
                    "Condition met: circuit breaker active",
                    extra={"event": "risk_breaker_active"},
                )
            return active
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in RiskManager.is_circuit_breaker_active: %s",
                exc,
                extra={"event": "risk_breaker_active_error"},
                exc_info=exc,
            )
            return False

    def bind_risk_state(self, risk_state: "RiskState | None") -> None:
        """Attach the live :class:`RiskState` instance used for gating."""

        self.risk_state = risk_state
        self._risk_state = risk_state

    def risk_gate_should_trade(self) -> tuple[bool, tuple[str, ...]]:
        """Return whether the micro risk state permits new trades.

        Returns:
            Tuple of ``(allowed, reasons)`` where ``reasons`` is an ordered
            tuple of active block codes.
        """

        state = self._risk_state
        if state is None:
            return True, ()
        allowed, reasons = state.should_trade()
        ordered = tuple(sorted(reasons))
        if not allowed:
            self._last_rejection = ",".join(ordered) if ordered else None
        else:
            self._last_rejection = None
        return allowed, ordered

    # Compatibility shim for StrategyRunner
    def validate_new_position(
        self,
        *,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        quantity: int,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> tuple[bool, str]:
        action: OrderSide = "BUY" if side.upper() == "LONG" else "SELL"
        signal = OrderSignal(
            symbol=symbol,
            side=action,
            quantity=quantity,
            price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        allowed, reason = self.check_order(signal, live_enabled=True)
        if not allowed:
            self._record_block(reason)
        return allowed, reason

    # ------------------------------------------------------------------
    def suggest_position_size(
        self,
        *,
        side: OrderSide,
        price: float,
        stop_loss: float | None,
        atr: float | None,
        requested_quantity: int,
        confidence: float | None = None,
        symbol: str | None = None,
    ) -> int:
        """Return quantity respecting per-trade risk sizing."""

        if requested_quantity <= 0 or price <= 0 or stop_loss is None:
            return 0

        sl_distance = abs(price - stop_loss)
        if atr is not None:
            try:
                atr_value = float(atr)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                atr_value = 0.0
            sl_distance = max(sl_distance, abs(atr_value))
        if sl_distance <= 0:
            return 0

        allowed_risk = max(self.account_balance, 0.0) * (
            self.settings.per_trade_risk_pct / 100.0
        )
        if allowed_risk <= 0:
            return 0

        confidence_value = 1.0
        if confidence is not None:
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                confidence_value = 0.0
        confidence_value = max(0.0, min(1.0, confidence_value))

        # [MODIFIED] Safe handling for lot size resolution
        try:
            lot_size = self._resolve_lot_size(symbol)
        except (RuntimeError, ValueError) as exc:
            self._logger.warning(
                f"Cannot size position for {symbol}: {exc}",
                extra={"event": "risk_sizing_missing_lot_info"}
            )
            return 0

        max_units_by_risk = int(allowed_risk // sl_distance)
        if max_units_by_risk < lot_size:
            return 0

        max_lots_by_risk = max_units_by_risk // lot_size
        scaled_lots = int(max_lots_by_risk * confidence_value)
        if scaled_lots == 0 and confidence_value > 0.0 and max_lots_by_risk > 0:
            scaled_lots = 1

        requested_lots = requested_quantity // lot_size
        if requested_lots == 0 and requested_quantity > 0:
            requested_lots = 1

        final_lots = min(requested_lots, scaled_lots)
        if final_lots <= 0:
            return 0

        return final_lots * lot_size

    def set_lot_size_provider(
        self, lookup: Callable[[str], int] | None, *, symbol: str | None = None
    ) -> None:
        """Configure the resolver used for lot-size aware sizing."""

        self._lot_size_lookup = lookup
        self._lot_size_symbol = symbol

    def _resolve_lot_size(self, symbol: str | None) -> int:
        lookup = self._lot_size_lookup
        target_symbol = symbol or self._lot_size_symbol
        if lookup is None or not callable(lookup):
            raise RuntimeError("Lot size provider not configured for risk manager")
        if not target_symbol:
            raise RuntimeError("Lot size symbol not specified for risk sizing")
        resolved = lookup(target_symbol)
        try:
            lot_size = int(resolved)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise RuntimeError("Lot size provider returned invalid value") from exc
        if lot_size <= 0:
            raise RuntimeError("Lot size provider returned non-positive value")
        return lot_size

    def record_fill(self, realized_pnl: float) -> None:  # pragma: no cover
        """Update realised PnL state and consecutive loss counters."""

        self._reset_daily_if_needed()
        self._refresh_realized_pnl()

    def record_rejection(self, reason: str) -> None:  # pragma: no cover
        """Apply a short cooldown after broker rejections."""

        code = canonical(reason)
        self._last_rejection = code
        if code in {"STALE", "COOLDOWN", "RISK_STATE", "TRADING_DISABLED", "MARGIN"}:
            return
        if code in SOFT:
            self._record_block(code)
            return
        if self.settings.cooldown_on_reject_seconds > 0:
            self._cooldown_until = datetime.now(timezone.utc) + timedelta(
                seconds=self.settings.cooldown_on_reject_seconds
            )
            self._switches.engage_cooldown(
                self.settings.cooldown_on_reject_seconds / 60.0
            )
        self._record_block(code)

    def cooldown_remaining(self) -> float:  # pragma: no cover
        remaining = 0.0
        if self._cooldown_until is not None:
            remaining = (
                self._cooldown_until - datetime.now(timezone.utc)
            ).total_seconds()
            if remaining <= 0:
                self._cooldown_until = None
                remaining = 0.0
        switch_remaining = self._switches.cooldown_remaining()
        combined = max(remaining, switch_remaining)
        if combined <= 0:
            return 0.0
        adjusted = combined - (COOLDOWN_SLOP_MS / 1000.0)
        return adjusted if adjusted > 0 else 0.0

    def reset_on_start(self, override: bool) -> None:  # pragma: no cover
        """Clear soft risk blocks and align override state.

        Args:
            override: Whether soft breakers should be bypassed post-reset.
        """

        self._soft_override = bool(override)
        self._cooldown_until = None
        self._last_rejection = None
        try:
            self._switches.engage_cooldown(0.0)
        except Exception:  # pragma: no cover - defensive
            pass
        self._set_cooldown_metric(0.0)

    def snapshot(self) -> RiskSnapshot:  # pragma: no cover
        """Return the current risk snapshot."""

        self._reset_daily_if_needed()
        self._refresh_realized_pnl()
        return self._build_snapshot()

    def is_green(self) -> bool:  # pragma: no cover
        """Return ``True`` if no breakers/cooldowns are active."""

        snap = self.snapshot()
        if snap.breaker_tripped:
            return False
        if snap.cooldown_remaining > 0:
            return False
        if snap.losses_in_row >= self.settings.max_consecutive_losses:
            return False
        if snap.max_day_loss > 0 and snap.day_loss >= snap.max_day_loss:
            return False
        if snap.daily_loss_limit > 0 and snap.daily_realized <= -snap.daily_loss_limit:
            return False
        return True

    def force_shadow(self, enabled: bool) -> None:  # pragma: no cover
        """Record whether shadow (paper) mode is forced by the breaker."""

        self._shadow_forced = bool(enabled)

    def validate_close_position(
        self, *, symbol: str, exit_price: float | None = None
    ) -> tuple[bool, str]:  # pragma: no cover
        """Allow closing positions unless a breaker is active."""

        self._reset_daily_if_needed()
        self._refresh_realized_pnl()
        if self._breaker_tripped:
            return False, self._breaker_reason or "BREAKER"
        return True, ""

    # ------------------------------------------------------------------
    def _check_per_trade_risk(self, signal: OrderSignal) -> bool:  # pragma: no cover
        stop_loss = signal.stop_loss
        if stop_loss is None:
            return False
        risk_per_unit = abs(signal.price - stop_loss)
        if risk_per_unit <= 0:
            return False
        allowed_risk = max(self.account_balance, 0.0) * (
            self.settings.per_trade_risk_pct / 100.0
        )
        if allowed_risk <= 0:
            return False
        return abs(risk_per_unit * signal.quantity) <= allowed_risk

    def _check_daily_loss_limit(self) -> bool:  # pragma: no cover
        loss = self._switches.day_loss()
        pct_cap = max(self.account_balance, 0.0) * (
            self._daily_loss_limit_pct() / 100.0
        )
        abs_cap = max(
            float(getattr(self.settings, "max_daily_loss_absolute", 0.0)), 0.0
        )
        if pct_cap > 0 and abs_cap > 0:
            cap = min(pct_cap, abs_cap)
        else:
            cap = max(pct_cap, abs_cap)
        cap = max(cap, self._switches.max_day_loss)
        if cap <= 0:
            return True
        if loss >= cap:
            self._trip_breaker(f"Daily loss {loss:.2f} exceeded limit {cap:.2f}")
            return False
        return True

    def _reset_daily_if_needed(self) -> None:  # pragma: no cover
        now = datetime.now(timezone.utc)
        if now >= self._trading_day_start + timedelta(days=1):
            self._trading_day_start = self._current_trading_day()
            self._breaker_tripped = False
            self._breaker_reason = None
            if self.settings.breaker_auto_shadow:
                self._shadow_forced = False
            self._last_rejection = None
            self._breaker_alerted = False
            self._last_pnl_snapshot = float(self.position_manager.get_realized_pnl())
            pct_cap = max(self.account_balance, 0.0) * (
                self._daily_loss_limit_pct() / 100.0
            )
            abs_cap = max(
                float(getattr(self.settings, "max_daily_loss_absolute", 0.0)), 0.0
            )
            if pct_cap > 0 and abs_cap > 0:
                self._switches.max_day_loss = min(pct_cap, abs_cap)
            else:
                self._switches.max_day_loss = max(pct_cap, abs_cap)
            self._switches.reset_day()

    def _current_trading_day(self) -> datetime:  # pragma: no cover
        now = datetime.now(timezone.utc)
        reset_hour = int(self.settings.trading_day_reset_hour_utc)
        anchor = now.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
        if now < anchor:
            anchor -= timedelta(days=1)
        return anchor

    def _refresh_realized_pnl(self) -> None:  # pragma: no cover
        """Refresh realized PnL tracking and synchronize metrics.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        current = float(self.position_manager.get_realized_pnl())
        delta = current - self._last_pnl_snapshot
        if abs(delta) < 1e-6:
            return
        self._switches.record_pnl(delta)
        self._last_pnl_snapshot = current
        try:
            METRICS.set_live_pnl(book="primary", value=current)
        except Exception:  # pragma: no cover - optional metrics
            self._logger.debug("Unable to update live PnL metric", exc_info=True)
        try:
            METRICS.set_pnl_breakdown(book="primary", realized=current)
        except Exception:  # pragma: no cover - optional metrics
            self._logger.debug("Unable to update realized PnL metric", exc_info=True)
        reason = self._switches.breach_reason()
        if reason:
            formatted = self._format_switch_reason(reason)
            self._trip_breaker(formatted)

    def _trip_breaker(self, reason: str) -> None:  # pragma: no cover
        if self._breaker_tripped:
            return
        self._breaker_tripped = True
        self._breaker_reason = reason
        self._last_trip_time = datetime.now(timezone.utc)
        self._schedule_breaker_alert(reason)
        if self.settings.breaker_auto_shadow:
            self._shadow_forced = True
        snapshot = self._build_snapshot()
        if self.alert_callback is not None:
            try:
                self.alert_callback(reason, snapshot)
            except Exception:  # pragma: no cover - defensive
                self._logger.debug("Risk alert callback failed", exc_info=True)

    def _schedule_breaker_alert(self, reason: str) -> None:
        """Schedule asynchronous alert dispatch when the breaker trips.

        Args:
            reason: Human-readable reason associated with the breaker trip.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RiskManager._schedule_breaker_alert",
            extra={"event": "risk_breaker_alert_schedule", "reason": reason},
        )
        if self.breaker_alert_sender is None:
            return
        if self._breaker_alerted:
            return
        try:
            loop = asyncio.get_running_loop()
            # [MODIFIED] Ensure the loop is actually running/usable
            if loop.is_closed():
                raise RuntimeError("Loop is closed")
        except RuntimeError:
            # This is expected behavior when running in a synchronous thread (like the Runner)
            # We log as debug, not warning, to avoid spamming logs.
            self._logger.debug(
                "Breaker alert skipped: no async loop in current thread",
                extra={"event": "risk_breaker_alert_no_loop", "reason": reason},
            )
            return
        try:
            task = loop.create_task(
                self._send_breaker_alert(reason),
                name="risk-breaker-alert",
            )
        except Exception as exc:  # pragma: no cover - event loop guard
            self._logger.error(
                "Failure in RiskManager._schedule_breaker_alert: %s",
                exc,
                extra={"event": "risk_breaker_alert_schedule_error"},
                exc_info=exc,
            )
            return
        self._breaker_alerted = True
        task.add_done_callback(
            lambda completed: self._log_alert_result(completed, reason)
        )

    async def _send_breaker_alert(self, reason: str) -> None:
        """Dispatch the breaker alert via the configured asynchronous sender.

        Args:
            reason: Text describing the breaker trigger condition.

        Returns:
            None.

        Raises:
            None.
        """

        sender = self.breaker_alert_sender
        if sender is None:
            return
        self._logger.info(
            "Condition met: risk breaker alert dispatch",
            extra={"event": "risk_breaker_alert_dispatch", "reason": reason},
        )
        try:
            await sender(reason)
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.error(
                "Failure in RiskManager._send_breaker_alert: %s",
                exc,
                extra={"event": "risk_breaker_alert_error", "reason": reason},
                exc_info=exc,
            )

    def _log_alert_result(self, task: asyncio.Future[Any], reason: str) -> None:
        """Log completion status for scheduled breaker alert dispatches.

        Args:
            task: Asyncio future representing the scheduled alert coroutine.
            reason: Breaker trigger reason associated with the alert.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RiskManager._log_alert_result",
            extra={"event": "risk_breaker_alert_result", "reason": reason},
        )
        if task.cancelled():
            self._logger.warning(
                "Breaker alert dispatch task cancelled",
                extra={"event": "risk_breaker_alert_cancelled", "reason": reason},
            )
            self._breaker_alerted = False
            return
        exc = task.exception()
        if exc is not None:
            self._logger.error(
                "Breaker alert dispatch raised: %s",
                exc,
                extra={"event": "risk_breaker_alert_exception", "reason": reason},
                exc_info=exc,
            )
            self._breaker_alerted = False

    def _format_switch_reason(self, reason: str) -> str:
        lowered = reason.lower()
        if "day loss" in lowered:
            loss = self._switches.day_loss()
            limit = self._switches.max_day_loss
            if limit > 0:
                return f"Daily loss limit reached ({loss:.2f}/{limit:.2f})"
            return "Daily loss limit reached"
        if "consecutive" in lowered:
            current_losses = self._switches.consecutive_losses()
            max_losses = self.settings.max_consecutive_losses
            if max_losses > 0:
                return (
                    "Consecutive loss limit reached " f"({current_losses}/{max_losses})"
                )
            return "Consecutive loss limit reached"
        return reason

    def _build_snapshot(self) -> RiskSnapshot:  # pragma: no cover
        switch_snap = self._switches.snapshot()
        cooldown = self.cooldown_remaining()
        self._set_cooldown_metric(cooldown)
        pct_cap = max(self.account_balance, 0.0) * (
            self._daily_loss_limit_pct() / 100.0
        )
        limit = max(pct_cap, 0.0)
        if switch_snap.max_day_loss > 0:
            if limit > 0:
                limit = min(limit, switch_snap.max_day_loss)
            else:
                limit = switch_snap.max_day_loss
        return RiskSnapshot(
            daily_realized=self._last_pnl_snapshot,
            daily_loss_limit=limit,
            day_loss=switch_snap.day_loss,
            max_day_loss=switch_snap.max_day_loss,
            losses_in_row=switch_snap.consecutive_losses,
            cooldown_remaining=cooldown,
            breaker_tripped=self._breaker_tripped,
            breaker_reason=self._breaker_reason,
            shadow_forced=self._shadow_forced,
            per_trade_risk_pct=self.settings.per_trade_risk_pct,
            last_rejection=self._last_rejection,
            timestamp=datetime.now(timezone.utc),
        )

    def _record_block(self, reason: str) -> None:  # pragma: no cover
        code = canonical(reason)
        self._last_rejection = code if code != "OK" else reason
        try:
            self._m_blocks.inc()
        except Exception:  # pragma: no cover - optional metrics
            pass
        self._logger.info(
            "Risk block",
            extra={
                "reason": code,
                "reason_code": self._last_rejection,
                "reason_text": reason,
            },
        )

    def _set_cooldown_metric(self, value: float) -> None:  # pragma: no cover
        try:
            self._m_cooldown.set(value)
        except Exception:  # pragma: no cover - optional metrics
            pass

    def _daily_loss_limit_pct(self) -> float:  # pragma: no cover
        return getattr(
            self.settings,
            "daily_loss_pct",
            getattr(self.settings, "daily_pnl_cap_pct", 0.0),
        )


@dataclass(slots=True)
class RiskState:
    """Microstructure-aware kill-switch and drawdown tracker.

    Args:
        quote_stale_ms_threshold: Maximum allowable quote age before trading halts.
        spread_widen_mult: Spread multiple vs 60s median before blocking trading.
        max_intraday_dd: Maximum permitted drawdown relative to the session peak.
        max_consecutive_losses: Maximum consecutive losing trades allowed.
    """

    quote_stale_ms_threshold: int
    spread_widen_mult: float
    max_intraday_dd: float
    max_consecutive_losses: int
    _window_ns: int = field(init=False, default=60_000_000_000)
    _last_tick_ns: int = field(init=False, default=-1)
    _spread_history: deque[tuple[int, float]] = field(init=False, default_factory=deque)
    _median_spread: float = field(init=False, default=0.0)
    _peak_equity: float = field(init=False, default=0.0)
    _current_equity: float = field(init=False, default=0.0)
    _last_realized: float = field(init=False, default=0.0)
    _consecutive_losses: int = field(init=False, default=0)
    _reasons: set[str] = field(init=False, default_factory=set)
    _initialized_equity: bool = field(init=False, default=False)
    _data_hub: "DataHub | None" = field(init=False, default=None, repr=False)
    _data_symbol: str | None = field(init=False, default=None, repr=False)

    def on_tick(self, bid: float, ask: float, ts_ns: int) -> None:
        """Update spread statistics and latency guards from the latest quote."""

        if ask <= bid:
            return
        spread = ask - bid
        self._last_tick_ns = ts_ns
        self._spread_history.append((ts_ns, spread))
        cutoff = ts_ns - self._window_ns
        while self._spread_history and self._spread_history[0][0] < cutoff:
            self._spread_history.popleft()
        spreads = [entry[1] for entry in self._spread_history]
        if spreads:
            self._median_spread = median(spreads)
        self._evaluate_spread(spread)
        self._reasons.discard("STALE")

    def on_trade_update(self, realized_pnl: float, unrealized_pnl: float) -> None:
        """Record realized/unrealized PnL for drawdown tracking.

        Args:
            realized_pnl: Aggregate realized profit and loss value.
            unrealized_pnl: Aggregate unrealized mark-to-market profit and loss.

        Returns:
            None.

        Raises:
            None.
        """

        try:
            METRICS.set_pnl_breakdown(
                book="primary", realized=realized_pnl, unrealized=unrealized_pnl
            )
        except Exception:  # pragma: no cover - optional metrics
            LOGGER.debug("Unable to update PnL breakdown", exc_info=True)
        equity = realized_pnl + unrealized_pnl
        if not self._initialized_equity:
            self._peak_equity = equity
            self._initialized_equity = True
        else:
            self._peak_equity = max(self._peak_equity, equity)
        self._current_equity = equity
        drawdown = equity - self._peak_equity
        dd_limit = -abs(self.max_intraday_dd)
        if drawdown <= dd_limit:
            self._reasons.add("DD")
        else:
            self._reasons.discard("DD")

        delta_realized = realized_pnl - self._last_realized
        if delta_realized < 0:
            self._consecutive_losses += 1
        elif delta_realized > 0:
            self._consecutive_losses = 0
        self._last_realized = realized_pnl
        if self._consecutive_losses >= self.max_consecutive_losses > 0:
            self._reasons.add("LOSER_STREAK")
        else:
            self._reasons.discard("LOSER_STREAK")

    def should_trade(self, now_ns: int | None = None) -> tuple[bool, frozenset[str]]:
        """Return whether trading is allowed and current block reasons."""

        if now_ns is None:
            now_ns = time.time_ns()
        stale_reason: str | None = None
        hub = self._data_hub
        symbol = self._data_symbol or ""
        if hub is not None and symbol:
            try:
                fresh, _meta = hub.is_fresh(symbol)
            except Exception:  # pragma: no cover - defensive
                fresh = True
            if not fresh:
                stale_reason = "STALE"
        else:
            if self._last_tick_ns < 0:
                stale_reason = "STALE"
            else:
                age_ms = (now_ns - self._last_tick_ns) / 1_000_000
                if age_ms > self.quote_stale_ms_threshold:
                    stale_reason = "STALE"

        if stale_reason:
            self._reasons.add(stale_reason)
        else:
            self._reasons.discard("STALE")
        allowed = not self._reasons
        return allowed, frozenset(self._reasons)

    def _evaluate_spread(self, spread: float) -> None:
        if self._median_spread <= 0:
            self._reasons.discard("WIDE_SPREAD")
            return
        if spread > self._median_spread * self.spread_widen_mult:
            self._reasons.add("WIDE_SPREAD")
        else:
            self._reasons.discard("WIDE_SPREAD")

    def attach_data_hub(
        self, hub: "DataHub | None", *, symbol: str | None = None
    ) -> None:
        """Attach the shared data hub for delegated freshness checks."""

        self._data_hub = hub
        if hub is None:
            self._data_symbol = None
            return
        if symbol is None:
            self._data_symbol = None
            return
        try:
            normalized = hub.normalize(symbol)
        except Exception:  # pragma: no cover - defensive
            normalized = str(symbol or "").strip().upper()
        self._data_symbol = normalized or None


__all__ = ["OrderSignal", "RiskManager", "RiskSnapshot", "RiskState"]
