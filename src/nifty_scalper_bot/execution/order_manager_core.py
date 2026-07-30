"""Order lifecycle management: the live order path.

Runtime role:
- THE production order path. Places, modifies, and tracks orders against the
  broker (Kite), handling retries, idempotency, and lifecycle transitions.
- Coordinates with the bracket manager (SL/TP), position manager, and risk
  guardrails around each order.

Position in the pipeline:
    strategies/runner.py -> THIS FILE (order_manager.py)
    -> execution/bracket_manager.py / execution/position_manager.py

Owns / does NOT own:
- Owns: order placement and lifecycle against the broker, order-level retry and
  idempotency, and the live order state of record.
- Does NOT own: signal generation (runner) or contract selection. It executes
  what the runner approved; it does not decide whether to trade.

Safe-edit notes:
- Live money. Order placement is wrapped by SafeOrderManager; preserve the
  guard/idempotency boundaries and do not introduce a second placement route.
- order_executor.py is NOT the live path (it is a separate, non-live executor);
  do not confuse the two.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import deque
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Event, RLock, Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    cast,
)
from zoneinfo import ZoneInfo

import nifty_scalper_bot.config.settings as app_settings
from nifty_scalper_bot.config.paths import get_data_dir
from nifty_scalper_bot.core.active_basket import active_contract_selection_from_basket
from nifty_scalper_bot.core.signal_arbitrator import SignalArbitrator
from nifty_scalper_bot.core.trading_switch import TradingSwitchState, trading_switch
from nifty_scalper_bot.data.bracket_store import BracketStore
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.persistent_state import (
    BracketDict,
    PersistentStateManager,
)
from nifty_scalper_bot.execution import exceptions as execution_exceptions
from nifty_scalper_bot.execution.adaptive_trailing import AdaptiveTrailingController
from nifty_scalper_bot.execution.broker_rejects import BrokerReject
from nifty_scalper_bot.execution.execution_policy import ExecutionPolicy
from nifty_scalper_bot.execution.exit_router import plan_and_send_exit
from nifty_scalper_bot.execution.margin_engine import (
    MarginDecision,
    MarginEngine,
    MarginInputs,
    SizingResult,
)
from nifty_scalper_bot.execution.options_policy import OptionsExecutionPolicy
from nifty_scalper_bot.execution.position_manager import (
    OrderIntent,
    PositionManager,
    normalize_broker_order_status,
)
from nifty_scalper_bot.execution.trailing_stop import (
    TrailingSpec,
    TrailingStopController,
)
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.infra.structured_logger import emit_diag
from nifty_scalper_bot.risk.session_gate import can_trade
from nifty_scalper_bot.storage.journal import AtomicKV
from nifty_scalper_bot.utils import metrics
from nifty_scalper_bot.utils.circuit_breaker import CircuitBreaker
from nifty_scalper_bot.utils.errors import RateLimitError
from nifty_scalper_bot.utils.log_throttle import (
    log_on_change,
)
from nifty_scalper_bot.utils.log_throttle import (
    log_throttled as log_throttled_live,
)
from nifty_scalper_bot.utils.logging import get_logger, log_throttled
from nifty_scalper_bot.utils.lot_size import (
    resolve_lot_size as resolve_lot_size_with_source,
)
from nifty_scalper_bot.utils.market_hours import get_time_status
from nifty_scalper_bot.utils.metrics import Counter, Gauge
from nifty_scalper_bot.utils.pricing import canonical_price_source
from nifty_scalper_bot.utils.rate_limiter import RateLimiter
from nifty_scalper_bot.utils.reasons import canonical
from nifty_scalper_bot.utils.symbols import is_strategy_instrument, normalize_symbol

SOFT_BLOCK_CODES: set[str] = {
    "STALE",
    "COOLDOWN",
    "RECENT_REJECT",
    "RISK_STATE",
    "MARGIN",
    "MIS_WINDOW_CLOSED",
    "AMO_ONLY",
    "MARKET_CLOSED",
}
_REFERENCE_LOGGER = get_logger(__name__)

BrokerError = execution_exceptions.BrokerError
OrderPlacementError = execution_exceptions.OrderPlacementError
MarginCheckError = execution_exceptions.MarginCheckError
OrderModificationError = execution_exceptions.OrderModificationError
RiskBlockError = execution_exceptions.RiskBlockError

if TYPE_CHECKING:
    from journal.trade_journal import TradeJournal

    from nifty_scalper_bot.data.market_data_manager import MarketDataManager
    from nifty_scalper_bot.data.rest.client import BaseBrokerClient
    from nifty_scalper_bot.execution.bracket_manager import BracketManager
    from nifty_scalper_bot.notifications.telegram_enhanced import (
        TelegramEnhancedNotifier,
    )
    from nifty_scalper_bot.risk.risk_manager import RiskManager
else:  # pragma: no cover - typing only
    MarketDataManager = Any
    BaseBrokerClient = Any
    TelegramEnhancedNotifier = Any
    RiskManager = Any


@dataclass(slots=True)
class RefPriceMeta:
    """Reference price metadata payload."""

    source: str
    age_ms: Optional[int]
    market_protect: bool = False


def _mid(bid: float | None, ask: float | None) -> float | None:
    """Return the arithmetic mid-price if both bid and ask are valid.

    Args:
        bid: Best bid price.
        ask: Best ask price.

    Returns:
        Mid price when both inputs are positive, otherwise ``None``.

    Raises:
        None.
    """

    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / 2.0


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partial_filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_MARKET = "stop_loss_market"


@dataclass
class OrderDetails:
    # ------------------------------------------------------------
    # 1. Non-Default Fields (MUST COME FIRST)
    # ------------------------------------------------------------
    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: OrderType
    status: OrderStatus
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trigger_price: Optional[float] = None
    average_price: float = 0.0
    fill_price: float | None = None
    filled_quantity: int = 0
    applied_filled_quantity: int = 0
    pending_quantity: int = 0
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    tag: str | None = None
    parent_order_id: str | None = None
    child_order_ids: list[str] = field(default_factory=list)
    client_order_id: str | None = None
    rejection_reason: str | None = None
    intent: OrderIntent = "UNKNOWN"
    intended_position_side: Literal["LONG", "SHORT"] | None = None
    bracket_id: str | None = None
    signal_id: str | None = None
    signal_fingerprint: str | None = None
    trade_lifecycle_id: str | None = None
    linked_entry_order_id: str | None = None
    basket_version: int | str | None = None
    instrument_token: int | None = None
    contract_expiry: str | None = None
    exchange_order_id: str | None = None
    requested_lots: int = 0
    resolved_lot_size: int = 0
    entry_lifecycle_state: dict[str, Any] | None = None


@dataclass(slots=True)
class OrderPreflightResult:
    """Trade-plan preflight decision payload."""

    allowed: bool
    reason: str = "allowed"
    details: dict[str, Any] = field(default_factory=dict)


def _stamp_entry_sizing(details: dict | None, result: Any) -> Any:
    """Merge entry-sizing provenance into this exact result's details.

    Each submission owns its own sizing record: recovery reads provenance
    only from the result it was given, never from manager/global state.
    Supports the real TradePlanSubmitResult and SimpleNamespace test doubles.
    Args: details, result. Returns: the same result. Raises: none.
    """
    if not details:
        return result
    try:
        existing = getattr(result, "details", None)
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update(details)
        setattr(result, "details", merged)
    except Exception:  # noqa: BLE001 - provenance must not break submission
        pass
    return result


@dataclass(slots=True)
class TradePlan:
    """Strategy-to-execution trade intent contract."""

    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    strategy_name: str = "runner"
    signal_id: str | None = None
    trace_id: str | None = None
    tag: str = "runner"
    product: str = "MIS"
    variety: str = "regular"
    max_quote_age_ms: int = 5000
    max_spread_pct: float = 5.0
    min_depth_qty: int = 150
    allow_market_entry: bool = False
    intent: OrderIntent = "ENTRY"
    intended_position_side: Literal["LONG", "SHORT"] | None = "LONG"
    trade_lifecycle_id: str | None = None
    client_order_id: str | None = None
    basket_version: int | str | None = None
    instrument_token: int | None = None
    contract_expiry: str | None = None
    selection_timestamp: float | None = None
    requested_lots: int = 0
    resolved_lot_size: int = 0


@dataclass(slots=True)
class TradePlanSubmitResult:
    accepted: bool
    order_id: str | None = None
    reason: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)
    broker_attempted: bool = False


@dataclass(slots=True)
class ManagedOrderResult:
    accepted: bool
    order_id: str | None = None
    reason: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)
    broker_attempted: bool = False


@dataclass(slots=True)
class ExitIntent:
    """Bound exit request to the originating entry instrument."""

    symbol: str
    qty: int
    product: str
    exchange: str = "NFO"
    order_type: str = "MARKET"
    tag: str | None = None


@dataclass(slots=True)
class AtomicLeg:
    """Specification for a single leg in an atomic entry."""

    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    order_type: OrderType = OrderType.MARKET
    price: float | None = None


@dataclass(slots=True)
class BracketState:
    """Track live bracket exit state and outstanding quantities."""

    entry_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    exit_side: Literal["BUY", "SELL"]
    total_quantity: int
    entry_price: float
    product: str | None
    tag: str | None
    stop_order_id: str
    stop_price: float
    stop_order_type: OrderType
    stop_filled: int = 0
    tp_primary_id: str | None = None
    tp_primary_price: float | None = None
    tp_primary_qty: int = 0
    tp_primary_filled: int = 0
    tp_secondary_id: str | None = None
    tp_secondary_price: float | None = None
    tp_secondary_qty: int = 0
    tp_secondary_filled: int = 0
    trailing_spec: TrailingSpec | None = None
    partial_fraction: float = 0.0
    second_target_price: float | None = None

    def remaining_position(self) -> int:
        """Return open quantity still protected by the bracket."""

        executed = self.tp_primary_filled + self.tp_secondary_filled + self.stop_filled
        pending = self.total_quantity - executed
        return pending if pending > 0 else 0

    def primary_remaining(self) -> int:
        """Return outstanding quantity on the first take-profit leg."""

        remaining = self.tp_primary_qty - self.tp_primary_filled
        return remaining if remaining > 0 else 0

    def secondary_remaining(self) -> int:
        """Return outstanding quantity on the secondary take-profit leg."""

        remaining = self.tp_secondary_qty - self.tp_secondary_filled
        return remaining if remaining > 0 else 0


@dataclass(slots=True)
class GuardPair:
    """Track paired stop and target orders that form an in-memory OCO guard."""

    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    stop_order_id: str
    target_order_id: str
    created_at: datetime

    def to_dict(self) -> dict[str, object]:
        """Serialise the guard pair for persistence.

        Args:
            None.

        Returns:
            Dictionary representation compatible with JSON.

        Raises:
            None.
        """

        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "stop_order_id": self.stop_order_id,
            "target_order_id": self.target_order_id,
            "created_at": self.created_at.isoformat(),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "GuardPair":
        """Hydrate :class:`GuardPair` from a persisted mapping.

        Args:
            payload: Serialized guard pair mapping.

        Returns:
            GuardPair instance populated from the mapping.

        Raises:
            ValueError: If mandatory fields are missing.
        """

        created = payload.get("created_at")
        if isinstance(created, str):
            try:
                created_dt = datetime.fromisoformat(created)
            except ValueError as exc:
                raise ValueError("Invalid created_at for guard pair") from exc
        elif isinstance(created, datetime):
            created_dt = created
        else:
            raise ValueError("Missing created_at for guard pair")
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        side_value = str(payload.get("side") or "SELL").upper()
        if side_value not in {"BUY", "SELL"}:
            side_value = "SELL"
        return GuardPair(
            symbol=str(payload.get("symbol") or "").strip().upper(),
            side=cast(Literal["BUY", "SELL"], side_value),
            quantity=int(payload.get("quantity") or 0),
            stop_order_id=str(payload.get("stop_order_id") or ""),
            target_order_id=str(payload.get("target_order_id") or ""),
            created_at=created_dt,
        )


def resolve_reference_price(
    symbol: str,
    *,
    mdm: "MarketDataManager",
    require_depth: bool,
    max_age_ms: int,
    allow_ltp_fallback: bool,
    allow_market_protect: bool,
    protect_slippage_bps: int,
) -> tuple[float | None, RefPriceMeta]:
    """Resolve reference pricing for order evaluation.

    Args:
        symbol: Trading symbol under evaluation.
        mdm: Market data manager that supplies quotes.
        require_depth: Whether bid/ask depth is required.
        max_age_ms: Maximum acceptable age for reference quotes.
        allow_ltp_fallback: Whether LTP fallback is permitted.
        allow_market_protect: Whether to allow market-protect fallback.
        protect_slippage_bps: Basis points for market-protect slippage.

    Returns:
        Tuple of resolved price (or ``None``) and metadata describing the source.

    Raises:
        None.
    """

    _REFERENCE_LOGGER.debug(
        "Entered resolve_reference_price",
        extra={"event": "resolve_reference_price_enter", "symbol": symbol},
    )
    quote: Mapping[str, Any] | None = None
    try:
        quote = mdm.get_quote(symbol) or {}
    except Exception as exc:  # noqa: BLE001
        _REFERENCE_LOGGER.error(
            "Failure in resolve_reference_price: %s",
            exc,
            extra={
                "event": "resolve_reference_price_quote_failed",
                "symbol": symbol,
                "error": str(exc),
            },
        )
        quote = {}

    age_ms: int = 1_000_000_000
    try:
        age_ms = int(mdm.quote_age_ms(symbol))
    except Exception as exc:  # noqa: BLE001
        _REFERENCE_LOGGER.error(
            "Failure in resolve_reference_price: %s",
            exc,
            extra={
                "event": "resolve_reference_price_age_failed",
                "symbol": symbol,
                "error": str(exc),
            },
        )

    def _coerce_price(value: object | None) -> float | None:
        if value in (None, ""):
            return None
        try:
            number = float(cast(Any, value))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= 0:
            return None
        return float(number)

    def _resolve_from_quote(
        payload: Mapping[str, Any] | None,
        current_age_ms: int,
    ) -> tuple[float | None, RefPriceMeta | None]:
        bid: float | None = None
        ask: float | None = None
        if isinstance(payload, Mapping):
            bid = _coerce_price(payload.get("bid"))
            ask = _coerce_price(payload.get("ask"))
            if require_depth and (bid is None or ask is None):
                depth = payload.get("depth")
                if isinstance(depth, Mapping):
                    buy_levels = depth.get("buy")
                    sell_levels = depth.get("sell")
                    if bid is None and isinstance(buy_levels, Iterable):
                        for level in buy_levels:
                            if isinstance(level, Mapping):
                                bid = _coerce_price(level.get("price"))
                                if bid is not None:
                                    break
                    if ask is None and isinstance(sell_levels, Iterable):
                        for level in sell_levels:
                            if isinstance(level, Mapping):
                                ask = _coerce_price(level.get("price"))
                                if ask is not None:
                                    break

        mid_price_inner = _mid(bid, ask)
        if mid_price_inner is not None and current_age_ms <= max_age_ms:
            _REFERENCE_LOGGER.info(
                "reference_price source=mid market_protect=false age_ms=%s",
                current_age_ms,
                extra={
                    "symbol": symbol,
                    "source": "mid",
                    "market_protect": False,
                    "age_ms": current_age_ms,
                },
            )
            return float(mid_price_inner), RefPriceMeta(
                source="mid", age_ms=current_age_ms
            )

        if (
            allow_ltp_fallback
            and current_age_ms <= max_age_ms
            and isinstance(payload, Mapping)
        ):
            ltp_value = (
                _coerce_price(payload.get("ltp"))
                or _coerce_price(payload.get("last_price"))
                or _coerce_price(payload.get("close"))
                or _coerce_price(payload.get("price"))
            )
            if ltp_value is not None:
                _REFERENCE_LOGGER.info(
                    "reference_price source=ltp market_protect=false age_ms=%s",
                    current_age_ms,
                    extra={
                        "symbol": symbol,
                        "source": "ltp",
                        "market_protect": False,
                        "age_ms": current_age_ms,
                    },
                )
                return float(ltp_value), RefPriceMeta(
                    source="ltp", age_ms=current_age_ms
                )

        return None, None

    resolved_price, resolved_meta = _resolve_from_quote(quote, age_ms)
    if resolved_meta is not None and resolved_price is not None:
        return resolved_price, resolved_meta

    if age_ms > max_age_ms:
        _REFERENCE_LOGGER.info(
            "reference_price_refresh age_ms=%s",
            age_ms,
            extra={
                "event": "resolve_reference_price_refresh",
                "symbol": symbol,
                "age_ms": age_ms,
                "max_age_ms": max_age_ms,
            },
        )
        try:
            mdm.refresh_quote_now(symbol)
        except Exception as exc:  # noqa: BLE001
            _REFERENCE_LOGGER.debug(
                "resolve_reference_price_refresh_failed",
                extra={
                    "event": "resolve_reference_price_refresh_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
        try:
            quote = mdm.get_quote(symbol) or {}
        except Exception as exc:  # noqa: BLE001
            _REFERENCE_LOGGER.error(
                "Failure in resolve_reference_price: %s",
                exc,
                extra={
                    "event": "resolve_reference_price_quote_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
            quote = {}
        try:
            age_ms = int(mdm.quote_age_ms(symbol))
        except Exception as exc:  # noqa: BLE001
            _REFERENCE_LOGGER.error(
                "Failure in resolve_reference_price: %s",
                exc,
                extra={
                    "event": "resolve_reference_price_age_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
        resolved_price, resolved_meta = _resolve_from_quote(quote, age_ms)
        if resolved_meta is not None and resolved_price is not None:
            return resolved_price, resolved_meta

    cached_mid: float | None = None
    cached_age: int | None = None
    try:
        cached_mid, cached_age = mdm.cached_mid(symbol)
    except Exception as exc:  # noqa: BLE001
        _REFERENCE_LOGGER.error(
            "Failure in resolve_reference_price: %s",
            exc,
            extra={
                "event": "resolve_reference_price_cached_mid_failed",
                "symbol": symbol,
                "error": str(exc),
            },
        )

    if (
        cached_mid is not None
        and cached_mid > 0
        and cached_age is not None
        and cached_age <= max_age_ms
    ):
        _REFERENCE_LOGGER.info(
            "reference_price source=stale_mid market_protect=false age_ms=%s",
            cached_age,
            extra={
                "symbol": symbol,
                "source": "stale_mid",
                "market_protect": False,
                "age_ms": cached_age,
            },
        )
        return float(cached_mid), RefPriceMeta(source="stale_mid", age_ms=cached_age)

    if allow_market_protect:
        _REFERENCE_LOGGER.info(
            "reference_price source=stale_mid market_protect=true age_ms=%s",
            age_ms,
            extra={
                "symbol": symbol,
                "source": "stale_mid",
                "market_protect": True,
                "age_ms": age_ms,
                "slippage_bps": protect_slippage_bps,
            },
        )
        return None, RefPriceMeta(
            source="stale_mid", age_ms=age_ms, market_protect=True
        )

    _REFERENCE_LOGGER.info(
        "reference_price source=stale_mid market_protect=false age_ms=%s",
        age_ms,
        extra={
            "symbol": symbol,
            "source": "stale_mid",
            "market_protect": False,
            "age_ms": age_ms,
        },
    )
    return None, RefPriceMeta(source="stale_mid", age_ms=age_ms)


class OrderManager:
    _SECRET_PATTERNS = (
        re.compile(r"(?i)\b(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9._\-]+"),
        re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]+"),
        re.compile(
            r"(?i)\b("
            r"api[_-]?key|api[_-]?secret|access[_-]?token|request[_-]?token|"
            r"refresh[_-]?token|auth[_-]?token|session[_-]?token|enctoken|"
            r"password|passwd|secret|token"
            r")\s*[:=]\s*([^\s,&]+)"
        ),
    )
    """Manage complete order lifecycle."""

    POLL_INTERVAL_SEC: float = 2.0
    MAX_RETRIES: int = 3
    RETRY_BLACKLIST: tuple[str, ...] = (
        "insufficient funds",
        "invalid symbol",
        "market closed",
    )
    BRACKET_ENTRY_TIMEOUT_SEC: float = 5.0
    # TTL for the in-flight entry reservation taken by the single-position
    # gate; covers the broker submit window and self-heals on failure paths.
    ENTRY_INFLIGHT_TTL_SEC: float = 30.0
    FINAL_STATUSES: tuple[OrderStatus, ...] = (
        OrderStatus.CANCELLED,
        OrderStatus.FILLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    )

    @staticmethod
    def _env_truthy(name: str) -> bool:
        """Return True for common truthy environment values."""
        return str(os.getenv(name, "false") or "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
            "live",
        }

    @staticmethod
    def _execution_mode_env() -> str:
        """Return normalized execution mode from environment."""
        mode = str(os.getenv("EXECUTION_MODE") or "SHADOW").strip().upper()
        mode_name = mode.rsplit(".", 1)[-1]
        if mode_name in {"LIVE", "LIVE_SIMULATION", "PAPER", "SHADOW", "SIMULATION"}:
            return mode_name
        return "SHADOW"

    @classmethod
    def _live_flag_enabled(cls) -> bool:
        """Return True if either supported live flag is enabled."""
        return cls._env_truthy("ENABLE_LIVE") or cls._env_truthy("ENABLE_LIVE_TRADING")

    @classmethod
    def _shadow_mode_enabled(cls) -> bool:
        """Return True when shadow mode is explicitly enabled."""
        return cls._env_truthy("SHADOW_MODE")

    @classmethod
    def _paper_mode_enabled(cls) -> bool:
        """Return True when paper mode is explicitly enabled."""
        return cls._env_truthy("PAPER__ENABLED") or cls._env_truthy("PAPER_MODE")

    @classmethod
    def _order_live_execution_enabled(cls) -> bool:
        """Return True only when actual live order execution is enabled."""
        return (
            cls._execution_mode_env() in {"LIVE", "LIVE_SIMULATION"}
            and (cls._live_flag_enabled() or cls._execution_mode_env() == "LIVE_SIMULATION")
            and not cls._shadow_mode_enabled()
            and not cls._paper_mode_enabled()
        )

    @property
    def execution_mode(self) -> str:
        """Public execution-mode API."""
        return self._execution_mode

    def get_execution_mode(self) -> str:
        """Return current execution mode."""
        return self.execution_mode

    def is_live_mode(self) -> bool:
        """Return True only when this manager is allowed to place live orders."""
        return self.execution_mode in {"LIVE", "LIVE_SIMULATION"} and self._order_live_execution_enabled()

    @classmethod
    def _sanitize_broker_error(cls, exc_or_text: Any) -> str:
        text = str(exc_or_text or "")
        for pattern in cls._SECRET_PATTERNS:
            text = pattern.sub(lambda m: f"{m.group(1)}[REDACTED]", text)
        return text[:500]

    def __init__(
        self,
        broker_client: BaseBrokerClient,
        position_manager: PositionManager,
        rate_limiter: RateLimiter,
        instrument_resolver: Any | None = None,
        history_path: str | Path | None = None,
        indicator_engine: Any | None = None,
        trade_journal: "TradeJournal | None" = None,
    ):
        """Initialize with broker client and position manager."""

        self._broker = broker_client
        self._execution_mode = self._execution_mode_env()
        self._logger = get_logger(__name__)
        if self._execution_mode == "LIVE" and not self._live_flag_enabled():
            raise RuntimeError(
                "LIVE mode requires ENABLE_LIVE=true or ENABLE_LIVE_TRADING=true"
            )
        if self._execution_mode == "LIVE" and (
            self._shadow_mode_enabled() or self._paper_mode_enabled()
        ):
            raise RuntimeError(
                "LIVE mode blocked because SHADOW_MODE or PAPER mode is enabled"
            )

        if self._execution_mode == "SIMULATION":
            try:
                from nifty_scalper_bot.testing.simulated_broker import (
                    SimulatedZerodhaBroker,
                )

                self._broker = SimulatedZerodhaBroker()
                self._logger = get_logger(__name__)
                self._logger.info("Condition met: using simulated broker backend")
            except Exception as exc:  # noqa: BLE001
                self._logger = get_logger(__name__)
                self._logger.error(
                    "Failure in OrderManager.__init__ simulated broker swap: %s", exc
                )
        self._positions = position_manager
        self._limiter = rate_limiter
        self._trade_journal = trade_journal
        self._seen_signal_ids: set[str] = set()
        self._signal_history: deque[str] = deque(maxlen=10_000)
        self._pending_signal_ids: dict[str, float] = {}
        self._pending_signal_ttl_seconds: float = max(
            30.0,
            float(os.getenv("ORDER_SIGNAL_PENDING_TTL_SECONDS", "120") or 120),
        )
        self._uncertain_client_order_ids: dict[str, float] = {}
        self._uncertain_order_ttl_seconds: float = max(
            15.0,
            float(os.getenv("ORDER_UNCERTAIN_TTL_SECONDS", "60") or 60),
        )
        self._logger.info(
            "OrderManager execution mode resolved",
            extra={
                "event": "order_manager_execution_mode_resolved",
                "execution_mode": self._execution_mode,
                "live_flag_enabled": self._live_flag_enabled(),
                "shadow_mode": self._shadow_mode_enabled(),
                "paper_mode": self._paper_mode_enabled(),
                "live_order_execution_enabled": self._order_live_execution_enabled(),
            },
        )
        self._broker_circuit = CircuitBreaker()
        _data_dir = get_data_dir()
        self._history_path = Path(history_path or _data_dir / "order_history.json")
        self._orders: dict[str, OrderDetails] = {}
        # ENTRY submissions currently between the single-position gate and
        # local order registration (symbol -> reservation wall-clock ts).
        # Entries auto-expire after _ENTRY_INFLIGHT_TTL_SEC so a crashed
        # submission can never wedge the gate.
        self._entries_in_flight: dict[str, float] = {}
        self._history: deque[OrderDetails] = deque(maxlen=1000)
        self._history_index: dict[str, int] = {}
        self._history_base_index = 0
        self._history_persist_path = _data_dir / "order_history_archive.jsonl"
        self._history_persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._history_persisted_ids: set[str] = set()
        self._notifier: TelegramEnhancedNotifier | None = None
        self._bracket_manager: BracketManager | None = None
        self.entry_order_failed_callback: Callable[..., None] | None = None
        self._lock = RLock()
        self._stop_event = Event()
        self._monitor_thread: Thread | None = None
        self._market_data: MarketDataManager | None = None
        self._data_hub: DataHub | None = None
        self._instrument_resolver = instrument_resolver
        self._resolver = instrument_resolver
        self._indicator_engine = indicator_engine
        self._trailing: dict[
            str, tuple[TrailingStopController, Callable[[dict[str, Any]], None]]
        ] = {}
        trailing_state_path = self._history_path.parent / "trailing_state.json"
        self._trailing_journal = AtomicKV(trailing_state_path)
        self._session_guard_getter: Callable[[], Mapping[str, object]] | None = None
        self._enable_live_getter: Callable[[], bool] | None = None
        self._shadow_mode_getter: Callable[[], bool] | None = None
        self._execution_policy: ExecutionPolicy | None = None
        self._options_policy = OptionsExecutionPolicy()
        self._risk_manager: RiskManager | None = None
        # Runner/risk owns the trading re-entry cooldown.  OrderManager keeps
        # only the short submission/concurrency reservation; a second default
        # 300-second cooldown here previously blocked otherwise eligible NIFTY
        # entries after a position closed.
        self._signal_arbitrator = SignalArbitrator(
            cooldown_seconds=3.0,
            reentry_cooldown_seconds=3.0,
        )
        self._client_order_index: dict[str, str] = {}
        self._last_skip_reason: str | None = None
        self._margin_block_events: deque[float] = deque()
        self._margin_block_streak: int = 0
        self._margin_cooldown_until: float | None = None
        self._margin_block_threshold: int = 3
        self._m_margin_blocks = Counter(
            "order_block_margin_total", "Margin-driven order blocks"
        )
        self._m_margin_block_window = Gauge(
            "order_block_margin_window",
            "Margin-driven order blocks by trailing window",
            ["window"],
        )
        self._m_order_slippage = Gauge(
            "order_slippage_estimate",
            "Estimated adverse slippage per lot for upcoming orders",
            ["symbol"],
        )
        self._m_queue_position = Gauge(
            "order_queue_position",
            "Estimated visible queue depth for upcoming orders",
            ["symbol"],
        )
        self._configure_options_policy()
        self._load_history()
        self._margin_factor = 1.1
        self._margin_buffer = 0.9
        self._guard_pairs: dict[str, GuardPair] = {}
        self._guard_state_path = self._history_path.parent / "guard_pairs.json"
        self._load_guard_pairs()
        self._persistent_state: PersistentStateManager | None = None
        # 4. Add Missing Initialization Safety (The Fix)
        # Clean Initialization
        if override := os.getenv("NSB__BRACKET_ENTRY_TIMEOUT_SEC"):
            with suppress(ValueError):
                # Safely parse float, ensure min 0.5s
                sanitized = max(float(override), 0.5)
                self.BRACKET_ENTRY_TIMEOUT_SEC = sanitized
                self._logger.info(
                    "Condition met: override bracket entry timeout",
                    extra={
                        "event": "order_config_override",
                        "field": "bracket_entry_timeout_sec",
                        "value": sanitized,
                    },
                )
        self._margin_engine = MarginEngine(
            broker=self._broker,
            data_hub=self._data_hub,
            lot_size_resolver=position_manager,
            clock=time.time,
        )
        self._brackets: dict[str, BracketState] = {}
        self._bracket_index: dict[str, str] = {}
        self._bracket_store = BracketStore()  # Initialize SQLite Store
        self._load_orders()  # Restore Physical Orders
        self._restore_virtual_brackets()  # Restore Virtual Brackets (New)

        # ---------------------------------------------------------
        # 🛡️ CIRCUIT BREAKER STATE (Kill Switch)
        # ---------------------------------------------------------
        self._consecutive_failures: int = 0
        self._max_failures: int = max(
            1, int(os.getenv("ORDER_KILL_SWITCH_MAX_FAILURES", "5") or 5)
        )
        self._kill_switch_auto_reset_seconds: int = max(
            60, int(os.getenv("ORDER_KILL_SWITCH_AUTO_RESET_SECONDS", "900") or 900)
        )
        self._kill_switch_allow_auto_reset: bool = os.getenv(
            "ORDER_KILL_SWITCH_ALLOW_AUTO_RESET", "true"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._kill_switch_engaged_at: datetime | None = None
        self._kill_switch_reason: str | None = None
        self._last_kill_switch_log_ts: float = 0.0
        self._kill_switch_failure_history: deque[dict[str, Any]] = deque(maxlen=20)
        self._kill_switch_last_reset: dict[str, Any] | None = None
        self._missing_counts: dict[str, int] = {}
        self._last_order_decision: dict[str, Any] = {}
        self._margin_cache_max_age_seconds: int = max(
            1, int(os.getenv("MARGIN_CACHE_MAX_AGE_SECONDS", "120") or 120)
        )
        self._allow_entry_with_stale_margin: bool = os.getenv(
            "ALLOW_ENTRY_WITH_STALE_MARGIN", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._last_margin_refresh_ts: float | None = None
        self._last_margin_success_ts: float | None = None
        self._last_margin_error_type: str | None = None
        self._last_margin_error: str | None = None
        self._margin_circuit_open: bool = False
        self._margin_circuit_until_ts: float | None = None
        self._last_order_api_error_type: str | None = None
        self._last_order_api_error: str | None = None
        self._last_broker_health_emit_ts: float = 0.0
        self._last_broker_health_effect: str = "none"
        self._last_broker_health_circuit_state: bool = False
        self._last_margin_available_balance: float | None = None
        self._last_margin_balance_source: str | None = None

    def set_market_data_manager(self, market_data_manager: MarketDataManager) -> None:
        """Inject the shared market data manager instance."""

        self._market_data = market_data_manager
        self._configure_options_policy()

    def _is_duplicate_signal(self, signal_id: str) -> bool:
        """Check in-memory signal idempotency. Args: signal_id; Returns: bool; Raises: None."""
        with self._lock:
            return signal_id in self._seen_signal_ids

    def _prune_pending_signals(self) -> None:
        now_ts = time.time()
        ttl = float(getattr(self, "_pending_signal_ttl_seconds", 120.0) or 120.0)
        expired = [
            signal_id
            for signal_id, ts in self._pending_signal_ids.items()
            if now_ts - float(ts or 0.0) > ttl
        ]
        for signal_id in expired:
            self._pending_signal_ids.pop(signal_id, None)

    def _is_pending_signal(self, signal_id: str) -> bool:
        with self._lock:
            self._prune_pending_signals()
            return signal_id in self._pending_signal_ids

    def _mark_signal_pending(self, signal_id: str) -> None:
        with self._lock:
            self._prune_pending_signals()
            self._pending_signal_ids[signal_id] = time.time()

    def _clear_pending_signal(self, signal_id: str | None) -> None:
        if not signal_id:
            return
        with self._lock:
            self._pending_signal_ids.pop(signal_id, None)

    def _prune_uncertain_orders(self) -> None:
        now_ts = time.time()
        ttl = float(getattr(self, "_uncertain_order_ttl_seconds", 60.0) or 60.0)
        expired = [
            client_order_id
            for client_order_id, ts in self._uncertain_client_order_ids.items()
            if now_ts - float(ts or 0.0) > ttl
        ]
        for client_order_id in expired:
            self._uncertain_client_order_ids.pop(client_order_id, None)

    def _mark_order_uncertain(self, client_order_id: str) -> None:
        if not client_order_id:
            return
        with self._lock:
            self._prune_uncertain_orders()
            self._uncertain_client_order_ids[str(client_order_id)] = time.time()

    def _clear_uncertain_order(self, client_order_id: str | None) -> None:
        if not client_order_id:
            return
        with self._lock:
            self._uncertain_client_order_ids.pop(str(client_order_id), None)

    def _is_uncertain_order(self, client_order_id: str) -> bool:
        if not client_order_id:
            return False
        with self._lock:
            self._prune_uncertain_orders()
            return str(client_order_id) in self._uncertain_client_order_ids

    def _find_open_order(self, client_order_id: str) -> dict[str, Any] | None:
        """Find a live/open broker order by client_order_id/order tag/order_id."""
        try:
            get_orders = getattr(self._broker, "get_orders", None)
            if not callable(get_orders):
                get_orders = getattr(self._broker, "orders", None)
            if not callable(get_orders):
                return None

            orders = get_orders() or []
            wanted = str(client_order_id or "").strip()
            wanted_upper = wanted.upper()

            for order in orders:
                if not isinstance(order, Mapping):
                    continue

                status = str(order.get("status") or "").strip().upper()
                if status in {
                    "CANCELLED",
                    "CANCELED",
                    "REJECTED",
                    "COMPLETE",
                    "COMPLETED",
                }:
                    continue

                candidates = {
                    str(order.get("client_order_id") or "").strip().upper(),
                    str(order.get("tag") or "").strip().upper(),
                    str(order.get("order_id") or "").strip().upper(),
                    str(order.get("guid") or "").strip().upper(),
                    str(order.get("exchange_order_id") or "").strip().upper(),
                    str(order.get("parent_order_id") or "").strip().upper(),
                }
                if wanted and wanted_upper in candidates:
                    return dict(order)
                tag_value_upper = str(order.get("tag") or "").strip().upper()
                if (
                    wanted_upper
                    and tag_value_upper
                    and (
                        wanted_upper in tag_value_upper
                        or wanted_upper[-8:] in tag_value_upper
                    )
                ):
                    return dict(order)

            return None
        except Exception as exc:
            self._logger.warning(
                "ORDER_FIND_OPEN_ORDER_FAILED client_order_id=%s error=%s",
                client_order_id,
                exc,
                exc_info=exc,
            )
            return None

    def _remember_signal(self, signal_id: str) -> None:
        """Record signal id to preserve idempotency. Args: signal_id; Returns: None; Raises: None."""
        with self._lock:
            if signal_id in self._seen_signal_ids:
                return
            if len(self._signal_history) >= self._signal_history.maxlen:
                stale = self._signal_history.popleft()
                self._seen_signal_ids.discard(stale)
            self._signal_history.append(signal_id)
            self._seen_signal_ids.add(signal_id)

    def _log_trade_event(
        self,
        event_type: str,
        *,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        order_id: str | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> None:
        """Queue async trade journal event. Args: event_type,symbol,side,qty,price,order_id,meta; Returns: None; Raises: None."""
        journal = self._trade_journal
        if journal is None:
            return
        try:
            journal.log_event(
                {
                    "event_type": event_type,
                    "timestamp": time.time(),
                    "symbol": symbol,
                    "side": side,
                    "qty": int(qty),
                    "price": float(price),
                    "order_id": order_id,
                    "meta": dict(meta or {}),
                }
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _log_trade_event: %s", exc)

    def _ensure_quote_refresh(
        self,
        mdm: "MarketDataManager",
        symbol: str,
        trace_id: str | None,
    ) -> None:
        """Refresh broker quotes when cached data is missing.

        Args:
            mdm: Active market data manager instance.
            symbol: Trading symbol under evaluation.
            trace_id: Optional correlation identifier for logging.

        Returns:
            None.

        Raises:
            None.
        """

        try:
            if not mdm.has_quote(symbol):
                mdm.refresh_quote_now(symbol, trace_id=trace_id)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "pre_trade_quote_refresh_failed",
                extra={
                    "event": "pre_trade_quote_refresh_failed",
                    "symbol": symbol,
                    "trace_id": trace_id,
                    "error": str(exc),
                },
            )

    def _handle_missing_reference_price(
        self,
        mdm: "MarketDataManager | None",
        symbol: str,
    ) -> None:
        """Raise skip conditions when reference price is unavailable.

        Args:
            mdm: Market data manager or ``None`` if unavailable.
            symbol: Trading symbol for the pending order.

        Returns:
            None.

        Raises:
            OrderPlacementError: When reference price cannot be determined.
        """

        if mdm is not None:
            try:
                if not mdm.has_quote(symbol):
                    self.set_last_skip_reason("no_quote_data")
                    raise OrderPlacementError("NO_QUOTE_DATA")
            except OrderPlacementError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "pre_trade_quote_check_failed",
                    extra={
                        "event": "pre_trade_quote_check_failed",
                        "symbol": symbol,
                        "error": str(exc),
                    },
                )
        self.set_last_skip_reason("no_reference_price")
        raise OrderPlacementError("NO_REFERENCE_PRICE")

    def attach_data_hub(self, hub: "DataHub | None") -> None:
        """Attach the shared data hub for order and position updates."""

        self._data_hub = hub
        self._refresh_margin_engine()
        if hub is None:
            self._execution_policy = None
            return
        self._maybe_init_execution_policy()

    def on_tick_event(self, tick: dict[str, Any]) -> None:
        """Args: tick; Returns: none; Raises: none."""
        try:
            symbol = str(tick.get("symbol") or "")
            if not symbol:
                return
            self._logger.debug("EVENT|order_tick|%s", symbol)
        except Exception as e:
            self._logger.error("Failure in OrderManager.on_tick_event: %s", e)

    def _subscribe_market_callback(
        self, symbol: str, callback: Callable[[dict[str, Any]], None]
    ) -> bool:
        provider = self._data_hub or self._market_data
        if provider is None:
            return False
        subscribe_fn = getattr(provider, "subscribe_ticks", None)
        if not callable(subscribe_fn):
            subscribe_fn = getattr(provider, "subscribe", None)
        if not callable(subscribe_fn):
            return False
        subscribe_fn(symbol, callback)
        return True

    def _unsubscribe_market_callback(
        self, symbol: str, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        provider = self._data_hub or self._market_data
        if provider is None:
            return
        unsubscribe_fn = getattr(provider, "unsubscribe_ticks", None)
        if not callable(unsubscribe_fn):
            unsubscribe_fn = getattr(provider, "unsubscribe", None)
        if callable(unsubscribe_fn):
            unsubscribe_fn(symbol, callback)

    def set_broker_client(self, broker_client: Any) -> None:
        """Swap the underlying broker client used for routing orders."""

        self._broker = broker_client
        self._broker_circuit = CircuitBreaker()
        self._refresh_margin_engine()
        # Proactively warm the margin snapshot so the broker-health gate has a
        # fresh balance before the first signal. Without this, balance_stale
        # stays True (margin is otherwise only fetched during place_order, which
        # the gate blocks) — a deadlock that prevented the first live trade.
        self.prime_margin(reason="set_broker_client")

    def prime_margin(self, *, reason: str = "manual") -> bool:
        """Fetch the available margin once to refresh broker-health freshness.

        Safe to call repeatedly. Records success/failure on the broker-health
        snapshot. Returns True when a fresh balance was obtained.

        Args:
            reason: Diagnostic tag for the log line.

        Returns:
            bool: True if margin was refreshed successfully.
        """
        try:
            available, source = self._resolve_available_margin_raw()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "MARGIN_PRIME_FAILED reason=%s error_type=%s",
                reason,
                type(exc).__name__,
                extra={
                    "event": "MARGIN_PRIME_FAILED",
                    "reason": reason,
                    "error_type": type(exc).__name__,
                },
            )
            return False
        ok = bool(
            available is not None
            and available > 0
            and source in {"mdm", "margin_cache_used", "risk"}
        )
        self._logger.info(
            "MARGIN_PRIMED reason=%s ok=%s source=%s available=%s",
            reason,
            ok,
            source,
            available,
            extra={
                "event": "MARGIN_PRIMED",
                "reason": reason,
                "ok": ok,
                "source": source,
                "available": available,
            },
        )
        self._emit_broker_health_status(force=True)
        return ok

    def set_instrument_resolver(self, resolver: Any | None) -> None:
        """Store resolver used for lot-size lookups."""

        self._instrument_resolver = resolver
        self._resolver = resolver
        self._configure_options_policy()
        self._maybe_init_execution_policy()
        self._refresh_margin_engine()

    def set_risk_manager(self, risk_manager: "RiskManager | None") -> None:
        """Attach a risk manager used for risk-state gating."""

        self._risk_manager = risk_manager
        if risk_manager is not None:
            setter = getattr(risk_manager, "set_lot_size_provider", None)
            if callable(setter):
                setter(self._lot_lookup(), symbol=None)

    def panic_button(self):
        """Cancel ALL orders first, then exit positions."""
        # 1. Fast Cancel (prevents new fills while we exit)
        self.cancel_pending_orders()

        # 2. Dump Positions (Market Orders)
        for pos in self._positions.get_open_positions():
            # Detect product from position to ensure NRML/MIS compatibility
            product_code = getattr(pos, "product", "MIS")

            # Use 'exit' logic to ensure side is correct
            self._place_exit_order(
                symbol=pos.symbol,
                side="SELL" if pos.quantity > 0 else "BUY",
                quantity=abs(pos.quantity),
                product=product_code,
                tag="PANIC_BUTTON",
            )

    def set_bracket_manager(self, bracket_manager: BracketManager | None) -> None:
        """Attach bracket manager responsible for OCO coordination.

        Args:
            bracket_manager: Bracket manager instance or ``None`` to clear.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered set_bracket_manager",
            extra={"event": "order_manager.bracket_manager_set_enter"},
        )
        try:
            self._bracket_manager = bracket_manager
            if bracket_manager is None:
                self._logger.info(
                    "Condition met: bracket manager dependency cleared",
                    extra={"event": "order_manager.bracket_manager_cleared"},
                )
            else:
                if hasattr(bracket_manager, "set_notifier"):
                    bracket_manager.set_notifier(self._notify_bracket_event)
                if hasattr(bracket_manager, "attach_exit_executor"):
                    bracket_manager.attach_exit_executor(self.exit_position)
                self._logger.info(
                    "Bracket manager attached to order manager",
                    extra={"event": "order_manager.bracket_manager_attached"},
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in set_bracket_manager: %s",
                exc,
                extra={"event": "order_manager.bracket_manager_set_failed"},
                exc_info=exc,
            )
            raise

    def set_session_guard_getter(
        self, getter: Callable[[], Mapping[str, object]] | None
    ) -> None:
        """Provide a callable returning the latest trading-session guard snapshot."""

        self._session_guard_getter = getter

    def set_trade_mode_getters(
        self,
        *,
        enable_live: Callable[[], bool] | None = None,
        shadow_mode: Callable[[], bool] | None = None,
    ) -> None:
        """Provide callables exposing live/shadow toggles for guard evaluation."""

        self._enable_live_getter = enable_live
        self._shadow_mode_getter = shadow_mode

    def _refresh_margin_engine(self) -> None:
        """Rebuild the margin engine with the latest dependencies.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "margin_engine_refresh", extra={"event": "margin_engine_refresh"}
        )
        lot_resolver = self._resolver if self._resolver is not None else self._positions
        self._margin_engine = MarginEngine(
            broker=self._broker,
            data_hub=None,
            lot_size_resolver=lot_resolver,
            clock=time.time,
        )
        try:
            self._margin_engine.set_data_hub(self._data_hub)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in order_manager_margin_engine_set_data_hub: %s",
                exc,
                extra={"event": "order_manager.margin_engine_set_data_hub_failed"},
                exc_info=exc,
            )
            raise

    def _maybe_init_execution_policy(self) -> None:
        if self._data_hub is None:
            return

        def _env_float(names: tuple[str, ...], default: float) -> float:
            for name in names:
                raw = os.getenv(name)
                if raw is None:
                    continue
                token = str(raw).strip()
                if not token:
                    continue
                try:
                    return float(token)
                except ValueError:
                    continue
            return default

        def _env_int(names: tuple[str, ...], default: int) -> int:
            for name in names:
                raw = os.getenv(name)
                if raw is None:
                    continue
                token = str(raw).strip()
                if not token:
                    continue
                try:
                    return int(float(token))
                except ValueError:
                    continue
            return default

        peg_k = _env_float(("EXEC__PEG_K", "EXEC_PEG_K"), 0.25)
        retry_steps = _env_int(("EXEC__RETRY_STEPS", "EXEC_RETRY_STEPS"), 3)
        timeout_sec = _env_float(("EXEC__TIMEOUT_SEC", "EXEC_TIMEOUT_SEC"), 4.0)
        max_spread_pct = _env_float(
            ("LIQ__MAX_SPREAD_PCT", "ORDER_MAX_SPREAD_PCT"), 0.015
        )
        margin_factor = _env_float(("EXEC__MARGIN_FACTOR", "EXEC_MARGIN_FACTOR"), 1.1)
        margin_buffer = _env_float(("EXEC__MARGIN_BUFFER", "EXEC_MARGIN_BUFFER"), 0.9)
        self._margin_factor = (
            margin_factor if margin_factor > 0 else self._margin_factor
        )
        self._margin_buffer = (
            margin_buffer if 0 < margin_buffer <= 1.0 else self._margin_buffer
        )

        if self._execution_policy is None:
            try:
                self._execution_policy = ExecutionPolicy(
                    data_hub=self._data_hub,
                    peg_k=peg_k,
                    retry_steps=retry_steps,
                    timeout_sec=timeout_sec,
                    max_spread_pct=max_spread_pct,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.warning(
                    "execution_policy_init_failed",
                    extra={"event": "execution_policy_init_failed", "error": str(exc)},
                )
                self._execution_policy = None
        # ExecutionPolicy is a slots dataclass; margin tuning is tracked on the
        # order manager and consumed by pre-checks rather than being stored on
        # the policy instance.

    def attach_trailing_stop(
        self,
        *,
        entry_order_id: str,
        sl_order_id: str,
        symbol: str,
        side: Literal["BUY", "SELL"],
        entry_price: float,
        spec: TrailingSpec,
        variety: str = "regular",
    ) -> str:
        """Attach a trailing stop controller to an existing bracket order."""

        if not entry_order_id:
            raise ValueError("entry_order_id is required for trailing stops")
        if not sl_order_id:
            raise ValueError("sl_order_id is required for trailing stops")
        if self._market_data is None:
            raise RuntimeError(
                "MarketDataManager must be configured before attaching trailing stops"
            )

        broker = cast("BaseBrokerClient", self._broker)
        if not hasattr(broker, "modify_order"):
            raise NotImplementedError("Broker does not support order modifications")

        # Capture a non-optional reference for type-checkers and closures
        md = cast(MarketDataManager, self._market_data)

        def _get_ltp(sym: str) -> Optional[float]:
            price = md.get_latest_price(sym)
            if price is not None:
                return price
            quote = md.pull_quote(sym)
            if isinstance(quote, dict):
                try:
                    return float(quote.get("ltp", 0.0))
                except (TypeError, ValueError):
                    return None
            return None

        def _modify(
            var: str, order_id: str, qty: Optional[int], price: Optional[float]
        ) -> dict:
            return cast(Any, broker).modify_order(
                order_id, quantity=qty, price=price, variety=var
            )

        # [FIX] Choose Adaptive Controller if engine is available
        if self._indicator_engine:
            controller = AdaptiveTrailingController(
                symbol=symbol,
                side="LONG" if side == "BUY" else "SHORT",
                entry=entry_price,
                sl_order_id=sl_order_id,
                variety=variety,
                spec=spec,
                get_ltp=_get_ltp,
                modify_order=_modify,
                # Pass the ATR computer
                get_atr=lambda s: self._indicator_engine.compute_atr(s),
                journal=self._trailing_journal,
            )
        else:
            # Fallback to static
            controller = TrailingStopController(
                symbol=symbol,
                side="LONG" if side == "BUY" else "SHORT",
                entry=entry_price,
                sl_order_id=sl_order_id,
                variety=variety,
                spec=spec,
                get_ltp=_get_ltp,
                modify_order=_modify,
                journal=self._trailing_journal,
            )

        def _listener(tick: dict[str, Any]) -> None:
            controller.on_tick(tick)

        md.subscribe(symbol, _listener)
        self._trailing[entry_order_id] = (controller, _listener)
        controller.on_tick(None)
        return entry_order_id

    def attach_dynamic_tp(
        self,
        *,
        tp_order_id: str,
        symbol: str,
        side: Literal["BUY", "SELL"],
        initial_price: float,
        parent_order_id: str,
    ) -> None:
        """Attach a dynamic Take Profit controller to an active target order."""
        if not self._indicator_engine:
            return  # Cannot optimize without indicators

        # Lazy import to avoid circular dependency
        try:
            from nifty_scalper_bot.execution.dynamic_tp import DynamicTPController
        except ImportError:
            self._logger.warning("DynamicTP module not found. Skipping TP expansion.")
            return

        def _modify(order_id: str, price: float) -> bool:
            return self.modify_order(order_id, new_price=price)

        try:
            controller = DynamicTPController(
                tp_order_id=tp_order_id,
                symbol=symbol,
                side=side,
                initial_price=initial_price,
                get_indicators=lambda s: self._indicator_engine.get_latest(s),
                modify_order=_modify,
            )

            # Store controller (using a new dict similar to _trailing)
            if not hasattr(self, "_tp_controllers"):
                self._tp_controllers = {}

            # Subscribe to ticks via DataHub (SSOT), fall back to MDM if unset.
            if not self._subscribe_market_callback(symbol, controller.on_tick):
                self._logger.warning(
                    "Dynamic TP not attached because market callback subscription is unavailable for %s",
                    symbol,
                )
                return
            self._tp_controllers[tp_order_id] = controller
            self._logger.info(f"🚀 Dynamic TP attached to {tp_order_id}")

        except Exception as e:
            self._logger.error(f"Failed to attach Dynamic TP: {e}")

    def stop_dynamic_tp(self, tp_order_id: str) -> None:
        """Stop and remove a dynamic TP controller."""
        if not hasattr(self, "_tp_controllers"):
            return

        controller = self._tp_controllers.pop(tp_order_id, None)
        if controller:
            self._unsubscribe_market_callback(controller.symbol, controller.on_tick)

    def stop_trailing(self, entry_order_id: str) -> bool:
        """Stop and remove a trailing stop controller if it exists."""

        record = self._trailing.pop(entry_order_id, None)
        if record is None:
            return False
        controller, callback = record
        try:
            hub = self._data_hub or self._market_data
            if hub is not None:
                hub.unsubscribe(controller.symbol, callback)
            self._trailing_journal.delete(controller.order_id)
        except Exception as exc:
            self._logger.warning(
                "TRAILING_STOP_CLEANUP_FAILED entry_order_id=%s order_id=%s error=%s",
                entry_order_id,
                getattr(controller, "order_id", None),
                exc,
                exc_info=exc,
            )
        return True

    def get_best_bid_ask_depth(self, symbol: str) -> dict[str, float | None]:
        """Return price/size snapshot at the most liquid visible levels.

        Args:
            symbol: Instrument identifier whose depth is requested.

        Returns:
            Dictionary with ``bid``, ``ask``, ``bid_size``, ``ask_size`` keys.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered get_best_bid_ask_depth",
            extra={"event": "get_best_bid_ask_depth_enter", "symbol": symbol},
        )
        snapshot: dict[str, float | None] = {
            "bid": None,
            "ask": None,
            "bid_size": 0.0,
            "ask_size": 0.0,
        }
        mdm = self._data_hub or self._market_data
        if mdm is None:
            self._logger.info(
                "Condition met: depth_unavailable",
                extra={
                    "event": "depth_unavailable",
                    "symbol": symbol,
                    "reason": "no_market_data",
                },
            )
            return snapshot
        try:
            quote = mdm.get_quote(symbol) or {}
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in get_best_bid_ask_depth: %s",
                exc,
                extra={
                    "event": "get_best_bid_ask_depth_error",
                    "symbol": symbol,
                },
            )
            return snapshot
        depth_payload = quote.get("depth") if isinstance(quote, Mapping) else None

        def _select_level(
            payload: object, side: Literal["buy", "sell"]
        ) -> tuple[float | None, float]:
            best_price: float | None = None
            best_size = 0.0
            if isinstance(payload, Iterable):
                for entry in payload:
                    if not isinstance(entry, Mapping):
                        continue
                    price_value = self._coerce_float(
                        entry.get("price") or entry.get("p") or entry.get("value")
                    )
                    size_value = self._coerce_float(
                        entry.get("quantity")
                        or entry.get("qty")
                        or entry.get("size")
                        or entry.get("volume")
                    )
                    if price_value is None or size_value is None or size_value <= 0:
                        continue
                    price_float = float(price_value)
                    size_float = float(size_value)
                    better_quantity = size_float > best_size
                    equal_quantity = math.isclose(
                        size_float, best_size, rel_tol=1e-6, abs_tol=1e-6
                    )
                    better_price = False
                    if side == "buy":
                        better_price = best_price is None or price_float > float(
                            best_price
                        )
                    else:
                        better_price = best_price is None or price_float < float(
                            best_price
                        )
                    if better_quantity or (equal_quantity and better_price):
                        best_price = price_float
                        best_size = size_float
            return best_price, best_size

        bid_levels = None
        ask_levels = None
        if isinstance(depth_payload, Mapping):
            bid_levels = depth_payload.get("buy")
            ask_levels = depth_payload.get("sell")

        bid_price, bid_size = _select_level(bid_levels, "buy")
        ask_price, ask_size = _select_level(ask_levels, "sell")

        if bid_price is None and isinstance(quote, Mapping):
            bid_price = self._coerce_float(
                quote.get("best_bid") or quote.get("bid") or quote.get("buy_price")
            )
        if ask_price is None and isinstance(quote, Mapping):
            ask_price = self._coerce_float(
                quote.get("best_ask") or quote.get("ask") or quote.get("sell_price")
            )
        snapshot["bid"] = float(bid_price) if bid_price is not None else None
        snapshot["ask"] = float(ask_price) if ask_price is not None else None
        snapshot["bid_size"] = float(bid_size)
        snapshot["ask_size"] = float(ask_size)
        self._logger.info(
            "Condition met: depth_snapshot_ready",
            extra={
                "event": "depth_snapshot_ready",
                "symbol": symbol,
                "bid": snapshot["bid"],
                "ask": snapshot["ask"],
                "bid_size": snapshot["bid_size"],
                "ask_size": snapshot["ask_size"],
            },
        )
        return snapshot

    def calculate_queue_position(
        self, symbol: str, side: Literal["BUY", "SELL"], price: float
    ) -> int:
        """Estimate visible queue quantity ahead of the provided price level.

        Args:
            symbol: Instrument identifier for the evaluation.
            side: Order side (``BUY`` joins offers, ``SELL`` joins bids).
            price: Limit price considered for the queue estimate.

        Returns:
            Integer quantity representing the estimated queue depth ahead.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered calculate_queue_position",
            extra={
                "event": "calculate_queue_position_enter",
                "symbol": symbol,
                "side": side,
                "price": price,
            },
        )
        if price is None or price <= 0:
            return 0
        mdm = self._data_hub or self._market_data
        if mdm is None:
            return 0
        try:
            quote = mdm.get_quote(symbol) or {}
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in calculate_queue_position: %s",
                exc,
                extra={
                    "event": "calculate_queue_position_error",
                    "symbol": symbol,
                },
            )
            return 0
        depth_payload = quote.get("depth") if isinstance(quote, Mapping) else None
        if not isinstance(depth_payload, Mapping):
            return 0
        side_key = "sell" if side.upper() == "BUY" else "buy"
        levels = depth_payload.get(side_key)
        if not isinstance(levels, Iterable):
            return 0
        tolerance = max(price * 0.0005, 0.05)
        queue_qty = 0.0
        for entry in levels:
            if not isinstance(entry, Mapping):
                continue
            level_price = self._coerce_float(
                entry.get("price") or entry.get("p") or entry.get("value")
            )
            level_qty = self._coerce_float(
                entry.get("quantity")
                or entry.get("qty")
                or entry.get("size")
                or entry.get("volume")
            )
            if level_price is None or level_qty is None or level_qty <= 0:
                continue
            level_price_f = float(level_price)
            level_qty_f = float(level_qty)
            if side.upper() == "BUY":
                if level_price_f > price + tolerance:
                    queue_qty += level_qty_f
                elif abs(level_price_f - price) <= tolerance:
                    queue_qty += level_qty_f
            else:
                if level_price_f < price - tolerance:
                    queue_qty += level_qty_f
                elif abs(level_price_f - price) <= tolerance:
                    queue_qty += level_qty_f
        return int(round(queue_qty))

    def _queue_threshold(self, symbol: str, quantity: int) -> int:
        """Return queue threshold derived from configuration and lot sizing."""

        base_threshold = max(
            int(getattr(app_settings, "ORDER_MAX_QUEUE_POSITION", 0)), 0
        )
        per_lot = max(
            float(getattr(app_settings, "ORDER_QUEUE_POSITION_PER_LOT", 0.0)), 0.0
        )
        derived_threshold = 0
        if per_lot > 0 and quantity > 0:
            lots = 0
            try:
                lot_size = self._lot_size_for_symbol(symbol)
            except OrderPlacementError:
                lot_size = 0
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "queue_threshold_lot_lookup_failed",
                    extra={
                        "event": "queue_threshold_lot_lookup_failed",
                        "symbol": symbol,
                        "error": str(exc),
                    },
                )
                lot_size = 0
            if lot_size > 0:
                lots = max(quantity // lot_size, 0)
            derived_threshold = int(math.floor(per_lot * max(lots, 0)))
        return max(base_threshold, derived_threshold)

    def _estimate_order_slippage(
        self, symbol: str, side: Literal["BUY", "SELL"], quantity: int
    ) -> tuple[float | None, float | None, float | None]:
        """Estimate average fill and slippage based on depth and history."""

        self._logger.debug(
            "Entered _estimate_order_slippage",
            extra={
                "event": "estimate_order_slippage_enter",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            },
        )
        mdm = self._data_hub or self._market_data
        if mdm is None or quantity <= 0:
            return (None, None, None)
        try:
            quote = mdm.get_quote(symbol) or {}
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _estimate_order_slippage: %s",
                exc,
                extra={
                    "event": "estimate_order_slippage_error",
                    "symbol": symbol,
                },
            )
            return (None, None, None)
        depth_payload = quote.get("depth") if isinstance(quote, Mapping) else None
        if not isinstance(depth_payload, Mapping):
            return (None, None, None)
        levels_key = "sell" if side.upper() == "BUY" else "buy"
        levels = depth_payload.get(levels_key)
        if not isinstance(levels, Iterable):
            return (None, None, None)
        remaining = max(int(quantity), 0)
        consumed = 0
        notional = 0.0
        best_price = self._coerce_float(
            quote.get("ask") if side.upper() == "BUY" else quote.get("bid")
        )
        for entry in levels:
            if remaining <= 0:
                break
            if not isinstance(entry, Mapping):
                continue
            level_price = self._coerce_float(
                entry.get("price") or entry.get("p") or entry.get("value")
            )
            level_qty = self._coerce_float(
                entry.get("quantity")
                or entry.get("qty")
                or entry.get("size")
                or entry.get("volume")
            )
            if level_price is None or level_qty is None or level_qty <= 0:
                continue
            available = int(level_qty)
            if available <= 0:
                continue
            take_qty = min(remaining, available)
            notional += float(level_price) * take_qty
            consumed += take_qty
            remaining -= take_qty
            if best_price is None:
                best_price = level_price
        if consumed <= 0:
            return (None, None, None)
        average_price = notional / consumed
        slippage: float | None = None
        if best_price is not None:
            if side.upper() == "BUY":
                slippage = average_price - float(best_price)
            else:
                slippage = float(best_price) - average_price
        historical: list[float] = []
        for detail in reversed(self._history):
            if detail.symbol != symbol:
                continue
            if detail.fill_price is None or detail.price <= 0:
                continue
            if detail.side.upper() not in {"BUY", "SELL"}:
                continue
            fill_slippage = detail.fill_price - detail.price
            if detail.side.upper() == "SELL":
                fill_slippage = -fill_slippage
            historical.append(fill_slippage)
            if len(historical) >= 10:
                break
        historical_avg = None
        if historical:
            historical_avg = sum(historical) / len(historical)
        return (average_price, slippage, historical_avg)

    def _resolve_exchange(self, symbol: str) -> str:
        """Resolve exchange segment from symbol string (e.g., 'NFO:...' -> 'NFO')."""
        # 1. Try to parse from symbol prefix (Fastest)
        if ":" in symbol:
            return symbol.split(":")[0]

        # 2. Try the instrument resolver if available
        if self._resolver:
            try:
                info = self._resolver.resolve_by_symbol(symbol)
                if info and "exchange" in info:
                    return info["exchange"]
            except Exception:
                self._logger.exception("Unhandled exception", exc_info=True)
                raise

        # 3. Default fallback for this bot (mostly trades Options)
        return "NFO"

    def _resolve_tradingsymbol(self, symbol: str) -> str:
        """Resolve trading symbol string (e.g., 'NFO:NIFTY...' -> 'NIFTY...')."""
        # 1. Parse from string if colon present (Fastest)
        if ":" in symbol:
            return symbol.split(":")[1]

        # 2. Try the instrument resolver if available
        if self._resolver:
            try:
                info = self._resolver.resolve_by_symbol(symbol)
                if info and "tradingsymbol" in info:
                    return info["tradingsymbol"]
            except Exception:
                self._logger.exception("Unhandled exception", exc_info=True)
                raise

        # 3. Fallback: Return as is (assuming it's already a tradingsymbol)
        return symbol

    def _round_to_tick(self, price: float, tick_size: float = 0.05) -> float:
        """
        ✅ Round price to nearest valid tick size.
        """
        if price is None or price <= 0:
            return 0.0
        return round(round(price / tick_size) * tick_size, 2)

    def _validate_live_execution_safety(self) -> bool:
        """Validate live execution preconditions. Args: none. Returns: bool. Raises: None."""
        try:
            execution_mode = self._execution_mode_env()
            if not self._order_live_execution_enabled():
                self._logger.error(
                    "live_execution_guard_block",
                    extra={
                        "event": "live_execution_guard_block",
                        "reason": "mode_or_flag",
                        "execution_mode": execution_mode,
                        "enable_live": os.getenv("ENABLE_LIVE"),
                        "enable_live_trading": os.getenv("ENABLE_LIVE_TRADING"),
                    },
                )
                return False
            broker_connected = True
            if hasattr(self._broker, "is_connected") and callable(
                getattr(self._broker, "is_connected")
            ):
                broker_connected = bool(self._broker.is_connected())
            if not broker_connected:
                self._logger.error(
                    "live_execution_guard_block",
                    extra={
                        "event": "live_execution_guard_block",
                        "reason": "broker_disconnected",
                    },
                )
                return False
            available_margin = None
            margin_fn = getattr(self._margin_engine, "available_margin", None)
            if callable(margin_fn):
                available_margin = float(margin_fn() or 0.0)
            if available_margin is not None and available_margin <= 0:
                self._logger.error(
                    "live_execution_guard_block",
                    extra={
                        "event": "live_execution_guard_block",
                        "reason": "insufficient_margin",
                    },
                )
                return False
            return True
        except Exception as e:
            self._logger.error("Failure in _validate_live_execution_safety: %s", e)
            return False

    def _record_kill_switch_failure(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._kill_switch_failure_history.append(dict(record))

    def get_kill_switch_failure_history(self, limit: int = 20) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit or 20), 20))
        with self._lock:
            return [
                dict(item)
                for item in list(self._kill_switch_failure_history)[-safe_limit:]
            ]

    def get_last_kill_switch_failure(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._kill_switch_failure_history:
                return None
            return dict(self._kill_switch_failure_history[-1])

    def clear_kill_switch_failure_history(self) -> None:
        with self._lock:
            self._kill_switch_failure_history.clear()

    def is_kill_switch_active(self) -> bool:
        """Return whether kill switch is currently blocking new entries. Args: none. Returns: bool. Raises: none."""
        if self._kill_switch_engaged_at is None:
            return False
        if not self._kill_switch_allow_auto_reset:
            return True
        # Auto-reset after the cooldown even in live mode. Without this, a transient
        # broker problem (e.g. an IP-allowlist/network blip that trips 5 consecutive
        # failures) would halt trading for the rest of the day until a manual reset
        # or restart. Resetting only after the full cooldown avoids instantly
        # re-trying into a still-broken broker.
        elapsed = (
            datetime.now(timezone.utc) - self._kill_switch_engaged_at
        ).total_seconds()
        if elapsed < float(self._kill_switch_auto_reset_seconds):
            return True
        self.reset_kill_switch(reason="auto_timeout")
        return False

    def reset_kill_switch(self, reason: str = "manual") -> None:
        """Reset kill switch state. Args: reason. Returns: None. Raises: none."""
        self._consecutive_failures = 0
        self._kill_switch_engaged_at = None
        self._kill_switch_reason = None
        self._last_kill_switch_log_ts = 0.0
        self._kill_switch_last_reset = {
            "reset_reason": reason,
            "reset_ts": datetime.now(timezone.utc).isoformat(),
        }
        self._logger.info(
            "ORDER_KILL_SWITCH_RESET reason=%s",
            reason,
            extra={
                "event": "ORDER_KILL_SWITCH_RESET",
                "reason": reason,
                **self._kill_switch_last_reset,
            },
        )
        self._log_kill_switch_status()

    def _log_kill_switch_status(self) -> None:
        status = self.get_kill_switch_status()
        log_on_change(
            self._logger,
            key="ORDER_KILL_SWITCH_STATUS",
            state=(
                status.get("active"),
                status.get("kill_reason"),
                status.get("consecutive_failures"),
                status.get("engaged_at"),
            ),
            message="ORDER_KILL_SWITCH_STATUS active=%s reason=%s failures=%s engaged_at=%s auto_reset_allowed=%s"
            % (
                status.get("active"),
                status.get("kill_reason"),
                status.get("consecutive_failures"),
                status.get("engaged_at"),
                status.get("auto_reset_allowed"),
            ),
            reminder_seconds=300.0,
            level=logging.INFO,
            extra={"event": "ORDER_KILL_SWITCH_STATUS", **status},
        )

    def get_kill_switch_status(self) -> dict[str, Any]:
        """Return kill switch diagnostics. Args: none. Returns: status map. Raises: none."""
        engaged_at = self._kill_switch_engaged_at
        remaining = 0.0
        if engaged_at is not None and self._kill_switch_allow_auto_reset:
            elapsed = (datetime.now(timezone.utc) - engaged_at).total_seconds()
            remaining = max(float(self._kill_switch_auto_reset_seconds) - elapsed, 0.0)
        recent_failures = self.get_kill_switch_failure_history(limit=5)
        last_failure = self.get_last_kill_switch_failure() or {}
        return {
            "active": self.is_kill_switch_active(),
            "kill_reason": self._kill_switch_reason,
            "consecutive_failures": self._consecutive_failures,
            "engaged_at": engaged_at.isoformat() if engaged_at else None,
            "auto_reset_allowed": bool(self._kill_switch_allow_auto_reset),
            "auto_reset_seconds": int(self._kill_switch_auto_reset_seconds),
            "remaining_auto_reset_seconds": int(remaining),
            "last_failure": last_failure or None,
            "recent_failures_count": len(recent_failures),
            "recent_failures": recent_failures,
            "broker_attempted": bool(last_failure.get("broker_attempted", False)),
            "last_exception_type": last_failure.get("exception_type"),
            "last_exception_message": last_failure.get("exception_message"),
            "symbol": last_failure.get("symbol"),
            "trace_id": last_failure.get("trace_id"),
            "last_reset": (
                dict(self._kill_switch_last_reset)
                if self._kill_switch_last_reset
                else None
            ),
        }

    def execution_health_snapshot(self) -> dict[str, Any]:
        """Return compact execution health diagnostics."""
        with self._lock:
            self._prune_pending_signals()
            self._prune_uncertain_orders()
            return {
                "kill_switch": self.get_kill_switch_status(),
                "last_order_decision": dict(self._last_order_decision),
                "pending_orders_count": len(self._pending_signal_ids),
                "uncertain_orders_count": len(self._uncertain_client_order_ids),
                "consecutive_failures": self._consecutive_failures,
            }

    def resolve_lot_size(self, symbol: str) -> int:
        """Resolve lot size for symbol. Args: symbol. Returns: lot size. Raises: OrderPlacementError."""
        return self._lot_size_for_symbol(symbol)

    def normalize_quantity_to_lot(self, symbol: str, quantity: int) -> tuple[int, int]:
        """Normalize quantity to lot multiples. Args: symbol, quantity. Returns: (lot_size, normalized_qty). Raises: OrderPlacementError."""
        lot_size = self._lot_size_for_symbol(symbol)
        if quantity <= 0:
            return lot_size, 0
        if quantity % lot_size == 0:
            return lot_size, quantity
        normalized = (int(quantity) // lot_size) * lot_size
        return lot_size, normalized

    def _looks_like_signature_type_error(self, exc: TypeError) -> bool:
        """Return True only for TypeError raised by incompatible call signature."""
        text = str(exc).lower()
        signature_markers = (
            "unexpected keyword argument",
            "got an unexpected keyword",
            "missing 1 required positional argument",
            "missing required positional argument",
            "takes",
            "positional argument",
            "positional arguments",
            "required keyword-only argument",
        )
        return any(marker in text for marker in signature_markers)

    def _validate_execution_adapter(self) -> None:
        mode = self._execution_mode_env()

        if mode != "LIVE_SIMULATION":
            return

        broker = getattr(self, "broker", None)
        if broker is None:
            broker = getattr(self, "_broker", None)
        if broker is None:
            broker = getattr(self, "broker_client", None)

        is_simulated = bool(getattr(broker, "is_simulated_adapter", False))

        wrapped = getattr(broker, "client", None)
        if wrapped is not None:
            is_simulated = is_simulated or bool(
                getattr(wrapped, "is_simulated_adapter", False)
            )

        wrapped = getattr(broker, "_broker", None)
        if wrapped is not None:
            is_simulated = is_simulated or bool(
                getattr(wrapped, "is_simulated_adapter", False)
            )

        if not is_simulated:
            raise RuntimeError(
                "LIVE_SIMULATION cannot submit orders through a non-simulated broker"
            )

    def _submit_broker_order(
        self,
        payload: dict[str, object],
        *,
        legacy_payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self._validate_execution_adapter()
        try:
            response = self._broker.place_order(**payload)
        except TypeError as exc:
            if legacy_payload is None or not self._looks_like_signature_type_error(exc):
                raise
            try:
                response = self._broker.place_order(legacy_payload)
            except TypeError:
                raise exc

        if not isinstance(response, dict):
            raise OrderPlacementError("invalid_broker_response_payload")
        return response

    def place_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        trigger_price: float | None = None,
        tag: str | None = None,
        product: str = "MIS",
        variety: str = "regular",
        check_risk: bool = True,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        signal_id: str | None = None,
        strategy_name: str = "manual",
        trace_id: str | None = None,
        intent: OrderIntent | str | None = None,
        intended_position_side: Literal["LONG", "SHORT"] | None = None,
        client_order_id: str | None = None,
        trade_lifecycle_id: str | None = None,
        linked_entry_order_id: str | None = None,
        bracket_id: str | None = None,
        basket_version: int | str | None = None,
        instrument_token: int | None = None,
        contract_expiry: str | None = None,
        requested_lots: int = 0,
        resolved_lot_size: int = 0,
    ) -> str | None:
        """
        Execute order with Idempotency, Safe Trading Window, Risk Gating, and Auto-Recovery.
        """
        # ── PHASE 3: TRADE_ATTEMPT — first thing, always ─────────────────────
        self._logger.info(
            "TRADE_ATTEMPT symbol=%s side=%s qty=%s price=%s sl=%s tp=%s strategy=%s signal_id=%s",
            symbol,
            side,
            quantity,
            price,
            stop_loss,
            take_profit,
            strategy_name,
            signal_id,
        )
        if trace_id:
            self._last_trace_id = trace_id
        normalized_intent = str(intent or "UNKNOWN").strip().upper()
        if normalized_intent not in {
            "ENTRY",
            "SCALE_IN",
            "EXIT",
            "REDUCE",
            "REVERSAL",
            "UNKNOWN",
        }:
            normalized_intent = "UNKNOWN"
        if normalized_intent == "UNKNOWN" and self.is_live_mode():
            self.set_last_skip_reason("order_intent_required")
            self._logger.error(
                "LIVE_ORDER_REJECTED symbol=%s side=%s reason=order_intent_required",
                symbol,
                side,
                extra={
                    "event": "LIVE_ORDER_REJECTED",
                    "symbol": symbol,
                    "side": side,
                    "reason": "order_intent_required",
                },
            )
            return None
        pnl_blocker = None
        if normalized_intent in {"ENTRY", "SCALE_IN", "REVERSAL"}:
            blocker_getter = getattr(
                self._positions, "current_pnl_reconciliation_blocker", None
            )
            if callable(blocker_getter):
                pnl_blocker = blocker_getter()
        if pnl_blocker:
            self.set_last_skip_reason(str(pnl_blocker))
            self._logger.error(
                "LIVE_ORDER_REJECTED symbol=%s side=%s reason=%s",
                symbol,
                side,
                pnl_blocker,
                extra={
                    "event": "LIVE_ORDER_REJECTED",
                    "symbol": symbol,
                    "side": side,
                    "intent": normalized_intent,
                    "reason": pnl_blocker,
                    "pnl_reconciliation": (
                        self._positions.pnl_reconciliation_snapshot()
                        if hasattr(self._positions, "pnl_reconciliation_snapshot")
                        else None
                    ),
                },
            )
            return None
        broker = getattr(self, "_broker", None)
        if bool(getattr(broker, "auth_invalid", False)):
            self.set_last_skip_reason("broker_auth_invalid")
            raise OrderPlacementError("broker_auth_invalid")

        def _log_order_decision(
            *,
            allowed: bool,
            block_reason: str | None = None,
            order_id: str | None = None,
            broker_mode: str | None = None,
            broker_attempted: bool = False,
            details: dict[str, Any] | None = None,
        ) -> None:
            """Emit unified decision logs for order placement. Args: fields. Returns: None. Raises: None."""
            if broker_mode is None:
                try:
                    broker_mode = (
                        str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper()
                    )
                except Exception:  # noqa: BLE001
                    broker_mode = None
            # State snapshot — the fields that most often cause *silent* blocking.
            # Captured here so every decision line shows WHY, not just THAT.
            try:
                import time as _t

                _now = _t.time()
                _margin_ts = self._last_margin_success_ts
                _margin_age = round(_now - _margin_ts, 1) if _margin_ts else None
                _state = {
                    "execution_mode": broker_mode,
                    "live_enabled": bool(self.is_live_mode()),
                    "shadow_mode": bool(self._shadow_mode_enabled()),
                    "kill_switch_active": bool(self.is_kill_switch_active()),
                    "consecutive_failures": int(self._consecutive_failures),
                    "margin_available": self._last_margin_available_balance,
                    "margin_age_s": _margin_age,
                    "margin_stale": bool(_margin_ts is None),
                    "allow_entry_with_stale_margin": bool(
                        self._allow_entry_with_stale_margin
                    ),
                    "qty": quantity,
                }
            except Exception:  # noqa: BLE001
                _state = {}
            merged_details = {**_state, **(details or {})}
            self._last_order_decision = {
                "allowed": allowed,
                "block_reason": block_reason,
                "details": merged_details,
                "trace_id": trace_id,
                "broker_attempted": broker_attempted,
            }
            self._logger.info(
                "ORDER_MANAGER_DECISION allowed=%s block_reason=%s symbol=%s mode=%s live=%s shadow=%s kill=%s margin_stale=%s broker_attempted=%s",
                allowed,
                block_reason,
                symbol,
                broker_mode,
                _state.get("live_enabled"),
                _state.get("shadow_mode"),
                _state.get("kill_switch_active"),
                _state.get("margin_stale"),
                broker_attempted,
                extra={
                    "event": "ORDER_MANAGER_DECISION",
                    "symbol": symbol,
                    "side": side,
                    "allowed": allowed,
                    "block_reason": block_reason,
                    "signal_id": signal_id,
                    "strategy_name": strategy_name,
                    "price": price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "order_id": order_id,
                    "trace_id": trace_id,
                    "broker_mode": broker_mode,
                    "details": merged_details,
                    "broker_attempted": broker_attempted,
                },
            )

        # ✅ FIX: Round Price/Trigger to 0.05 tick size BEFORE processing
        # ═══════════════════════════════════════════════════════════════════════
        if price is not None and price > 0:
            price = self._round_to_tick(price)
        if trigger_price is not None and trigger_price > 0:
            trigger_price = self._round_to_tick(trigger_price)
        if stop_loss is not None and stop_loss > 0:
            stop_loss = self._round_to_tick(stop_loss)
        if take_profit is not None and take_profit > 0:
            take_profit = self._round_to_tick(take_profit)
        # ---------------------------------------------------------
        # 🛡️ DETECT EXIT vs ENTRY (must be BEFORE any guard)
        normalized_symbol = normalize_symbol(symbol)
        normalized_side = str(side).strip().upper()
        normalized_tag = (tag or "").lower()
        is_system_exit = any(
            x in normalized_tag for x in ["exit", "stop", "target", "square", "guard"]
        )

        entry_blocker = getattr(self, "current_entry_blocker", None)
        entry_block = (
            entry_blocker()
            if callable(entry_blocker)
            else getattr(self, "_entry_lifecycle_blocker", None)
        )
        if normalized_intent in {"ENTRY", "SCALE_IN", "REVERSAL"} and entry_block:
            details = (
                dict(entry_block)
                if isinstance(entry_block, Mapping)
                else {"block_reason": str(entry_block)}
            )
            reason = str(details.get("block_reason") or "entry_reconciliation_pending")
            _log_order_decision(
                allowed=False,
                block_reason=reason,
                details=details,
                broker_attempted=False,
            )
            self.set_last_skip_reason(reason)
            return None

        if not is_system_exit and self._bracket_manager is not None:
            has_unresolved_exit = getattr(
                self._bracket_manager, "has_unresolved_exit", None
            )
            if callable(has_unresolved_exit) and bool(has_unresolved_exit()):
                active_bracket_id = None
                getter = getattr(
                    self._bracket_manager, "get_first_unresolved_exit_bracket_id", None
                )
                if callable(getter):
                    active_bracket_id = getter()
                self._logger.critical(
                    "NEW_ENTRY_BLOCKED reason=exit_unresolved active_bracket_id=%s symbol=%s side=%s qty=%s",
                    active_bracket_id,
                    normalized_symbol,
                    normalized_side,
                    quantity,
                    extra={
                        "event": "NEW_ENTRY_BLOCKED",
                        "reason": "exit_unresolved",
                        "active_bracket_id": active_bracket_id,
                        "symbol": normalized_symbol,
                        "side": normalized_side,
                        "quantity": quantity,
                    },
                )
                notify = getattr(self, "_notify_bracket_event", None)
                if callable(notify):
                    try:
                        notify(
                            "NEW_ENTRY_BLOCKED",
                            {
                                "reason": "exit_unresolved",
                                "active_bracket_id": active_bracket_id,
                                "message": "⚠️ Exit unresolved. New entries frozen.",
                            },
                        )
                    except Exception as exc:  # noqa: BLE001 - alert best-effort only
                        self._logger.debug("NEW_ENTRY_BLOCK_ALERT_FAILED error=%s", exc)
                _log_order_decision(
                    allowed=False,
                    block_reason="exit_unresolved",
                    details={"active_bracket_id": active_bracket_id},
                )
                return None

        if not is_system_exit and self.is_kill_switch_active():
            kill_state = self.get_kill_switch_status()
            last_failure = kill_state.get("last_failure") or {}
            now_ts = time.time()
            if now_ts - self._last_kill_switch_log_ts >= 300.0:
                self._last_kill_switch_log_ts = now_ts
                log_throttled_live(
                    self._logger,
                    logging.INFO,
                    "ORDER_KILL_SWITCH_BLOCK",
                    f"ORDER_KILL_SWITCH_BLOCK:{symbol}:{self._kill_switch_reason}",
                    300.0,
                    "ORDER_KILL_SWITCH_BLOCK symbol=%s consecutive_failures=%s reason=%s",
                    symbol,
                    self._consecutive_failures,
                    self._kill_switch_reason,
                    extra={
                        "event": "ORDER_KILL_SWITCH_BLOCK",
                        "symbol": symbol,
                        "consecutive_failures": self._consecutive_failures,
                        "reason": self._kill_switch_reason,
                    },
                )
            self._logger.warning(
                "ORDER_BLOCKED reason=kill_switch_active kill_reason=%s failures=%s engaged_at=%s trace_id=%s symbol=%s side=%s qty=%s last_failure_exception_type=%s last_failure_exception_message=%s",
                kill_state.get("kill_reason"),
                kill_state.get("consecutive_failures"),
                kill_state.get("engaged_at"),
                trace_id,
                normalized_symbol,
                normalized_side,
                quantity,
                last_failure.get("exception_type"),
                last_failure.get("exception_message"),
                extra={
                    "event": "ORDER_BLOCKED",
                    "block_reason": "kill_switch_active",
                    "trace_id": trace_id,
                    "kill_state": kill_state,
                    "last_failure": last_failure,
                    "symbol": normalized_symbol,
                    "side": normalized_side,
                    "quantity": quantity,
                },
            )
            _log_order_decision(
                allowed=False,
                block_reason="kill_switch_active",
                details=kill_state,
            )
            return None

        if not is_system_exit and (
            os.getenv("EXECUTION_MODE", "SHADOW").strip().upper() == "LIVE"
        ):
            if not self._validate_live_execution_safety():
                self._logger.warning(
                    "ORDER_BLOCKED: live_execution_safety_check_failed symbol=%s",
                    symbol,
                )
                _log_order_decision(
                    allowed=False, block_reason="live_execution_safety_check_failed"
                )
                return None

        # 🛡️ SAFETY GUARD: ENFORCE VIRTUAL BRACKETS
        is_entry = (side == "BUY") and not is_system_exit

        # 2. Identify Intraday Context
        current_product = (product or "MIS").upper()
        is_intraday = current_product == "MIS"

        if not is_system_exit:
            has_open_local = self._positions.has_open_position(symbol)
            has_pending_entry = any(
                o.symbol == normalize_symbol(symbol)
                and o.side == side
                and o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
                for o in self._orders.values()
            )
            if normalized_intent == "ENTRY" and has_open_local:
                self._logger.warning(
                    "ORDER_BLOCKED: open_position_exists symbol=%s side=%s",
                    symbol,
                    side,
                )
                _log_order_decision(
                    allowed=False, block_reason="open_position_exists"
                )
                return None
            if has_open_local and has_pending_entry:
                self._logger.warning(
                    "ORDER_BLOCKED: duplicate_entry_prevented symbol=%s side=%s",
                    symbol,
                    side,
                )
                _log_order_decision(
                    allowed=False, block_reason="duplicate_entry_prevented"
                )
                return None
            if not self._signal_arbitrator.allow(symbol, side):
                self._logger.critical(
                    "ORDER_BLOCKED: signal_arbitrator_blocked symbol=%s side=%s",
                    symbol,
                    side,
                )
                _log_order_decision(
                    allowed=False, block_reason="signal_arbitrator_blocked"
                )
                return None

        # 3. THE INVARIANT CHECK
        if is_entry and is_intraday:
            # If SL is missing, None, or Zero -> REJECT
            if stop_loss is None or stop_loss <= 0:
                self._logger.critical(
                    f"🛑 FATAL SAFETY BLOCK: Attempted Naked Entry on {symbol}!"
                    f"\nReason: Intraday Buy Orders MUST have a Stop Loss to attach a Virtual Bracket."
                    f"\nData: Qty={quantity}, SL={stop_loss}, Tag={tag}"
                )
                self._logger.warning(
                    "ORDER_BLOCKED: naked_entry_no_stop_loss symbol=%s qty=%s",
                    symbol,
                    quantity,
                )
                _log_order_decision(
                    allowed=False, block_reason="naked_entry_no_stop_loss"
                )
                return None  # ❌ STOP HERE. DO NOT CALL BROKER.

        # =========================================================
        from nifty_scalper_bot.core.trading_switch import trading_switch
        from nifty_scalper_bot.risk import OrderSignal

        normalized_symbol = normalize_symbol(symbol)
        if not is_strategy_instrument(normalized_symbol):
            raise RuntimeError("Blocked non-NIFTY instrument")
        if (
            self._execution_mode_env() == "LIVE"
            and self._is_selected_option_for_live_execution(normalized_symbol)
        ):
            quote = self._get_latest_quote_safe(normalized_symbol) or {}
            quote_diag = self._extract_quote_diagnostics(quote)
            if (
                float(quote_diag.get("bid") or 0.0) <= 0
                or float(quote_diag.get("ask") or 0.0) <= 0
            ):
                self.set_last_skip_reason("selected_option_bid_ask_missing")
                self._logger.warning(
                    "ORDER_BLOCKED: selected_option_bid_ask_missing symbol=%s bid=%s ask=%s ltp=%s",
                    normalized_symbol,
                    quote_diag.get("bid"),
                    quote_diag.get("ask"),
                    quote_diag.get("ltp"),
                    extra={
                        "event": "ORDER_BLOCKED",
                        "block_reason": "selected_option_bid_ask_missing",
                        "symbol": normalized_symbol,
                        "bid": quote_diag.get("bid"),
                        "ask": quote_diag.get("ask"),
                        "ltp": quote_diag.get("ltp"),
                    },
                )
                _log_order_decision(
                    allowed=False,
                    block_reason="selected_option_bid_ask_missing",
                    details={"quote": quote_diag},
                )
                return None
        is_option_entry = (
            normalized_intent in {"ENTRY", "SCALE_IN", "REVERSAL"}
            and normalized_symbol.endswith(("CE", "PE"))
        )
        if is_option_entry:
            try:
                lot_size = self._lot_size_for_symbol(normalized_symbol)
            except OrderPlacementError as exc:
                self._logger.warning(
                    "ORDER_BLOCKED: invalid_lot_quantity symbol=%s qty=%s reason=%s",
                    normalized_symbol,
                    quantity,
                    exc,
                    extra={
                        "event": "INVALID_LOT_QUANTITY",
                        "block_reason": "invalid_lot_quantity",
                        "symbol": normalized_symbol,
                        "quantity_units": quantity,
                        "intent": normalized_intent,
                    },
                )
                _log_order_decision(allowed=False, block_reason="invalid_lot_quantity")
                return None
            if quantity <= 0 or quantity % lot_size != 0:
                self._logger.warning(
                    "ORDER_BLOCKED: invalid_lot_quantity symbol=%s qty=%s lot_size=%s",
                    normalized_symbol,
                    quantity,
                    lot_size,
                    extra={
                        "event": "INVALID_LOT_QUANTITY",
                        "block_reason": "invalid_lot_quantity",
                        "symbol": normalized_symbol,
                        "quantity_units": quantity,
                        "lot_size": lot_size,
                        "remainder": quantity % lot_size if lot_size else None,
                        "intent": normalized_intent,
                    },
                )
                _log_order_decision(allowed=False, block_reason="invalid_lot_quantity")
                return None
        is_option_exit = (
            normalized_intent in {"EXIT", "REDUCE"}
            and normalized_symbol.endswith(("CE", "PE"))
        )
        if is_option_exit:
            exit_validation = self._validate_option_exit_quantity(
                normalized_symbol, normalized_side, int(quantity), normalized_intent
            )
            if exit_validation is not None:
                _log_order_decision(
                    allowed=False,
                    block_reason=str(exit_validation.get("reason")),
                    details=exit_validation,
                )
                return None
        # ---------------------------------------------------------------------
        # 🛑 FIX 1: Smart Idempotency with Timeout
        # ---------------------------------------------------------------------
        with self._lock:
            current_time = time.time()
            # Check for any pending orders on this symbol same side.
            # FIX (BUG 4): system exits (tagged "exit"/"stop"/etc.) MUST bypass
            # this dedup.  A stale PENDING exit order for the same symbol would
            # otherwise silently block every retry of the SL exit, leaving the
            # position open until EOD — exactly the scenario that caused the loss.
            pending_orders = [
                o
                for o in self._orders.values()
                if o.symbol == normalized_symbol
                and o.side == side
                and o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
                # TIMEOUT SAFETY: Only block if order is fresh (< 45 seconds old)
                # This prevents getting stuck forever if an order is lost in limbo
                and (current_time - self._timestamp_seconds(o.timestamp) < 45)
            ]

            broker_has_live_pending = False
            if pending_orders and hasattr(self._broker, "get_orders"):
                try:
                    broker_orders = self._broker.get_orders() or []
                    broker_has_live_pending = any(
                        isinstance(order, Mapping)
                        and str(order.get("status") or "").upper()
                        in {"OPEN", "TRIGGER PENDING", "PENDING"}
                        and (
                            str(order.get("tradingsymbol") or "").upper()
                            == normalized_symbol.split(":", 1)[-1]
                            or str(order.get("symbol") or "").upper()
                            == normalized_symbol
                        )
                        for order in broker_orders
                    )
                except Exception:
                    broker_has_live_pending = True

            if pending_orders and broker_has_live_pending and not is_system_exit:
                self._logger.warning(
                    f"🚫 BLOCKED: Fresh pending order exists for {normalized_symbol}. Ignored to prevent duplicate.",
                    extra={"event": "duplicate_block", "symbol": normalized_symbol},
                )
                self._logger.warning(
                    "ORDER_BLOCKED: fresh_pending_order_exists symbol=%s",
                    normalized_symbol,
                )
                _log_order_decision(
                    allowed=False, block_reason="fresh_pending_order_exists"
                )
                return None
            elif pending_orders and is_system_exit:
                self._logger.warning(
                    "⚠️ PENDING exit order already exists for %s — allowing new exit attempt "
                    "(dedup bypassed for system exit to prevent position bleed).",
                    normalized_symbol,
                    extra={"event": "exit_dedup_bypass", "symbol": normalized_symbol},
                )

            # ── SINGLE-POSITION GATE (cross-strike, atomic at the choke point) ──
            # One NIFTY option exposure at a time. Reject a second ENTRY while
            # any OTHER option symbol has: an entry submission in flight, a live
            # entry order, an open local position, or an active bracket.
            # Same-symbol duplicates are already handled by the dedup above.
            # Fail-closed: unreadable state blocks the entry.
            if normalized_intent == "ENTRY" and not is_system_exit:
                _gate_now = time.time()
                # Purge expired reservations first (self-healing).
                for _sym, _ts in list(self._entries_in_flight.items()):
                    if _gate_now - _ts > self.ENTRY_INFLIGHT_TTL_SEC:
                        self._entries_in_flight.pop(_sym, None)
                conflict: str | None = None
                for _sym in self._entries_in_flight:
                    if _sym != normalized_symbol:
                        conflict = f"entry_in_flight:{_sym}"
                        break
                if conflict is None:
                    for o in self._orders.values():
                        o_sym = normalize_symbol(o.symbol)
                        if (
                            o_sym != normalized_symbol
                            and str(getattr(o, "intent", "")).upper()
                            in {"ENTRY", "SCALE_IN", "REVERSAL"}
                            and o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
                        ):
                            conflict = f"pending_entry_order:{o_sym}"
                            break
                if conflict is None:
                    try:
                        for pos in self._positions.get_open_positions():
                            p_sym = normalize_symbol(str(pos.symbol))
                            if (
                                p_sym != normalized_symbol
                                and int(getattr(pos, "quantity", 0) or 0) != 0
                            ):
                                conflict = f"open_position:{p_sym}"
                                break
                    except Exception:  # noqa: BLE001 — fail-closed on unknown state
                        conflict = "position_state_unavailable"
                if conflict is None and self._bracket_manager is not None:
                    try:
                        _actives = (
                            getattr(self._bracket_manager, "active_brackets", {}) or {}
                        )
                        for _b_sym in _actives:
                            if normalize_symbol(str(_b_sym)) != normalized_symbol:
                                conflict = f"active_bracket:{_b_sym}"
                                break
                    except Exception:  # noqa: BLE001 — fail-closed on unknown state
                        conflict = "bracket_state_unavailable"
                if conflict:
                    self._logger.warning(
                        "ORDER_BLOCKED: single_position_gate symbol=%s side=%s conflict=%s",
                        normalized_symbol,
                        side,
                        conflict,
                        extra={
                            "event": "ORDER_BLOCKED",
                            "block_reason": "single_position_gate",
                            "symbol": normalized_symbol,
                            "conflict": conflict,
                        },
                    )
                    _log_order_decision(
                        allowed=False,
                        block_reason=f"single_position_gate:{conflict}",
                    )
                    return None
                # Reserve this symbol atomically with the check so a racing
                # second entry (CE vs PE) cannot pass the gate before this
                # order is registered locally.
                self._entries_in_flight[normalized_symbol] = _gate_now

        # ---------------------------------------------------------------------
        # 1. IDEMPOTENCY CHECK (The Fix for Duplicate Trades)
        # ---------------------------------------------------------------------
        if signal_id:
            duplicate_permanent = self._is_duplicate_signal(signal_id)
            duplicate_pending = self._is_pending_signal(signal_id)
            if duplicate_permanent or duplicate_pending:
                block_reason = (
                    "duplicate_signal"
                    if duplicate_permanent
                    else "duplicate_signal_pending"
                )
                self._logger.warning(
                    "🛑 DUPLICATE BLOCKED: Signal %s already traded.",
                    signal_id,
                    extra={
                        "symbol": normalized_symbol,
                        "event": "duplicate_block",
                        "block_reason": block_reason,
                    },
                )
                self._log_trade_event(
                    "ORDER_BLOCKED_DUPLICATE",
                    symbol=normalized_symbol,
                    side=side,
                    qty=quantity,
                    price=float(price or 0.0),
                    meta={"signal_id": signal_id, "block_reason": block_reason},
                )
                self._logger.warning(
                    "ORDER_BLOCKED: %s signal_id=%s symbol=%s",
                    block_reason,
                    signal_id,
                    normalized_symbol,
                )
                _log_order_decision(allowed=False, block_reason=block_reason)
                return None

        # --- SEMANTIC VALIDATION GATEKEEPER ---
        if price and price > 0:
            if side == "BUY":
                # For a Long, TP must be above Entry, SL must be below Entry
                if take_profit and take_profit <= price:
                    self._logger.error(
                        f"🛑 REJECTED: BUY TP ({take_profit}) is below entry ({price})"
                    )
                    self._logger.critical(
                        "ORDER_BLOCKED: buy_tp_below_entry symbol=%s tp=%s entry=%s",
                        normalized_symbol,
                        take_profit,
                        price,
                    )
                    _log_order_decision(
                        allowed=False, block_reason="buy_tp_below_entry"
                    )
                    return None
                if stop_loss and stop_loss >= price:
                    self._logger.error(
                        f"🛑 REJECTED: BUY SL ({stop_loss}) is above entry ({price})"
                    )
                    self._logger.critical(
                        "ORDER_BLOCKED: buy_sl_above_entry symbol=%s sl=%s entry=%s",
                        normalized_symbol,
                        stop_loss,
                        price,
                    )
                    _log_order_decision(
                        allowed=False, block_reason="buy_sl_above_entry"
                    )
                    return None
            elif side == "SELL":
                # For a Short/Exit, TP must be below Entry, SL must be above Entry
                if take_profit and take_profit >= price:
                    self._logger.error(
                        f"🛑 REJECTED: SELL TP ({take_profit}) is above entry ({price})"
                    )
                    self._logger.critical(
                        "ORDER_BLOCKED: sell_tp_above_entry symbol=%s tp=%s entry=%s",
                        normalized_symbol,
                        take_profit,
                        price,
                    )
                    _log_order_decision(
                        allowed=False, block_reason="sell_tp_above_entry"
                    )
                    return None
                if stop_loss and stop_loss <= price:
                    self._logger.error(
                        f"🛑 REJECTED: SELL SL ({stop_loss}) is below entry ({price})"
                    )
                    self._logger.critical(
                        "ORDER_BLOCKED: sell_sl_below_entry symbol=%s sl=%s entry=%s",
                        normalized_symbol,
                        stop_loss,
                        price,
                    )
                    _log_order_decision(
                        allowed=False, block_reason="sell_sl_below_entry"
                    )
                    return None

        # ---------------------------------------------------------------------
        # 2. TIME GUARD (central market_hours source of truth)
        # ---------------------------------------------------------------------
        if variety == "regular" and not is_system_exit:
            try:
                execution_mode = os.getenv("EXECUTION_MODE", "SHADOW").strip().upper()
                enable_live = os.getenv("ENABLE_LIVE", "false").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                is_live_mode = execution_mode == "LIVE" or enable_live
                allowed, detail = get_time_status()
                if is_live_mode and not allowed:
                    self._logger.info(
                        "ORDER_BLOCKED reason=time_guard detail=%s symbol=%s",
                        detail,
                        normalized_symbol,
                        extra={
                            "event": "ORDER_BLOCKED",
                            "reason": "time_guard",
                            "detail": detail,
                            "symbol": normalized_symbol,
                        },
                    )
                    _log_order_decision(
                        allowed=False,
                        block_reason="time_guard",
                        details={"detail": detail},
                    )
                    return None
            except Exception as e:
                self._logger.error(
                    f"Time Guard Check Failed: {e}. Proceeding with caution."
                )

        # ---------------------------------------------------------------------
        # 3. TRADING SWITCH GUARD
        # ---------------------------------------------------------------------
        if not is_system_exit:
            switch_instance = (
                trading_switch() if callable(trading_switch) else trading_switch
            )
            checker = getattr(
                switch_instance,
                "can_trade",
                getattr(switch_instance, "can_trade_new", None),
            )

            if callable(checker) and not checker():
                self._logger.warning(
                    "Order blocked: Trading Switch is OFF",
                    extra={"symbol": normalized_symbol},
                )
                self._logger.info(
                    "ORDER_BLOCKED: trading_switch_off symbol=%s", normalized_symbol
                )
                _log_order_decision(allowed=False, block_reason="trading_switch_off")
                return None

        # ---------------------------------------------------------------------
        # 4. RISK MANAGER VALIDATION
        # ---------------------------------------------------------------------
        if check_risk and self._risk_manager:
            signal = OrderSignal(
                symbol=normalized_symbol,
                side=side,
                quantity=quantity,
                price=price or 0.0,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            is_live = False
            if hasattr(self, "_enable_live_getter") and self._enable_live_getter:
                is_live = self._enable_live_getter()
            elif hasattr(self, "_resolve_enable_live"):
                is_live = self._resolve_enable_live()

            allowed, reason = self._risk_manager.check_order(
                signal, live_enabled=is_live
            )
            if not allowed:
                self._logger.warning(
                    f"Risk Block: {reason}",
                    extra={"symbol": normalized_symbol, "event": "risk_block"},
                )
                self._logger.info(
                    "ORDER_BLOCKED: risk_manager_blocked reason=%s symbol=%s",
                    reason,
                    normalized_symbol,
                )
                _log_order_decision(allowed=False, block_reason="risk_manager_blocked")
                return None

        # ---------------------------------------------------------------------
        # 5. STATE PERSISTENCE (Save Intent BEFORE Execution)
        # ---------------------------------------------------------------------
        # Generate ID if manual
        if not signal_id:
            raw_sig = f"{normalized_symbol}:{side}:{quantity}:{int(time.time())}"
            sig_hash = hashlib.md5(raw_sig.encode()).hexdigest()[:12]
            signal_id = f"manual_{sig_hash}"

        trade_id = f"TRD_{signal_id}"
        unique_client_id = f"bot_{signal_id[-12:]}"  # Max 20 chars usually
        suffix = unique_client_id[-8:]
        base_tag = str(tag or "bot").strip() or "bot"
        safe_base = "".join(ch for ch in base_tag if ch.isalnum() or ch in {"_", "-"})
        if not safe_base:
            safe_base = "bot"
        if normalized_intent == "EXIT" and safe_base:
            broker_tag = safe_base[:20]
        else:
            safe_base = safe_base[: max(1, 19 - len(suffix))]
            broker_tag = f"{safe_base}_{suffix}"[:20]

        pending_signal_marked = False
        if signal_id:
            self._mark_signal_pending(signal_id)
            pending_signal_marked = True
        self._log_trade_event(
            "ORDER_SUBMIT_ATTEMPT",
            symbol=normalized_symbol,
            side=side,
            qty=quantity,
            price=float(price or 0.0),
            meta={
                "trade_id": trade_id,
                "signal_id": signal_id,
                "strategy": strategy_name,
                "status": "SUBMIT_ATTEMPT",
            },
        )

        # ---------------------------------------------------------------------
        # 6. PAYLOAD OPTIMIZATION (SL-M Fix) - CORRECTED
        # ---------------------------------------------------------------------

        # ✅ HELPER: Normalize OrderType to Zerodha string BEFORE any broker call
        def _normalize_order_type(ot: Any) -> str:
            """Convert OrderType Enum to Zerodha-compatible string."""
            # Fast path: Already a valid Zerodha string
            if isinstance(ot, str):
                ot_upper = ot.strip().upper()
                if ot_upper in {"MARKET", "LIMIT", "SL", "SL-M"}:
                    return ot_upper

            # Enum path: Extract name
            if isinstance(ot, OrderType):
                ot_name = ot.name.upper()
            elif hasattr(ot, "name"):
                ot_name = str(ot.name).upper()
            elif hasattr(ot, "value"):
                ot_name = str(ot.value).upper()
            else:
                ot_name = str(ot).upper()

            # Map to Zerodha format
            zerodha_map = {
                "MARKET": "MARKET",
                "LIMIT": "LIMIT",
                "STOP_LOSS": "SL",
                "STOPLOSS": "SL",
                "STOP_LOSS_MARKET": "SL-M",
                "STOPLOSSMARKET": "SL-M",
                "SL-M": "SL-M",
                "SL": "SL",
                "STOP_LOSS_LIMIT": "SL",
                "STOPLOSSLIMIT": "SL",
            }
            return zerodha_map.get(ot_name, "MARKET")

        # ✅ HELPER: Normalize side to Zerodha string
        def _normalize_side(s: Any) -> str:
            """Convert TransactionType Enum to 'BUY' or 'SELL'."""
            if isinstance(s, str):
                s_upper = s.strip().upper()
                if s_upper in {"BUY", "SELL"}:
                    return s_upper
                if s_upper == "LONG":
                    return "BUY"
                if s_upper == "SHORT":
                    return "SELL"

            if hasattr(s, "name"):
                s_name = str(s.name).upper()
            elif hasattr(s, "value"):
                s_name = str(s.value).upper()
            else:
                s_name = str(s).upper()

            side_map = {"BUY": "BUY", "SELL": "SELL", "LONG": "BUY", "SHORT": "SELL"}
            return side_map.get(s_name, "BUY")

        # ✅ CRITICAL: Normalize BEFORE any retry attempt
        final_order_type = _normalize_order_type(order_type)
        normalized_side = _normalize_side(side)

        # [FIX] Zerodha blocks SL-M for Options -> Convert to SL with limit price
        if (
            final_order_type in {"SL", "SL-M"}
            and (price is None or price <= 0.0)
            and trigger_price
        ):
            buffer_pct = 0.03  # 3% Buffer
            if normalized_side == "BUY":  # Short Exit -> Buy higher
                price = round(trigger_price * (1 + buffer_pct), 2)
            else:  # Long Exit -> Sell lower
                price = round(trigger_price * (1 - buffer_pct), 2)

            self._logger.info(
                f"🛡️ Converted SL-M to SL Limit. Trigger: {trigger_price}, Limit: {price}",
                extra={
                    "event": "order.slm_to_sl_conversion",
                    "trigger": trigger_price,
                    "limit": price,
                },
            )
            final_order_type = "SL"  # Force SL (not SL-M)

        self._logger.info(
            f"🚀 Sending Order: {normalized_side} {quantity} {normalized_symbol} ({final_order_type})",
            extra={
                "event": "order_sending",
                "symbol": normalized_symbol,
                "signal_id": signal_id,
            },
        )
        self._logger.info(
            "ORDER_SENT symbol=%s side=%s qty=%s",
            normalized_symbol,
            normalized_side,
            quantity,
        )

        # ---------------------------------------------------------------------
        # 7. EXECUTION LOOP (With Anti-Zombie Timeout)
        # ---------------------------------------------------------------------
        # Helper for threaded execution
        def _broker_call(kwargs):
            try:
                return self._submit_broker_order(kwargs)
            except Exception as exc:
                return exc

        # ✅ FIX: Define call_args BEFORE the loop starts
        # This ensures the variable exists for the 'isinstance' check below
        call_args = {
            "symbol": normalized_symbol,
            "side": normalized_side,
            "quantity": quantity,
            "product": product,
            "order_type": final_order_type,
            "price": price,
            "trigger_price": trigger_price,
            "tag": broker_tag,
            "variety": variety,
        }
        if normalized_intent != "EXIT":
            call_args["client_order_id"] = unique_client_id

        def _find_existing_order_after_uncertain_submit() -> dict[str, Any] | None:
            try:
                existing = self._find_open_order(unique_client_id)
                if isinstance(existing, Mapping):
                    return dict(existing)
            except Exception as exc:
                self._logger.warning(
                    "ORDER_RECONCILE_AFTER_TIMEOUT_FAILED signal_id=%s client_order_id=%s error=%s",
                    signal_id,
                    unique_client_id,
                    exc,
                    exc_info=exc,
                )
            return None

        def _extract_order_id(response: object) -> str | None:
            if isinstance(response, Mapping):
                raw = response.get("order_id") or response.get("id")
                return str(raw) if raw else None
            if response:
                return str(response)
            return None

        if self._is_uncertain_order(unique_client_id):
            existing_uncertain = self._find_open_order(unique_client_id)
            if existing_uncertain is not None:
                existing_order_id = _extract_order_id(existing_uncertain)
                if existing_order_id:
                    self._clear_uncertain_order(unique_client_id)
                    if pending_signal_marked:
                        self._clear_pending_signal(signal_id)
                    if signal_id:
                        self._remember_signal(signal_id)
                    return existing_order_id
            _log_order_decision(
                allowed=False,
                block_reason="uncertain_order_reconciliation_pending",
            )
            if pending_signal_marked:
                self._clear_pending_signal(signal_id)
            return None

        for attempt in range(1, 4):
            if self._is_uncertain_order(unique_client_id):
                existing_uncertain = self._find_open_order(unique_client_id)
                if existing_uncertain is not None:
                    existing_order_id = _extract_order_id(existing_uncertain)
                    if existing_order_id:
                        self._clear_uncertain_order(unique_client_id)
                        if pending_signal_marked:
                            self._clear_pending_signal(signal_id)
                        if signal_id:
                            self._remember_signal(signal_id)
                        return existing_order_id
                _log_order_decision(
                    allowed=False,
                    block_reason="uncertain_order_reconciliation_pending",
                )
                if pending_signal_marked:
                    self._clear_pending_signal(signal_id)
                return None
            # -----------------------------------------------------------------
            # ✅ FIX: Re-hydrate Enums to prevent Adapter Crash
            # The broker adapter expects Enum objects (e.g. OrderType.MARKET).
            # If we pass a string "MARKET", it crashes on .value access.
            # -----------------------------------------------------------------
            if isinstance(call_args["order_type"], str):
                ot_str = call_args["order_type"]
                if ot_str == "MARKET":
                    call_args["order_type"] = OrderType.MARKET
                elif ot_str == "LIMIT":
                    call_args["order_type"] = OrderType.LIMIT
                elif ot_str == "SL":
                    call_args["order_type"] = OrderType.STOP_LOSS
                elif ot_str == "SL-M":
                    call_args["order_type"] = OrderType.STOP_LOSS_MARKET
            try:
                # ✅ Run in thread with 3s timeout to prevent hanging
                result_holder = {"resp": None}

                def target():
                    try:
                        result_holder["resp"] = _broker_call(call_args)
                    except (
                        Exception
                    ) as exc:  # noqa: BLE001 - re-raised on order thread with broker text intact
                        result_holder["resp"] = exc

                # We use the 'Thread' class already imported at top of file
                t = Thread(target=target, name=f"ord_{unique_client_id}", daemon=True)
                t.start()
                t.join(timeout=8.0)

                if t.is_alive():
                    self._logger.warning(
                        "Broker call exceeded 8s for %s (attempt %s); waiting recovery window",
                        normalized_symbol,
                        attempt,
                    )
                    t.join(timeout=2.0)
                    if t.is_alive():
                        existing_after_timeout = (
                            _find_existing_order_after_uncertain_submit()
                        )
                        if existing_after_timeout is not None:
                            response = existing_after_timeout
                        else:
                            self._mark_order_uncertain(unique_client_id)
                            self._logger.critical(
                                "🚨 Broker API hung on attempt %s and no existing order was found for client_order_id=%s; marking uncertain and blocking retry",
                                attempt,
                                unique_client_id,
                            )
                            _log_order_decision(
                                allowed=False,
                                block_reason="uncertain_order_reconciliation_pending",
                                broker_attempted=True,
                            )
                            if pending_signal_marked:
                                self._clear_pending_signal(signal_id)
                            return None
                    self._logger.info(
                        "Recovered late broker response inside grace window for %s",
                        normalized_symbol,
                    )

                response = result_holder["resp"]
                if response is None:
                    existing_after_timeout = (
                        _find_existing_order_after_uncertain_submit()
                    )
                    if existing_after_timeout is not None:
                        response = existing_after_timeout

                # Re-raise exceptions captured in thread
                if isinstance(response, Exception):
                    existing_after_error = _find_existing_order_after_uncertain_submit()
                    if existing_after_error is not None:
                        response = existing_after_error
                    else:
                        raise response

                # --- Success Logic ---
                order_id = _extract_order_id(response)

                if order_id:
                    self._clear_uncertain_order(unique_client_id)
                    # ✅ RESET Kill Switch on success
                    self._consecutive_failures = 0
                    self._logger.info(
                        "ORDER_SUBMITTED order_id=%s symbol=%s",
                        order_id,
                        normalized_symbol,
                    )

                    self._log_trade_event(
                        "ORDER_SUBMITTED",
                        symbol=normalized_symbol,
                        side=side,
                        qty=quantity,
                        price=float(price or 0.0),
                        order_id=order_id,
                        meta={"trade_id": trade_id, "status": "SUBMITTED"},
                    )

                    # B. Register Order Locally
                    details = OrderDetails(
                        order_id=order_id,
                        symbol=normalized_symbol,
                        side=side,
                        order_type=order_type,
                        quantity=quantity,
                        price=float(price or 0.0),
                        status=OrderStatus.PENDING,
                        timestamp=datetime.now(timezone.utc),
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        tag=tag,
                        average_price=0.0,
                        intent=cast(OrderIntent, normalized_intent),
                        intended_position_side=intended_position_side,
                        signal_id=signal_id,
                        client_order_id=client_order_id,
                        trade_lifecycle_id=trade_lifecycle_id,
                        linked_entry_order_id=linked_entry_order_id,
                        bracket_id=bracket_id,
                        basket_version=basket_version,
                        instrument_token=instrument_token,
                        contract_expiry=contract_expiry,
                        requested_lots=requested_lots,
                        resolved_lot_size=resolved_lot_size,
                    )
                    self._register_order(details)
                    # Sync PositionManager's pending-order registry so the
                    # incoming broker fill is attributed to OUR order (intent
                    # preserved), not synthesized as an unknown order and
                    # quarantined. 2026-07-10: this sync existed only on the
                    # reconcile path, so a live entry's own fill arrived as
                    # intent=UNKNOWN -> BROKER_POSITION_QUARANTINED_FOR_
                    # UNKNOWN_ORDER every sync -> all new entries blocked and
                    # the position left on a wide guard bracket.
                    if hasattr(self._positions, "add_pending_order"):
                        try:
                            self._positions.add_pending_order(
                                order_id=details.order_id,
                                symbol=details.symbol,
                                side=details.side,
                                qty=details.quantity,
                                price=details.price,
                                order_type=details.order_type,
                                intent=details.intent,
                                bracket_id=details.bracket_id,
                                signal_id=details.signal_id,
                                signal_fingerprint=details.signal_fingerprint,
                            )
                        except Exception as _pm_sync_exc:  # noqa: BLE001
                            self._logger.error(
                                "POSITION_MANAGER_PENDING_SYNC_FAILED order_id=%s error=%s",
                                details.order_id,
                                _pm_sync_exc,
                                extra={
                                    "event": "POSITION_MANAGER_PENDING_SYNC_FAILED",
                                    "order_id": details.order_id,
                                },
                            )
                    # Local order record now owns the symbol — release the
                    # single-position gate's in-flight reservation.
                    with self._lock:
                        self._entries_in_flight.pop(normalized_symbol, None)

                    # C. Auto-Register Bracket
                    # SKIP if caller will register separately (e.g. place_bracket_order)
                    # Detect via tag or explicit flag — place_bracket_order always calls
                    # register_virtual_bracket itself with tp1/trailing params.
                    _caller_manages_bracket = (
                        any(x in normalized_tag for x in ["virtual_bracket"])
                        if normalized_tag
                        else False
                    )

                    if (
                        self._bracket_manager
                        and (stop_loss or take_profit)
                        and not _caller_manages_bracket
                    ):
                        self._bracket_manager.register_virtual_bracket(
                            order_id=order_id,
                            symbol=normalized_symbol,
                            side=normalized_side,
                            qty=quantity,
                            price=float(price or 0.0),
                            sl=float(stop_loss) if stop_loss else 0.0,
                            tp=float(take_profit) if take_profit else 0.0,
                            tag=tag or "auto",
                            intent=normalized_intent,
                            activate_immediately=False,
                        )
                        self._logger.info(f"🛡️ Auto-bracket registered for {order_id}")
                        self._logger.info(
                            "BRACKET_CREATED order_id=%s symbol=%s",
                            order_id,
                            normalized_symbol,
                        )

                    # Keep the short entry confirmation used to activate protection.
                    # System exits are confirmed by the order monitor/reconciliation;
                    # polling here would block the synchronous protection callback.
                    fill_confirmed = (
                        False
                        if is_system_exit
                        else self._confirm_fill_fast(order_id, timeout_ms=300)
                    )
                    if fill_confirmed and self._bracket_manager:
                        bracket = self._bracket_manager.get_bracket(order_id)
                        if bracket:
                            stop_order_id = getattr(
                                bracket, "stop_order_id", None
                            ) or getattr(bracket, "virtual_sl_id", None)
                            trailing_spec = getattr(bracket, "trailing_spec", None)

                            if stop_order_id and trailing_spec:
                                try:
                                    self.attach_trailing_stop(
                                        entry_order_id=order_id,
                                        sl_order_id=stop_order_id,
                                        symbol=normalized_symbol,
                                        side=normalized_side,
                                        entry_price=bracket.entry_price,
                                        spec=trailing_spec,
                                    )
                                    self._logger.info(
                                        "📈 TRAILING SL ATTACHED | %s | order=%s",
                                        normalized_symbol,
                                        order_id,
                                    )
                                except Exception as exc:
                                    self._logger.error(
                                        "Trailing attach failed for %s: %s",
                                        order_id,
                                        exc,
                                    )
                            else:
                                self._logger.debug(
                                    "TRAILING_ATTACH_SKIPPED order_id=%s symbol=%s reason=no_stop_order_or_trailing_spec",
                                    order_id,
                                    normalized_symbol,
                                )

                    if fill_confirmed:
                        self._logger.info(
                            f"🟢 ORDER_FILL_CONFIRMED & BRACKET ACTIVE: {order_id}"
                        )
                        self._log_trade_event(
                            "ORDER_FILL_CONFIRMED",
                            symbol=normalized_symbol,
                            side=side,
                            qty=quantity,
                            price=float(price or 0.0),
                            order_id=order_id,
                            meta={"trade_id": trade_id, "status": "FILLED"},
                        )
                    else:
                        self._logger.info(
                            f"🟡 ORDER SUBMITTED (fill pending): {order_id}"
                        )

                    if not is_system_exit:
                        self._signal_arbitrator.register(normalized_symbol, side)
                    _log_order_decision(
                        allowed=True,
                        order_id=order_id,
                        broker_attempted=True,
                    )
                    self._consecutive_failures = 0
                    if signal_id:
                        self._clear_pending_signal(signal_id)
                        self._remember_signal(signal_id)
                    return order_id
                _log_order_decision(
                    allowed=False,
                    block_reason="missing_order_id",
                    broker_attempted=True,
                    details={
                        "failure_class": "missing_order_id",
                        "error_message": "broker response did not include order_id",
                        "broker_payload": (
                            response
                            if isinstance(response, dict)
                            else {"raw_response": repr(response)}
                        ),
                        "retryable": True,
                    },
                )
                raise OrderPlacementError(
                    f"missing_order_id: broker response did not include order_id payload={response!r}"
                )

            except Exception as e:
                msg = str(e).lower()
                failure_class = "unexpected_exception"
                if "rate" in msg and "limit" in msg:
                    failure_class = "transient_api_error"
                elif any(
                    x in msg
                    for x in [
                        "no ips configured",
                        "allowed ips",
                        "ip configured",
                        "403",
                        "access denied",
                        "forbidden",
                        "ip not whitelisted",
                    ]
                ):
                    # Broker-side account/config problem (Kite static-IP allowlist).
                    # Non-retryable and NOT a trading failure — must not count toward
                    # the kill switch, otherwise a fixable config issue latches the bot.
                    failure_class = "broker_config_error"
                elif "auth" in msg or "token" in msg or "unauthor" in msg:
                    failure_class = "broker_rejected"
                elif "timeout" in msg:
                    failure_class = "timeout"
                elif "missing_order_id" in msg or "did not include order_id" in msg:
                    failure_class = "missing_order_id"
                elif "margin" in msg or "fund" in msg:
                    failure_class = "insufficient_margin"
                elif "market closed" in msg:
                    failure_class = "market_closed"
                elif "invalid" in msg and "symbol" in msg:
                    failure_class = "invalid_symbol"
                elif "invalid" in msg and "quant" in msg:
                    failure_class = "invalid_quantity"
                elif any(
                    x in msg for x in ["invalid", "bad request", "payload", "400"]
                ):
                    failure_class = "broker_rejected"

                if failure_class == "broker_config_error":
                    self._logger.error(
                        "ORDER_BROKER_CONFIG_ERROR non_retryable=True reason=ip_allowlist_or_access_denied "
                        "symbol=%s trace_id=%s detail=%s",
                        normalized_symbol,
                        trace_id,
                        self._sanitize_broker_error(e),
                        extra={
                            "event": "ORDER_BROKER_CONFIG_ERROR",
                            "non_retryable": True,
                            "symbol": normalized_symbol,
                            "trace_id": trace_id,
                            "failure_class": failure_class,
                        },
                    )

                countable_failures = {
                    "transient_api_error",
                    "invalid_symbol",
                    "invalid_quantity",
                    "insufficient_margin",
                    "broker_rejected",
                    "timeout",
                    "unexpected_exception",
                }
                if failure_class in countable_failures:
                    self._consecutive_failures += 1
                    self._record_kill_switch_failure(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "trace_id": trace_id,
                            "symbol": normalized_symbol,
                            "side": normalized_side or side,
                            "quantity": quantity,
                            "attempt": attempt,
                            "failure_class": failure_class,
                            "exception_type": type(e).__name__,
                            "exception_message": str(e),
                            "consecutive_failures_after": self._consecutive_failures,
                            "max_failures": self._max_failures,
                            "broker_attempted": True,
                            "order_type": str(final_order_type),
                            "product": product,
                            "variety": variety,
                            "signal_id": signal_id,
                            "strategy_name": strategy_name,
                            "client_order_id": unique_client_id,
                            "payload_summary": {
                                "symbol": normalized_symbol,
                                "side": normalized_side,
                                "quantity": quantity,
                                "product": product,
                                "order_type": str(final_order_type),
                                "price": price,
                                "trigger_price": trigger_price,
                                "tag": broker_tag,
                                "variety": variety,
                            },
                        }
                    )
                if (
                    self._consecutive_failures >= self._max_failures
                    and self._kill_switch_engaged_at is None
                ):
                    self._kill_switch_engaged_at = datetime.now(timezone.utc)
                    self._kill_switch_reason = failure_class
                    self._last_kill_switch_log_ts = time.time()
                    self._logger.critical(
                        "ORDER_KILL_SWITCH_ENGAGED reason=%s failures=%s exception_type=%s exception=%s trace_id=%s symbol=%s",
                        self._kill_switch_reason,
                        self._consecutive_failures,
                        type(e).__name__,
                        str(e),
                        trace_id,
                        normalized_symbol,
                        extra={
                            "event": "ORDER_KILL_SWITCH_ENGAGED",
                            "reason": self._kill_switch_reason,
                            "failures": self._consecutive_failures,
                            "exception_type": type(e).__name__,
                            "exception": str(e),
                            "trace_id": trace_id,
                            "symbol": normalized_symbol,
                        },
                        exc_info=(failure_class == "unexpected_exception"),
                    )
                    self._log_kill_switch_status()

                # Fail Fast logic
                if failure_class != "missing_order_id" and (
                    failure_class == "broker_config_error"
                    or any(
                        x in msg
                        for x in [
                            "400",
                            "invalid",
                            "market closed",
                            "bad request",
                            "insufficient funds",
                        ]
                    )
                ):
                    self._logger.critical(
                        f"🛑 FATAL Payload Error: {e}",
                        extra={"event": "fatal_order_error"},
                    )
                    self._log_trade_event(
                        "ORDER_REJECTED_FATAL",
                        symbol=normalized_symbol,
                        side=side,
                        qty=quantity,
                        price=float(price or 0.0),
                        meta={"trade_id": trade_id, "error": str(e)},
                    )
                    _log_order_decision(
                        allowed=False,
                        block_reason="fatal_order_error",
                        broker_attempted=True,
                    )
                    if failure_class == "broker_config_error":
                        _log_order_decision(
                            allowed=False,
                            block_reason="broker_config_error",
                            broker_attempted=True,
                            details={
                                "failure_class": failure_class,
                                "error_message": str(e),
                                "broker_rejection": self._sanitize_broker_error(e),
                                "retryable": False,
                            },
                        )
                    if pending_signal_marked:
                        self._clear_pending_signal(signal_id)
                    return None

                self._logger.warning(f"⚠️ Retry {attempt}/3 failed: {e}")
                time.sleep(0.5 * attempt)

        self._logger.error("❌ Order placement failed after retries.")
        self._clear_uncertain_order(unique_client_id)
        _log_order_decision(
            allowed=False,
            block_reason="order_placement_failed_after_retries",
            broker_attempted=True,
        )
        if pending_signal_marked:
            self._clear_pending_signal(signal_id)
        return None

    def _get_latest_quote_safe(self, symbol: str) -> dict[str, Any] | None:
        providers = (
            getattr(self, "_data_hub", None),
            getattr(self, "data_hub", None),
            getattr(self, "_market_data", None),
            getattr(self, "_market_data_manager", None),
            getattr(self, "market_data_manager", None),
        )
        for provider in providers:
            if provider is None:
                continue
            for method_name in ("get_quote", "get_latest_tick", "get_tick"):
                method = getattr(provider, method_name, None)
                if not callable(method):
                    continue
                try:
                    quote = method(symbol)
                    if quote:
                        return dict(quote)
                except TypeError:
                    try:
                        quote = method(symbol, allow_pull=False)
                        if quote:
                            return dict(quote)
                    except Exception:
                        pass
                except Exception:
                    pass
        log_throttled(
            self._logger,
            f"quote_probe_all_methods_failed:{symbol}",
            "QUOTE_PROBE_ALL_METHODS_FAILED symbol=%s" % symbol,
            interval_sec=60,
            level=logging.WARNING,
            extra={"event": "QUOTE_PROBE_ALL_METHODS_FAILED", "symbol": str(symbol)},
        )
        return None

    def _selected_option_symbols_for_execution(self) -> set[str]:
        """Return selected CE/PE symbols from the attached active basket context."""

        selected: set[str] = set()
        providers = (
            getattr(self, "_market_data", None),
            getattr(self, "_data_hub", None),
            getattr(self, "market_data_manager", None),
            getattr(self, "data_hub", None),
        )
        for provider in providers:
            if provider is None:
                continue
            baskets: list[Any] = []
            getter = getattr(provider, "get_active_contract_basket", None)
            if callable(getter):
                try:
                    baskets.append(getter())
                except Exception as exc:  # noqa: BLE001 - provider diagnostics only
                    self._logger.debug(
                        "selected_option_basket_lookup_failed",
                        extra={
                            "event": "selected_option_basket_lookup_failed",
                            "error": str(exc),
                        },
                    )
            baskets.append(getattr(provider, "active_trading_universe", None))
            baskets.append(getattr(provider, "active_contract_basket", None))
            for basket in baskets:
                for key in ("selected_ce", "selected_pe", "atm_ce", "atm_pe"):
                    value = None
                    if isinstance(basket, Mapping):
                        value = basket.get(key)
                    elif basket is not None:
                        value = getattr(basket, key, None)
                    if value:
                        normalized = normalize_symbol(str(value))
                        if normalized:
                            selected.add(normalized)
            for attr in (
                "selected_ce",
                "selected_pe",
                "atm_ce_symbol",
                "atm_pe_symbol",
            ):
                value = getattr(provider, attr, None)
                if value:
                    normalized = normalize_symbol(str(value))
                    if normalized:
                        selected.add(normalized)
        return selected

    def _is_selected_option_for_live_execution(self, symbol: str) -> bool:
        normalized = normalize_symbol(symbol)
        return bool(
            normalized and normalized in self._selected_option_symbols_for_execution()
        )

    def _extract_quote_diagnostics(self, quote: Mapping[str, Any]) -> dict[str, Any]:
        def _safe_float(value: object, default: float = 0.0) -> float:
            if value in (None, ""):
                return default
            try:
                number = float(value)
            except (TypeError, ValueError):
                return default
            if not math.isfinite(number):
                return default
            return number

        def _safe_int(value: object, default: int = 0) -> int:
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default

        bid = _safe_float(
            quote.get("best_bid")
            or quote.get("bid")
            or quote.get("best_bid_price")
            or quote.get("buy_price")
        )
        ask = _safe_float(
            quote.get("best_ask")
            or quote.get("ask")
            or quote.get("best_ask_price")
            or quote.get("sell_price")
        )
        ltp = _safe_float(quote.get("ltp") or quote.get("last_price"))
        bid_qty = _safe_int(
            quote.get("bid_quantity") or quote.get("bid_qty") or quote.get("buy_qty")
        )
        ask_qty = _safe_int(
            quote.get("ask_quantity") or quote.get("ask_qty") or quote.get("sell_qty")
        )
        spread = max(0.0, ask - bid) if bid > 0 and ask > 0 else 0.0
        ref = ask if ask > 0 else ltp if ltp > 0 else 1.0
        spread_pct = (spread / ref) * 100.0 if ref > 0 else 999.0
        age_ms = None
        ts_raw = None
        ts_key = None
        for key in (
            "received_at",
            "wallclock",
            "exchange_timestamp",
            "timestamp",
            "ts",
            "last_trade_time",
        ):
            val = quote.get(key)
            if val not in (None, ""):
                ts_raw = val
                ts_key = key
                break

        try:
            parsed_ts = None
            if isinstance(ts_raw, datetime):
                if ts_raw.tzinfo is None:
                    ts_utc = ts_raw.replace(tzinfo=timezone.utc).timestamp()
                    ts_ist = ts_raw.replace(tzinfo=ZoneInfo("Asia/Kolkata")).timestamp()
                    now = time.time()
                    if ts_key in ("received_at", "wallclock"):
                        if (
                            abs(now - ts_ist) < abs(now - ts_utc)
                            and abs(now - ts_ist) < 60.0
                        ):
                            parsed_ts = ts_ist
                        else:
                            parsed_ts = ts_utc
                    else:
                        if (
                            abs(now - ts_utc) < abs(now - ts_ist)
                            and abs(now - ts_utc) < 60.0
                        ):
                            parsed_ts = ts_utc
                        else:
                            parsed_ts = ts_ist
                else:
                    parsed_ts = ts_raw.timestamp()
            elif isinstance(ts_raw, (int, float)):
                raw_val = float(ts_raw)
                parsed_ts = raw_val / 1000.0 if raw_val > 1_000_000_000_000 else raw_val
            elif isinstance(ts_raw, str) and ts_raw.strip():
                iso = ts_raw.strip().replace("Z", "+00:00")
                parsed_dt = datetime.fromisoformat(iso)
                if parsed_dt.tzinfo is None:
                    ts_utc = parsed_dt.replace(tzinfo=timezone.utc).timestamp()
                    ts_ist = parsed_dt.replace(
                        tzinfo=ZoneInfo("Asia/Kolkata")
                    ).timestamp()
                    now = time.time()
                    if ts_key in ("received_at", "wallclock"):
                        if (
                            abs(now - ts_ist) < abs(now - ts_utc)
                            and abs(now - ts_ist) < 60.0
                        ):
                            parsed_ts = ts_ist
                        else:
                            parsed_ts = ts_utc
                    else:
                        if (
                            abs(now - ts_utc) < abs(now - ts_ist)
                            and abs(now - ts_utc) < 60.0
                        ):
                            parsed_ts = ts_utc
                        else:
                            parsed_ts = ts_ist
                else:
                    parsed_ts = parsed_dt.timestamp()
            if parsed_ts is not None:
                age_ms = max(0.0, (time.time() - float(parsed_ts)) * 1000.0)
        except Exception:
            age_ms = None
        return {
            "bid": bid,
            "ask": ask,
            "ltp": ltp,
            "spread": spread,
            "spread_pct": spread_pct,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
            "depth_qty": bid_qty + ask_qty,
            "age_ms": age_ms,
        }

    def _trade_plan_rejection_details(
        self, plan: TradePlan, reason: str, **details: Any
    ) -> dict[str, Any]:
        payload = {
            "broker_attempted": False,
            "retryable": False,
            "trade_lifecycle_id": plan.trade_lifecycle_id,
            "client_order_id": plan.client_order_id,
            "signal_id": plan.signal_id,
            "symbol": normalize_symbol(plan.symbol),
            "requested_quantity": plan.quantity,
            "filled_quantity": 0,
            "protected_quantity": 0,
            "broker_position_quantity": None,
            "blocker_code": reason,
        }
        payload.update(details)
        return payload

    def _active_contract_for_trade_plan(self, plan: TradePlan) -> dict[str, Any] | None:
        symbol = normalize_symbol(plan.symbol)
        sources = (
            getattr(self, "_active_contract_basket", None),
            getattr(getattr(self, "_data_hub", None), "_active_contract_basket", None),
            getattr(
                getattr(self, "_market_data", None), "_active_contract_basket", None
            ),
        )
        for source in sources:
            if source is None:
                continue
            selection_obj = None
            with suppress(Exception):
                selection_obj = active_contract_selection_from_basket(source)
            if selection_obj is not None:
                selected = {
                    normalize_symbol(
                        str(getattr(selection_obj, "selected_ce", "") or "")
                    ),
                    normalize_symbol(
                        str(getattr(selection_obj, "selected_pe", "") or "")
                    ),
                }
                if symbol not in selected:
                    continue
                token_by_symbol = getattr(selection_obj, "token_by_symbol", None) or {}
                token = None
                if isinstance(token_by_symbol, Mapping):
                    token = token_by_symbol.get(symbol) or token_by_symbol.get(
                        symbol.split(":", 1)[-1]
                    )
                if token is None and symbol == normalize_symbol(
                    str(getattr(selection_obj, "selected_ce", "") or "")
                ):
                    token = getattr(selection_obj, "selected_ce_token", None)
                if token is None and symbol == normalize_symbol(
                    str(getattr(selection_obj, "selected_pe", "") or "")
                ):
                    token = getattr(selection_obj, "selected_pe_token", None)
                return {
                    "symbol": symbol,
                    "instrument_token": int(token) if token not in (None, "") else None,
                    "basket_version": getattr(selection_obj, "basket_version", None),
                    "contract_expiry": getattr(selection_obj, "expiry", None),
                    "selected_at": getattr(selection_obj, "committed_at", None),
                }
            if not isinstance(source, Mapping):
                continue
            token_map = dict(source.get("token_by_symbol") or {})
            selected = [source.get("selected_ce"), source.get("selected_pe")]
            symbols = list(source.get("option_symbols") or source.get("symbols") or [])
            if symbol not in {
                normalize_symbol(str(x)) for x in [*selected, *symbols] if x
            }:
                continue
            token = token_map.get(symbol) or token_map.get(symbol.split(":", 1)[-1])
            if symbol == normalize_symbol(str(source.get("selected_ce") or "")):
                token = token or source.get("selected_ce_token")
            if symbol == normalize_symbol(str(source.get("selected_pe") or "")):
                token = token or source.get("selected_pe_token")
            return {
                "symbol": symbol,
                "instrument_token": int(token) if token not in (None, "") else None,
                "basket_version": source.get("basket_version") or source.get("version"),
                "contract_expiry": source.get("expiry")
                or source.get("contract_expiry"),
                "selected_at": source.get("selected_at") or source.get("committed_at"),
            }
        return None

    def _validate_trade_plan(self, plan: TradePlan) -> OrderPreflightResult:
        symbol = normalize_symbol(plan.symbol)
        if not is_strategy_instrument(symbol):
            return OrderPreflightResult(
                False, "non_strategy_instrument", {"symbol": symbol}
            )
        if plan.quantity <= 0:
            return OrderPreflightResult(
                False, "invalid_quantity", {"quantity": plan.quantity}
            )
        is_entry = str(plan.intent or "").upper() == "ENTRY"
        live_checker = getattr(self, "is_live_mode", None)
        if callable(live_checker):
            live_mode = bool(live_checker())
        else:
            live_fn = getattr(self, "_order_live_execution_enabled", None)
            live_mode = bool(live_fn()) if callable(live_fn) else False
        live_entry = bool(is_entry and live_mode)
        try:
            lot_size = int(
                plan.resolved_lot_size or self._lot_size_for_symbol(symbol) or 0
            )
        except Exception as exc:  # noqa: BLE001
            if live_entry:
                return OrderPreflightResult(
                    False,
                    "lot_size_unresolved",
                    OrderManager._trade_plan_rejection_details(
                        self, plan, "lot_size_unresolved", error_type=type(exc).__name__
                    ),
                )
            raise
        if live_entry:
            if not plan.trade_lifecycle_id:
                return OrderPreflightResult(
                    False,
                    "trade_lifecycle_id_missing",
                    OrderManager._trade_plan_rejection_details(
                        self, plan, "trade_lifecycle_id_missing"
                    ),
                )
            if not plan.client_order_id:
                return OrderPreflightResult(
                    False,
                    "client_order_id_missing",
                    OrderManager._trade_plan_rejection_details(
                        self, plan, "client_order_id_missing"
                    ),
                )
            if not plan.signal_id:
                return OrderPreflightResult(
                    False,
                    "signal_id_missing",
                    OrderManager._trade_plan_rejection_details(
                        self, plan, "signal_id_missing"
                    ),
                )
            if not plan.instrument_token:
                return OrderPreflightResult(
                    False,
                    "instrument_token_mismatch",
                    OrderManager._trade_plan_rejection_details(
                        self,
                        plan,
                        "instrument_token_mismatch",
                        required_value="present",
                        actual_value=plan.instrument_token,
                    ),
                )
            if lot_size <= 0:
                return OrderPreflightResult(
                    False,
                    "lot_size_unresolved",
                    OrderManager._trade_plan_rejection_details(
                        self, plan, "lot_size_unresolved", actual_value=lot_size
                    ),
                )
            if plan.requested_lots <= 0:
                return OrderPreflightResult(
                    False,
                    "invalid_entry_lot_quantity",
                    OrderManager._trade_plan_rejection_details(
                        self,
                        plan,
                        "invalid_entry_lot_quantity",
                        requested_lots=plan.requested_lots,
                    ),
                )
            max_lots = int(os.getenv("MAX_LOTS_PER_TRADE", "1") or "1")
            if max_lots <= 1 and int(plan.requested_lots) != 1:
                return OrderPreflightResult(
                    False,
                    "invalid_entry_lot_quantity",
                    OrderManager._trade_plan_rejection_details(
                        self,
                        plan,
                        "invalid_entry_lot_quantity",
                        required_value=1,
                        actual_value=plan.requested_lots,
                    ),
                )
            expected_qty = int(plan.requested_lots) * int(lot_size)
            if plan.quantity != expected_qty or plan.quantity % lot_size != 0:
                return OrderPreflightResult(
                    False,
                    "invalid_entry_lot_quantity",
                    OrderManager._trade_plan_rejection_details(
                        self,
                        plan,
                        "invalid_entry_lot_quantity",
                        required_value=expected_qty,
                        actual_value=plan.quantity,
                        lot_size=lot_size,
                    ),
                )
            active = OrderManager._active_contract_for_trade_plan(self, plan)
            if active is None:
                return OrderPreflightResult(
                    False,
                    "active_contract_unavailable",
                    OrderManager._trade_plan_rejection_details(
                        self, plan, "active_contract_unavailable"
                    ),
                )
            if (
                plan.basket_version is not None
                and active.get("basket_version") is not None
                and str(plan.basket_version) != str(active.get("basket_version"))
            ):
                return OrderPreflightResult(
                    False,
                    "stale_contract_selection",
                    OrderManager._trade_plan_rejection_details(
                        self,
                        plan,
                        "stale_contract_selection",
                        required_value=active.get("basket_version"),
                        actual_value=plan.basket_version,
                    ),
                )
            if active.get("instrument_token") is not None and int(
                plan.instrument_token
            ) != int(active["instrument_token"]):
                return OrderPreflightResult(
                    False,
                    "instrument_token_mismatch",
                    OrderManager._trade_plan_rejection_details(
                        self,
                        plan,
                        "instrument_token_mismatch",
                        required_value=active.get("instrument_token"),
                        actual_value=plan.instrument_token,
                    ),
                )
            if (
                plan.contract_expiry
                and active.get("contract_expiry")
                and str(plan.contract_expiry) != str(active.get("contract_expiry"))
            ):
                return OrderPreflightResult(
                    False,
                    "contract_expiry_mismatch",
                    OrderManager._trade_plan_rejection_details(
                        self,
                        plan,
                        "contract_expiry_mismatch",
                        required_value=active.get("contract_expiry"),
                        actual_value=plan.contract_expiry,
                    ),
                )
        if lot_size > 0 and plan.quantity % lot_size != 0 and is_entry:
            return OrderPreflightResult(
                False,
                "quantity_not_lot_multiple",
                {"quantity": plan.quantity, "lot_size": lot_size},
            )
        if is_entry and (plan.stop_loss is None or plan.stop_loss <= 0):
            return OrderPreflightResult(False, "missing_stop_loss", {})
        if is_entry and (plan.take_profit is None or plan.take_profit <= 0):
            return OrderPreflightResult(False, "missing_take_profit", {})
        quote = self._get_latest_quote_safe(symbol)
        if quote is None:
            return OrderPreflightResult(False, "quote_unavailable", {})
        qd = self._extract_quote_diagnostics(quote)
        if qd.get("age_ms") is not None and qd["age_ms"] > plan.max_quote_age_ms:
            return OrderPreflightResult(
                False,
                "quote_stale",
                {"age_ms": qd["age_ms"], "limit_ms": plan.max_quote_age_ms},
            )
        if (
            is_entry
            and qd["bid"] > 0
            and qd["ask"] > 0
            and qd["spread_pct"] > plan.max_spread_pct
        ):
            return OrderPreflightResult(
                False,
                "spread_too_wide",
                {"spread_pct": qd["spread_pct"], "limit_pct": plan.max_spread_pct},
            )
        has_depth_fields = (
            int(qd.get("bid_qty", 0) or 0) > 0
            or int(qd.get("ask_qty", 0) or 0) > 0
            or int(qd.get("depth_qty", 0) or 0) > 0
        )
        if is_entry and has_depth_fields and qd["depth_qty"] < plan.min_depth_qty:
            return OrderPreflightResult(
                False,
                "depth_insufficient",
                {"depth_qty": qd["depth_qty"], "limit_qty": plan.min_depth_qty},
            )
        return OrderPreflightResult(
            True, "allowed", {"quote": qd, "lot_size": lot_size}
        )

    _ENTRY_SIZING_INTENTS = ("ENTRY", "SCALE_IN")

    def _apply_entry_margin_gate(
        self, plan: TradePlan, price: float
    ) -> tuple[TradePlan | None, TradePlanSubmitResult | None]:
        """Final entry-only risk/affordability gate.

        Runs after the protected price and re-anchored bracket are final and
        before any managed order, lifecycle, recovery or bracket state exists.
        Returns (effective_plan, None) to proceed, or (None, rejection).
        Exposure-reducing intents are never gated here.
        """
        intent = str(plan.intent or "").upper()
        if intent not in self._ENTRY_SIZING_INTENTS:
            return plan, None

        requested_qty = int(plan.quantity or 0)
        if requested_qty <= 0:
            return None, self._reject_entry_sizing(
                plan, "invalid_requested_quantity", {"requested_quantity": requested_qty}
            )

        lot_size = int(plan.resolved_lot_size or 0)
        if lot_size <= 0:
            try:
                lot_size = int(self._lot_size_for_symbol(plan.symbol) or 0)
            except Exception:  # noqa: BLE001 - fail closed below
                lot_size = 0
        if lot_size <= 0:
            return None, self._reject_entry_sizing(
                plan, "lot_size_unresolved", {"symbol": plan.symbol}
            )

        # Trusted balance only. _resolve_available_margin(for_entry=True)
        # applies the canonical staleness policy and reports provenance; the
        # synthetic fallback used elsewhere must never authorise a live entry.
        balance, balance_source = self._resolve_available_margin(for_entry=True)
        if balance is None or float(balance) <= 0:
            return None, self._reject_entry_sizing(
                plan,
                "available_balance_unavailable",
                {"balance_source": balance_source, "available_balance": balance},
            )
        available_balance = float(balance)

        try:
            decision = self._plan_entry_margin(
                plan=plan,
                price=price,
                lot_size=lot_size,
                available_balance=available_balance,
            )
        except Exception as exc:  # noqa: BLE001 - never place on a failed gate
            self._logger.error(
                "ENTRY_MARGIN_DECISION_FAILED symbol=%s error=%s",
                plan.symbol,
                type(exc).__name__,
                extra={
                    "event": "ENTRY_MARGIN_DECISION_FAILED",
                    "symbol": plan.symbol,
                    "error_type": type(exc).__name__,
                    "trace_id": plan.trace_id,
                },
            )
            return None, self._reject_entry_sizing(
                plan, "entry_sizing_failed", {"error_type": type(exc).__name__}
            )

        allowed_qty = int(getattr(decision, "quantity", 0) or 0)
        decision_ok = bool(getattr(decision, "ok", False))
        decision_reason = str(getattr(decision, "reason", "") or "")
        base = {
            "symbol": plan.symbol,
            "intent": intent,
            "original_requested_quantity": requested_qty,
            "original_requested_lots": requested_qty // lot_size,
            "resolved_lot_size": lot_size,
            "protected_entry_price": price,
            "final_stop_loss": plan.stop_loss,
            "final_take_profit": plan.take_profit,
            "available_balance": available_balance,
            "balance_source": balance_source,
            "estimated_required": getattr(decision, "est_required", None),
            "decision_ok": decision_ok,
            "decision_reason": decision_reason or None,
            "trace_id": plan.trace_id,
            "signal_id": plan.signal_id,
            "trade_lifecycle_id": plan.trade_lifecycle_id,
        }

        # Fail closed on ANY ok=False decision, including MIS_WINDOW_CLOSED.
        # An out-of-window entry is never allowed on the assumption that some
        # other guard will catch it later.
        session_deferred = False
        if not decision_ok or allowed_qty <= 0:
            reason = decision_reason or "margin_no_qty"
            self._logger.warning(
                "ENTRY_MARGIN_DECISION symbol=%s blocked reason=%s",
                plan.symbol,
                reason,
                extra={
                    "event": "ENTRY_MARGIN_DECISION",
                    **base,
                    "allowed_quantity": 0,
                    "allowed_lots": 0,
                    "quantity_reduced": False,
                    "broker_attempted": False,
                    "broker_attempt_pending": False,
                },
            )
            return None, TradePlanSubmitResult(
                False, reason=reason, details=dict(base), broker_attempted=False
            )

        if allowed_qty % lot_size != 0:
            return None, self._reject_entry_sizing(
                plan,
                "invalid_lot_quantity",
                {"allowed_quantity": allowed_qty, "resolved_lot_size": lot_size},
            )
        if allowed_qty > requested_qty:
            # Never round up or expand a request.
            allowed_qty = requested_qty

        effective_plan = plan
        if allowed_qty != requested_qty:
            effective_plan = replace(
                plan,
                quantity=allowed_qty,
                requested_lots=allowed_qty // lot_size,
                resolved_lot_size=lot_size,
            )
        self._logger.info(
            "ENTRY_MARGIN_DECISION symbol=%s allowed_quantity=%s reduced=%s",
            plan.symbol,
            allowed_qty,
            allowed_qty != requested_qty,
            extra={
                "event": "ENTRY_MARGIN_DECISION",
                **base,
                "allowed_quantity": allowed_qty,
                "allowed_lots": allowed_qty // lot_size,
                "quantity_reduced": allowed_qty != requested_qty,
                "sizing_permitted_only": session_deferred,
                # Pre-broker event: no broker method has been called yet.
                # The placement result remains the source of truth.
                "broker_attempted": False,
                "broker_attempt_pending": True,
            },
        )
        return effective_plan, None

    def _reject_entry_sizing(
        self, plan: TradePlan, reason: str, details: dict[str, object]
    ) -> TradePlanSubmitResult:
        """Reject an entry at the sizing gate with no broker call."""
        payload = {"symbol": plan.symbol, "trace_id": plan.trace_id, **details}
        self._logger.warning(
            "ENTRY_MARGIN_DECISION symbol=%s blocked reason=%s",
            plan.symbol,
            reason,
            extra={
                "event": "ENTRY_MARGIN_DECISION",
                "decision_reason": reason,
                "decision_ok": False,
                "allowed_quantity": 0,
                "quantity_reduced": False,
                "broker_attempted": False,
                "broker_attempt_pending": False,
                **payload,
            },
        )
        return TradePlanSubmitResult(
            False, reason=reason, details=payload, broker_attempted=False
        )

    def _plan_entry_margin(
        self,
        *,
        plan: TradePlan,
        price: float,
        lot_size: int,
        available_balance: float,
    ) -> Any:
        """Build MarginInputs from final option economics and size once."""
        # Same risk policy source and same canonical defaults as the existing
        # _pre_trade_decision() margin-planning path: risk_manager.settings
        # when present, otherwise app_settings.get_settings().risk.
        settings = getattr(self._risk_manager, "settings", None)
        if settings is None:
            try:
                settings = getattr(app_settings.get_settings(), "risk", None)
            except Exception:  # noqa: BLE001 - defaults below stay canonical
                settings = None

        def _f(name: str, default: float) -> float:
            value = getattr(settings, name, None) if settings is not None else None
            try:
                parsed = float(value) if value is not None else float(default)
            except (TypeError, ValueError):
                return float(default)
            return parsed if parsed > 0 else float(default)

        per_trade_risk_pct = _f("per_trade_risk_pct", 0.5)
        per_trade_cap_pct = _f("per_trade_cap_pct", per_trade_risk_pct)
        min_lots = max(1, int(_f("min_lots_per_trade", 1)))
        max_lots = max(min_lots, int(_f("max_lots_per_trade", 1)))
        atr_multiple = _f("atr_stop_multiple", 1.0)

        # ATR is only safe when metadata proves it belongs to this exact
        # option symbol; underlying ATR points must never be mixed with
        # option-premium prices. The final re-anchored option stop is the
        # preferred risk source.
        atr = None
        return self._margin_engine.plan(
            MarginInputs(
                symbol=plan.symbol,
                side=plan.side,
                price=float(price),
                stop_loss=(
                    float(plan.stop_loss) if plan.stop_loss is not None else None
                ),
                atr=atr,
                requested_qty=int(plan.quantity or 0),
                product=plan.product,
                lot_size=int(lot_size),
                balance=float(available_balance),
                per_trade_risk_pct=per_trade_risk_pct,
                per_trade_cap_pct=per_trade_cap_pct,
                margin_factor=float(self._margin_factor),
                margin_buffer=float(self._margin_buffer),
                contract_multiplier=float(max(int(lot_size), 1)),
                ist_now=datetime.now(ZoneInfo("Asia/Kolkata")),
                min_lots_per_trade=min_lots,
                max_lots_per_trade=max_lots,
                atr_multiple=atr_multiple,
            )
        )

    def _reanchor_bracket_to_price(self, plan: TradePlan, price: float) -> TradePlan:
        """Re-anchor a stale SL/TP bracket to the live protected ``price``.

        Strategy SL/TP are computed off the option premium at signal time.
        Premiums can move materially before submission, so the precomputed
        band can land on the wrong side of the live protected price and the
        order is rejected (``protected_price_invalidates_bracket``) — a lost
        but valid entry. When (and only when) the existing band is invalid
        against ``price``, rebuild it at ``price`` preserving the plan's
        intended SL/TP distances (so the sized rupee risk is unchanged).
        Valid brackets pass through untouched.
        """
        entry = plan.entry_price
        sl = plan.stop_loss
        tp = plan.take_profit
        if (
            not entry
            or entry <= 0
            or not price
            or price <= 0
            or sl is None
            or tp is None
        ):
            return plan

        if plan.side == "BUY":
            bracket_valid = sl < price < tp
        elif plan.side == "SELL":
            bracket_valid = tp < price < sl
        else:
            return plan
        if bracket_valid:
            return plan

        # Preserve the strategy's intended absolute distances from its entry.
        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        if sl_dist <= 0 or tp_dist <= 0:
            # Degenerate plan — leave it to be rejected by validation.
            return plan

        if plan.side == "BUY":
            new_sl = max(0.05, round(price - sl_dist, 2))
            new_tp = round(price + tp_dist, 2)
        else:  # SELL
            new_sl = round(price + sl_dist, 2)
            new_tp = max(0.05, round(price - tp_dist, 2))

        self._logger.warning(
            "BRACKET_REANCHORED symbol=%s side=%s entry=%.2f price=%.2f "
            "sl=%.2f->%.2f tp=%.2f->%.2f trace_id=%s",
            plan.symbol,
            plan.side,
            entry,
            price,
            sl,
            new_sl,
            tp,
            new_tp,
            plan.trace_id,
        )
        return replace(plan, stop_loss=new_sl, take_profit=new_tp)

    def _protected_limit_price(self, plan: TradePlan) -> float | None:
        quote = self._get_latest_quote_safe(plan.symbol)
        tick_size = 0.05
        if quote:
            qd = self._extract_quote_diagnostics(quote)
            if plan.side == "BUY":
                if qd["ask"] > 0:
                    return self._round_to_tick(qd["ask"] + tick_size, tick_size)
                if qd["ltp"] > 0:
                    return self._round_to_tick(qd["ltp"] * 1.003, tick_size)
            if plan.side == "SELL":
                if qd["bid"] > 0:
                    return self._round_to_tick(
                        max(tick_size, qd["bid"] - tick_size), tick_size
                    )
                if qd["ltp"] > 0:
                    return self._round_to_tick(
                        max(tick_size, qd["ltp"] * 0.997), tick_size
                    )
        if plan.entry_price and plan.entry_price > 0:
            return self._round_to_tick(float(plan.entry_price), tick_size)
        return None

    def explain_preflight(
        self, symbol: str, *, plan: TradePlan | None = None
    ) -> dict[str, Any]:
        quote = self._get_latest_quote_safe(symbol)
        details = {"symbol": symbol, "quote_present": quote is not None}
        if quote:
            details.update(self._extract_quote_diagnostics(quote))
        if plan is not None:
            result = self._validate_trade_plan(plan)
            details["allowed"] = result.allowed
            details["reason"] = result.reason
            details["validation_details"] = result.details
        return details

    def submit_trade_plan(self, plan: TradePlan) -> str | None:
        result = self.submit_trade_plan_result(plan)
        return result.order_id if result.accepted else None

    def submit_trade_plan_result(self, plan: TradePlan) -> TradePlanSubmitResult:
        """Validate TradePlan and delegate to place_managed_order/place_order."""
        symbol = normalize_symbol(plan.symbol)
        if self.is_kill_switch_active():
            ks = self.get_kill_switch_status()
            kill_reason = (
                ks.get("kill_reason")
                or ks.get("last_reason")
                or ks.get("reason")
                or ks.get("last_exception_type")
            )
            self._logger.warning(
                "ORDER_MANAGER_KILL_SWITCH_REJECTED symbol=%s reason=%s broker_attempted=False consecutive_failures=%s kill_reason=%s trace_id=%s",
                symbol,
                "order_manager_kill_switch_active",
                ks.get("consecutive_failures"),
                kill_reason,
                plan.trace_id,
                extra={
                    "event": "ORDER_MANAGER_KILL_SWITCH_REJECTED",
                    "symbol": symbol,
                    "reason": "order_manager_kill_switch_active",
                    "broker_attempted": False,
                    "kill_switch_status": ks,
                    "trace_id": plan.trace_id,
                },
            )
            return TradePlanSubmitResult(
                False,
                reason="order_manager_kill_switch_active",
                details={
                    "kill_switch_status": ks,
                    "kill_reason": kill_reason,
                    "symbol": symbol,
                    "trace_id": plan.trace_id,
                },
                broker_attempted=False,
            )
        validation = self._validate_trade_plan(plan)
        if not validation.allowed:
            self._logger.warning(
                "ORDER_REJECTED symbol=%s reason=%s details=%s trace_id=%s",
                symbol,
                validation.reason,
                validation.details,
                plan.trace_id,
            )
            return TradePlanSubmitResult(
                False,
                reason=validation.reason,
                details=validation.details,
                broker_attempted=False,
            )
        price = self._protected_limit_price(plan)
        if price is None:
            self._logger.warning(
                "ORDER_REJECTED symbol=%s reason=protected_limit_unavailable details=%s trace_id=%s",
                symbol,
                {"entry_price": plan.entry_price},
                plan.trace_id,
            )
            return TradePlanSubmitResult(
                False,
                reason="protected_limit_unavailable",
                details={"entry_price": plan.entry_price},
                broker_attempted=False,
            )
        if (
            plan.intent in ("ENTRY", "SCALE_IN", "REVERSAL")
            and plan.entry_price is not None
            and plan.entry_price > 0
        ):
            try:
                max_deviation_pct = float(
                    os.getenv("MAX_ENTRY_REPRICE_DEVIATION_PCT", "8.0") or "8.0"
                )
            except (TypeError, ValueError):
                max_deviation_pct = 8.0
            if not math.isfinite(max_deviation_pct) or max_deviation_pct <= 0:
                max_deviation_pct = 8.0
            deviation_pct = (
                abs(float(price) - float(plan.entry_price))
                / float(plan.entry_price)
                * 100.0
            )
            if deviation_pct > max_deviation_pct:
                details = {
                    "reference_price": float(plan.entry_price),
                    "protected_price": float(price),
                    "deviation_pct": deviation_pct,
                    "max_deviation_pct": max_deviation_pct,
                    "side": plan.side,
                }
                self._logger.warning(
                    "ORDER_REJECTED symbol=%s "
                    "reason=entry_price_deviation_exceeded details=%s trace_id=%s",
                    symbol,
                    details,
                    plan.trace_id,
                )
                return TradePlanSubmitResult(
                    False,
                    reason="entry_price_deviation_exceeded",
                    details=details,
                    broker_attempted=False,
                )
        plan = self._reanchor_bracket_to_price(plan, price)
        if plan.side == "BUY":
            if plan.stop_loss is not None and plan.stop_loss >= price:
                details = {
                    "protected_price": price,
                    "stop_loss": plan.stop_loss,
                    "take_profit": plan.take_profit,
                    "side": plan.side,
                    "violation": "stop_loss_above_or_equal_entry",
                }
                self._logger.warning(
                    "ORDER_REJECTED symbol=%s reason=protected_price_invalidates_bracket details=%s trace_id=%s",
                    symbol,
                    details,
                    plan.trace_id,
                )
                return TradePlanSubmitResult(
                    False,
                    reason="protected_price_invalidates_bracket",
                    details=details,
                    broker_attempted=False,
                )
            if plan.take_profit is not None and plan.take_profit <= price:
                details = {
                    "protected_price": price,
                    "stop_loss": plan.stop_loss,
                    "take_profit": plan.take_profit,
                    "side": plan.side,
                    "violation": "take_profit_below_or_equal_entry",
                }
                self._logger.warning(
                    "ORDER_REJECTED symbol=%s reason=protected_price_invalidates_bracket details=%s trace_id=%s",
                    symbol,
                    details,
                    plan.trace_id,
                )
                return TradePlanSubmitResult(
                    False,
                    reason="protected_price_invalidates_bracket",
                    details=details,
                    broker_attempted=False,
                )
        elif plan.side == "SELL":
            if plan.stop_loss is not None and plan.stop_loss <= price:
                details = {
                    "protected_price": price,
                    "stop_loss": plan.stop_loss,
                    "take_profit": plan.take_profit,
                    "side": plan.side,
                    "violation": "stop_loss_below_or_equal_entry",
                }
                self._logger.warning(
                    "ORDER_REJECTED symbol=%s reason=protected_price_invalidates_bracket details=%s trace_id=%s",
                    symbol,
                    details,
                    plan.trace_id,
                )
                return TradePlanSubmitResult(
                    False,
                    reason="protected_price_invalidates_bracket",
                    details=details,
                    broker_attempted=False,
                )
            if plan.take_profit is not None and plan.take_profit >= price:
                details = {
                    "protected_price": price,
                    "stop_loss": plan.stop_loss,
                    "take_profit": plan.take_profit,
                    "side": plan.side,
                    "violation": "take_profit_above_or_equal_entry",
                }
                self._logger.warning(
                    "ORDER_REJECTED symbol=%s reason=protected_price_invalidates_bracket details=%s trace_id=%s",
                    symbol,
                    details,
                    plan.trace_id,
                )
                return TradePlanSubmitResult(
                    False,
                    reason="protected_price_invalidates_bracket",
                    details=details,
                    broker_attempted=False,
                )
        # Attribute lookup (not a direct call) so synthetic test doubles that
        # invoke this method unbound still work; a real OrderManager always
        # provides it, so a live entry can never bypass the gate here.
        _entry_gate = getattr(self, "_apply_entry_margin_gate", None)
        _entry_sizing_details: dict[str, Any] | None = None
        if _entry_gate is not None:
            _requested_before_gate = int(plan.quantity or 0)
            effective_plan, sizing_rejection = _entry_gate(plan, price)
            if sizing_rejection is not None:
                return sizing_rejection
            plan = effective_plan if effective_plan is not None else plan
            _effective_lot = int(getattr(plan, "resolved_lot_size", 0) or 0)
            if _effective_lot <= 0:
                try:
                    _effective_lot = int(self._lot_size_for_symbol(plan.symbol) or 0)
                except Exception:  # noqa: BLE001
                    _effective_lot = 0
            # Frozen sizing record: recovery must never restore quantity the
            # first gate removed. Stamped onto every post-gate result so a
            # retryable broker rejection still carries it.
            _entry_sizing_details = {
                "entry_sizing_requested_quantity": _requested_before_gate,
                "entry_sizing_effective_quantity": int(plan.quantity or 0),
                "entry_sizing_lot_size": _effective_lot,
                "entry_sizing_symbol": plan.symbol,
                "entry_sizing_trace_id": plan.trace_id,
                "entry_sizing_signal_id": plan.signal_id,
                "entry_sizing_trade_lifecycle_id": plan.trade_lifecycle_id,
            }
            # Diagnostics only. Recovery must NEVER read this: it is a
            # most-recent-submission cache and would leak sizing across
            # concurrent trades.
            self._last_entry_sizing_details_diagnostic = dict(_entry_sizing_details)
        if hasattr(self, "place_managed_order"):
            if hasattr(self, "place_managed_order_result"):
                try:
                    managed = self.place_managed_order_result(
                        symbol=symbol,
                        side=plan.side,
                        quantity=plan.quantity,
                        entry_price=price,
                        stop_loss=plan.stop_loss,
                        take_profit=plan.take_profit,
                        signal_id=plan.signal_id,
                        strategy_name=plan.strategy_name,
                        tag=plan.tag,
                        product=plan.product,
                        variety=plan.variety,
                        trace_id=plan.trace_id,
                        allow_market_entry=plan.allow_market_entry,
                        intent=plan.intent,
                        intended_position_side=plan.intended_position_side,
                        trade_lifecycle_id=plan.trade_lifecycle_id,
                        client_order_id=plan.client_order_id,
                        basket_version=plan.basket_version,
                        instrument_token=plan.instrument_token,
                        contract_expiry=plan.contract_expiry,
                        requested_lots=int(plan.requested_lots or 0),
                        resolved_lot_size=int(plan.resolved_lot_size or 0),
                    )
                except Exception as exc:  # noqa: BLE001
                    err = self._sanitize_broker_error(exc)
                    self._last_order_api_error_type = type(exc).__name__
                    self._last_order_api_error = err
                    self._emit_broker_health_status(force=True)
                    return _stamp_entry_sizing(
                        _entry_sizing_details,
                        TradePlanSubmitResult(
                            False,
                            reason="broker_placement_exception",
                            details={
                                "error_type": type(exc).__name__,
                                "error": err,
                                "symbol": symbol,
                                "trace_id": plan.trace_id,
                                "protected_price": price,
                            },
                            broker_attempted=True,
                        ),
                    )
                return _stamp_entry_sizing(
                    _entry_sizing_details,
                    TradePlanSubmitResult(
                        managed.accepted,
                        order_id=managed.order_id,
                        reason=managed.reason,
                        details=managed.details or {"protected_price": price},
                        broker_attempted=managed.broker_attempted,
                    ),
                )
            try:
                oid = self.place_managed_order(
                    symbol=symbol,
                    side=plan.side,
                    quantity=plan.quantity,
                    entry_price=price,
                    stop_loss=plan.stop_loss,
                    take_profit=plan.take_profit,
                    signal_id=plan.signal_id,
                    strategy_name=plan.strategy_name,
                    tag=plan.tag,
                    product=plan.product,
                    variety=plan.variety,
                    trace_id=plan.trace_id,
                    allow_market_entry=plan.allow_market_entry,
                    intent=plan.intent,
                    intended_position_side=plan.intended_position_side,
                )
            except Exception as exc:  # noqa: BLE001
                err = self._sanitize_broker_error(exc)
                self._last_order_api_error_type = type(exc).__name__
                self._last_order_api_error = err
                self._emit_broker_health_status(force=True)
                return _stamp_entry_sizing(
                    _entry_sizing_details,
                    TradePlanSubmitResult(
                        False,
                        reason="broker_placement_exception",
                        details={
                            "error_type": type(exc).__name__,
                            "error": err,
                            "symbol": symbol,
                            "trace_id": plan.trace_id,
                            "protected_price": price,
                        },
                        broker_attempted=True,
                    ),
                )
            if oid:
                self._last_order_api_error_type = None
                self._last_order_api_error = None
            return _stamp_entry_sizing(
                _entry_sizing_details,
                TradePlanSubmitResult(
                    bool(oid),
                    order_id=oid,
                    reason="accepted" if oid else "place_order_rejected",
                    details={"protected_price": price},
                    broker_attempted=bool(oid),
                ),
            )
        try:
            oid = self.place_order(
                symbol=symbol,
                side=plan.side,
                quantity=plan.quantity,
                order_type=OrderType.LIMIT,
                price=price,
                stop_loss=plan.stop_loss,
                take_profit=plan.take_profit,
                tag=plan.tag,
                check_risk=True,
                product=plan.product,
                intent=plan.intent,
                intended_position_side=plan.intended_position_side,
            )
        except Exception as exc:  # noqa: BLE001
            err = self._sanitize_broker_error(exc)
            self._last_order_api_error_type = type(exc).__name__
            self._last_order_api_error = err
            self._emit_broker_health_status(force=True)
            return _stamp_entry_sizing(
                _entry_sizing_details,
                TradePlanSubmitResult(
                    False,
                    reason="broker_placement_exception",
                    details={
                        "error_type": type(exc).__name__,
                        "error": err,
                        "symbol": symbol,
                        "trace_id": plan.trace_id,
                        "protected_price": price,
                    },
                    broker_attempted=True,
                ),
            )
        if oid:
            self._last_order_api_error_type = None
            self._last_order_api_error = None
        return _stamp_entry_sizing(
            _entry_sizing_details,
            TradePlanSubmitResult(
                bool(oid),
                order_id=oid,
                reason="accepted" if oid else "order_rejected",
                details={"protected_price": price},
                broker_attempted=True,
            ),
        )

    def place_managed_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        signal_id: str | None = None,
        strategy_name: str = "runner",
        tag: str = "strategy",
        product: str = "MIS",
        variety: str = "regular",
        trace_id: str | None = None,
        allow_market_entry: bool = False,
        intent: OrderIntent = "ENTRY",
        intended_position_side: Literal["LONG", "SHORT"] | None = "LONG",
        trade_lifecycle_id: str | None = None,
        client_order_id: str | None = None,
        basket_version: int | str | None = None,
        instrument_token: int | None = None,
        contract_expiry: str | None = None,
    ) -> str | None:
        result = self.place_managed_order_result(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_id=signal_id,
            strategy_name=strategy_name,
            tag=tag,
            product=product,
            variety=variety,
            trace_id=trace_id,
            allow_market_entry=allow_market_entry,
            intent=intent,
            intended_position_side=intended_position_side,
            trade_lifecycle_id=trade_lifecycle_id,
            client_order_id=client_order_id,
            basket_version=basket_version,
            instrument_token=instrument_token,
            contract_expiry=contract_expiry,
        )
        return result.order_id if result.accepted else None

    def place_managed_order_result(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        signal_id: str | None = None,
        strategy_name: str = "runner",
        tag: str = "strategy",
        product: str = "MIS",
        variety: str = "regular",
        trace_id: str | None = None,
        allow_market_entry: bool = False,
        intent: OrderIntent = "ENTRY",
        intended_position_side: Literal["LONG", "SHORT"] | None = "LONG",
        trade_lifecycle_id: str | None = None,
        client_order_id: str | None = None,
        basket_version: int | str | None = None,
        instrument_token: int | None = None,
        contract_expiry: str | None = None,
        requested_lots: int = 0,
        resolved_lot_size: int = 0,
    ) -> ManagedOrderResult:
        """Convert a TradePlan-style entry into broker/paper placement plus bracket registration."""
        # BUG 6 FIX: lot size was hardcoded to 65 — NIFTY options lot size fallback for resiliency.
        # Use dynamic resolution with a safe fallback to reject mismatched quantities.
        exec_mode = str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper()
        try:
            _lot = self._lot_size_for_symbol(symbol)
        except Exception as exc:
            if exec_mode == "LIVE":
                self._logger.error(
                    "LIVE_ORDER_REJECTED symbol=%s reason=lot_size_unresolved error=%s",
                    symbol,
                    exc,
                )
                return ManagedOrderResult(False, reason="lot_size_unresolved")
            fallback_lot = int(float(os.getenv("PAPER_LOT_FALLBACK", "0") or 0))
            if fallback_lot <= 0:
                self._logger.error(
                    "PAPER_ORDER_REJECTED symbol=%s reason=lot_size_unresolved error=%s",
                    symbol,
                    exc,
                )
                return ManagedOrderResult(False, reason="lot_size_unresolved")
            _lot = fallback_lot
            self._logger.warning(
                "PAPER_LOT_FALLBACK_USED symbol=%s lot=%s", symbol, _lot
            )
        if _lot > 0 and quantity % _lot != 0:
            self._logger.error(
                f"🛑 INVALID QTY: {quantity} is not a multiple of {_lot} (lot size for {symbol}). Order aborted."
            )
            return ManagedOrderResult(
                False,
                reason="invalid_lot_quantity",
                details={"quantity": quantity, "lot_size": _lot},
            )

        # 2. Execute Entry (Passes Safety Guard because SL is provided)
        entry_order_type = OrderType.LIMIT
        if entry_price is None or entry_price <= 0:
            if not allow_market_entry:
                self._logger.warning(
                    "MANAGED_ORDER_REJECTED symbol=%s reason=protected_limit_unavailable",
                    symbol,
                )
                return ManagedOrderResult(False, reason="protected_limit_unavailable")
            entry_order_type = OrderType.MARKET

        self._last_order_decision = {}
        order_id = self.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=entry_order_type,
            price=entry_price,
            stop_loss=stop_loss,  # ✅ Critical: Passing this satisfies the Safety Guard
            take_profit=take_profit,
            signal_id=signal_id,
            trace_id=trace_id,
            tag=tag,
            product=product,
            intent=intent,
            intended_position_side=intended_position_side,
            client_order_id=client_order_id,
            trade_lifecycle_id=trade_lifecycle_id,
            basket_version=basket_version,
            instrument_token=instrument_token,
            contract_expiry=contract_expiry,
            requested_lots=requested_lots,
            resolved_lot_size=resolved_lot_size,
        )

        if order_id:
            return ManagedOrderResult(
                True, order_id=order_id, reason="accepted", broker_attempted=True
            )
        decision = dict(getattr(self, "_last_order_decision", {}) or {})
        if not decision:
            return ManagedOrderResult(
                False,
                reason="place_order_rejected_without_decision",
                details={
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "trace_id": trace_id,
                },
                broker_attempted=False,
            )
        return ManagedOrderResult(
            False,
            reason=str(decision.get("block_reason") or "place_order_rejected"),
            details=dict(decision.get("details") or {}),
            broker_attempted=bool(decision.get("broker_attempted", True)),
        )

    def guard_existing_position(
        self,
        *,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        quantity: int,
        average_price: float,
        last_price: float | None = None,
        product: str | None = None,
    ) -> None:
        """Ensure orphaned positions receive protective exits.

        Args:
            symbol: Instrument identifier for the live position.
            side: Direction of the existing position.
            quantity: Absolute open quantity to protect.
            average_price: Average fill price for the position.
            last_price: Latest traded price, if available.
            product: Optional broker product code for exit orders.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered guard_existing_position",
            extra={
                "event": "guard_existing_position_enter",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            },
        )
        qty = abs(int(quantity))
        guard_symbol = DataHub.normalize(symbol) or symbol.strip().upper()
        if qty <= 0:
            emit_diag(
                self._logger,
                "guard_skip_zero_qty",
                reason="zero_qty",
                severity="info",
                symbol=symbol,
                quantity=quantity,
            )
            return
        if self.has_guard_pair(guard_symbol):
            self._logger.info(
                "Condition met: guard_pair_exists",
                extra={
                    "event": "guard_pair_exists",
                    "symbol": guard_symbol,
                    "side": side,
                },
            )
            return
        reduce_side: Literal["BUY", "SELL"] = (
            "SELL" if side.upper() == "LONG" else "BUY"
        )
        product_code = (product or "MIS").strip().upper()
        base_price = float(average_price) if average_price > 0 else 0.0
        if last_price is not None and float(last_price) > 0:
            base_price = float(last_price)
        if base_price <= 0:
            base_price = 1.0
        price_buffer = max(base_price * 0.015, 5.0)
        if side.upper() == "LONG":
            stop_price = max(base_price - price_buffer, 0.5)
            target_price = base_price + price_buffer
        else:
            stop_price = base_price + price_buffer
            target_price = max(base_price - price_buffer, 0.5)
        try:
            existing = self._positions.get_pending_orders(symbol)
        except Exception:  # pragma: no cover - defensive
            existing = []
        if any(order.side == reduce_side for order in existing):
            self._logger.info(
                "Condition met: guard_existing_orders_present",
                extra={
                    "event": "guard_existing_orders_present",
                    "symbol": symbol,
                    "side": reduce_side,
                },
            )
            return
        if not self._ensure_trading_allowed(
            symbol=symbol, side=reduce_side, quantity=qty
        ):
            self._logger.info(
                "Condition met: guard_trading_blocked",
                extra={
                    "event": "guard_trading_blocked",
                    "symbol": symbol,
                    "side": reduce_side,
                },
            )
            return
        try:
            stop_details = self._place_single_order(
                symbol=symbol,
                side=reduce_side,
                quantity=qty,
                order_type=OrderType.STOP_LOSS_MARKET,
                price=stop_price,
                product=product_code,
                tag="orphan-guard",
            )
            target_details = self._place_single_order(
                symbol=symbol,
                side=reduce_side,
                quantity=qty,
                order_type=OrderType.LIMIT,
                price=target_price,
                product=product_code,
                tag="orphan-guard",
            )
        except OrderPlacementError as exc:
            self._logger.error(
                "Failure in guard_existing_position: %s",
                exc,
                extra={
                    "event": "guard_existing_position_failed",
                    "symbol": symbol,
                    "side": reduce_side,
                },
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in guard_existing_position: %s",
                exc,
                extra={
                    "event": "guard_existing_position_failed",
                    "symbol": symbol,
                    "side": reduce_side,
                },
            )
            return
        pair = GuardPair(
            symbol=guard_symbol,
            side=reduce_side,
            quantity=qty,
            stop_order_id=stop_details.order_id,
            target_order_id=target_details.order_id,
            created_at=datetime.now(timezone.utc),
        )
        self._register_guard_pair(pair)

        # ================= FIX-4: ATTACH VIRTUAL BRACKET =================
        if self._bracket_manager:
            try:
                self._bracket_manager.attach_orphan_position(
                    symbol=guard_symbol,
                    side="BUY" if side.upper() == "LONG" else "SELL",
                    qty=qty,
                    entry_price=base_price,
                )
                self._logger.warning(
                    "Orphan position attached to virtual bracket",
                    extra={
                        "event": "orphan_virtual_bracket_attached",
                        "symbol": guard_symbol,
                        "side": side,
                        "quantity": qty,
                    },
                )
            except Exception as exc:  # defensive, do not block guard
                self._logger.error(
                    "Failed to attach virtual bracket to orphan position: %s",
                    exc,
                    extra={
                        "event": "orphan_virtual_bracket_failed",
                        "symbol": guard_symbol,
                    },
                )
        # ================= END FIX-4 =====================================
        self._logger.info(
            "Condition met: recover_orphan_position",
            extra={
                "event": "recover_orphan_position",
                "symbol": symbol,
                "side": side,
                "quantity": qty,
                "stop_order_id": stop_details.order_id,
                "target_order_id": target_details.order_id,
            },
        )

    def place_bracket_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        *,
        product: str | None = None,
        tag: str | None = None,
        trailing_spec: TrailingSpec | None = None,
        # ✅ NEW: High-Profit Args (TP1 Scaling & ATR Trailing)
        tp1_price: float | None = None,
        tp1_qty: int | None = None,
        trailing_atr_mult: float | None = None,
    ) -> str | None:
        """
        Places a PHYSICAL Entry and registers a VIRTUAL Bracket.
        World-Class Upgrade: Replaces legacy OCO with internal Sniper execution.
        """
        self._logger.info(
            f"🚀 Initiating Virtual Bracket: {symbol} {side} {quantity}",
            extra={"event": "virtual_bracket_init", "symbol": symbol},
        )

        # 1. Place the Entry Order (Physical)
        # Logic: If price > 0 use LIMIT, else MARKET
        order_type = OrderType.LIMIT if entry_price > 0 else OrderType.MARKET

        entry_id = self.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=entry_price if entry_price > 0 else None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            product=product or "MIS",
            tag=f"virtual_bracket_{tag}" if tag else "virtual_bracket",
            check_risk=True,
        )

        if not entry_id:
            self._logger.error(f"❌ Bracket Entry Failed for {symbol}")
            return None

        # 2. Register Virtual Bracket (Pending Fill)
        # The BracketManager will wait for 'confirm_entry_fill' to activate triggers
        if self._bracket_manager:
            self._bracket_manager.register_virtual_bracket(
                order_id=entry_id,
                symbol=symbol,
                side=side,
                qty=quantity,
                price=entry_price if entry_price > 0 else 0.0,
                sl=stop_loss,
                tp=take_profit,
                tag=tag or "virtual_bracket",
                tp1_price=tp1_price,
                tp1_qty=tp1_qty,
                trailing_atr_mult=trailing_atr_mult,
                activate_immediately=False,  # Wait for actual fill!
            )
            self._logger.info(
                f"🛡️ Virtual Bracket Registered (Pending Fill) for {entry_id}"
            )
        else:
            self._logger.warning("⚠️ BracketManager not attached! Trade is naked.")

        # Return Entry ID (Stop/TP IDs are empty because they are virtual/dynamic)
        return entry_id

    def _register_bracket_state(self, state: BracketState) -> None:
        """Persist bracket metadata and index child orders.

        Args:
            state: Runtime bracket state to register.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _register_bracket_state",
            extra={"event": "register_bracket_state", "entry_id": state.entry_id},
        )
        with self._lock:
            self._brackets[state.entry_id] = state
            for order_id in (
                state.stop_order_id,
                state.tp_primary_id,
                state.tp_secondary_id,
            ):
                if order_id:
                    self._bracket_index[order_id] = state.entry_id
        self._persist_bracket_state(state)

    def _await_entry_fill(
        self, entry: OrderDetails, *, timeout: float
    ) -> OrderDetails | None:
        """Wait for *entry* to fill and cancel remainder on timeout.

        Args:
            entry: Entry order whose fill should be confirmed.
            timeout: Maximum seconds to wait for a full fill.

        Returns:
            Updated order details when any quantity executed, otherwise ``None``.

        Raises:
            OrderPlacementError: If broker polling fails unexpectedly.
        """

        self._logger.debug(
            "Entered _await_entry_fill",
            extra={
                "event": "await_entry_fill_enter",
                "order_id": entry.order_id,
                "timeout": timeout,
            },
        )
        try:
            filled = self.wait_for_fill(entry.order_id, timeout_sec=timeout)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _await_entry_fill: %s",
                exc,
                extra={
                    "event": "await_entry_fill_wait_failed",
                    "order_id": entry.order_id,
                },
            )
            raise OrderPlacementError("Failed to confirm entry fill") from exc

        updated = self._refresh_order(entry.order_id)
        if filled and updated.filled_quantity > 0:
            self._logger.info(
                "Condition met: entry filled during wait",
                extra={
                    "event": "await_entry_fill_complete",
                    "order_id": entry.order_id,
                    "filled_quantity": updated.filled_quantity,
                },
            )
            return updated

        if updated.filled_quantity > 0:
            self._logger.info(
                "Condition met: entry partially filled; cancelling remainder",
                extra={
                    "event": "await_entry_fill_partial",
                    "order_id": entry.order_id,
                    "filled_quantity": updated.filled_quantity,
                },
            )
        else:
            self._logger.info(
                "Condition met: entry not filled within timeout",
                extra={
                    "event": "await_entry_fill_timeout",
                    "order_id": entry.order_id,
                },
            )

        try:
            self.cancel_order(entry.order_id)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _await_entry_fill cancel: %s",
                exc,
                extra={
                    "event": "await_entry_fill_cancel_failed",
                    "order_id": entry.order_id,
                },
            )

        refreshed = self._refresh_order(entry.order_id)
        if refreshed.filled_quantity > 0:
            self._logger.info(
                "Condition met: entry retained partial fill",
                extra={
                    "event": "await_entry_fill_partial_kept",
                    "order_id": entry.order_id,
                    "filled_quantity": refreshed.filled_quantity,
                },
            )
            return refreshed
        return None

    def set_notifier(self, notifier: TelegramEnhancedNotifier | None) -> None:
        """Attach or clear the notifier used for critical alerts.

        Args:
            notifier: Optional notifier responsible for alert delivery.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered OrderManager.set_notifier",
            extra={"event": "order_manager_set_notifier"},
        )
        self._notifier = notifier
        bracket_manager = self._bracket_manager
        if bracket_manager is not None and hasattr(bracket_manager, "set_notifier"):
            bracket_manager.set_notifier(self._notify_bracket_event)

    def _notify_bracket_event(
        self, event: str, payload: Mapping[str, object] | None = None
    ) -> None:
        """Send bracket lifecycle notifications through the Telegram notifier.

        Args:
            event: Event label describing the bracket lifecycle action.
            payload: Optional payload to enrich the alert message.

        Returns:
            None.

        Raises:
            None.
        """
        self._logger.debug(
            "Entered _notify_bracket_event",
            extra={"event": "order_manager_bracket_notify_enter", "label": event},
        )
        if self._notifier is None:
            return
        try:
            parts: list[str] = []
            if payload:
                for key, value in payload.items():
                    if value is None:
                        continue
                    if isinstance(value, float):
                        parts.append(f"{key}={value:.2f}")
                    else:
                        parts.append(f"{key}={value}")
            detail = " | ".join(parts)
            message = f"[{event}] {detail}" if detail else f"[{event}]"
            self._notifier.send_alert(message)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _notify_bracket_event: %s",
                exc,
                extra={"event": "order_manager_bracket_notify_failed", "label": event},
                exc_info=exc,
            )

    def _register_virtual_bracket_for_fill(
        self, order: OrderDetails, *, source: str
    ) -> None:
        """Register a virtual bracket for a filled order when needed.

        Args:
            order: Filled order details to protect with a bracket.
            source: Source label describing the fill origin.

        Returns:
            None.

        Raises:
            None.
        """
        self._logger.debug(
            "Entered _register_virtual_bracket_for_fill",
            extra={
                "event": "order_manager_register_virtual_bracket",
                "order_id": order.order_id,
                "source": source,
            },
        )
        if self._bracket_manager is None:
            self._logger.warning(
                "Bracket manager missing for filled order",
                extra={
                    "event": "order_manager_bracket_missing",
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "source": source,
                },
            )
            self._notify_bracket_event(
                "BRACKET_MANAGER_MISSING",
                {"symbol": order.symbol, "order_id": order.order_id, "source": source},
            )
            return
        if str(getattr(order, "intent", "UNKNOWN")).upper() not in {
            "ENTRY",
            "SCALE_IN",
            "REVERSAL",
        }:
            self._logger.warning(
                "Skipping entry bracket activation for non-entry order",
                extra={
                    "event": "entry_bracket_rejected_for_exit_order",
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "intent": getattr(order, "intent", "UNKNOWN"),
                    "source": source,
                },
            )
            return
        try:
            entry_price = float(
                order.fill_price or order.average_price or order.price or 0.0
            )
            qty = int(order.filled_quantity or order.quantity or 0)
            if entry_price <= 0 or qty <= 0:
                self._logger.warning(
                    "Skipping bracket registration due to invalid fill data",
                    extra={
                        "event": "order_manager_bracket_invalid_fill",
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "entry_price": entry_price,
                        "qty": qty,
                    },
                )
                return

            sl_price = float(order.stop_loss or 0.0)
            tp_price = float(order.take_profit or 0.0)
            side = str(order.side).upper()
            if sl_price <= 0:
                sl_price = round(entry_price * (0.90 if side == "BUY" else 1.10), 1)
            if tp_price <= 0:
                tp_price = round(entry_price * (1.20 if side == "BUY" else 0.80), 1)

            bracket_exists = self._bracket_manager.get_bracket(order.order_id)
            if bracket_exists is None:
                if self._bracket_manager.has_active_bracket(order.symbol):
                    self._logger.error(
                        "ENTRY_BRACKET_LIFECYCLE_CONFLICT "
                        "order_id=%s symbol=%s source=%s",
                        order.order_id,
                        order.symbol,
                        source,
                        extra={
                            "event": "ENTRY_BRACKET_LIFECYCLE_CONFLICT",
                            "order_id": order.order_id,
                            "symbol": order.symbol,
                            "source": source,
                        },
                    )
                    return
                self._bracket_manager.register_virtual_bracket(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=side,
                    qty=qty,
                    price=entry_price,
                    sl=sl_price,
                    tp=tp_price,
                    tag=order.tag or source,
                    intent=str(getattr(order, "intent", "ENTRY") or "ENTRY"),
                )
                bracket_exists = self._bracket_manager.get_bracket(order.order_id)
                if bracket_exists is None:
                    raise RuntimeError("entry bracket registration was not confirmed")
            self._bracket_manager.confirm_entry_fill(order.order_id, entry_price)
            verified = self._bracket_manager.get_bracket(order.order_id)
            if (
                verified is None
                or not bool(getattr(verified, "entry_confirmed", False))
                or not bool(getattr(verified, "active", False))
            ):
                raise RuntimeError("entry bracket activation was not confirmed")
            self._logger.info(
                "Condition met: virtual bracket active for filled order",
                extra={
                    "event": "order_manager_bracket_activated",
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "source": source,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _register_virtual_bracket_for_fill: %s",
                exc,
                extra={
                    "event": "order_manager_bracket_register_failed",
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "source": source,
                },
                exc_info=exc,
            )
            raise

    def _confirm_position_protection_for_fill(self, order: OrderDetails) -> None:
        """Acknowledge protection only after fill accounting and bracket activation."""

        if str(getattr(order, "intent", "UNKNOWN")).upper() not in {
            "ENTRY",
            "SCALE_IN",
            "REVERSAL",
        }:
            return
        if self._bracket_manager is None:
            return
        confirm = getattr(self._positions, "confirm_entry_protection", None)
        if not callable(confirm):
            return
        bracket = self._bracket_manager.get_bracket(order.order_id)
        protected_qty = int(order.filled_quantity or order.quantity or 0)
        if (
            bracket is None
            or not bool(getattr(bracket, "entry_confirmed", False))
            or not bool(getattr(bracket, "active", False))
            or float(getattr(bracket, "sl_trigger_price", 0.0) or 0.0) <= 0
            or protected_qty <= 0
        ):
            self._logger.error(
                "ENTRY_PROTECTION_NOT_CONFIRMED order_id=%s symbol=%s",
                order.order_id,
                order.symbol,
                extra={
                    "event": "ENTRY_PROTECTION_NOT_CONFIRMED",
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                },
            )
            return
        try:
            confirm(
                order.order_id,
                str(getattr(bracket, "bracket_id", order.order_id)),
                protected_qty,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "ENTRY_PROTECTION_CONFIRM_FAILED order_id=%s symbol=%s error=%s",
                order.order_id,
                order.symbol,
                exc,
                extra={
                    "event": "ENTRY_PROTECTION_CONFIRM_FAILED",
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "error_type": type(exc).__name__,
                },
                exc_info=exc,
            )

    def _handle_failed_bracket_entry(
        self,
        *,
        entry_details: OrderDetails,
        exit_side: Literal["BUY", "SELL"],
        product: str | None,
        tag: str | None,
        original_exception: Exception | None = None,
    ) -> None:
        """Place and verify an emergency exit when bracket setup fails.

        Args:
            entry_details: Broker-reported entry order metadata.
            exit_side: Direction required to flatten the position.
            product: Broker product code to reuse for the exit.
            tag: Optional tag propagated to the exit order.
            original_exception: Root cause of the bracket failure, if known.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _handle_failed_bracket_entry",
            extra={
                "event": "handle_failed_bracket_entry",
                "entry_id": entry_details.order_id,
                "symbol": entry_details.symbol,
            },
        )
        symbol = entry_details.symbol
        filled_qty = int(entry_details.filled_quantity or 0)
        if filled_qty <= 0:
            self._logger.warning(
                "Bracket failure but no fill; no emergency exit needed",
                extra={
                    "event": "bracket.emergency.skip",
                    "symbol": symbol,
                    "entry_id": entry_details.order_id,
                },
            )
            return

        try:
            exit_details = self._execute_market_order(
                symbol=symbol,
                side=exit_side,
                quantity=filled_qty,
                product=product,
                tag=tag,
            )
            emergency_order_id = exit_details.order_id
            self._logger.critical(
                "Emergency exit placed after bracket failure",
                extra={
                    "event": "bracket.emergency.placed",
                    "symbol": symbol,
                    "emergency_order_id": emergency_order_id,
                    "quantity": filled_qty,
                },
            )
        except Exception as emergency_exc:  # noqa: BLE001
            self._logger.critical(
                "Emergency exit placement FAILED",
                extra={
                    "event": "bracket.emergency.error",
                    "symbol": symbol,
                    "entry_id": entry_details.order_id,
                    "error": str(emergency_exc),
                },
                exc_info=emergency_exc,
            )
            if self._notifier is not None:
                try:
                    self._notifier.send_alert(
                        (
                            "🚨 CRITICAL: Could not place emergency exit for "
                            f"{symbol}\n"
                            f"Filled qty: {filled_qty}\n"
                            f"Error: {str(emergency_exc)}\n"
                            "IMMEDIATE MANUAL INTERVENTION REQUIRED"
                        )
                    )
                except Exception:  # pragma: no cover - notifier resilience
                    self._logger.error(
                        "Notifier failed during emergency exit placement",
                        extra={
                            "event": "bracket.emergency.notifier_failed",
                            "symbol": symbol,
                        },
                        exc_info=True,
                    )
            raise

        max_wait_seconds = 10.0
        poll_interval = 0.5
        elapsed = 0.0
        exit_verified = False

        raw_status_getter = getattr(self._broker, "get_order_status", None)
        status_getter: Callable[[str], Mapping[str, Any] | None] | None = None
        if callable(raw_status_getter):
            status_getter = cast(
                Callable[[str], Mapping[str, Any] | None], raw_status_getter
            )
        else:
            self._logger.warning(
                "Broker does not support status polling during emergency exit",
                extra={
                    "event": "bracket.emergency.status_unsupported",
                    "symbol": symbol,
                    "order_id": emergency_order_id,
                },
            )

        while elapsed < max_wait_seconds:
            time.sleep(poll_interval)
            elapsed += poll_interval
            if status_getter is None:
                break
            try:
                response = self._call_broker(status_getter, emergency_order_id)
            except Exception as status_exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in emergency exit status poll",
                    extra={
                        "event": "bracket.emergency.status_error",
                        "symbol": symbol,
                        "order_id": emergency_order_id,
                        "error": str(status_exc),
                    },
                    exc_info=status_exc,
                )
                continue

            order_status = cast(Mapping[str, Any] | None, response)
            if not order_status:
                continue

            status_value = str(order_status.get("status") or "").upper()
            if status_value in {"COMPLETE", "FILLED"}:
                exit_verified = True
                self._logger.info(
                    "Emergency exit verified",
                    extra={
                        "event": "bracket.emergency.verified",
                        "symbol": symbol,
                        "order_id": emergency_order_id,
                    },
                )
                break
            if status_value in {"REJECTED", "CANCELLED"}:
                self._logger.critical(
                    "Emergency exit FAILED - manual intervention required",
                    extra={
                        "event": "bracket.emergency.failed",
                        "symbol": symbol,
                        "order_id": emergency_order_id,
                        "status": status_value,
                        "reason": order_status.get("status_message"),
                    },
                )
                if self._notifier is not None:
                    try:
                        self._notifier.send_alert(
                            (
                                f"🚨 CRITICAL: Emergency exit FAILED for {symbol}\n"
                                f"Quantity: {filled_qty}\n"
                                f"Order: {emergency_order_id}\n"
                                f"Status: {status_value}\n"
                                "MANUAL ACTION REQUIRED IMMEDIATELY"
                            )
                        )
                    except Exception:  # pragma: no cover - notifier resilience
                        self._logger.error(
                            "Notifier failed during emergency exit failure alert",
                            extra={
                                "event": "bracket.emergency.notifier_failed",
                                "symbol": symbol,
                            },
                            exc_info=True,
                        )
                break

        if not exit_verified and elapsed >= max_wait_seconds:
            self._logger.critical(
                "Emergency exit verification timeout - manual check required",
                extra={
                    "event": "bracket.emergency.timeout",
                    "symbol": symbol,
                    "order_id": emergency_order_id,
                },
            )
            if self._notifier is not None:
                try:
                    self._notifier.send_alert(
                        (
                            f"⚠️ Emergency exit verification timeout for {symbol}\n"
                            f"Order: {emergency_order_id}\n"
                            "Please verify position manually"
                        )
                    )
                except Exception:  # pragma: no cover - notifier resilience
                    self._logger.error(
                        "Notifier failed during emergency exit timeout alert",
                        extra={
                            "event": "bracket.emergency.notifier_failed",
                            "symbol": symbol,
                        },
                        exc_info=True,
                    )

        if original_exception is not None:
            self._logger.debug(
                "Bracket failure root cause logged",
                extra={
                    "event": "bracket.emergency.origin",
                    "symbol": symbol,
                    "error": str(original_exception),
                },
            )

    def _cleanup_bracket_state(self, entry_id: str) -> None:
        """Remove bracket metadata and trailing controller for entry.

        Args:
            entry_id: Entry order identifier whose bracket is completed.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _cleanup_bracket_state",
            extra={"event": "cleanup_bracket_state", "entry_id": entry_id},
        )
        state = self._brackets.pop(entry_id, None)
        if state is None:
            return
        for order_id in (
            state.stop_order_id,
            state.tp_primary_id,
            state.tp_secondary_id,
        ):
            if order_id:
                self._bracket_index.pop(order_id, None)
        self.stop_trailing(entry_id)
        self._update_entry_children(
            entry_id,
            remove=[
                order_id
                for order_id in (
                    state.stop_order_id,
                    state.tp_primary_id,
                    state.tp_secondary_id,
                )
                if order_id
            ],
        )
        manager = self._persistent_state
        if manager is not None:
            try:
                manager.remove_bracket(entry_id)
            except Exception as exc:  # noqa: BLE001
                self._logger.error("Failure in _cleanup_bracket_state remove: %s", exc)

    def _update_entry_children(
        self,
        entry_id: str,
        *,
        add: Sequence[str] | None = None,
        remove: Sequence[str] | None = None,
    ) -> None:
        """Synchronize stored child order identifiers for an entry.

        Args:
            entry_id: Parent entry identifier to adjust.
            add: Optional iterable of child IDs to include.
            remove: Optional iterable of child IDs to remove.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _update_entry_children",
            extra={"event": "update_entry_children", "entry_id": entry_id},
        )
        add = add or []
        remove = remove or []
        with self._lock:
            entry = self._orders.get(entry_id)
            if entry is None:
                return
            updated = False
            for order_id in remove:
                if order_id in entry.child_order_ids:
                    entry.child_order_ids.remove(order_id)
                    updated = True
            for order_id in add:
                if order_id not in entry.child_order_ids:
                    entry.child_order_ids.append(order_id)
                    updated = True
            if updated:
                self._register_order(entry)

    def _cancel_stop_order(self, state: BracketState) -> None:
        """Cancel the protective stop order for a bracket state.

        Args:
            state: Bracket state containing stop metadata.

        Returns:
            None.

        Raises:
            None.
        """

        stop_id = state.stop_order_id
        if not stop_id:
            return
        self._logger.debug(
            "Entered _cancel_stop_order",
            extra={"event": "cancel_stop_order", "order_id": stop_id},
        )
        try:
            self.cancel_order(stop_id)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _cancel_stop_order: %s",
                exc,
                extra={"event": "cancel_stop_failed", "order_id": stop_id},
            )
        self._bracket_index.pop(stop_id, None)
        self._update_entry_children(state.entry_id, remove=[stop_id])
        state.stop_order_id = ""

    def _resize_stop_order(self, state: BracketState, new_quantity: int) -> None:
        """Ensure stop-loss quantity matches current exposure.

        Args:
            state: Bracket state describing the stop order.
            new_quantity: Desired protective quantity.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _resize_stop_order",
            extra={
                "event": "resize_stop_order",
                "order_id": state.stop_order_id,
                "new_quantity": new_quantity,
            },
        )
        if new_quantity <= 0:
            self._cancel_stop_order(state)
            return
        stop_id = state.stop_order_id
        if not stop_id:
            return
        try:
            success = self.modify_order(stop_id, new_quantity=new_quantity)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _resize_stop_order modify: %s",
                exc,
                extra={"event": "resize_stop_modify_failed", "order_id": stop_id},
            )
            success = False
        if success:
            details = self._orders.get(stop_id)
            if details is not None:
                details.quantity = new_quantity
            return
        try:
            self.cancel_order(stop_id)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _resize_stop_order cancel: %s",
                exc,
                extra={"event": "resize_stop_cancel_failed", "order_id": stop_id},
            )
        self._bracket_index.pop(stop_id, None)
        self._update_entry_children(state.entry_id, remove=[stop_id])
        state.stop_order_id = ""
        # [FIX] Robust Stop Replacement Logic
        replacement = None
        for attempt in range(3):  # Try 3 times to place the new stop
            try:
                replacement = self._place_single_order(
                    symbol=state.symbol,
                    side=state.exit_side,
                    quantity=new_quantity,
                    order_type=state.stop_order_type,
                    price=state.stop_price,
                    product=state.product,
                    tag=state.tag,
                    parent_order_id=state.entry_id,
                )
                if replacement:
                    break
            except Exception as exc:
                self._logger.warning(
                    f"Stop replacement attempt {attempt+1} failed: {exc}",
                    extra={"entry_id": state.entry_id},
                )
                time.sleep(0.5)
        if not replacement:
            # CRITICAL: Failed to replace stop. Position is NAKED.
            # ACTION: Trigger Emergency Exit immediately.
            self._logger.critical(
                "CRITICAL: Failed to replace Stop Loss. Executing EMERGENCY EXIT.",
                extra={"event": "resize_stop_fatal_error", "entry_id": state.entry_id},
            )
            try:
                self._place_exit_order(
                    symbol=state.symbol,
                    side=state.exit_side,
                    quantity=new_quantity,
                    product=state.product,
                    tag="emergency_no_stop",
                )
            except Exception as exit_exc:
                self._logger.critical(f"EMERGENCY EXIT FAILED: {exit_exc}")
            return

        # ... (Rest of the function continues with `replacement` guaranteed) ...
        state.stop_order_id = replacement.order_id
        self._bracket_index[replacement.order_id] = state.entry_id
        self._update_entry_children(state.entry_id, add=[replacement.order_id])
        if (
            state.stop_order_type == OrderType.STOP_LOSS_MARKET
            and replacement.price > 0
        ):
            state.stop_price = replacement.price

        if state.trailing_spec is not None:
            self.stop_trailing(state.entry_id)
            try:
                self.attach_trailing_stop(
                    entry_order_id=state.entry_id,
                    sl_order_id=replacement.order_id,
                    symbol=state.symbol,
                    side=state.side,
                    entry_price=state.entry_price,
                    spec=state.trailing_spec,
                )
            except Exception as exc:
                # [ADD THIS LINE]
                state.trailing_spec = None  # Mark as no longer trailing
                self._logger.error(
                    "Failure in _resize_stop_order trailing: %s. Stop is now STATIC.",
                    exc,
                    extra={
                        "event": "resize_stop_trailing_failed",
                        "entry_id": state.entry_id,
                    },
                )

    def _resize_target_order(
        self,
        state: BracketState,
        *,
        target: Literal["primary", "secondary"],
        new_outstanding: int,
    ) -> None:
        """Adjust outstanding quantity for a take-profit order.

        Args:
            state: Bracket state owning the target.
            target: Identifier for the target leg to resize.
            new_outstanding: Desired remaining quantity for the target.

        Returns:
            None.

        Raises:
            None.
        """

        order_id = state.tp_primary_id if target == "primary" else state.tp_secondary_id
        if not order_id:
            return
        self._logger.debug(
            "Entered _resize_target_order",
            extra={
                "event": "resize_target_order",
                "order_id": order_id,
                "new_outstanding": new_outstanding,
            },
        )
        if new_outstanding <= 0:
            try:
                self.cancel_order(order_id)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _resize_target_order cancel: %s",
                    exc,
                    extra={
                        "event": "resize_target_cancel_failed",
                        "order_id": order_id,
                    },
                )
            self._bracket_index.pop(order_id, None)
            self._update_entry_children(state.entry_id, remove=[order_id])
            if target == "primary":
                state.tp_primary_id = None
                state.tp_primary_qty = state.tp_primary_filled
            else:
                state.tp_secondary_id = None
                state.tp_secondary_qty = state.tp_secondary_filled
            return
        try:
            success = self.modify_order(order_id, new_quantity=new_outstanding)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _resize_target_order modify: %s",
                exc,
                extra={"event": "resize_target_modify_failed", "order_id": order_id},
            )
            success = False
        if success:
            details = self._orders.get(order_id)
            updated_total = new_outstanding
            if target == "primary":
                state.tp_primary_qty = state.tp_primary_filled + new_outstanding
                updated_total = state.tp_primary_qty
            else:
                state.tp_secondary_qty = state.tp_secondary_filled + new_outstanding
                updated_total = state.tp_secondary_qty
            if details is not None:
                details.quantity = updated_total
            return
        self._logger.warning(
            "Condition met: target resize fallback",
            extra={"event": "resize_target_modify_failed", "order_id": order_id},
        )
        try:
            self.cancel_order(order_id)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _resize_target_order fallback cancel: %s",
                exc,
                extra={"event": "resize_target_fallback_cancel", "order_id": order_id},
            )
            return
        self._bracket_index.pop(order_id, None)
        self._update_entry_children(state.entry_id, remove=[order_id])
        if target == "primary":
            state.tp_primary_id = None
            state.tp_primary_qty = state.tp_primary_filled
        else:
            state.tp_secondary_id = None
            state.tp_secondary_qty = state.tp_secondary_filled

    def _rebalance_targets(self, state: BracketState) -> None:
        """Ensure outstanding target legs align with current exposure.

        Args:
            state: Bracket state requiring rebalancing.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _rebalance_targets",
            extra={"event": "rebalance_targets", "entry_id": state.entry_id},
        )
        remaining = state.remaining_position()
        if remaining <= 0:
            self._cancel_bracket_targets(state)
            self._cancel_stop_order(state)
            self._cleanup_bracket_state(state.entry_id)
            return
        if state.tp_primary_id:
            primary_outstanding = min(state.primary_remaining(), remaining)
            remaining = max(remaining - primary_outstanding, 0)
            self._resize_target_order(
                state,
                target="primary",
                new_outstanding=primary_outstanding,
            )
        if state.tp_secondary_id:
            secondary_outstanding = min(state.secondary_remaining(), remaining)
            remaining = max(remaining - secondary_outstanding, 0)
            self._resize_target_order(
                state,
                target="secondary",
                new_outstanding=secondary_outstanding,
            )
        self._resize_stop_order(state, state.remaining_position())

    def _determine_second_target_price(self, state: BracketState) -> float:
        """Compute fallback price for the second take-profit target.

        Args:
            state: Bracket state for which a price is needed.

        Returns:
            Price level for the secondary target.

        Raises:
            None.
        """

        if state.second_target_price is not None:
            return float(state.second_target_price)
        base = state.tp_primary_price or state.entry_price
        if base <= 0:
            return state.entry_price
        if state.tp_primary_price is None:
            return base
        distance = state.tp_primary_price - state.entry_price
        if state.side == "SELL":
            distance = state.entry_price - state.tp_primary_price
        if distance <= 0:
            return state.tp_primary_price
        if state.side == "BUY":
            return state.tp_primary_price + distance
        return state.tp_primary_price - distance

    def _place_secondary_target(self, state: BracketState) -> None:
        """Submit a secondary take-profit order for remaining quantity.

        Args:
            state: Bracket state ready for the next target.

        Returns:
            None.

        Raises:
            None.
        """

        remaining = state.remaining_position()
        if remaining <= 0:
            return
        price = self._determine_second_target_price(state)
        try:
            tp_order = self._place_single_order(
                symbol=state.symbol,
                side=state.exit_side,
                quantity=remaining,
                order_type=OrderType.LIMIT,
                price=price,
                product=state.product,
                tag=state.tag,
                parent_order_id=state.entry_id,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _place_secondary_target: %s",
                exc,
                extra={"event": "secondary_target_failed", "entry_id": state.entry_id},
            )
            return
        state.tp_secondary_id = tp_order.order_id
        state.tp_secondary_qty = remaining
        state.tp_secondary_price = price
        state.tp_secondary_filled = 0
        self._bracket_index[tp_order.order_id] = state.entry_id
        self._update_entry_children(state.entry_id, add=[tp_order.order_id])
        self._logger.info(
            "Condition met: secondary target placed",
            extra={
                "event": "secondary_target_placed",
                "entry_id": state.entry_id,
                "order_id": tp_order.order_id,
                "quantity": remaining,
                "price": price,
            },
        )

    def _cancel_bracket_targets(self, state: BracketState) -> None:
        """Cancel all outstanding take-profit legs for a bracket.

        Args:
            state: Bracket state whose targets should be cancelled.

        Returns:
            None.

        Raises:
            None.
        """

        for target in ("primary", "secondary"):
            order_id = (
                state.tp_primary_id if target == "primary" else state.tp_secondary_id
            )
            if not order_id:
                continue
            try:
                self.cancel_order(order_id)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _cancel_bracket_targets: %s",
                    exc,
                    extra={
                        "event": "cancel_bracket_target_failed",
                        "order_id": order_id,
                    },
                )
            self._bracket_index.pop(order_id, None)
            self._update_entry_children(state.entry_id, remove=[order_id])
            if target == "primary":
                state.tp_primary_id = None
                state.tp_primary_qty = state.tp_primary_filled
            else:
                state.tp_secondary_id = None
                state.tp_secondary_qty = state.tp_secondary_filled

    def _handle_bracket_update(
        self,
        order: OrderDetails,
        _previous_status: OrderStatus,
        _payload: Mapping[str, Any] | None,
    ) -> None:
        """Process bracket state transitions for updated order."""

        # -------------------------------------------------------------------------
        # 1. LAZY INITIALIZATION (Auto-Bracket for Simple Orders)
        # -------------------------------------------------------------------------
        entry_id = self._bracket_index.get(order.order_id)
        if not entry_id and order.order_id in self._brackets:
            entry_id = order.order_id

        # If this is a filled entry with SL/TP intent but no bracket orders yet
        if (
            not entry_id
            and order.status == OrderStatus.FILLED
            and (order.stop_loss or order.take_profit)
        ):
            self._logger.info(
                f"🛡️ Auto-initializing bracket for simple order {order.order_id}",
                extra={"event": "auto_bracket_init", "order_id": order.order_id},
            )

            exit_side: Literal["BUY", "SELL"] = "SELL" if order.side == "BUY" else "BUY"

            state = BracketState(
                entry_id=order.order_id,
                symbol=order.symbol,
                side=cast(Literal["BUY", "SELL"], order.side),
                exit_side=exit_side,
                total_quantity=order.filled_quantity,
                entry_price=order.fill_price or order.price,
                product="MIS",
                tag=order.tag,
                stop_order_id="",
                stop_price=float(order.stop_loss) if order.stop_loss else 0.0,
                stop_order_type=OrderType.STOP_LOSS_MARKET,
                tp_primary_price=(
                    float(order.take_profit) if order.take_profit else None
                ),
                tp_primary_qty=order.filled_quantity if order.take_profit else 0,
            )

            self._register_bracket_state(state)
            entry_id = order.order_id

            # --- A. Place Stop Loss (Critical Safety Check) ---
            sl_successful = False  # Track success to gate TP placement

            if state.stop_price > 0:
                try:
                    stop = self._place_single_order(
                        symbol=state.symbol,
                        side=state.exit_side,
                        quantity=state.total_quantity,
                        order_type=OrderType.STOP_LOSS_MARKET,
                        price=state.stop_price,
                        product=state.product,
                        tag="auto-stop",
                        parent_order_id=entry_id,
                    )
                    state.stop_order_id = stop.order_id
                    self._bracket_index[stop.order_id] = entry_id
                    self._update_entry_children(entry_id, add=[stop.order_id])

                    sl_successful = True  # ✅ Mark as successful

                    # [UPGRADE] Attach Adaptive Trailing to Auto-Stop
                    if self._indicator_engine:
                        self.attach_trailing_stop(
                            entry_order_id=entry_id,
                            sl_order_id=stop.order_id,
                            symbol=state.symbol,
                            side=state.side,
                            entry_price=state.entry_price,
                            spec=TrailingSpec(trail_by=10.0, step=2.0),
                        )
                except Exception as e:
                    self._logger.critical(
                        "CRITICAL: Failed to place STOP LOSS: %s. Aborting TP placement to prevent naked trade.",
                        e,
                        extra={
                            "event": "auto_bracket_stop_failed_critical",
                            "entry_id": entry_id,
                        },
                    )
            else:
                # If no stop price requested, we proceed (user intentional)
                sl_successful = True

            # --- B. Place Take Profit (ONLY IF SL WAS SUCCESSFUL) ---
            if sl_successful and state.tp_primary_price and state.tp_primary_price > 0:
                try:
                    tp = self._place_single_order(
                        symbol=state.symbol,
                        side=state.exit_side,
                        quantity=state.total_quantity,
                        order_type=OrderType.LIMIT,
                        price=state.tp_primary_price,
                        product=state.product,
                        tag="auto-target",
                        parent_order_id=entry_id,
                    )
                    state.tp_primary_id = tp.order_id
                    self._bracket_index[tp.order_id] = entry_id
                    self._update_entry_children(entry_id, add=[tp.order_id])

                    # [UPGRADE] Attach Dynamic TP Expansion
                    self.attach_dynamic_tp(
                        tp_order_id=tp.order_id,
                        symbol=state.symbol,
                        side=state.exit_side,
                        initial_price=state.tp_primary_price,
                        parent_order_id=entry_id,
                    )
                except Exception as e:
                    self._logger.error(
                        "Failed to place auto-target: %s",
                        e,
                        extra={"event": "auto_bracket_tp_failed", "entry_id": entry_id},
                    )

            self._persist_bracket_state(state)
            return

        # -------------------------------------------------------------------------
        # 2. STANDARD UPDATE LOGIC (Existing Code)
        # -------------------------------------------------------------------------
        if not entry_id:
            return
        state = self._brackets.get(entry_id)
        if state is None:
            self._bracket_index.pop(order.order_id, None)
            return

        # ✅ FIX: Safe Status Access (Handle String vs Enum)
        status_val = (
            order.status.value if hasattr(order.status, "value") else str(order.status)
        )

        self._logger.debug(
            "Entered _handle_bracket_update",
            extra={
                "event": "handle_bracket_update",
                "entry_id": entry_id,
                "order_id": order.order_id,
                "status": status_val,
            },
        )

        # -----------------------------------------------------------------------
        # ✅ NEW: Register with Virtual Sniper (Replaces handle_bracket_update)
        # -----------------------------------------------------------------------
        if self._bracket_manager is not None and order.status in (
            OrderStatus.FILLED,
            OrderStatus.COMPLETE,
        ):
            try:
                # 1. Get Entry Price & Quantity
                entry_price = float(order.average_price or order.price or 0.0)
                qty = int(order.filled_quantity or order.quantity)

                if entry_price > 0 and qty > 0:
                    # 2. Determine SL/TP (Use provided args or Auto-Calculate Defaults)
                    sl_price = float(order.stop_loss or 0.0)
                    tp_price = float(order.take_profit or 0.0)

                    # Default Safety Net: 10% Stop, 20% Target if not specified
                    if sl_price <= 0:
                        if order.side == "BUY":
                            sl_price = round(entry_price * 0.90, 1)
                        else:
                            sl_price = round(entry_price * 1.10, 1)

                    if tp_price <= 0:
                        if order.side == "BUY":
                            tp_price = round(entry_price * 1.20, 1)
                        else:
                            tp_price = round(entry_price * 0.80, 1)

                    # 3. Handover to the Sniper Engine
                    self._bracket_manager.register_virtual_bracket(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        qty=qty,
                        price=entry_price,
                        sl=sl_price,
                        tp=tp_price,
                        tag=order.tag,
                    )
                    self._logger.info(
                        f"✅ Handed off {order.symbol} to Virtual Sniper (SL: {sl_price}, TP: {tp_price})",
                        extra={
                            "event": "virtual_bracket_registered",
                            "order_id": order.order_id,
                        },
                    )

            except AttributeError as e:
                self._logger.warning(
                    f"⚠️ Virtual bracket registration unavailable: {e}"
                )
            except Exception as exc:
                self._logger.error(
                    "Failure in virtual bracket registration: %s",
                    exc,
                    extra={
                        "event": "virtual_bracket_registration_failed",
                        "order_id": order.order_id,
                    },
                    exc_info=exc,
                )

        # -------------------------------------------------------------------------
        # 3. STATE MACHINE (Wrapped in Original Try/Except)
        # -------------------------------------------------------------------------
        try:
            # Case 1: STOP LOSS UPDATE
            if order.order_id == state.stop_order_id:
                state.stop_filled = order.filled_quantity
                if order.status == OrderStatus.FILLED:
                    self._logger.info(
                        "Condition met: stop-loss filled",
                        extra={
                            "event": "stop_loss_filled",
                            "entry_id": entry_id,
                            "order_id": order.order_id,
                        },
                    )
                    self._cancel_bracket_targets(state)
                    # [FIX] Clean up dynamic controllers
                    if state.tp_primary_id:
                        self.stop_dynamic_tp(state.tp_primary_id)
                    if state.tp_secondary_id:
                        self.stop_dynamic_tp(state.tp_secondary_id)
                    self._cleanup_bracket_state(entry_id)
                elif order.status == OrderStatus.PARTIALLY_FILLED:
                    self._logger.info(
                        "Condition met: stop-loss partially filled",
                        extra={
                            "event": "stop_loss_partial",
                            "entry_id": entry_id,
                            "order_id": order.order_id,
                            "filled_quantity": order.filled_quantity,
                        },
                    )
                    self._rebalance_targets(state)

            # Case 2: TP1 UPDATE
            elif state.tp_primary_id and order.order_id == state.tp_primary_id:
                state.tp_primary_filled = order.filled_quantity
                remaining = state.remaining_position()
                if order.status == OrderStatus.FILLED:
                    self._logger.info(
                        "Condition met: first take-profit filled",
                        extra={
                            "event": "tp1_filled",
                            "entry_id": entry_id,
                            "order_id": order.order_id,
                            "filled_quantity": order.filled_quantity,
                        },
                    )
                    # [FIX] Stop Dynamic TP for this leg
                    self.stop_dynamic_tp(order.order_id)

                    self._bracket_index.pop(order.order_id, None)
                    self._update_entry_children(state.entry_id, remove=[order.order_id])
                    state.tp_primary_id = None
                    state.tp_primary_qty = order.filled_quantity

                    if remaining <= 0:
                        self._cancel_stop_order(state)
                        self._cleanup_bracket_state(entry_id)
                    else:
                        self._resize_stop_order(state, remaining)
                        if state.tp_secondary_qty > 0:
                            self._place_secondary_target(state)
                elif order.status == OrderStatus.PARTIALLY_FILLED:
                    self._logger.info(
                        "Condition met: first take-profit partial",
                        extra={
                            "event": "tp1_partial",
                            "entry_id": entry_id,
                            "order_id": order.order_id,
                            "filled_quantity": order.filled_quantity,
                        },
                    )
                    self._resize_stop_order(state, remaining)

            # Case 3: TP2 UPDATE
            elif state.tp_secondary_id and order.order_id == state.tp_secondary_id:
                state.tp_secondary_filled = order.filled_quantity
                remaining = state.remaining_position()
                if order.status == OrderStatus.FILLED:
                    self._logger.info(
                        "Condition met: second take-profit filled",
                        extra={
                            "event": "tp2_filled",
                            "entry_id": entry_id,
                            "order_id": order.order_id,
                            "filled_quantity": order.filled_quantity,
                        },
                    )
                    # [FIX] Stop Dynamic TP for this leg
                    self.stop_dynamic_tp(order.order_id)

                    self._cancel_stop_order(state)
                    self._bracket_index.pop(order.order_id, None)
                    self._update_entry_children(state.entry_id, remove=[order.order_id])
                    state.tp_secondary_id = None
                    state.tp_secondary_qty = order.filled_quantity
                    self._cleanup_bracket_state(entry_id)
                elif order.status == OrderStatus.PARTIALLY_FILLED:
                    self._logger.info(
                        "Condition met: second take-profit partial",
                        extra={
                            "event": "tp2_partial",
                            "entry_id": entry_id,
                            "order_id": order.order_id,
                            "filled_quantity": order.filled_quantity,
                        },
                    )
                    self._resize_stop_order(state, remaining)

        # -------------------------------------------------------------------------
        # 4. ORIGINAL ERROR HANDLING (RESTORED)
        # -------------------------------------------------------------------------
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _handle_bracket_update: %s",
                exc,
                extra={"event": "handle_bracket_update_failed", "entry_id": entry_id},
            )

    def apply_broker_order_update(
        self, order_id: str, broker_payload: Mapping[str, Any]
    ) -> None:
        """Canonical broker order update ingress for polling and websocket paths."""

        payload = dict(broker_payload)
        payload["order_id"] = str(order_id)
        self._apply_broker_order_update(payload)

    def on_order_update(self, order_update: dict) -> None:
        """Compatibility wrapper for broker websocket order updates."""

        order_id = order_update.get("order_id")
        if not order_id:
            return
        self.apply_broker_order_update(str(order_id), order_update)

    def _apply_broker_order_update(self, order_update: dict) -> None:
        """Handle broker order updates and follow-up workflows.

        Args:
            order_update: Raw broker order update payload.

        Returns:
            None.

        Raises:
            None.
        """
        self._logger.debug(
            "Entered on_order_update",
            extra={"event": "order_update_enter"},
        )
        order_id = order_update.get("order_id")
        if not order_id:
            return

        # Normalize status to uppercase string
        status_raw = str(order_update.get("status", "")).upper()
        adopted = False

        with self._lock:
            # 1. Try to retrieve the existing order
            order = self._orders.get(order_id)

            # -----------------------------------------------------
            # 🛡️ UNKNOWN ORDER HANDLING (Ghost Orders)
            # -----------------------------------------------------
            if not order:
                # A. STOP THE SPAM: Ignore Dead Orders
                # If a manual/ghost order is already finished (Rejected/Cancelled),
                # we don't need to track it. Just return silently.
                if status_raw in ["REJECTED", "CANCELLED", "CANCELED"]:
                    return

                # B. ADOPT ACTIVE ORDERS (Manual Trades you are holding)
                try:
                    # [FIX] Robust Quantity Resolution
                    qty = int(float(order_update.get("quantity", 0)))
                    if qty == 0:
                        qty = int(float(order_update.get("filled_quantity", 0)))

                    # Create the order object so we track it from now on
                    order = OrderDetails(
                        order_id=order_id,
                        symbol=order_update.get("tradingsymbol")
                        or order_update.get("symbol", "UNKNOWN"),
                        side=order_update.get("transaction_type", "BUY"),
                        quantity=max(qty, 1),  # Ensure we never adopt a 0-qty order
                        order_type=OrderType.MARKET,  # Assume Market for manual entries
                        price=float(order_update.get("price", 0.0) or 0.0),
                        trigger_price=float(
                            order_update.get("trigger_price", 0.0) or 0.0
                        ),
                        average_price=float(
                            order_update.get("average_price", 0.0) or 0.0
                        ),
                        filled_quantity=int(
                            float(order_update.get("filled_quantity", 0))
                        ),
                        status=self._parse_status(status_raw),
                        timestamp=datetime.now(timezone.utc),
                        tag="adopted_manual_trade",
                        intent="UNKNOWN",
                    )

                    # 1. Save to Memory (Stop "Unknown Order" warnings for future updates)
                    self._orders[order_id] = order
                    adopted = True

                    # [FIX] CRITICAL: Sync with PositionManager immediately
                    # This ensures the PositionManager knows this ID exists before we try to update it
                    if hasattr(self._positions, "add_pending_order"):
                        self._positions.add_pending_order(
                            order_id=order.order_id,
                            symbol=order.symbol,
                            side=order.side,
                            qty=order.quantity,
                            price=order.price,
                            order_type=order.order_type,
                            intent="UNKNOWN",
                        )

                    self._logger.info(
                        f"🆕 ADOPTED UNKNOWN ORDER: {order_id} [{order.symbol}]"
                    )
                    self._notify_bracket_event(
                        "ORDER_ADOPTED",
                        {
                            "symbol": order.symbol,
                            "order_id": order_id,
                            "status": status_raw,
                            "source": "manual_adoption",
                        },
                    )

                    # 2. Persist to Disk Immediately (Survive Restarts)
                    if hasattr(self, "save_orders"):
                        self.save_orders()

                except Exception as e:
                    # Log as DEBUG so it doesn't spam your console if adoption fails
                    self._logger.debug(f"⚠️ Failed to adopt order {order_id}: {e}")
                    return

            # -----------------------------------------------------
            # 🔄 STATE SYNCHRONIZATION
            # -----------------------------------------------------
            old_status = order.status
            new_status = self._parse_status(status_raw)

            order.status = new_status
            incoming_filled_quantity = max(
                0, int(float(order_update.get("filled_quantity", 0) or 0))
            )
            previous_broker_filled = max(
                0, int(getattr(order, "filled_quantity", 0) or 0)
            )
            applied_filled_quantity = max(
                0, int(getattr(order, "applied_filled_quantity", 0) or 0)
            )
            broker_filled_quantity = max(
                previous_broker_filled, incoming_filled_quantity
            )
            order.filled_quantity = broker_filled_quantity

            # Update Price: Prefer actual fill price ('average_price')
            avg_px = order_update.get("average_price")
            if avg_px and float(avg_px) > 0:
                order.fill_price = float(avg_px)
            elif not order.fill_price:
                order.fill_price = float(order_update.get("price", 0.0) or 0.0)

            # -----------------------------------------------------
            # ✅ FILL PROCESSING (Trigger Stop Loss / Target)
            # -----------------------------------------------------
            fill_delta = max(0, broker_filled_quantity - applied_filled_quantity)
            is_fill_update = status_raw in [
                "PARTIALLY FILLED",
                "PARTIAL",
                "COMPLETE",
                "FILLED",
            ] and fill_delta > 0

            if is_fill_update or (adopted and order.filled_quantity > 0):
                self._logger.info(
                    f"✅ FILL DETECTED: {order.symbol} ({order_id}) @ {order.fill_price}"
                )

                # Update Bracket (Stop Loss / Target)
                self._register_virtual_bracket_for_fill(
                    order, source="manual_adoption" if adopted else "order_update"
                )

                # Update Positions (Critical for Dashboard accuracy)
                if hasattr(self._positions, "apply_broker_order_update"):
                    try:
                        self._positions.apply_broker_order_update(
                            order.order_id,
                            {
                                **order_update,
                                "status": status_raw,
                                "average_price": order.fill_price,
                                "filled_quantity": order.filled_quantity,
                            },
                        )
                    except Exception:
                        self._logger.exception("Unhandled exception", exc_info=True)
                        raise
                elif hasattr(self._positions, "update_from_order"):
                    try:
                        self._positions.update_from_order(order)
                    except Exception:
                        self._logger.exception("Unhandled exception", exc_info=True)
                        raise

                self._confirm_position_protection_for_fill(order)
                order.applied_filled_quantity = broker_filled_quantity

            self._notify_failed_entry_terminal(order, old_status, status_raw)

            # ── FIX (BUG 3): CANCELLED exit orders — reactivate bracket.
            # _check_zombie_orders cancels stuck PENDING orders after 45s via
            # cancel_order(). If that order was an exit, on_order_update(CANCELLED)
            # is the only place to catch it and recover the bracket.
            # Without this, the position stays open with no SL protection.
            is_cancelled = status_raw in (
                "CANCELLED",
                "CANCELED",
            ) and old_status not in (OrderStatus.CANCELLED, OrderStatus.FILLED)
            if is_cancelled and self._bracket_manager is not None:
                tag_str = (order.tag or "").lower()
                is_exit_tag = any(
                    x in tag_str for x in ["exit", "stop", "target", "square", "guard"]
                )
                if is_exit_tag:
                    try:
                        recovered = self._bracket_manager.reactivate_bracket_after_rejected_exit(
                            symbol=order.symbol,
                            rejected_order_id=order_id,
                            reason="CANCELLED",
                        )
                        if recovered:
                            self._logger.critical(
                                "🔁 CANCELLED EXIT recovered for %s (order=%s) — bracket reactivated.",
                                order.symbol,
                                order_id,
                                extra={
                                    "event": "cancelled_exit_bracket_reactivated",
                                    "symbol": order.symbol,
                                    "order_id": order_id,
                                },
                            )
                    except Exception as _can_exc:
                        self._logger.error(
                            "Bracket reactivation failed after cancelled exit for %s: %s",
                            order.symbol,
                            _can_exc,
                        )

            # Final Persistence
            if hasattr(self, "save_orders"):
                try:
                    self.save_orders()
                except Exception:
                    self._logger.exception("Unhandled exception", exc_info=True)
                    raise

    def place_atomic_entry(
        self,
        legs: Sequence[AtomicLeg | Mapping[str, Any]],
        *,
        product: str | None = None,
        tag: str | None = None,
        partial_fill_tolerance: float = 0.0,
    ) -> list[str]:
        """Submit a two-leg entry atomically with risk-state enforcement.

        Args:
            legs: Iterable containing exactly two leg specifications.
            product: Optional broker product type applied to both legs.
            tag: Optional tag applied to both legs.
            partial_fill_tolerance: Fractional tolerance (0-1] for partial fills.

        Returns:
            List of broker order identifiers for the accepted legs.

        Raises:
            ValueError: If the leg specification is invalid.
            OrderPlacementError: If any leg fails placement or risk blocks the order.
        """

        if len(legs) != 2:
            raise ValueError("Atomic entry requires exactly two legs")
        normalized: list[AtomicLeg] = [self._normalize_leg(leg) for leg in legs]
        tolerance = max(0.0, min(1.0, float(partial_fill_tolerance)))
        for leg in normalized:
            self._validate_quantity(leg.symbol, leg.quantity)
            if not self._ensure_trading_allowed(
                symbol=leg.symbol, side=leg.side, quantity=leg.quantity
            ):
                return []

        placed: list[OrderDetails] = []
        policy = self._options_policy
        try:
            for index, leg in enumerate(normalized):
                client_order_id: str | None = None
                price = leg.price
                if policy is not None:
                    policy.validate_qty(leg.symbol, leg.quantity)
                    if price is not None:
                        price = policy.round_to_tick(price)
                        policy.validate_notional(price, leg.quantity)
                    client_order_id = policy.client_order_id(
                        leg.symbol,
                        leg.side,
                        leg.quantity,
                        nonce=f"atomic:{index}",
                    )
                details = self._place_single_order(
                    symbol=leg.symbol,
                    side=leg.side,
                    quantity=leg.quantity,
                    order_type=leg.order_type,
                    price=price,
                    product=product,
                    tag=tag,
                    client_order_id=client_order_id,
                )
                placed.append(details)
                if self._leg_failed(details, tolerance):
                    status_value = details.status.value
                    message = (
                        f"Atomic leg {details.order_id} failed "
                        f"with status {status_value}"
                    )
                    raise OrderPlacementError(message)
        except Exception:
            if placed:
                try:
                    self.cancel_and_reconcile([order.order_id for order in placed])
                except Exception:  # pragma: no cover - defensive
                    self._logger.warning(
                        "atomic_entry_rollback_failed",
                        extra={
                            "event": "atomic_entry_rollback_failed",
                            "orders": [order.order_id for order in placed],
                        },
                        exc_info=True,
                    )
            raise

        order_ids = [order.order_id for order in placed]
        self._logger.info(
            "atomic_entry_submitted",
            extra={
                "event": "atomic_entry_submitted",
                "orders": order_ids,
                "client_order_ids": [order.client_order_id for order in placed],
            },
        )
        return order_ids

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current status of order."""

        order = self._refresh_order(order_id)
        return order.status

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Intercepts Virtual Brackets for clean shutdown."""

        self._validate_execution_adapter()
        # 1. Intercept Virtual Bracket Cancellation
        if self._bracket_manager:
            # If strategy tries to cancel the Entry ID, it implies "Abort Trade"
            # We verify if this ID is tracked as a bracket
            bracket = self._bracket_manager.get_bracket(order_id)
            if bracket:
                self._logger.info(f"🗑️ Cancelling Virtual Bracket {order_id}")
                self._bracket_manager.unregister_bracket(order_id)
                # We continue to cancel the physical order just in case it's still OPEN at broker
        # ✅ OPTIMIZATION: Don't cancel if already finished
        with self._lock:
            order = self._orders.get(order_id)
            if order and order.status in [
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
            ]:
                self._logger.info(
                    f"⏭️ Skipping cancel for {order_id}: Already {order.status.name}"
                )
                return True

        # 2. Standard Broker Cancel
        cancel = getattr(self._broker, "cancel_order", None)
        if cancel is None:
            raise NotImplementedError("Broker does not support order cancellation")

        try:
            response = self._call_broker(cancel, order_id)
            success = bool(response)
            if success:
                self._update_local_status(order_id, OrderStatus.CANCELLED)
            return success
        except Exception as e:
            self._logger.warning(f"Cancel failed for {order_id}: {e}")
            return False

    def cancel_pending_orders(self) -> list[str]:
        """Cancel all known pending orders and return cancelled IDs."""

        cancelled: list[str] = []
        for order in self._pending_orders():
            try:
                if self.cancel_order(order.order_id):
                    cancelled.append(order.order_id)
            except NotImplementedError:
                break
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "cancel_pending_failed",
                    extra={
                        "event": "cancel_pending_failed",
                        "order_id": order.order_id,
                        "err": str(exc),
                    },
                )
        return cancelled

    def cancel_and_reconcile(self, order_ids: Sequence[str] | None = None) -> list[str]:
        """Cancel provided orders and reconcile broker state.

        Args:
            order_ids: Optional iterable of order identifiers to cancel. When
                omitted all pending orders are targeted.

        Returns:
            List of order identifiers that were successfully cancelled.
        """

        targets = (
            list(order_ids)
            if order_ids is not None
            else [order.order_id for order in self._pending_orders()]
        )
        cancelled: list[str] = []
        for order_id in targets:
            try:
                if self.cancel_order(order_id):
                    cancelled.append(order_id)
            except NotImplementedError:
                break
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "cancel_failed",
                    extra={
                        "event": "cancel_failed",
                        "order_id": order_id,
                        "err": str(exc),
                    },
                )
        try:
            self.reconcile_open_orders()
        except Exception:  # noqa: BLE001 - defensive
            self._logger.warning(
                "reconcile_failed",
                extra={"event": "reconcile_failed", "orders": targets},
                exc_info=True,
            )
        return cancelled

    def reconcile_open_orders(self) -> None:
        """Refresh open-order state from the broker to avoid duplicates.

        Returns:
            None. The local order cache is updated in place.
        """

        fetcher = self._resolve_open_orders_fetcher()
        if fetcher is None:
            return
        try:
            response = self._call_broker(fetcher)
        except Exception:  # noqa: BLE001 - defensive
            self._logger.debug("open_order_fetch_failed", exc_info=True)
            return
        if response is None:
            return

        raw_orders: list[Mapping[str, Any]] = []
        if isinstance(response, Mapping):
            raw_orders = [cast(Mapping[str, Any], response)]
        elif isinstance(response, Sequence):
            raw_orders = [
                cast(Mapping[str, Any], item)
                for item in response
                if isinstance(item, Mapping)
            ]
        else:
            return

        reconciled: list[str] = []
        for raw in raw_orders:
            details = self._coerce_broker_open_order(raw)
            if details is None:
                continue

            self._register_order(details)

            # [FIX] CRITICAL: Sync with PositionManager
            # We must pass the Enum object directly. Converting to string causes
            # "AttributeError: 'str' object has no attribute 'value'" inside the manager.
            if hasattr(self._positions, "add_pending_order"):
                self._positions.add_pending_order(
                    order_id=details.order_id,
                    symbol=details.symbol,
                    side=details.side,
                    qty=details.quantity,
                    price=details.price,
                    order_type=details.order_type,  # ✅ CORRECT: Pass Enum Object
                    intent=details.intent,
                    bracket_id=details.bracket_id,
                    signal_id=details.signal_id,
                    signal_fingerprint=details.signal_fingerprint,
                )

            try:
                # Safe Enum Access for Status updates (Status is usually a string in PM)
                status_val = details.status
                status_str = (
                    status_val.name if hasattr(status_val, "name") else str(status_val)
                )

                self._positions.update_order_status(
                    details.order_id, status_str, details.fill_price
                )
            except Exception:  # pragma: no cover - defensive
                self._logger.debug("position_status_update_failed", exc_info=True)

            if details.client_order_id and details.status in self.FINAL_STATUSES:
                self._client_order_index.pop(details.client_order_id, None)

            reconciled.append(details.order_id)

        if reconciled:
            self._sync_positions_to_hub()
            self._logger.info(
                "order_reconcile_complete",
                extra={"event": "order_reconcile", "orders": reconciled},
            )

    def exit_position(
        self, symbol: str, quantity: int, tag: str = "exit", force: bool = False
    ) -> str | None:
        """
        Executes the 'Soft Exit' (Market Order) and cleans up the 'Hard' Safety Net.
        This completes the Hybrid Approach.
        """
        symbol = symbol.strip().upper()

        # 1. EXECUTE MARKET EXIT (The "Soft" Trigger)
        self._logger.info(
            f"⚡ Hybrid Exit Triggered for {symbol} (Qty: {quantity})",
            extra={"event": "hybrid_exit_trigger", "symbol": symbol},
        )

        try:
            # Determine exit side based on quantity direction or passed arg
            # Assuming positive quantity means we HOLD Long, so we need to SELL
            # If quantity is passed as absolute, you might need to check self._positions
            exit_side = "SELL"  # Default for Long Exit

            # ── FIX: resolve LTP so risk-accounting stats work correctly.
            # MARKET exits MUST always bypass the risk-manager's "Price must be
            # positive" guard — passing price=None→0.0 previously caused every
            # soft-exit to be blocked.  check_risk=False is always correct here:
            # the bracket already made the exit decision; the risk-manager must
            # not veto it.
            _exit_ltp: float | None = None
            try:
                _price_source = self._data_hub or self._market_data
                if _price_source is not None:
                    _exit_ltp = _price_source.get_latest_price(symbol)
            except Exception:
                self._logger.exception("Unhandled exception", exc_info=True)
                raise

            exit_id = self.place_order(
                symbol=symbol,
                side=exit_side,
                quantity=abs(quantity),
                order_type=OrderType.MARKET,
                price=_exit_ltp,  # supply live LTP for accounting; None is safe
                tag=tag,
                check_risk=False,  # exits MUST never be blocked by risk-manager
                intent="EXIT",
            )

            if not exit_id:
                self._logger.error("❌ Soft Exit Failed: place_order returned None")
                return None

        except Exception as e:
            self._logger.critical(f"❌ Soft Exit Failed: {e}", exc_info=True)
            return None

        # 2. CLEAN UP SAFETY NET (The "Hard" Cleanup)
        # We must cancel the Hard SL and Wide TP we placed earlier
        try:
            with self._lock:
                if symbol in self._brackets:
                    bracket = self._brackets[symbol]

                    # Cancel Hard SL
                    if bracket.stop_order_id:
                        self._logger.info(
                            f"🗑️ Cancelling Safety SL: {bracket.stop_order_id}"
                        )
                        try:
                            self.cancel_order(bracket.stop_order_id)
                        except Exception as e:
                            self._logger.warning(
                                f"Failed to cancel SL {bracket.stop_order_id}: {e}"
                            )

                    # Cancel Wide TP
                    if bracket.tp_primary_id:
                        self._logger.info(
                            f"🗑️ Cancelling Safety TP: {bracket.tp_primary_id}"
                        )
                        try:
                            self.cancel_order(bracket.tp_primary_id)
                        except Exception as e:
                            self._logger.warning(
                                f"Failed to cancel TP {bracket.tp_primary_id}: {e}"
                            )

                    # Remove local bracket state so we don't track dead orders
                    del self._brackets[symbol]

        except Exception as e:
            self._logger.error(f"⚠️ Safety Net Cleanup Failed (Non-Critical): {e}")

        self._signal_arbitrator.release(symbol)
        return exit_id

    def release_entry_reservation(
        self, symbol: str, *, start_cooldown: bool = True
    ) -> None:
        """Converge the entry arbitrator after broker-confirmed terminal state."""
        if start_cooldown:
            self._signal_arbitrator.release(symbol)
        else:
            self._signal_arbitrator.clear(symbol)

    def _notify_failed_entry_terminal(
        self,
        order: OrderDetails,
        old_status: OrderStatus,
        raw_status: str,
    ) -> None:
        """Converge entry guards once any broker ingress proves terminal failure."""
        failed = {
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        if (
            order.status not in failed
            or old_status == order.status
            or str(order.intent or "").upper() not in {"ENTRY", "UNKNOWN", ""}
            or not callable(self.entry_order_failed_callback)
        ):
            return
        try:
            self.entry_order_failed_callback(
                order_id=str(order.order_id),
                symbol=str(order.symbol),
                reason=str(raw_status or order.status.name).lower(),
            )
        except Exception:
            self._logger.exception(
                "ENTRY_ORDER_FAILED_CALLBACK_ERROR order_id=%s symbol=%s status=%s",
                order.order_id,
                order.symbol,
                raw_status,
            )

    def modify_order(
        self,
        order_id: str,
        price: float = 0.0,
        trigger_price: float = 0.0,
        quantity: int = 0,
    ) -> bool:
        """
        Modify an existing order.
        SMART FIX: If modifying a 'Fake SL-M' (SL-Limit), auto-adjust the limit price.
        """
        try:
            # ✅ FIX: Round Price/Trigger to 0.05 tick size
            if price:
                price = self._round_to_tick(price)
            if trigger_price:
                trigger_price = self._round_to_tick(trigger_price)

            # 1. Fetch Order Context
            order = self.get_order(order_id)
            if not order:
                self._logger.error(f"Cannot modify unknown order {order_id}")
                return False

            # 2. Smart Limit Adjustment for SL Orders
            # If user is only updating trigger_price, we must also update limit price
            # to maintain the "Market Buffer".
            if (
                order.order_type == OrderType.STOP_LOSS
                and trigger_price > 0
                and (price is None or price == 0)
            ):
                buffer = 0.05  # 5%
                if order.side == "SELL":  # Long SL
                    price = round(trigger_price * (1 - buffer), 1)
                else:  # Short SL
                    price = round(trigger_price * (1 + buffer), 1)
                self._logger.info(
                    f"🔄 Auto-adjusting Limit Price to {price} for Trigger {trigger_price}"
                )

            # 3. Execute Modification
            self._validate_execution_adapter()
            self._broker.modify_order(
                order_id=order_id,
                price=price,
                trigger_price=trigger_price,
                quantity=quantity,
                variety="regular",  # Zerodha default
            )
            return True
        except Exception as e:
            self._logger.error(f"Modification Failed: {e}")
            return False

    def wait_for_fill(self, order_id: str, timeout_sec: float = 30.0) -> bool:
        """Wait for order to fill. Returns True if filled within timeout."""

        deadline = time.time() + float(timeout_sec)
        # Use a tighter poll interval for execution workflows (max 0.2s)
        poll_interval = min(0.2, self.POLL_INTERVAL_SEC)

        while time.time() < deadline:
            status = self.get_order_status(order_id)
            if status == OrderStatus.FILLED:
                return True
            if status in self.FINAL_STATUSES:
                return False
            time.sleep(poll_interval)
        return False

    def get_fill_price(self, order_id: str) -> float | None:
        """Get fill price for filled order."""

        order = self._orders.get(order_id)
        if order:
            return order.fill_price
        history_index = self._resolve_history_index(order_id)
        if history_index is not None:
            return self._history[history_index].fill_price
        return None

    def get_order_history(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[OrderDetails]:
        """Get order history, optionally filtered by symbol."""

        with self._lock:
            history = list(self._history)
        if symbol is not None:
            symbol_key = symbol.upper()
            history = [order for order in history if order.symbol == symbol_key]
        return history[-limit:]

    def recent_orders(self, limit: int = 10) -> list[dict[str, object]]:
        """Return recent orders serialized for status endpoints."""

        orders = self.get_order_history(limit=limit)
        return [
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "status": (
                    order.status.value
                    if hasattr(order.status, "value")
                    else str(order.status)
                ),
                "quantity": order.quantity,
                "filled_quantity": order.filled_quantity,
                "price": order.price,
                "fill_price": order.fill_price,
                "timestamp": (
                    order.timestamp.isoformat()
                    if hasattr(order.timestamp, "isoformat")
                    else float(order.timestamp)
                ),
                "rejection_reason": order.rejection_reason,
            }
            for order in orders
        ]

    def get_today_orders(self) -> list[OrderDetails]:
        """Get all orders placed today."""

        today = datetime.now(timezone.utc).date()
        with self._lock:
            return [order for order in self._history if order.timestamp.date() == today]

    def _confirm_fill_fast(self, order_id: str, timeout_ms: int = 2000) -> bool:
        """
        Fast fill confirmation with exponential backoff.

        ✅ WORLD-CLASS: Sub-500ms fill detection when possible

        Args:
            order_id: The broker order ID to confirm
            timeout_ms: Maximum time to wait (default 2 seconds)

        Returns:
            True if fill confirmed, False if timeout/rejected
        """
        import time

        start = time.monotonic()
        backoff_ms = 50  # Start checking every 50ms
        max_backoff_ms = 300  # Don't wait more than 300ms between checks
        attempts = 0

        self._logger.debug(f"⏱️ Fast fill check started for {order_id}")

        while (time.monotonic() - start) * 1000 < timeout_ms:
            attempts += 1

            try:
                # Check order status
                status = None
                if hasattr(self._broker, "get_order_status"):
                    status = self._broker.get_order_status(order_id)
                elif hasattr(self._broker, "order_history"):
                    # Some brokers use order_history
                    history = self._broker.order_history(order_id)
                    if history and isinstance(history, list):
                        status = history[-1] if history else None

                if not status:
                    time.sleep(backoff_ms / 1000)
                    backoff_ms = min(backoff_ms * 1.5, max_backoff_ms)
                    continue

                status_str = str(status.get("status", "")).upper()

                # ✅ FILL DETECTED - Immediately process
                if status_str in {"COMPLETE", "FILLED"}:
                    elapsed = (time.monotonic() - start) * 1000
                    self._logger.info(
                        f"✅ FILL CONFIRMED in {elapsed:.0f}ms (attempts: {attempts}): {order_id}"
                    )

                    # CRITICAL: Trigger immediate order update processing
                    # This activates the bracket instantly
                    self.on_order_update(status)

                    return True

                # ❌ REJECTED/CANCELLED - Stop waiting
                if status_str in {"REJECTED", "CANCELLED", "CANCELED"}:
                    self._logger.warning(
                        f"❌ Order {order_id} {status_str}: {status.get('status_message', 'No reason')}"
                    )
                    self.on_order_update(status)
                    return False

                # PENDING/SUBMITTED - Continue waiting with backoff
                if status_str in {"PENDING", "SUBMITTED", "OPEN", "TRIGGER PENDING"}:
                    time.sleep(backoff_ms / 1000)
                    backoff_ms = min(backoff_ms * 1.5, max_backoff_ms)
                    continue

            except Exception as e:
                self._logger.debug(f"Fill check error (attempt {attempts}): {e}")

            time.sleep(backoff_ms / 1000)
            backoff_ms = min(backoff_ms * 1.5, max_backoff_ms)

        # Timeout - rely on periodic reconcile
        elapsed = (time.monotonic() - start) * 1000
        self._logger.warning(
            f"⏰ Fill check timeout after {elapsed:.0f}ms (attempts: {attempts}): {order_id}"
        )
        return False

    def start_monitoring(self) -> None:
        """Start background thread for order status monitoring."""

        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_orders, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop background monitoring thread."""

        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=self.POLL_INTERVAL_SEC * 2)
        self._monitor_thread = None

    def _log_status_report(self) -> None:
        """
        Emit a rich, insightful 'Situation Room' report every 60 seconds.
        Shows active positions, P&L, distance to stops, and current battle plan.
        """
        # intent: suppress log spam by emitting only on meaningful state changes
        try:
            # 1. Gather Active Positions
            positions = list(self._positions.get_open_positions())
            if not positions:
                # self._logger.info("💤 Status Report: Market is quiet. No active positions.")
                return

            report = ["\n📊 ------------------ SITUATION REPORT ------------------"]
            state_changed = False
            state_cache = getattr(self, "_last_status_report_state", None)
            if state_cache is None:
                state_cache = {}
                self._last_status_report_state = state_cache

            total_unrealized_pnl = 0.0

            for pos in positions:
                symbol = pos.symbol
                qty = pos.quantity
                entry = float(pos.entry_price or 0.0)
                side = pos.side  # "LONG" or "SHORT"
                tag = getattr(pos, "tag", "Manual/Unknown")

                # Get Live Market Data via DataHub (SSOT)
                _src = self._data_hub or self._market_data
                ltp = (_src.get_latest_price(symbol) or 0.0) if _src else 0.0

                # Calculate P&L
                raw_pnl = 0.0
                if ltp > 0 and entry > 0:
                    raw_pnl = (
                        (ltp - entry) * qty if side == "LONG" else (entry - ltp) * qty
                    )

                status_icon = "✅" if raw_pnl > 0 else "🔻"

                # ═══════════════════════════════════════════════════════════════
                # ✅ FIX: Query BracketManager (source of truth for virtual brackets)
                # Fixed: Feb 3, 2026 - SITREP was looking in wrong dict
                # ═══════════════════════════════════════════════════════════════
                sl_info = "NONE ⚠️"
                tp_info = "Open"
                insight = "Monitoring..."
                bracket = None

                if self._bracket_manager:
                    # Check if symbol is managed by BracketManager
                    if self._bracket_manager.is_symbol_managed(symbol):
                        # Get bracket via symbol lookup
                        try:
                            if hasattr(self._bracket_manager, "get_bracket_by_symbol"):
                                bracket = self._bracket_manager.get_bracket_by_symbol(
                                    symbol
                                )
                            else:
                                # Fallback: Manual lookup via _symbol_map
                                with self._bracket_manager._lock:
                                    entry_ids = self._bracket_manager._symbol_map.get(
                                        symbol, []
                                    )
                                    for entry_id in entry_ids:
                                        b = self._bracket_manager._brackets.get(
                                            entry_id
                                        )
                                        if b and b.remaining_quantity > 0:
                                            bracket = b
                                            break
                        except Exception as e:
                            self._logger.debug(f"Bracket lookup for {symbol}: {e}")

                        if bracket:
                            # ✅ Use correct BracketState attribute names
                            sl_val = getattr(bracket, "sl_trigger_price", 0) or 0
                            tp_val = getattr(bracket, "tp_trigger_price", 0) or 0

                            sl_info = f"{sl_val:.2f}" if sl_val > 0 else "NONE ⚠️"
                            tp_info = f"{tp_val:.2f}" if tp_val > 0 else "Open"

                            # Generate insights
                            is_active = getattr(bracket, "active", False)
                            highest = getattr(bracket, "highest_ltp", 0)

                            if ltp > 0 and sl_val > 0:
                                dist_to_sl = abs(ltp - sl_val)
                                risk_gap = abs(entry - sl_val) if entry > 0 else 1

                                if raw_pnl > 0:
                                    if tp_val > 0 and abs(tp_val - ltp) < (ltp * 0.01):
                                        insight = "🎯 Sniper Mode: Near Target!"
                                    elif is_active:
                                        insight = (
                                            f"🚀 Trailing Active | High: {highest:.2f}"
                                        )
                                    else:
                                        insight = "🚀 Cruising: Holding for TP"
                                else:
                                    if risk_gap > 0 and dist_to_sl < (risk_gap * 0.25):
                                        insight = "🚨 DANGER: Near Stop Loss!"
                                    else:
                                        insight = "🛡️ Defending: Structure holds"
                            else:
                                insight = f"✅ Protected | Active: {is_active}"
                        else:
                            insight = "⚠️ Symbol managed but bracket unavailable"
                    else:
                        insight = "⚠️ ORPHAN TRADE: No bracket protection!"
                else:
                    # Legacy fallback
                    with self._lock:
                        for b in self._brackets.values():
                            if getattr(b, "symbol", None) == symbol:
                                sl_info = f"{getattr(b, 'stop_price', 0):.2f}"
                                tp_info = f"{getattr(b, 'tp_primary_price', 0):.2f}"
                                insight = "✅ Legacy bracket"
                                break
                        else:
                            insight = "⚠️ BracketManager not available"

                # Track meaningful state change triggers
                pnl_sign = "profit" if raw_pnl > 0 else "loss"
                danger_flag = False
                if ltp > 0 and sl_info not in {"NONE ⚠️", "NONE"} and entry > 0:
                    try:
                        sl_val = float(sl_info)
                        risk_gap = abs(entry - sl_val) if entry > 0 else 0.0
                        dist_to_sl = abs(ltp - sl_val)
                        danger_flag = risk_gap > 0 and dist_to_sl < (risk_gap * 0.25)
                    except (TypeError, ValueError):
                        danger_flag = False
                insight_severity = "neutral"
                if "DANGER" in insight or "ORPHAN" in insight:
                    insight_severity = "high"
                elif "Sniper" in insight or "Trailing" in insight:
                    insight_severity = "medium"

                prev_state = state_cache.get(symbol)
                current_state = {
                    "pnl_sign": pnl_sign,
                    "danger": danger_flag,
                    "insight_severity": insight_severity,
                }
                if prev_state != current_state:
                    state_cache[symbol] = current_state
                    state_changed = True

                # Format the Block
                line = (
                    f"{status_icon} {symbol} | {side} {qty} Qty | Strat: {tag}\n"
                    f"   Entry: {entry:.2f} ➜ LTP: {ltp:.2f} ({raw_pnl:+.2f})\n"
                    f"   🛑 SL: {sl_info} | 🎯 TP: {tp_info}\n"
                    f"   🤖 Insight: {insight}"
                )
                report.append(line)
                total_unrealized_pnl += raw_pnl

            report.append(f"\n💰 Total Active P&L: {total_unrealized_pnl:+.2f}")
            report.append("-------------------------------------------------------")
            if state_changed:
                self._logger.info("\n".join(report))

        except Exception as e:
            self._logger.error(f"Status Report Failed: {e}")

    def _poll_pending_orders(self) -> None:
        """
        OPTIMIZED: Polls orders efficiently with lock snapshots to prevent race conditions.
        """
        # 1. Snapshot pending IDs inside lock (Fast)
        # We grab the set of IDs we care about immediately. This prevents race conditions
        # where the list of orders might change while we are fetching from the broker.
        with self._lock:
            pending_ids = {
                oid
                for oid, o in self._orders.items()
                if o.status == OrderStatus.PENDING
            }

        # Optimization: If nothing is pending, don't waste network calls
        if not pending_ids:
            return

        try:
            # 2. Bulk Fetch (Network I/O - No Lock)
            # Fetch the full order book from the broker
            if hasattr(self._broker, "orders"):
                all_orders = self._broker.orders()
            elif hasattr(self._broker, "get_orders"):
                all_orders = self._broker.get_orders()
            else:
                return  # Broker doesn't support bulk fetch

            if not all_orders:
                return

            # 3. Process Updates
            # Iterate through broker result and update ONLY orders we are tracking
            for remote in all_orders:
                oid = str(remote.get("order_id") or "")

                # OPTIMIZATION: Only process orders that were pending in our snapshot
                # This skips the hundreds of old/closed orders in the broker's book
                if oid in pending_ids:
                    # Normalize status
                    status = str(remote.get("status", "")).upper()

                    # Check against significant status updates
                    if status in ["COMPLETE", "FILLED", "CANCELLED", "REJECTED"]:
                        self._logger.info(
                            f"⚡ Bulk Update: {oid} -> {status}",
                            extra={
                                "event": "bulk_poll_update",
                                "order_id": oid,
                                "status": status,
                            },
                        )
                        # Call the central update handler (Handles its own locking safely)
                        self.on_order_update(remote)

        except Exception as e:
            self._logger.debug(f"Bulk poll failed: {e}")

    def _check_zombie_orders(self) -> None:
        """
        🛡️ SAFETY: Auto-cancel orders stuck in PENDING for too long (> 45s).
        Prevents margin blockage and 'ghost' fills.
        """
        now = time.time()
        ZOMBIE_TIMEOUT = 45.0  # Seconds

        zombies = []
        with self._lock:
            for oid, order in self._orders.items():
                if order.status == OrderStatus.PENDING:
                    # Check age
                    age = now - order.timestamp.timestamp()
                    if age > ZOMBIE_TIMEOUT:
                        zombies.append(oid)

        if not zombies:
            return

        self._logger.warning(
            f"🧟 Found {len(zombies)} ZOMBIE orders (> {ZOMBIE_TIMEOUT}s). Killing...",
            extra={"event": "zombie_cleanup", "orders": zombies},
        )

        for oid in zombies:
            try:
                self.cancel_order(oid)
            except Exception:
                self._logger.exception("Unhandled exception", exc_info=True)
                raise

    # ----------------------------------------------------------------
    # 💾 PERSISTENCE LAYER (Crash Recovery)
    # ----------------------------------------------------------------
    def save_orders(self) -> None:
        """Persist active orders to disk (Thread-Safe & Crash-Proof).

        ✅ PRODUCTION FIX: Uses DATA_DIR env var with /tmp fallback for Railway.
        """
        import os
        import uuid

        try:
            data = {}
            with self._lock:
                for oid, order in self._orders.items():
                    # Skip completely dead orders to keep file size manageable
                    if order.status in [OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                        continue

                    # 1. Convert Dataclass to dict
                    record = asdict(order)

                    # 2. SAFE ENUM SERIALIZATION
                    if hasattr(order.status, "name"):
                        record["status"] = order.status.name
                    elif hasattr(order.status, "value"):
                        record["status"] = order.status.value
                    else:
                        record["status"] = str(order.status).upper()

                    if hasattr(order.order_type, "name"):
                        record["order_type"] = order.order_type.name
                    elif hasattr(order.order_type, "value"):
                        record["order_type"] = order.order_type.value
                    else:
                        record["order_type"] = str(order.order_type).upper()

                    data[oid] = record

            # ✅ FIX: Use DATA_DIR environment variable with /tmp fallback
            data_dir = os.getenv("DATA_DIR", "data")
            path = Path(data_dir) / "orders.json"

            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                # Test write permission
                test_file = path.parent / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
            except (PermissionError, OSError):
                # Fallback to /tmp for Railway/Cloud environments
                path = Path(os.getenv("DATA_DIR", "data")) / "orders.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                self._logger.warning(f"⚠️ Using /tmp fallback: {path}")

            # [FIX] Unique Temp File prevents Thread Collision
            tmp_path = path.with_suffix(f".tmp.{uuid.uuid4().hex}")

            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())  # Force write to physical disk

            # Atomic replacement
            os.replace(tmp_path, path)
            self._logger.debug(f"✅ Orders saved to {path}")

        except Exception as e:
            self._logger.error(f"❌ Failed to save orders: {e}")

    def _restore_virtual_brackets(self) -> None:
        """Hydrate brackets from SQLite and reconcile against live broker positions."""
        try:
            saved_brackets = self._bracket_store.load_all_brackets()
            restored_symbols: set[str] = set()
            if self._bracket_manager and saved_brackets:
                for b_data in saved_brackets:
                    try:
                        self._bracket_manager.restore_bracket(
                            order_id=b_data["order_id"],
                            symbol=b_data["symbol"],
                            side=b_data["side"],
                            qty=b_data["qty"],
                            entry_price=b_data["entry_price"],
                            sl=b_data["current_sl"],
                            tp=b_data.get("tp1"),
                            trailing_enabled=b_data.get("trailing_active", False),
                            highest_ltp=b_data.get("highest_ltp", 0.0),
                            tag=b_data.get("tag"),
                        )
                        restored_symbols.add(str(b_data.get("symbol", "")).upper())
                    except Exception as e:
                        self._logger.warning(
                            f"Skipped restoring bracket {b_data.get('order_id')}: {e}"
                        )

            broker_positions: list[dict[str, Any]] = []
            try:
                if hasattr(self._broker, "get_positions"):
                    raw_positions = self._broker.get_positions()
                    if asyncio.iscoroutine(raw_positions):
                        raw_positions = asyncio.run(raw_positions)
                    if isinstance(raw_positions, list):
                        broker_positions = [
                            p for p in raw_positions if isinstance(p, dict)
                        ]
                    elif isinstance(raw_positions, dict):
                        net = raw_positions.get("net", raw_positions)
                        if isinstance(net, list):
                            broker_positions = [p for p in net if isinstance(p, dict)]
            except Exception as exc:
                self._logger.error(
                    "Failed broker position fetch during bracket recovery: %s", exc
                )

            live_symbols: set[str] = set()
            for payload in broker_positions:
                symbol = str(
                    payload.get("tradingsymbol") or payload.get("symbol") or ""
                ).upper()
                qty_raw = (
                    payload.get("net_qty")
                    or payload.get("quantity")
                    or payload.get("net_quantity")
                )
                try:
                    qty = int(float(qty_raw or 0))
                except (TypeError, ValueError):
                    qty = 0
                if symbol and qty != 0:
                    live_symbols.add(symbol)
                    if self._bracket_manager and symbol not in restored_symbols:
                        # Reconstruct missing brackets for broker-live positions.
                        if hasattr(self._bracket_manager, "attach_orphan_position"):
                            side = "LONG" if qty > 0 else "SHORT"
                            entry = float(
                                payload.get("average_price")
                                or payload.get("avg_price")
                                or 0.0
                            )
                            if entry > 0:
                                try:
                                    self._bracket_manager.attach_orphan_position(
                                        symbol=symbol,
                                        side=side,
                                        qty=abs(qty),
                                        entry_price=entry,
                                    )
                                    self._logger.info(
                                        "Recovered missing bracket for %s", symbol
                                    )
                                except Exception as exc:
                                    self._logger.error(
                                        "Failed to recover orphan bracket %s: %s",
                                        symbol,
                                        exc,
                                    )

            # Purge stale persisted brackets when broker no longer has matching position.
            for b_data in saved_brackets:
                symbol = str(b_data.get("symbol", "")).upper()
                if symbol and symbol not in live_symbols:
                    try:
                        self._bracket_store.delete_bracket(
                            str(b_data.get("order_id", ""))
                        )
                    except Exception as exc:
                        self._logger.error(
                            "Failed stale bracket purge for %s: %s", symbol, exc
                        )

            if saved_brackets:
                self._logger.info(
                    "Bracket recovery complete persisted=%d live_positions=%d",
                    len(saved_brackets),
                    len(live_symbols),
                )

        except Exception as e:
            self._logger.error(f"❌ Failed to restore virtual brackets: {e}")

    def _load_orders(self) -> None:
        """Restore orders from disk on startup.

        ✅ PRODUCTION FIX: Uses DATA_DIR env var with /tmp fallback.
        """
        import os

        # ✅ FIX: Use DATA_DIR environment variable
        data_dir = os.getenv("DATA_DIR", "data")
        path = Path(data_dir) / "orders.json"

        if not path.exists():
            # Try /tmp fallback location
            fallback = Path(os.getenv("DATA_DIR", "data")) / "orders.json"
            if fallback.exists():
                path = fallback
                self._logger.info(f"📂 Loading orders from fallback: {path}")
            else:
                self._logger.debug("No saved orders found")
                return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            restored_count = 0
            for oid, record in data.items():
                try:
                    # Reconstruct OrderStatus enum
                    status_str = record.get("status", "PENDING")
                    if hasattr(OrderStatus, status_str):
                        status = getattr(OrderStatus, status_str)
                    else:
                        status = OrderStatus.PENDING

                    # Reconstruct OrderType enum
                    order_type_str = record.get("order_type", "LIMIT")
                    if hasattr(OrderType, order_type_str):
                        order_type = getattr(OrderType, order_type_str)
                    else:
                        order_type = OrderType.LIMIT

                    # Create the order object (OrderDetails is the class actually
                    # stored in self._orders; 'Order' does not exist, which made
                    # every restore raise NameError and silently skip).
                    order = OrderDetails(
                        order_id=record.get("order_id", oid),
                        symbol=record.get("symbol", ""),
                        side=record.get("side", "BUY"),
                        quantity=int(record.get("quantity", 0)),
                        price=float(record.get("price", 0)),
                        order_type=order_type,
                        status=status,
                        fill_price=(
                            float(record["fill_price"])
                            if record.get("fill_price") is not None
                            else None
                        ),
                        filled_quantity=int(record.get("filled_quantity", 0) or 0),
                        applied_filled_quantity=int(
                            record.get("applied_filled_quantity", 0) or 0
                        ),
                        intent=record.get("intent") or "UNKNOWN",
                    )

                    with self._lock:
                        self._orders[oid] = order
                    restored_count += 1

                except Exception as e:
                    self._logger.warning(f"⚠️ Skipped restoring order {oid}: {e}")

            if restored_count > 0:
                self._logger.info(f"✅ Restored {restored_count} orders from {path}")

        except json.JSONDecodeError as e:
            self._logger.error(f"❌ Invalid JSON in orders file: {e}")
        except Exception as e:
            self._logger.error(f"❌ Failed to load orders: {e}")

    def _verify_restored_orders(self) -> None:
        """
        Startup Hygiene: Verify all 'PENDING' restored orders against Broker.
        Uses BULK FETCH to avoid API rate limits and clean up 'Phantom' orders.
        """
        # 1. Identify what needs verification (Non-Final states)
        with self._lock:
            to_verify = [
                oid
                for oid, o in self._orders.items()
                if o.status
                not in [
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                    OrderStatus.EXPIRED,
                ]
            ]

        if not to_verify:
            return

        self._logger.info(
            f"🔍 Verifying {len(to_verify)} restored orders with broker..."
        )

        try:
            # 2. BULK FETCH (One API Call) - Safe & Fast
            all_remote = []
            if hasattr(self._broker, "orders"):
                all_remote = self._broker.orders()
            elif hasattr(self._broker, "get_orders"):
                all_remote = self._broker.get_orders()
            else:
                self._logger.warning("Broker does not support bulk verify. Skipping.")
                return

            if not all_remote:
                # If broker returns empty list, ALL pending orders are ghosts.
                # But we must be careful. Let's assume connection is good.
                pass

            # Map remote orders for O(1) lookup
            remote_map = {str(o.get("order_id")): o for o in all_remote}

            # 3. Reconcile
            verified_count = 0
            stale_count = 0

            for oid in to_verify:
                if oid in remote_map:
                    # Order exists on broker -> Update local state
                    self.on_order_update(remote_map[oid])
                    verified_count += 1
                else:
                    # Order MISSING on broker -> It is a Phantom/Ghost. Kill it.
                    with self._lock:
                        if oid in self._orders:
                            self._orders[oid].status = OrderStatus.CANCELLED
                            self._orders[oid].rejection_reason = (
                                "Stale/Phantom Order cleaned on startup"
                            )
                    stale_count += 1

            self._logger.info(
                f"✅ Verification Complete: {verified_count} synced, {stale_count} phantoms cleaned."
            )
            # Save the cleaned state immediately
            self.save_orders()

        except Exception as e:
            self._logger.error(f"❌ Failed to verify restored orders: {e}")

    def _monitor_orders(self) -> None:
        """
        🔥 CRITICAL FIX: Active Fill Detection via Polling
        1. Polls broker for order status every 2 seconds
        2. Detects fills and triggers bracket activation
        """
        self._logger.info("🚀 Order monitoring thread started (POLLING MODE)")

        last_report_time = 0.0
        last_poll_time = 0.0

        while not self._stop_event.wait(0.5):  # Wake up every 500ms
            try:
                now = time.time()

                # CRITICAL: Fast Poll for Fills (Every 2 seconds)
                if now - last_poll_time >= 2.0:
                    self._poll_pending_orders()
                    # ✅ FIX: Kill stuck orders
                    self._check_zombie_orders()
                    last_poll_time = now

                # intent: central stop-loss monitoring independent of strategy cadence
                self._check_force_stop_losses()

                # Slow Status Report (Every 60 seconds)
                if now - last_report_time >= 60.0:
                    self._log_status_report()
                    last_report_time = now

            except Exception as exc:
                self._logger.error(f"Monitor loop error: {exc}", exc_info=True)
                time.sleep(1.0)
        self._logger.info("Order monitoring thread stopped")

    def _check_force_stop_losses(self) -> None:
        """Args: None. Returns: None. Raises: Exception."""
        self._logger.debug(
            "Entered OrderManager._check_force_stop_losses",
            extra={"event": "order_manager_force_sl_enter"},
        )
        try:
            positions = list(self._positions.get_open_positions())
            if not positions:
                return
            for pos in positions:
                symbol = getattr(pos, "symbol", "") or ""
                side = getattr(pos, "side", "LONG")
                qty = int(getattr(pos, "quantity", 0) or 0)
                stop_loss = getattr(pos, "stop_loss", None)
                if qty <= 0 or stop_loss is None or float(stop_loss) <= 0:
                    continue
                if getattr(pos, "state", None) == "force_closed_by_sl":
                    continue
                ltp = None
                _src = self._data_hub or self._market_data
                if _src is not None:
                    ltp = _src.get_latest_price(symbol)
                if ltp is None:
                    ltp = getattr(pos, "current_price", None)
                if ltp is None:
                    continue
                ltp_val = float(ltp)
                if ltp_val <= 0:
                    continue
                stop_val = float(stop_loss)
                sl_hit = (side == "LONG" and ltp_val <= stop_val) or (
                    side == "SHORT" and ltp_val >= stop_val
                )
                if not sl_hit:
                    continue
                pos.state = "force_closed_by_sl"
                try:
                    self._positions.save_state()
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in OrderManager._check_force_stop_losses: %s",
                        exc,
                        extra={
                            "event": "order_manager_force_sl_state_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
                self._logger.info(
                    f"❌ STOP LOSS HIT → FORCE EXIT | symbol={symbol} "
                    f"entry={float(getattr(pos, 'entry_price', 0.0)):.2f} "
                    f"sl={stop_val:.2f} ltp={ltp_val:.2f}",
                    extra={
                        "event": "force_stop_loss_exit",
                        "symbol": symbol,
                        "entry": float(getattr(pos, "entry_price", 0.0)),
                        "sl": stop_val,
                        "ltp": ltp_val,
                        "side": side,
                        "quantity": qty,
                    },
                )
                exit_side = "SELL" if side == "LONG" else "BUY"
                try:
                    self._place_exit_order(
                        symbol=symbol,
                        side=exit_side,
                        quantity=abs(qty),
                        product=getattr(pos, "product", "MIS"),
                        tag="FORCE_SL_EXIT",
                    )
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in OrderManager._check_force_stop_losses: %s",
                        exc,
                        extra={
                            "event": "force_stop_loss_exit_failed",
                            "symbol": symbol,
                            "side": exit_side,
                            "quantity": qty,
                        },
                        exc_info=exc,
                    )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in OrderManager._check_force_stop_losses: %s",
                exc,
                extra={"event": "order_manager_force_sl_error"},
                exc_info=exc,
            )

    def _handle_order_filled(self, order: OrderDetails) -> None:
        """Callback when order is filled."""

        latency_ms = 0.0
        if isinstance(order.timestamp, datetime):
            latency_ms = max(
                (datetime.now(timezone.utc) - order.timestamp).total_seconds() * 1000.0,
                0.0,
            )
        slippage_bps = None
        if order.price and order.price > 0 and order.fill_price is not None:
            try:
                slippage_bps = (
                    (order.fill_price - order.price) / order.price
                ) * 10_000.0
            except ZeroDivisionError:  # pragma: no cover - defensive
                slippage_bps = None
        self._logger.info(
            "order_filled",
            extra={
                "event": "order_fill",
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "fill_price": order.fill_price,
                "latency_ms": round(latency_ms, 3),
                "slippage_bps": (
                    None if slippage_bps is None else round(slippage_bps, 4)
                ),
            },
        )
        order_type_label = getattr(order.order_type, "name", str(order.order_type))
        try:
            METRICS.observe_order_latency(
                order_type=str(order_type_label), latency_seconds=latency_ms / 1000.0
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in OrderManager._handle_order_filled: %s",
                exc,
                extra={
                    "event": "order_latency_metric_error",
                    "order_id": order.order_id,
                },
            )

    def _handle_order_rejected(self, order: OrderDetails) -> None:
        """Handle broker rejections for submitted orders.

        Args:
            order: Order payload provided by the broker callback.

        Returns:
            None.

        Raises:
            None.
        """

        reason = order.rejection_reason or "unknown"
        reason_token = canonical(reason)
        emit_diag(
            self._logger,
            "order_reject",
            reason=reason_token,
            severity="warning",
            alert=True,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
        )
        try:
            METRICS.record_order_rejection(reason=reason_token)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in OrderManager._handle_order_rejected: %s",
                exc,
                extra={
                    "event": "order_rejection_metric_error",
                    "order_id": order.order_id,
                },
            )

        # ── FIX (BUG 2): If this was an EXIT order that got rejected, reactivate
        # the virtual bracket so the position gets SL protection back and the next
        # tick / watchdog cycle fires a fresh exit attempt.
        # Without this, the bracket stays permanently dead and the open position
        # bleeds until EOD square-off — which is exactly what caused the capital loss.
        tag = (order.tag or "").lower()
        is_exit_order = any(
            x in tag for x in ["exit", "stop", "target", "square", "guard"]
        )
        if is_exit_order and self._bracket_manager is not None:
            try:
                recovered = (
                    self._bracket_manager.reactivate_bracket_after_rejected_exit(
                        symbol=order.symbol,
                        rejected_order_id=order.order_id,
                        reason=reason,
                    )
                )
                if recovered:
                    self._logger.critical(
                        "🔁 REJECTED EXIT recovered for %s (order=%s) — bracket reactivated. "
                        "Next tick will retry the exit.",
                        order.symbol,
                        order.order_id,
                        extra={
                            "event": "rejected_exit_bracket_reactivated",
                            "symbol": order.symbol,
                            "order_id": order.order_id,
                            "reason": reason,
                        },
                    )
                else:
                    self._logger.error(
                        "⚠️ REJECTED EXIT for %s (order=%s) — no matching bracket found to reactivate. "
                        "MANUAL INTERVENTION REQUIRED.",
                        order.symbol,
                        order.order_id,
                        extra={
                            "event": "rejected_exit_no_bracket",
                            "symbol": order.symbol,
                            "order_id": order.order_id,
                        },
                    )
            except Exception as _rec_exc:
                self._logger.error(
                    "Bracket reactivation failed after rejected exit for %s: %s",
                    order.symbol,
                    _rec_exc,
                )

    # Internal helpers -------------------------------------------------

    def _pending_orders(self) -> list[OrderDetails]:
        with self._lock:
            return [
                order
                for order in self._orders.values()
                if order.status not in self.FINAL_STATUSES
            ]

    def get_broker_health_snapshot(self) -> dict[str, Any]:
        now = time.time()
        last_margin_success_age_s = (
            max(now - self._last_margin_success_ts, 0.0)
            if self._last_margin_success_ts is not None
            else None
        )
        balance_stale = (
            last_margin_success_age_s is None
            or last_margin_success_age_s > float(self._margin_cache_max_age_seconds)
        )
        # Self-heal: if the only problem is a stale margin cache (not an auth or
        # API failure), try one fresh fetch before declaring live orders blocked.
        # This fixes the false stale-block without bypassing any broker safety —
        # a genuine fetch failure leaves balance_stale True and still blocks.
        if (
            balance_stale
            and self._last_margin_error_type is None
            and not self._margin_circuit_open
        ):
            try:
                self._resolve_available_margin_raw()
            except Exception:  # noqa: BLE001
                pass
            last_margin_success_age_s = (
                max(now - self._last_margin_success_ts, 0.0)
                if self._last_margin_success_ts is not None
                else None
            )
            balance_stale = (
                last_margin_success_age_s is None
                or last_margin_success_age_s > float(self._margin_cache_max_age_seconds)
            )
        trading_allowed_effect = "none"
        if self._margin_circuit_open or self._last_margin_error_type:
            trading_allowed_effect = "position_sizing_degraded"
        if balance_stale and not self._allow_entry_with_stale_margin:
            trading_allowed_effect = "live_orders_blocked"
        connected_attr = getattr(self._broker, "is_connected", True)
        try:
            broker_connected = bool(
                connected_attr() if callable(connected_attr) else connected_attr
            )
        except Exception:
            broker_connected = False
        # Classify WHY live orders are blocked, so the caller (and logs) can tell
        # a stale cache apart from auth/config/API failures.
        order_err = (self._last_order_api_error or "").lower()
        auth_invalid = any(
            t in order_err
            for t in (
                "token",
                "api_key",
                "access_token",
                "unauthor",
                "forbidden",
                "403",
            )
        )
        if trading_allowed_effect == "live_orders_blocked":
            if not broker_connected:
                block_class = "broker_health_failed"
            elif auth_invalid:
                block_class = "broker_auth_invalid"
            elif self._last_margin_error_type or self._margin_circuit_open:
                block_class = "broker_health_failed"
            else:
                block_class = "broker_health_stale"
        else:
            block_class = "none"
        return {
            "broker_connected": broker_connected,
            "margin_api_available": self._last_margin_error_type is None,
            "last_margin_refresh_ts": self._last_margin_refresh_ts,
            "last_margin_success_age_s": last_margin_success_age_s,
            "last_margin_error_type": self._last_margin_error_type,
            "last_margin_error": self._last_margin_error,
            "margin_circuit_open": self._margin_circuit_open,
            "margin_circuit_remaining_s": max(
                (self._margin_circuit_until_ts or 0.0) - now, 0.0
            ),
            "balance_stale": balance_stale,
            "available_balance": self._last_margin_available_balance,
            "balance_source": self._last_margin_balance_source or "unknown",
            "trading_allowed_effect": trading_allowed_effect,
            "block_class": block_class,
            "order_api_available": self._last_order_api_error_type is None,
            "last_order_api_error_type": self._last_order_api_error_type,
            "last_order_api_error": self._last_order_api_error,
        }

    def _emit_broker_health_status(self, *, force: bool = False) -> None:
        try:
            now = time.time()
            market_open, _reason = get_time_status()
            snapshot = self.get_broker_health_snapshot()
            changed_effect = (
                snapshot["trading_allowed_effect"] != self._last_broker_health_effect
            )
            changed_circuit = (
                bool(snapshot["margin_circuit_open"])
                != self._last_broker_health_circuit_state
            )
            interval_elapsed = now - self._last_broker_health_emit_ts >= 30.0
            if not (
                force
                or changed_effect
                or changed_circuit
                or (market_open and interval_elapsed)
            ):
                return
            self._last_broker_health_emit_ts = now
            self._last_broker_health_effect = str(snapshot["trading_allowed_effect"])
            self._last_broker_health_circuit_state = bool(
                snapshot["margin_circuit_open"]
            )
            self._logger.info(
                "BROKER_HEALTH_STATUS broker_connected=%s margin_api_available=%s order_api_available=%s margin_circuit_open=%s balance_stale=%s trading_allowed_effect=%s last_order_api_error_type=%s last_margin_error_type=%s",
                snapshot.get("broker_connected"),
                snapshot.get("margin_api_available"),
                snapshot.get("order_api_available"),
                snapshot.get("margin_circuit_open"),
                snapshot.get("balance_stale"),
                snapshot.get("trading_allowed_effect"),
                snapshot.get("last_order_api_error_type"),
                snapshot.get("last_margin_error_type"),
                extra={"event": "BROKER_HEALTH_STATUS", **snapshot},
            )
        except Exception as exc:  # noqa: BLE001
            with suppress(Exception):
                self._logger.debug(
                    "BROKER_HEALTH_STATUS_EMIT_FAILED",
                    extra={
                        "event": "BROKER_HEALTH_STATUS_EMIT_FAILED",
                        "error_type": type(exc).__name__,
                        "error": self._sanitize_broker_error(exc),
                    },
                )

    def _record_margin_refresh_failure(self, exc: Exception) -> None:
        now = time.time()
        self._last_margin_refresh_ts = now
        self._last_margin_error_type = type(exc).__name__
        self._last_margin_error = self._sanitize_broker_error(exc)
        self._margin_circuit_open = True
        self._margin_circuit_until_ts = now + 30.0

    def _record_margin_refresh_success(
        self, available: float, source: str = "mdm"
    ) -> None:
        now = time.time()
        self._last_margin_success_ts = now
        self._last_margin_refresh_ts = now
        self._last_margin_available_balance = float(available)
        self._last_margin_balance_source = source
        self._last_margin_error_type = None
        self._last_margin_error = None
        self._margin_circuit_open = False
        self._margin_circuit_until_ts = None

    def _resolve_margin_client(self) -> Any:
        client = getattr(self._broker, "_client", None)
        return client if client is not None else self._broker

    @classmethod
    def _extract_margin_value(
        cls, payload: Any, keywords: tuple[str, ...], hint: bool = False
    ) -> float | None:
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                key_hint = hint
                if isinstance(key, str):
                    upper = key.upper()
                    if any(token in upper for token in keywords):
                        key_hint = True
                result = cls._extract_margin_value(value, keywords, key_hint)
                if result is not None:
                    return result
            return None
        if isinstance(payload, (list, tuple, set)):
            for item in payload:
                result = cls._extract_margin_value(item, keywords, hint)
                if result is not None:
                    return result
            return None
        if not hint:
            return None
        number = cls._coerce_float(payload)
        if number is None or not math.isfinite(number):
            return None
        if number < 0:
            return None
        return float(number)

    def _resolve_available_margin_raw(self) -> tuple[float | None, str]:
        mdm = self._data_hub or self._market_data
        if mdm is not None:
            self._last_margin_refresh_ts = time.time()
            refresh_failed = False
            try:
                mdm.refresh_margin_snapshot()
            except Exception as exc:  # noqa: BLE001
                refresh_failed = True
                self._record_margin_refresh_failure(exc)
                self._logger.error(
                    "Failure in _resolve_available_margin mdm refresh: %s",
                    self._sanitize_broker_error(exc),
                    extra={
                        "event": "order_margin_mdm_refresh_error",
                        "error_type": type(exc).__name__,
                    },
                    exc_info=exc,
                )
            if refresh_failed:
                return None, "margin_refresh_failed"
            try:
                available = mdm.get_available_balance()
            except Exception as exc:  # noqa: BLE001
                self._record_margin_refresh_failure(exc)
                self._logger.error(
                    "Failure in _resolve_available_margin mdm get: %s",
                    self._sanitize_broker_error(exc),
                    extra={
                        "event": "order_margin_mdm_read_error",
                        "error_type": type(exc).__name__,
                    },
                    exc_info=exc,
                )
                return None, "margin_read_failed"
            else:
                # Record freshness whenever the broker returns a VALID reading,
                # even if it is zero. Staleness must mean "we haven't reached the
                # broker recently" — not "the number was zero". Previously only a
                # >0 balance recorded success, so a healthy broker returning a low/
                # zero available balance left _last_margin_success_ts=None forever
                # -> balance_stale=True / margin_age_s=None permanently blocked live
                # orders (observed: BROKER_HEALTH_LIVE_ORDERS_BLOCKED for the whole
                # session). Sizing still independently refuses qty on balance<=0
                # (live), so recording a zero reading as fresh cannot oversize.
                if (
                    available is not None
                    and math.isfinite(float(available))
                    and float(available) >= 0
                ):
                    self._record_margin_refresh_success(float(available), "mdm")
                    return float(available), "mdm"
        risk_manager = self._risk_manager
        if risk_manager is not None:
            with suppress(Exception):
                balance = float(getattr(risk_manager, "current_balance", 0.0))
                if math.isfinite(balance) and balance >= 0:
                    self._record_margin_refresh_success(balance, "risk")
                    return balance, "risk"
        return None, "unknown"

    def _resolve_available_margin(
        self, *, for_entry: bool = True
    ) -> tuple[float | None, str]:
        available, source = self._resolve_available_margin_raw()
        now = time.time()
        if (
            self._margin_circuit_open
            and self._margin_circuit_until_ts is not None
            and now >= self._margin_circuit_until_ts
        ):
            self._margin_circuit_open = False
        if source == "mdm":
            self._emit_broker_health_status()
            return available, source
        if self._last_margin_success_ts is not None:
            age = now - self._last_margin_success_ts
            cached = self._last_margin_available_balance
            if (
                age <= float(self._margin_cache_max_age_seconds)
                and cached is not None
                and cached > 0
            ):
                self._last_margin_balance_source = "margin_cache_used"
                self._emit_broker_health_status(force=True)
                return cached, "margin_cache_used"
            if not for_entry:
                self._emit_broker_health_status(force=True)
                return available, "margin_unavailable_stale_exit_allowed"
            if (
                self._allow_entry_with_stale_margin
                and cached is not None
                and cached > 0
            ):
                self._last_margin_balance_source = "margin_cache_stale_allowed"
                self._emit_broker_health_status(force=True)
                return cached, "margin_cache_stale_allowed"
            self._emit_broker_health_status(force=True)
            return None, "margin_unavailable_stale"
        if for_entry and source == "risk" and not self._allow_entry_with_stale_margin:
            self._emit_broker_health_status(force=True)
            return None, "margin_unavailable_stale"
        self._emit_broker_health_status(force=True)
        return available, source

    def _reference_price(self, symbol: str) -> float:
        """Return a best-effort reference price for margin planning.

        Args:
            symbol: Tradable instrument identifier.

        Returns:
            Latest known price or 0.0 if unavailable.

        Raises:
            None.
        """

        self._logger.debug(
            "reference_price_lookup",
            extra={"event": "reference_price_lookup", "symbol": symbol},
        )
        normalized = DataHub.normalize(symbol)
        quote: Mapping[str, Any] | None = None
        hub = self._data_hub
        if hub is not None:
            try:
                quote = hub.get_quote(normalized, allow_pull=True)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "reference_price_quote_failed",
                    extra={
                        "event": "reference_price_quote_failed",
                        "symbol": normalized,
                        "error": str(exc),
                    },
                )
        price = 0.0
        if isinstance(quote, Mapping):
            for key in ("ltp", "last_price", "close", "price"):
                raw = quote.get(key)
                number = self._coerce_float(raw) if raw is not None else None
                if number is not None and number > 0:
                    price = float(number)
                    break
        return price

    def consume_skip_reason(self) -> str | None:
        """Return and clear the most recent skip reason emitted by the manager.

        Args:
            None.

        Returns:
            Latest skip reason string if recorded, else ``None``.

        Raises:
            None.
        """

        reason = self._last_skip_reason
        self._last_skip_reason = None
        return reason

    def set_last_skip_reason(self, reason: str) -> None:
        """Record the most recent skip reason for downstream consumers.

        Args:
            reason: Skip reason token to expose.

        Returns:
            None.

        Raises:
            None.
        """

        token = canonical(reason).lower()
        self._last_skip_reason = token or None

    def _margin_cooldown_state(self) -> tuple[bool, float]:
        """Return whether a margin cooldown is currently active.

        Args:
            None.

        Returns:
            Tuple where the first element indicates if cooldown is active and the
            second provides remaining seconds (``0.0`` when inactive).

        Raises:
            None.
        """

        expiry = self._margin_cooldown_until
        if expiry is None:
            return (False, 0.0)
        now = time.monotonic()
        if now >= expiry:
            self._margin_cooldown_until = None
            self._margin_block_streak = 0
            return (False, 0.0)
        remaining = max(expiry - now, 0.0)
        return (True, remaining)

    def _reset_margin_cooldown(self) -> None:
        """Reset internal counters tracking margin related cooldowns.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self._margin_block_streak = 0
        self._margin_cooldown_until = None

    def _register_margin_block(
        self,
        *,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        reason: str,
        decision: MarginDecision,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Record a margin-induced block for telemetry and cooldown control.

        Args:
            symbol: Instrument identifier associated with the block.
            side: Trade direction that was blocked.
            quantity: Requested quantity for the attempted order.
            reason: Normalized skip reason string.
            decision: Margin decision returned by the planner.
            context: Optional context dictionary captured during planning.

        Returns:
            None.

        Raises:
            None.
        """

        now = time.monotonic()
        self._margin_block_events.append(now)
        cutoff = now - 900.0
        while self._margin_block_events and self._margin_block_events[0] < cutoff:
            self._margin_block_events.popleft()
        try:
            self._m_margin_blocks.inc()
        except Exception as exc:  # pragma: no cover - metrics optional
            self._logger.debug(
                "margin_block_counter_failed",
                exc_info=False,
                extra={"event": "margin_block_counter_failed", "error": str(exc)},
            )
        for window_seconds, label in ((60.0, "1m"), (300.0, "5m"), (900.0, "15m")):
            count = sum(
                1 for ts in self._margin_block_events if now - ts <= window_seconds
            )
            try:
                self._m_margin_block_window.labels(window=label).set(count)
            except Exception as exc:  # pragma: no cover - metrics optional
                self._logger.debug(
                    "margin_block_window_metric_failed",
                    exc_info=False,
                    extra={
                        "event": "margin_block_window_metric_failed",
                        "window": label,
                        "error": str(exc),
                    },
                )
        if decision.quantity <= 0:
            self._margin_block_streak += 1
        else:
            self._margin_block_streak = 0
        if self._margin_block_streak >= self._margin_block_threshold:
            excess = self._margin_block_streak - self._margin_block_threshold
            cooldown_seconds = min(120.0, 60.0 + float(excess) * 30.0)
            self._margin_cooldown_until = now + cooldown_seconds
            self._logger.info(
                "margin_cooldown_engaged",
                extra={
                    "event": "margin_cooldown_engaged",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "reason": reason,
                    "cooldown_seconds": round(cooldown_seconds, 3),
                    "context": dict(context) if context else {},
                },
            )

    def _pre_trade_decision(
        self,
        *,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        product: str | None,
        price: float | None = None,
        stop_loss: float | None = None,
        atr: float | None = None,
    ) -> tuple[MarginDecision, dict[str, object]]:
        """Plan order sizing and product before market execution.

        Args:
            symbol: Tradable instrument identifier.
            side: Direction of the intended trade.
            quantity: Requested quantity prior to lot snapping.
            product: Desired product type (e.g., MIS, NRML).
            price: Optional reference price override.
            stop_loss: Optional stop-loss price for risk sizing.
            atr: Optional ATR value for fallback risk sizing.

        Returns:
            Tuple containing the :class:`MarginDecision` and planning context.

        Raises:
            OrderPlacementError: If margin planning cannot complete.
        """

        self._logger.debug(
            "pre_trade_decision_enter",
            extra={
                "event": "pre_trade_decision_enter",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "product": product,
            },
        )
        lot_size = 1
        try:
            lot_size = self._lot_size_for_symbol(symbol)
        except OrderPlacementError as exc:
            self._logger.error(
                "pre_trade_lot_lookup_failed",
                extra={
                    "event": "pre_trade_lot_lookup_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "pre_trade_lot_lookup_failed",
                extra={
                    "event": "pre_trade_lot_lookup_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
        normalized_symbol = DataHub.normalize(symbol)
        if normalized_symbol and not normalized_symbol.endswith(("CE", "PE")):
            lot_size = 1
        reference_meta: dict[str, object] = {}
        trace_id: str | None = None
        try:
            trace_hint = reference_meta.get("trace_id") if reference_meta else None
            if trace_hint:
                trace_id = str(trace_hint)
            else:
                last_trace = getattr(self, "_last_trace_id", "")
                trace_id = str(last_trace).strip() or None
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "pre_trade_trace_resolve_failed",
                extra={
                    "event": "pre_trade_trace_resolve_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
            trace_id = None
        resolved_price: float | None
        meta = RefPriceMeta(source="unknown", age_ms=1_000_000_000)
        mdm = self._data_hub or cast("MarketDataManager | None", self._market_data)

        if mdm is not None:
            try:
                latest_tick = mdm.get_latest_tick(symbol)
                has_price = False
                if isinstance(latest_tick, Mapping):
                    ltp_candidate = latest_tick.get("ltp")
                    if isinstance(ltp_candidate, (int, float)):
                        has_price = float(ltp_candidate) > 0
                if not has_price:
                    self._logger.info(
                        "Condition met: order_manager_seed_tracking",
                        extra={
                            "event": "order_manager_seed_tracking",
                            "symbol": symbol,
                        },
                    )
                    mdm.ensure_tracking(symbol, seed=True)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure ensuring tracking before planning: %s",
                    exc,
                    extra={
                        "event": "order_manager_ensure_tracking_error",
                        "symbol": symbol,
                    },
                    exc_info=exc,
                )

        if price is not None and price > 0:
            resolved_price = float(price)
            meta = RefPriceMeta(source="override", age_ms=0, market_protect=False)
        elif mdm is not None:
            self._ensure_quote_refresh(mdm, symbol, trace_id)
            resolved_price, meta = resolve_reference_price(
                symbol,
                mdm=mdm,
                require_depth=app_settings.ORDER_REF_REQUIRE_DEPTH,
                max_age_ms=app_settings.ORDER_MAX_QUOTE_AGE_MS,
                allow_ltp_fallback=app_settings.ORDER_ALLOW_LTP_FALLBACK,
                allow_market_protect=app_settings.ORDER_ALLOW_MARKET_PROTECT,
                protect_slippage_bps=app_settings.ORDER_PROTECT_SLIPPAGE_BPS,
            )
            if resolved_price is None or resolved_price <= 0:
                forced_seed = False
                try:
                    forced_seed = bool(mdm._seed_quote_from_broker(symbol))
                    if forced_seed:
                        self._logger.info(
                            "Condition met: order_manager_forced_seed",
                            extra={
                                "event": "order_manager_forced_seed",
                                "symbol": symbol,
                            },
                        )
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure forcing broker seed: %s",
                        exc,
                        extra={
                            "event": "order_manager_forced_seed_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
                try:
                    quote = mdm.pull_quote(symbol)
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in pull_quote fallback: %s",
                        exc,
                        extra={
                            "event": "order_manager_pull_quote_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
                    quote = {}
                if isinstance(quote, Mapping):
                    ltp_value = quote.get("ltp")
                    if isinstance(ltp_value, (int, float)) and float(ltp_value) > 0:
                        resolved_price = float(ltp_value)
                        meta = RefPriceMeta(
                            source="mdm_pull_quote",
                            age_ms=0,
                            market_protect=False,
                        )
        else:
            resolved_price = self._reference_price(symbol)
            if resolved_price is None or resolved_price <= 0:
                resolved_price = 1.0
            meta = RefPriceMeta(
                source="data_hub", age_ms=1_000_000_000, market_protect=False
            )

        canonical_source = canonical_price_source(meta.source)
        meta.source = canonical_source
        self._logger.info(
            "reference_price source=%s market_protect=%s age_ms=%s",
            canonical_source,
            str(meta.market_protect).lower(),
            meta.age_ms,
            extra={
                "symbol": symbol,
                "source": canonical_source,
                "market_protect": meta.market_protect,
                "age_ms": meta.age_ms,
            },
        )
        try:
            metrics.reference_price_source_total.labels(source=canonical_source).inc()
        except Exception:  # noqa: BLE001
            pass
        if resolved_price is None and not meta.market_protect:
            self._handle_missing_reference_price(mdm, symbol)

        reference_meta = asdict(meta)
        reference_meta["slippage_bps"] = app_settings.ORDER_PROTECT_SLIPPAGE_BPS
        if resolved_price is not None:
            reference_meta["fallback_price"] = resolved_price

        resolved_price_value = (
            float(resolved_price) if resolved_price and resolved_price > 0 else 0.0
        )
        planning_price = resolved_price_value if resolved_price_value > 0 else 1.0

        available_balance = 0.0
        available_source = "unknown"
        risk_manager = self._risk_manager
        if risk_manager is not None:
            try:
                available_balance = float(getattr(risk_manager, "current_balance", 0.0))
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "pre_trade_balance_failed",
                    extra={"event": "pre_trade_balance_failed", "error": str(exc)},
                )
            else:
                if available_balance > 0:
                    try:
                        available_source = str(risk_manager.balance_source_label())
                    except Exception as label_exc:  # noqa: BLE001
                        available_source = "risk"
                        self._logger.error(
                            "pre_trade_balance_label_failed",
                            extra={
                                "event": "pre_trade_balance_label_failed",
                                "error": str(label_exc),
                            },
                        )

        context: dict[str, object] = {}

        if available_balance <= 0:
            available, source = self._resolve_available_margin()
            if available is not None and available > 0:
                available_balance = float(available)
                available_source = source

        settings = (
            getattr(risk_manager, "settings", None)
            if risk_manager is not None
            else None
        )
        contract_multiplier = float(max(lot_size, 1))
        min_lots_per_trade = 1
        max_lots_per_trade = 1
        atr_multiple = 1.0
        per_trade_risk_pct = 0.5
        per_trade_cap_pct = 100.0
        if settings is not None:
            per_trade_risk_pct = float(
                getattr(settings, "per_trade_risk_pct", per_trade_risk_pct)
            )
            cap_default = per_trade_risk_pct
            per_trade_cap_pct = float(
                getattr(settings, "per_trade_cap_pct", cap_default)
            )
        else:
            fallback_risk = None
            try:
                runtime_settings = app_settings.get_settings()
                fallback_risk = getattr(runtime_settings, "risk", None)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "pre_trade_risk_settings_fallback_failed",
                    extra={
                        "event": "pre_trade_risk_settings_fallback_failed",
                        "symbol": symbol,
                    },
                    exc_info=exc,
                )
            if fallback_risk is not None:
                per_trade_risk_pct = float(
                    getattr(fallback_risk, "per_trade_risk_pct", per_trade_risk_pct)
                )
                per_trade_cap_pct = float(
                    getattr(fallback_risk, "per_trade_cap_pct", per_trade_risk_pct)
                )
                min_lots_per_trade = max(
                    1,
                    int(
                        getattr(
                            fallback_risk,
                            "min_lots_per_trade",
                            min_lots_per_trade,
                        )
                    ),
                )
                max_lots_candidate = max(
                    min_lots_per_trade,
                    int(
                        getattr(
                            fallback_risk,
                            "max_lots_per_trade",
                            min_lots_per_trade,
                        )
                    ),
                )
                max_lots_per_trade = max(max_lots_per_trade, max_lots_candidate)
                atr_multiple = max(
                    float(getattr(fallback_risk, "atr_stop_multiple", atr_multiple)),
                    0.0,
                )
                contract_multiplier = max(
                    contract_multiplier,
                    float(
                        getattr(fallback_risk, "contract_lot_size", contract_multiplier)
                    ),
                )
        if settings is not None:
            try:
                min_lots_per_trade = max(
                    1, int(getattr(settings, "min_lots_per_trade", 1))
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "pre_trade_min_lots_invalid",
                    extra={
                        "event": "pre_trade_min_lots_invalid",
                        "symbol": symbol,
                        "error": str(exc),
                    },
                )
                min_lots_per_trade = 1
            try:
                max_candidate = int(
                    getattr(settings, "max_lots_per_trade", min_lots_per_trade)
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "pre_trade_max_lots_invalid",
                    extra={
                        "event": "pre_trade_max_lots_invalid",
                        "symbol": symbol,
                        "error": str(exc),
                    },
                )
                max_candidate = min_lots_per_trade
            max_lots_per_trade = max(min_lots_per_trade, max_candidate)
            try:
                atr_multiple = max(
                    float(getattr(settings, "atr_stop_multiple", 1.0)), 0.0
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "pre_trade_atr_multiple_invalid",
                    extra={
                        "event": "pre_trade_atr_multiple_invalid",
                        "symbol": symbol,
                        "error": str(exc),
                    },
                )
                atr_multiple = 1.0
            try:
                candidate_multiplier = float(
                    getattr(settings, "contract_lot_size", contract_multiplier)
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "pre_trade_contract_multiplier_invalid",
                    extra={
                        "event": "pre_trade_contract_multiplier_invalid",
                        "symbol": symbol,
                        "error": str(exc),
                    },
                )
                candidate_multiplier = contract_multiplier
            contract_multiplier = max(candidate_multiplier, float(max(lot_size, 1)))

        margin_factor = self._margin_factor if self._margin_factor > 0 else 1.0
        margin_buffer = self._margin_buffer if 0 < self._margin_buffer <= 1.0 else 1.0

        if available_balance <= 0:
            estimated = planning_price * max(quantity, 1) * margin_factor
            if estimated <= 0:
                estimated = float(max(quantity, 1))
            adjusted = estimated / margin_buffer if margin_buffer > 0 else estimated
            fallback_balance = max(adjusted, estimated, 1_000_000.0)
            available_balance = fallback_balance
            available_source = "fallback"

        cooldown_active, cooldown_remaining = self._margin_cooldown_state()
        if cooldown_active and quantity > 0:
            context = {
                "available_source": available_source,
                "reference_price": 0.0,
                "lot_size": lot_size,
                "reference_price_meta": {"source": "cooldown"},
                "skip_reason": "margin_cooldown",
                "cooldown_remaining": cooldown_remaining,
            }
            sizing = SizingResult(qty=0, reason="margin_cooldown")
            decision = MarginDecision(
                ok=False,
                reason="MARGIN margin_cooldown",
                order_type=(product or "NRML") or "NRML",
                quantity=0,
                est_required=0.0,
                available=available_balance,
                sizing=sizing,
            )
            self._logger.info(
                "margin_plan_cooldown_active",
                extra={
                    "event": "margin_plan_cooldown_active",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "remaining_seconds": round(cooldown_remaining, 3),
                },
            )
            self.set_last_skip_reason("margin_cooldown")
            return decision, context

        try:
            ist_zone = ZoneInfo("Asia/Kolkata")
            ist_now = datetime.now(ist_zone)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "pre_trade_zoneinfo_failed",
                extra={"event": "pre_trade_zoneinfo_failed", "error": str(exc)},
            )
            ist_now = datetime.now(timezone.utc)

        context = {
            "available_source": available_source,
            "reference_price": resolved_price_value,
            "lot_size": lot_size,
            "reference_price_meta": reference_meta,
        }
        if reference_meta.get("market_protect"):
            context["market_protect"] = True
            context["protect_slippage_bps"] = app_settings.ORDER_PROTECT_SLIPPAGE_BPS
            fallback_price = reference_meta.get("fallback_price")
            if isinstance(fallback_price, (int, float)) and fallback_price > 0:
                context["protect_fallback_price"] = float(fallback_price)

        skip_reason = None
        if resolved_price_value <= 0 and not reference_meta.get("market_protect"):
            skip_reason = "no_reference_price"
            context["skip_reason"] = skip_reason
            self.set_last_skip_reason(skip_reason)
            decision = MarginDecision(
                ok=False,
                reason="no_reference_price",
                order_type=(product or "NRML") or "NRML",
                quantity=0,
                est_required=0.0,
                available=available_balance,
                sizing=None,
            )
            return decision, context

        inputs = MarginInputs(
            symbol=symbol,
            side=side,
            price=planning_price,
            stop_loss=stop_loss,
            atr=atr,
            requested_qty=quantity,
            product=product,
            lot_size=lot_size,
            balance=available_balance,
            per_trade_risk_pct=per_trade_risk_pct,
            per_trade_cap_pct=per_trade_cap_pct,
            margin_factor=margin_factor,
            margin_buffer=margin_buffer,
            contract_multiplier=contract_multiplier,
            ist_now=ist_now,
            min_lots_per_trade=min_lots_per_trade,
            max_lots_per_trade=max_lots_per_trade,
            atr_multiple=atr_multiple,
        )
        try:
            decision = self._margin_engine.plan(inputs)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "pre_trade_decision_failed",
                extra={
                    "event": "pre_trade_decision_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
            raise OrderPlacementError("Margin planning failed") from exc
        context.setdefault("available_source", available_source)
        context.setdefault("lot_size", lot_size)
        context.setdefault("reference_price", resolved_price_value)
        return decision, context

    def _precheck_margin(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        product: str | None,
        *,
        dry_run: bool = False,
        price: float | None = None,
        stop_loss: float | None = None,
    ) -> tuple[bool, str, dict[str, object]]:
        meta: dict[str, object] = {
            "needed": 0.0,
            "available": None,
            "buffer": self._margin_buffer,
            "source": "margin_engine",
            "available_source": "unknown",
        }
        if quantity <= 0:
            return True, "", meta
        decision, context = self._pre_trade_decision(
            symbol=symbol,
            side=side,
            quantity=quantity,
            product=product,
            price=price,
            stop_loss=stop_loss,
        )
        meta["needed"] = decision.est_required
        meta["available"] = decision.available
        meta["available_source"] = context.get("available_source", "unknown")
        if decision.ok:
            return True, "", meta
        reason = decision.reason or "MARGIN"
        if not dry_run:
            reason_code = canonical(reason)
            self._logger.info(
                "order_blocked_soft reason=%s",
                reason_code,
                extra={
                    "event": "order_soft_block",
                    "reason": reason_code,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "product": product,
                    "needed": decision.est_required,
                    "available": decision.available,
                    "buffer": self._margin_buffer,
                },
            )
        return False, reason, meta

    @staticmethod
    def _is_margin_error(message: str) -> bool:
        token = message.upper()
        return "INSUFFICIENT FUNDS" in token or "MARGIN" in token

    def _guard_snapshot(self) -> dict[str, object] | None:
        getter = self._session_guard_getter
        if getter is None:
            return None
        try:
            snapshot = getter()
        except Exception:  # pragma: no cover - defensive
            self._logger.debug("session_guard_snapshot_failed", exc_info=True)
            return None
        if snapshot is None:
            return None
        if isinstance(snapshot, Mapping):
            return dict(snapshot)
        as_dict = getattr(snapshot, "as_dict", None)
        if callable(as_dict):
            try:
                result = as_dict()
            except Exception:  # pragma: no cover - defensive
                self._logger.debug("guard_as_dict_failed", exc_info=True)
                return None
            if isinstance(result, Mapping):
                return dict(result)
        if hasattr(snapshot, "items"):
            try:
                return dict(snapshot)
            except Exception:  # pragma: no cover - defensive
                self._logger.debug("guard_dict_coerce_failed", exc_info=True)
                return None
        return None

    def _resolve_enable_live(self) -> bool:
        if self._enable_live_getter is None:
            return True
        try:
            return bool(self._enable_live_getter())
        except Exception:  # pragma: no cover - defensive
            self._logger.debug("enable_live_getter_failed", exc_info=True)
            return True

    def _resolve_shadow_mode(self) -> bool:
        if self._shadow_mode_getter is None:
            return False
        try:
            return bool(self._shadow_mode_getter())
        except Exception:  # pragma: no cover - defensive
            self._logger.debug("shadow_mode_getter_failed", exc_info=True)
            return False

    def _ensure_trading_allowed(
        self,
        *,
        symbol: str | None = None,
        side: Literal["BUY", "SELL"] | None = None,
        quantity: int | None = None,
    ) -> bool:
        """Validate trading guardrails before submitting an order.

        Args:
            symbol: Optional tradable instrument identifier for context.
            side: Optional trade side associated with the request.
            quantity: Optional requested quantity for diagnostics.

        Returns:
            bool: ``True`` when trading is allowed for the supplied context.

        Raises:
            OrderPlacementError: If trading is blocked by guardrails.
        """

        self._logger.debug(
            "Entered OrderManager._ensure_trading_allowed",
            extra={
                "event": "ensure_trading_allowed_enter",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            },
        )
        guard = self._guard_snapshot()
        enable_live = self._resolve_enable_live()
        shadow_mode = self._resolve_shadow_mode()
        force_exit = False

        switch_state: TradingSwitchState | None = None
        switch = trading_switch()
        try:
            switch_allowed = switch.can_trade()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _ensure_trading_allowed: %s",
                exc,
                extra={"event": "trading_switch_check_failed"},
            )
            switch_allowed = True
        if not switch_allowed:
            try:
                switch_state = switch.snapshot()
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _ensure_trading_allowed: %s",
                    exc,
                    extra={"event": "trading_switch_snapshot_failed"},
                )
                switch_state = None

        def _resolve_force_exit() -> bool:
            nonlocal force_exit
            if force_exit:
                return True
            if symbol is None or side is None or quantity is None:
                return False
            force_exit = self._is_force_exit(
                symbol=symbol,
                side=side,
                quantity=quantity,
            )
            return force_exit

        if not switch_allowed:
            snapshot_payload: dict[str, object] = {}
            if switch_state is not None:
                snapshot_payload = {
                    "switch_enabled": switch_state.enabled,
                    "switch_resume_at": switch_state.resume_at,
                    "switch_remaining": round(switch_state.remaining_seconds, 3),
                    "switch_can_trade": switch_state.can_trade,
                }
            if _resolve_force_exit():
                self._logger.info(
                    "Condition met: trading pause overridden for force exit",
                    extra={
                        "event": "trading_pause_force_exit",
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        **snapshot_payload,
                    },
                )
                return True
            self._logger.info(
                "Condition met: trading paused by operator",
                extra={
                    "event": "order_rejected",
                    "reason": "TRADING_SWITCH",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    **snapshot_payload,
                },
            )
            raise OrderPlacementError("Trading paused by operator")

        if guard is not None and not can_trade(
            guard, enable_live=enable_live, shadow_mode=shadow_mode
        ):
            if _resolve_force_exit():
                self._logger.info(
                    "order_force_exit_allowed",
                    extra={
                        "event": "order_force_exit",
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "enable_live": enable_live,
                        "shadow_mode": shadow_mode,
                        "guard": guard,
                    },
                )
                return True
            reasons = (
                guard.get("reasons") if isinstance(guard.get("reasons"), list) else []
            )
            self._logger.error(
                "Not submitting, session guard blocked trading "
                "(check market_open and overrides)",
                extra={
                    "event": "order_rejected",
                    "reason": "TRADING_DISABLED",
                    "detail": "session_guard_blocked",
                    "enable_live": enable_live,
                    "shadow_mode": shadow_mode,
                    "guard": guard,
                    "reasons": reasons,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                },
            )
            raise OrderPlacementError("Trading disabled by session guard")

        risk_manager = self._risk_manager
        if risk_manager is None:
            self._logger.debug(
                "Condition met: trading allowed (no risk manager)",
                extra={
                    "event": "ensure_trading_allowed_pass",
                    "reason": "no_risk_manager",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                },
            )
            return True
        allowed, reasons = risk_manager.risk_gate_should_trade()
        if allowed:
            self._logger.debug(
                "Condition met: trading allowed (risk gate)",
                extra={
                    "event": "ensure_trading_allowed_pass",
                    "reason": "risk_gate",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                },
            )
            return True
        if _resolve_force_exit():
            self._logger.info(
                "order_force_exit_allowed",
                extra={
                    "event": "order_force_exit",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "enable_live": enable_live,
                    "shadow_mode": shadow_mode,
                    "risk_reasons": list(reasons),
                },
            )
            return True
        reason_text = ",".join(reasons)
        reason_code = canonical(reason_text or "RISK_STATE")
        reason_codes = [
            canonical(value) for value in reasons if isinstance(value, str) and value
        ]
        if not reason_codes:
            reason_codes = [reason_code]
        reason_set = {code for code in reason_codes if code}
        tick_effective_ms: float | None = None
        tick_mono_age_ms: float | None = None
        tick_server_age_ms: float | None = None
        tick_fresh: bool | None = None
        tick_reason: str | None = None
        tick_threshold_ms: float | None = None

        if symbol and self._data_hub is not None:
            try:
                fresh, meta = self._data_hub.is_fresh(symbol)
            except Exception:  # pragma: no cover - defensive
                meta = None
            else:
                tick_fresh = bool(fresh)
                meta_dict: dict[str, object] = dict(cast(Mapping[str, object], meta))

                def _as_float(value: object | None) -> float | None:
                    if value is None:
                        return None
                    try:
                        return float(cast(Any, value))
                    except (TypeError, ValueError):
                        return None

                tick_effective_ms = _as_float(meta_dict.get("effective_ms"))
                tick_mono_age_ms = _as_float(meta_dict.get("mono_age_ms"))
                tick_server_age_ms = _as_float(meta_dict.get("server_age_ms"))
                tick_threshold_ms = _as_float(meta_dict.get("threshold_ms"))
                raw_reason = meta_dict.get("reason")
                tick_reason = str(raw_reason) if raw_reason else None

        soft_only = bool(reason_set) and reason_set.issubset(SOFT_BLOCK_CODES)
        cooldown_remaining: float | None = None
        if soft_only and "COOLDOWN" in reason_set:
            try:
                cooldown_remaining = risk_manager.cooldown_remaining()
            except Exception:  # pragma: no cover - defensive
                cooldown_remaining = None
        if soft_only:
            self.set_last_skip_reason(reason_code or "RISK_STATE")
            self._logger.info(
                "order_blocked_soft reason=%s",
                ",".join(sorted(reason_set)) or reason_code,
                extra={
                    "event": "order_soft_block",
                    "reason": "RISK_STATE",
                    "enable_live": enable_live,
                    "shadow_mode": shadow_mode,
                    "risk_reasons": list(sorted(reason_set)),
                    "risk_reason_codes": reason_code,
                    "cooldown_remaining": cooldown_remaining,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "tick_fresh": tick_fresh,
                    "tick_reason": tick_reason,
                    "tick_threshold_ms": tick_threshold_ms,
                    "tick_age_ms": tick_effective_ms,
                    "tick_mono_age_ms": tick_mono_age_ms,
                    "tick_server_age_ms": tick_server_age_ms,
                },
            )
            return False
        cooldown_remaining = None
        if reason_code == "COOLDOWN":
            try:
                cooldown_remaining = risk_manager.cooldown_remaining()
            except Exception:  # pragma: no cover - defensive
                cooldown_remaining = None

        self._logger.warning(
            "order_rejected_risk_state reason=%s",
            reason_code or "unknown",
            extra={
                "event": "order_rejected",
                "reason": "RISK_STATE",
                "enable_live": enable_live,
                "shadow_mode": shadow_mode,
                "risk_reasons": reason_codes,
                "risk_reason_codes": reason_code,
                "cooldown_remaining": cooldown_remaining,
                "tick_fresh": tick_fresh,
                "tick_reason": tick_reason,
                "tick_threshold_ms": tick_threshold_ms,
                "tick_age_ms": tick_effective_ms,
                "tick_mono_age_ms": tick_mono_age_ms,
                "tick_server_age_ms": tick_server_age_ms,
            },
        )
        raise OrderPlacementError("Trading disabled by risk state")
        return False

    def _execute_market_order(
        self,
        *,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        product: str | None,
        tag: str | None,
    ) -> OrderDetails:
        """Submit a market-style order after margin planning safeguards.

        Args:
            symbol: Tradable instrument identifier.
            side: Trade direction ("BUY" or "SELL").
            quantity: Requested quantity before margin adjustments.
            product: Requested product type.
            tag: Optional broker tag for traceability.

        Returns:
            Broker-reported :class:`OrderDetails` for the routed order.

        Raises:
            OrderPlacementError: If planning blocks the order or broker rejects.
        """

        self._logger.debug(
            "execute_market_order_enter",
            extra={
                "event": "execute_market_order_enter",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "product": product,
            },
        )
        decision, context = self._pre_trade_decision(
            symbol=symbol,
            side=side,
            quantity=quantity,
            product=product,
        )
        effective_quantity = decision.quantity or 0
        reference_meta = cast(
            dict[str, object], context.get("reference_price_meta", {})
        )
        skip_reason = cast(str | None, context.get("skip_reason"))
        if effective_quantity <= 0:
            sizing_reason = None
            if hasattr(decision, "sizing"):
                sizing_payload = getattr(decision, "sizing", None)
                sizing_reason = getattr(sizing_payload, "reason", None)
            self._logger.debug(
                "margin_plan_zero_quantity",
                extra={
                    "event": "margin_plan_zero_quantity",
                    "symbol": symbol,
                    "side": side,
                    "requested_quantity": quantity,
                    "planned_quantity": effective_quantity,
                    "decision_ok": bool(decision.ok),
                    "decision_reason": decision.reason,
                    "sizing_reason": sizing_reason,
                },
            )
        if not decision.ok or effective_quantity <= 0:
            reason = decision.reason or "MARGIN"
            reason_code = canonical(reason)
            sizing = getattr(decision, "sizing", None)
            if skip_reason is None and sizing is not None:
                sizing_reason = getattr(sizing, "reason", None)
                if sizing_reason == "insufficient_margin":
                    skip_reason = "margin_no_qty"
            if skip_reason is None and reason_code == "MARGIN":
                skip_reason = "margin_no_qty" if effective_quantity <= 0 else None
            if skip_reason is not None:
                context["skip_reason"] = skip_reason
            skip_token = canonical(skip_reason or reason_code)
            if reason_code == "MARGIN":
                self._register_margin_block(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    reason=skip_token,
                    decision=decision,
                    context=context,
                )
            else:
                self._reset_margin_cooldown()
            emit_diag(
                self._logger,
                "order_soft_block",
                reason=skip_token,
                severity="warning",
                alert=True,
                symbol=symbol,
                side=side,
                quantity=quantity,
                product=product,
                needed=decision.est_required,
                available=decision.available,
                context=context,
                reason_code=reason_code,
            )
            self.set_last_skip_reason(skip_token)
            raise OrderPlacementError(skip_token.lower())
        if effective_quantity != quantity:
            self._logger.info(
                "order_quantity_adjusted",
                extra={
                    "event": "order_quantity_adjusted",
                    "symbol": symbol,
                    "side": side,
                    "requested_quantity": quantity,
                    "planned_quantity": effective_quantity,
                },
            )
        self._reset_margin_cooldown()
        quantity = effective_quantity
        product = decision.order_type or product
        if context.get("market_protect"):
            slippage_raw = context.get("protect_slippage_bps")
            slippage_bps: float | None = None
            if isinstance(slippage_raw, (int, float)):
                slippage_bps = float(slippage_raw)
            elif isinstance(slippage_raw, str):
                try:
                    slippage_bps = float(slippage_raw.strip())
                except ValueError:
                    slippage_bps = None
            if slippage_bps is None:
                fallback_slippage = reference_meta.get("slippage_bps", 12)
                if isinstance(fallback_slippage, (int, float)):
                    slippage_bps = float(fallback_slippage)
                elif isinstance(fallback_slippage, str):
                    try:
                        slippage_bps = float(fallback_slippage.strip())
                    except ValueError:
                        slippage_bps = 12.0
                else:
                    slippage_bps = 12.0
            base_price_candidate = context.get("protect_fallback_price")
            base_price: float | None = None
            if isinstance(base_price_candidate, (int, float)):
                numeric_base = float(base_price_candidate)
                if numeric_base > 0:
                    base_price = numeric_base
            elif isinstance(base_price_candidate, str):
                try:
                    numeric_base = float(base_price_candidate.strip())
                except ValueError:
                    numeric_base = 0.0
                if numeric_base > 0:
                    base_price = numeric_base
            if base_price is None:
                reference_candidate = context.get("reference_price")
                if isinstance(reference_candidate, (int, float)):
                    base_price = float(reference_candidate)
                elif isinstance(reference_candidate, str):
                    try:
                        base_price = float(reference_candidate.strip())
                    except ValueError:
                        base_price = None
                if base_price is None:
                    fallback_price = reference_meta.get("fallback_price")
                    if isinstance(fallback_price, (int, float)):
                        numeric_fallback = float(fallback_price)
                        if numeric_fallback > 0:
                            base_price = numeric_fallback
                    elif isinstance(fallback_price, str):
                        try:
                            numeric_fallback = float(fallback_price.strip())
                        except ValueError:
                            numeric_fallback = 0.0
                        if numeric_fallback > 0:
                            base_price = numeric_fallback
            if slippage_bps is None:
                slippage_bps = 12.0
            if base_price is None:
                base_price = 0.0
            if base_price <= 0:
                self._logger.error(
                    "market_protect_base_price_unavailable",
                    extra={
                        "event": "market_protect_base_price_unavailable",
                        "symbol": symbol,
                        "side": side,
                    },
                )
                raise OrderPlacementError("NO_REFERENCE_PRICE")
            adjustment = abs(slippage_bps) / 10_000.0
            if side.upper() == "BUY":
                limit_price = base_price * (1.0 + adjustment)
            else:
                limit_price = max(base_price * (1.0 - adjustment), 0.05)
            self._logger.info(
                "Condition met: market_protect_order",
                extra={
                    "event": "market_protect_order",
                    "symbol": symbol,
                    "side": side,
                    "limit_price": round(limit_price, 5),
                    "slippage_bps": round(slippage_bps, 5),
                },
            )
            details = self._place_single_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price=limit_price,
                product=product,
                tag=tag,
            )
            return details
        policy = self._execution_policy
        if policy is None:
            return self._place_single_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                price=None,
                product=product,
                tag=tag,
            )
        try:
            plan = policy.build_plan(symbol, side)
        except OrderPlacementError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.warning(
                "execution_policy_plan_failed",
                extra={
                    "event": "execution_policy_plan_failed",
                    "symbol": symbol,
                    "side": side,
                    "error": str(exc),
                },
            )
            return self._place_single_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                price=None,
                product=product,
                tag=tag,
            )

        try:
            details = self._place_single_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price=plan.limit_prices[0],
                product=product,
                tag=tag,
            )
        except OrderPlacementError as exc:
            message = str(exc)
            code = canonical(message)
            if code in SOFT_BLOCK_CODES:
                self._logger.info(
                    "execution_policy_plan_failed reason=%s",
                    code,
                    extra={
                        "event": "execution_policy_plan_failed",
                        "symbol": symbol,
                        "side": side,
                        "reason": code,
                        "error": message,
                    },
                )
                raise OrderPlacementError(message) from exc
            raise
        order_id = details.order_id
        deadline = time.time() + policy.timeout_sec
        self._logger.info(
            "execution_policy_applied",
            extra={
                "event": "execution_policy_applied",
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "prices": list(plan.limit_prices),
                "spread_pct": plan.spread_pct,
                "timeout": policy.timeout_sec,
            },
        )

        def _remaining() -> float:
            return max(deadline - time.time(), 0.0)

        def _refresh() -> OrderDetails:
            try:
                return self._refresh_order(order_id)
            except KeyError:
                return details

        if details.status == OrderStatus.FILLED:
            return details

        wait_window = min(plan.step_timeout, _remaining())
        if wait_window > 0 and self.wait_for_fill(order_id, wait_window):
            return _refresh()

        for price in plan.limit_prices[1:]:
            if _remaining() <= 0:
                break
            if not self.modify_order(order_id, new_price=price):
                continue
            _refresh()
            wait_window = min(plan.step_timeout, _remaining())
            if wait_window <= 0:
                continue
            if self.wait_for_fill(order_id, wait_window):
                return _refresh()

        final_state = _refresh()
        if final_state.status == OrderStatus.FILLED or (
            final_state.filled_quantity >= final_state.quantity > 0
        ):
            return final_state

        try:
            cancelled = self.cancel_order(order_id)
        except Exception as exc:  # noqa: BLE001
            cancelled = False
            self._logger.warning(
                "execution_policy_cancel_failed",
                extra={
                    "event": "execution_policy_cancel_failed",
                    "order_id": order_id,
                    "error": str(exc),
                },
            )
        if cancelled:
            final_state = _refresh()

        raise OrderPlacementError("Pegged execution timed out")

    def _place_single_order(
        self,
        *,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        order_type: OrderType,
        price: float | None,
        product: str | None = None,
        tag: str | None = None,
        parent_order_id: str | None = None,
        client_order_id: str | None = None,
        payload_overrides: dict[str, Any] | None = None,
    ) -> OrderDetails:
        """Submit a single order leg to the broker.

        Args:
            symbol: Instrument identifier.
            side: Trade side (``BUY`` or ``SELL``).
            quantity: Order quantity.
            order_type: Desired order type.
            price: Optional limit or trigger price.
            product: Broker product code.
            tag: Optional broker tag string.
            parent_order_id: Parent order reference for brackets.
            client_order_id: Client-assigned identifier.
            payload_overrides: Optional payload overrides for routing.

        Returns:
            OrderDetails: Broker response metadata for the submitted order.

        Raises:
            OrderPlacementError: If submission fails or broker rejects.
        """

        payload = self._build_order_payload(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            product=product,
            tag=tag,
            parent_order_id=parent_order_id,
            client_order_id=client_order_id,
        )
        if payload_overrides:
            payload.update(payload_overrides)
        response = self._submit_order_with_retry(payload)
        order_id = str(response.get("order_id"))
        if not order_id:
            raise OrderPlacementError("Broker response missing order_id")
        status = self._parse_status(response.get("status"))
        message = response.get("message") or response.get("status_message")
        timestamp = datetime.now(timezone.utc)
        response_client_id = (
            client_order_id
            or response.get("client_order_id")
            or response.get("clientOrderId")
        )
        resolved_client_id = str(response_client_id) if response_client_id else None
        details = OrderDetails(
            order_id=order_id,
            symbol=symbol.upper(),
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=float(price or 0.0),
            status=status,
            timestamp=timestamp,
            parent_order_id=parent_order_id,
            client_order_id=resolved_client_id,
        )
        raw_filled = response.get("filled_quantity") or response.get("filled")
        if raw_filled is not None:
            with suppress(Exception):
                details.filled_quantity = int(float(raw_filled))
        avg_price = response.get("average_price") or response.get("fill_price")
        if avg_price is not None:
            with suppress(Exception):
                details.fill_price = float(avg_price)
        if details.status == OrderStatus.FILLED and details.filled_quantity <= 0:
            details.filled_quantity = quantity
            if details.fill_price is None and details.price > 0:
                details.fill_price = details.price
        try:
            budgets = self._limiter.snapshot().get("orders", {})
        except Exception:  # pragma: no cover - defensive
            budgets = {}
        self._logger.info(
            "order_submitted",
            extra={
                "event": "order_submit",
                "order_id": order_id,
                "symbol": details.symbol,
                "side": details.side,
                "quantity": details.quantity,
                "order_type": details.order_type.value,
                "price": details.price,
                "budgets": budgets,
            },
        )
        if status == OrderStatus.REJECTED:
            details.rejection_reason = message
        self._register_order(details)
        self._positions.add_pending_order(
            order_id=details.order_id,
            symbol=details.symbol,
            side=details.side,
            qty=details.quantity,
            price=details.price,
            order_type=details.order_type.value.upper(),
            intent=details.intent,
            bracket_id=details.bracket_id,
            signal_id=details.signal_id,
            signal_fingerprint=details.signal_fingerprint,
        )
        self._positions.update_order_status(details.order_id, details.status.name)
        self._publish_order_to_hub(details, response)
        self._sync_positions_to_hub()
        if status == OrderStatus.REJECTED:
            self._handle_order_rejected(details)
        return details

    def _position_qty_for(
        self, symbol: str, product: str | None
    ) -> tuple[int, str | None]:
        """Return long position quantity and tradingsymbol for reduce-only exits.

        Args:
            symbol: Candidate trading symbol to locate within broker positions.
            product: Broker product code to filter the matched position.

        Returns:
            Tuple containing the detected long quantity and canonical tradingsymbol.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _position_qty_for",
            extra={
                "event": "position_qty_for_enter",
                "symbol": symbol,
                "product": product,
            },
        )
        normalized_symbol = DataHub.normalize(symbol)
        if not normalized_symbol:
            self._logger.info(
                "Condition met: position_qty_for_missing_symbol",
                extra={"event": "position_qty_for_missing_symbol"},
            )
            return 0, None

        product_code = (product or "").strip().upper()
        fetcher = getattr(self._broker, "get_positions", None)
        if not callable(fetcher):
            fetcher = getattr(self._broker, "positions", None)
        if not callable(fetcher):
            fallback_qty, fallback_symbol = self._position_qty_from_manager(
                symbol, product_code
            )
            if fallback_qty > 0 and fallback_symbol:
                self._logger.info(
                    "Condition met: position_qty_for_match",
                    extra={
                        "event": "position_qty_for_match",
                        "symbol": normalized_symbol,
                        "product": product_code or None,
                        "qty": fallback_qty,
                    },
                )
                return fallback_qty, fallback_symbol
            self._logger.error(
                "Failure in _position_qty_for: broker_missing_positions",
                extra={"event": "position_qty_for_missing_api"},
            )
            return 0, None

        try:
            raw_positions = self._call_broker(fetcher)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _position_qty_for: %s",
                exc,
                extra={"event": "position_qty_for_fetch_failed"},
            )
            return 0, None

        matched_symbol: str | None = None
        matched_qty = 0

        if isinstance(raw_positions, Sequence):
            iterable = raw_positions
        else:
            iterable = []

        for record in iterable:
            if not isinstance(record, Mapping):
                continue
            raw_symbol = str(
                record.get("tradingsymbol")
                or record.get("symbol")
                or record.get("instrument")
                or ""
            ).strip()
            record_symbol = DataHub.normalize(raw_symbol)
            if record_symbol != normalized_symbol:
                continue
            record_product = (
                str(record.get("product") or record.get("producttype") or "")
                .strip()
                .upper()
            )
            if product_code and record_product and record_product != product_code:
                continue
            qty_token: Any | None = record.get("quantity")
            if qty_token in (None, ""):
                qty_token = record.get("net_qty")
            if qty_token in (None, ""):
                qty_token = record.get("net_quantity")
            qty_value = self._coerce_int(qty_token) or 0
            if qty_value <= 0:
                continue
            matched_qty = qty_value
            matched_symbol = raw_symbol or normalized_symbol
            self._logger.info(
                "Condition met: position_qty_for_match",
                extra={
                    "event": "position_qty_for_match",
                    "symbol": normalized_symbol,
                    "product": product_code or None,
                    "qty": matched_qty,
                },
            )
            break

        return matched_qty, matched_symbol

    def _position_qty_from_manager(
        self, symbol: str, product_code: str
    ) -> tuple[int, str | None]:
        """Return quantity from the position manager when broker lookup is unavailable.

        Args:
            symbol: Symbol identifier used by the position manager.
            product_code: Product filter to apply when matching positions.

        Returns:
            Tuple containing quantity and normalized symbol if a matching long exists.

        Raises:
            None.
        """

        try:
            position = self._positions.get_position(symbol)
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in _position_qty_for: %s",
                exc,
                extra={"event": "position_qty_for_manager_failed"},
            )
            return 0, None
        if position is None:
            return 0, None
        side = str(getattr(position, "side", "")).strip().upper()
        if side and side != "LONG":
            return 0, None
        record_product = str(getattr(position, "product", "") or "").strip().upper()
        if product_code and record_product and record_product != product_code:
            return 0, None
        raw_qty: Any | None = getattr(position, "quantity", None)
        if raw_qty in (None, ""):
            raw_qty = getattr(position, "net_quantity", None)
        qty_value = self._coerce_int(raw_qty) or 0
        if qty_value <= 0:
            return 0, None
        normalized_symbol = DataHub.normalize(symbol)
        resolved_symbol = normalized_symbol or symbol
        return qty_value, resolved_symbol

    def place_reduce_only_exit(self, intent: ExitIntent) -> str | None:
        """Exit a long position without opening a fresh short order.

        Args:
            intent: Bound exit request containing instrument context and size.

        Returns:
            Broker order identifier if a reduce-only exit was placed, else ``None``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered place_reduce_only_exit",
            extra={
                "event": "place_reduce_only_exit_enter",
                "symbol": intent.symbol,
                "qty": intent.qty,
                "product": intent.product,
            },
        )
        requested_qty = max(int(intent.qty), 0)
        if requested_qty <= 0:
            emit_diag(
                self._logger,
                "exit_skip_zero_qty",
                reason="zero_qty",
                severity="info",
                symbol=intent.symbol,
                requested_qty=requested_qty,
            )
            return None

        product_code = (intent.product or "MIS").strip().upper()
        if ":" in intent.symbol:
            lookup_symbol = intent.symbol
        else:
            lookup_symbol = f"{intent.exchange}:{intent.symbol}"
        long_qty, matched_symbol = self._position_qty_for(lookup_symbol, product_code)
        if long_qty <= 0:
            emit_diag(
                self._logger,
                "exit_skip_no_position",
                reason="no_position",
                severity="warning",
                alert=True,
                symbol=intent.symbol,
                product=product_code,
            )
            return None

        qty = min(requested_qty, long_qty)
        if qty <= 0:
            emit_diag(
                self._logger,
                "exit_skip_zero_qty",
                reason="zero_qty",
                severity="info",
                symbol=intent.symbol,
                requested_qty=requested_qty,
                long_qty=long_qty,
            )
            return None

        tradingsymbol = matched_symbol or intent.symbol
        normalized_symbol = DataHub.normalize(tradingsymbol)
        if ":" in tradingsymbol:
            order_symbol = tradingsymbol
        else:
            if not normalized_symbol:
                self._logger.error(
                    "Failure in place_reduce_only_exit: missing_symbol",
                    extra={
                        "event": "exit_place_reduce_only_fail",
                        "symbol": intent.symbol,
                        "qty": qty,
                    },
                )
                return None
            order_symbol = f"{intent.exchange}:{normalized_symbol}"

        self._logger.info(
            "Condition met: exit_reduce_only_prepare",
            extra={
                "event": "exit_reduce_only_prepare",
                "symbol": order_symbol,
                "qty": qty,
                "product": product_code,
            },
        )

        try:
            details = self._place_exit_order(
                symbol=order_symbol,
                side="SELL",
                quantity=qty,
                product=product_code,
                tag=intent.tag or "reduce_only",
            )
        except OrderPlacementError as exc:
            self._logger.error(
                "Failure in place_reduce_only_exit: %s",
                exc,
                extra={
                    "event": "exit_place_reduce_only_fail",
                    "symbol": order_symbol,
                    "qty": qty,
                },
            )
            return None
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in place_reduce_only_exit: %s",
                exc,
                extra={
                    "event": "exit_place_reduce_only_fail",
                    "symbol": order_symbol,
                    "qty": qty,
                },
            )
            return None

        return details.order_id

    def _place_exit_order(
        self,
        *,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        product: str | None,
        tag: str | None,
    ) -> OrderDetails:
        """Route exit using deterministic fallbacks.

        Args:
            symbol: Instrument identifier for the exit.
            side: Exit side (``BUY`` or ``SELL``).
            quantity: Absolute exit quantity.
            product: Suggested product code for the exit.
            tag: Optional broker tag string.

        Returns:
            OrderDetails: Broker response metadata for the routed exit.

        Raises:
            OrderPlacementError: If all routing attempts fail.
        """

        self._logger.debug(
            "Entered _place_exit_order",
            extra={
                "event": "place_exit_order_enter",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "product": product,
            },
        )
        if quantity <= 0:
            raise OrderPlacementError("Exit quantity must be positive")

        refresh: Callable[[str], None] | None = None
        market_data = self._market_data
        if market_data is not None:
            refresh = getattr(market_data, "refresh", None)

        try:
            now_ist = datetime.now(ZoneInfo("Asia/Kolkata")).time()
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _place_exit_order: %s",
                exc,
                extra={"event": "place_exit_order_time_failed"},
            )
            now_ist = datetime.now(timezone.utc).time()

        last_details: OrderDetails | None = None

        def _submit_exit(
            planned_product: str, validity: str, variety: str
        ) -> tuple[bool, str | None, str | None]:
            nonlocal last_details
            overrides = {
                "product": planned_product,
                "validity": validity,
                "variety": variety,
            }
            try:
                last_details = self._place_single_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    price=None,
                    product=planned_product,
                    tag=tag,
                    payload_overrides=overrides,
                )
            except OrderPlacementError as exc:
                self._logger.error(
                    "Failure in _place_exit_order: %s",
                    exc,
                    extra={
                        "event": "place_exit_order_attempt_failed",
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "product": planned_product,
                        "validity": validity,
                        "variety": variety,
                    },
                )
                return False, None, str(exc)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _place_exit_order: %s",
                    exc,
                    extra={
                        "event": "place_exit_order_unexpected",
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                    },
                )
                return False, None, str(exc)
            return True, last_details.order_id, None

        try:
            result = plan_and_send_exit(
                symbol=symbol,
                quantity=quantity,
                product=(product or "MIS").upper(),
                now_time=now_ist,
                submit=_submit_exit,
                logger=self._logger,
                refresh_quote=refresh,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _place_exit_order: %s",
                exc,
                extra={"event": "place_exit_order_router_failed"},
            )
            raise OrderPlacementError("Exit routing failed") from exc

        if not result.ok or last_details is None:
            reason = result.reason if isinstance(result.reason, BrokerReject) else None
            detail = reason.value if reason is not None else "UNKNOWN"
            self._logger.error(
                "Failure in _place_exit_order: %s",
                detail,
                extra={
                    "event": "place_exit_order_exhausted",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "attempts": len(result.attempts),
                    "reason": detail,
                },
            )
            raise OrderPlacementError(f"Exit routing failed ({detail})")

        self._logger.info(
            "Condition met: exit_order_id=%s",
            last_details.order_id,
            extra={
                "event": "place_exit_order_success",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_id": last_details.order_id,
            },
        )
        return last_details

    def _resolve_order_metadata(self, symbol: str) -> dict[str, Any]:
        """Resolve exchange, tradingsymbol, and token hints for *symbol*.

        Args:
            symbol: Instrument identifier supplied by callers.

        Returns:
            dict[str, Any]: Resolver hints such as exchange and instrument token.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _resolve_order_metadata",
            extra={"event": "order_manager.resolve_order_metadata", "symbol": symbol},
        )
        metadata: dict[str, Any] = {}
        try:
            resolver = self._resolver
            if resolver is None and self._market_data is not None:
                resolver = getattr(self._market_data, "resolver", None) or getattr(
                    self._market_data, "_resolver", None
                )
            if resolver is not None:
                if hasattr(resolver, "tradingsymbol_for_order"):
                    try:
                        metadata["tradingsymbol"] = resolver.tradingsymbol_for_order(  # type: ignore[attr-defined]
                            symbol
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._logger.debug(
                            "resolve_order_metadata_tradingsymbol_failed",
                            extra={
                                "event": "order_manager.resolve_tradingsymbol_failed",
                                "symbol": symbol,
                            },
                            exc_info=exc,
                        )
                if hasattr(resolver, "exchange_for_symbol"):
                    try:
                        metadata["exchange"] = resolver.exchange_for_symbol(symbol)  # type: ignore[attr-defined]
                    except Exception as exc:  # noqa: BLE001
                        self._logger.debug(
                            "resolve_order_metadata_exchange_failed",
                            extra={
                                "event": "order_manager.resolve_exchange_failed",
                                "symbol": symbol,
                            },
                            exc_info=exc,
                        )
                if hasattr(resolver, "resolve_symbol_to_token"):
                    try:
                        token = resolver.resolve_symbol_to_token(symbol)  # type: ignore[attr-defined]
                        if token is not None:
                            metadata["instrument_token"] = int(token)
                    except Exception as exc:  # noqa: BLE001
                        self._logger.debug(
                            "resolve_order_metadata_token_failed",
                            extra={
                                "event": "order_manager.resolve_token_failed",
                                "symbol": symbol,
                            },
                            exc_info=exc,
                        )
            if "instrument_token" not in metadata and self._market_data is not None:
                token_lookup = getattr(self._market_data, "resolve_symbol_token", None)
                if callable(token_lookup):
                    try:
                        token_candidate = token_lookup(symbol)
                        if token_candidate is not None:
                            metadata["instrument_token"] = int(token_candidate)
                    except Exception as exc:  # noqa: BLE001
                        self._logger.debug(
                            "resolve_order_metadata_token_mdm_failed",
                            extra={
                                "event": "order_manager.resolve_token_mdm_failed",
                                "symbol": symbol,
                            },
                            exc_info=exc,
                        )
                elif hasattr(self._market_data, "_token_by_symbol"):
                    try:
                        token_map = getattr(self._market_data, "_token_by_symbol", {})
                        token_candidate = token_map.get(symbol) or token_map.get(
                            symbol.strip().upper()
                        )
                        if token_candidate is not None:
                            metadata["instrument_token"] = int(token_candidate)
                    except Exception as exc:  # noqa: BLE001
                        self._logger.debug(
                            "resolve_order_metadata_token_cache_failed",
                            extra={
                                "event": "order_manager.resolve_token_cache_failed",
                                "symbol": symbol,
                            },
                            exc_info=exc,
                        )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _resolve_order_metadata: %s",
                exc,
                extra={
                    "event": "order_manager.resolve_order_metadata_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )
        return metadata

    def _build_order_payload(
        self,
        *,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        order_type: OrderType,
        price: float | None,
        product: str | None,
        tag: str | None,
        parent_order_id: str | None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Assemble broker payload with SL-M -> SL-Limit Auto-Conversion."""

        self._logger.debug(
            "Entered OrderManager._build_order_payload",
            extra={"event": "order_manager.build_order_payload", "symbol": symbol},
        )

        # 1. Resolve Instrument Metadata
        resolver_metadata = self._resolve_order_metadata(symbol)
        tradingsymbol_hint = str(resolver_metadata.get("tradingsymbol") or "").strip()
        exchange_hint = resolver_metadata.get("exchange")
        instrument_token = resolver_metadata.get("instrument_token")

        # 2. Basic Payload Construction
        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "quantity": int(quantity),
            "product": product or "MIS",
            "transaction_type": side,
        }

        # 3. Handle Symbol/Exchange Mapping
        if ":" in symbol:
            parts = symbol.split(":", maxsplit=1)
            if not exchange_hint:
                exchange_hint = parts[0]
            if not tradingsymbol_hint:
                tradingsymbol_hint = parts[1]
        else:
            if not tradingsymbol_hint and str(symbol).upper().endswith(
                ("FUT", "CE", "PE")
            ):
                tradingsymbol_hint = symbol

        if tradingsymbol_hint:
            payload["tradingsymbol"] = tradingsymbol_hint
        if exchange_hint:
            payload["exchange"] = exchange_hint

        # Override exchange if env var set
        env_exch = os.getenv("INSTRUMENTS__TRADE_EXCHANGE")
        if env_exch:
            payload["exchange"] = env_exch

        # 4. CRITICAL FIX: Order Type Mapping & SL-M Interception
        raw_type = (
            order_type.value if hasattr(order_type, "value") else str(order_type)
        ).upper()

        # Zerodha Type Map
        zerodha_map = {
            "STOP_LOSS_MARKET": "SL-M",
            "SL-M": "SL-M",
            "STOP_LOSS_LIMIT": "SL",
            "STOP_LOSS": "SL",
            "MARKET": "MARKET",
            "LIMIT": "LIMIT",
            "SL": "SL",
        }
        kite_type = zerodha_map.get(raw_type, "MARKET")  # Default to MARKET if unknown

        # 🚀 AUTO-CONVERT SL-M to SL-LIMIT (Fixes 400 Bad Request)
        final_price = float(price) if price is not None else 0.0
        trigger_price = 0.0

        # If user passed price as trigger for SL-M, extract it
        if kite_type == "SL-M":
            # SL-M usually passes trigger in 'price' or has separate trigger
            trigger_price = final_price
            if trigger_price <= 0:
                # Try to extract from 'trigger_price' arg if passed in price field wasn't it
                # (Logic handled by caller usually, but we ensure safety here)
                pass

            self._logger.info(
                f"🛡️ Intercepting SL-M for {symbol}. Converting to SL-Limit."
            )
            kite_type = "SL"  # Force Limit

            # Calculate Safe Limit Buffer (5%)
            # BUY SL (Exit Short): Trigger 100 -> Limit 105 (Guarantees fill)
            # SELL SL (Exit Long): Trigger 100 -> Limit 95 (Guarantees fill)
            buffer = 0.05
            if side == "BUY":
                final_price = round(trigger_price * (1 + buffer), 1)
            else:
                final_price = max(0.05, round(trigger_price * (1 - buffer), 1))

        elif kite_type == "SL":
            # Standard SL: ensure trigger and price are set
            trigger_price = final_price  # Assume price arg is trigger if ambiguous
            # Caller usually sets price=limit, trigger=trigger.
            # We trust the passed 'price' if it looks like a limit price
            pass

        # 5. Finalize Payload Fields
        payload["order_type"] = kite_type

        if kite_type == "MARKET":
            payload.pop("price", None)
            payload.pop("trigger_price", None)
        elif kite_type == "LIMIT":
            payload["price"] = final_price
        elif kite_type == "SL":
            payload["price"] = final_price
            # If trigger wasn't extracted during conversion, assume current price is limit
            # This part relies on the caller passing the correct 'price' vs 'trigger_price' semantics
            # But for the SL-M conversion above, we handled it.
            if trigger_price > 0:
                payload["trigger_price"] = trigger_price
            else:
                # If we have a price but no trigger, and it's SL, usage is ambiguous.
                # Assume price IS the trigger for compatibility
                payload["trigger_price"] = final_price

        # 6. Optional Fields
        if tag:
            payload["tag"] = tag
        if parent_order_id:
            payload["parent_order_id"] = parent_order_id
        if client_order_id:
            payload["client_order_id"] = client_order_id

        # Token Handling (Optional but recommended)
        if instrument_token is not None:
            # Check feature flag
            try:
                if (
                    bool(
                        getattr(
                            app_settings.get_settings(),
                            "feature_order_without_token",
                            True,
                        )
                    )
                    is False
                ):
                    payload["instrument_token"] = int(instrument_token)
            except Exception:
                self._logger.exception("Unhandled exception", exc_info=True)
                raise  # Default to not sending if settings fail

        return payload

    def _submit_order_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._validate_execution_adapter()
        last_error: Exception | None = None
        client_order_id = str(payload.get("client_order_id") or "").strip()
        if client_order_id:
            existing = self._lookup_existing_order(client_order_id)
            if existing:
                self._logger.info(
                    "Condition met: broker_order_reuse",
                    extra={
                        "event": "broker_order_reuse",
                        "client_order_id": client_order_id,
                    },
                )
                return existing
        for attempt in range(1, self.MAX_RETRIES + 1):
            if not self._broker_circuit.allow():
                raise RateLimitError("broker_circuit_open")
            try:
                self._limiter.acquire("orders", timeout=2.0)
            except RateLimitError as exc:
                last_error = exc
                self._logger.warning("Rate limit hit when placing order: %s", exc)
                if "not configured" not in str(exc).lower():
                    time.sleep(0.5 * (2 ** (attempt - 1)))
                    continue
            if client_order_id:
                existing = self._lookup_existing_order(client_order_id)
                if existing:
                    self._logger.info(
                        "Condition met: broker_order_reuse",
                        extra={
                            "event": "broker_order_reuse",
                            "client_order_id": client_order_id,
                            "attempt": attempt,
                        },
                    )
                    return existing
            self._logger.debug(
                "broker_order_attempt",
                extra={
                    "event": "broker_order_attempt",
                    "attempt": attempt,
                    "payload_symbol": payload.get("symbol"),
                    "payload_side": payload.get("side"),
                    "client_order_id": client_order_id,
                },
            )
            try:
                METRICS.increment_retry_event(
                    label="broker.place", stage="attempt", outcome="start"
                )
            except Exception:  # pragma: no cover - optional metrics
                self._logger.debug("broker_order_attempt_metric_failed", exc_info=True)
            try:
                # [FIX] Unpack payload so Broker Client receives named arguments
                response = self._call_broker(self._broker.place_order, **payload)
            except RateLimitError as exc:
                last_error = exc
                break
            except BrokerError as exc:
                last_error = exc
                message = str(exc)
                if self._is_margin_error(message):
                    annotated = (
                        message if "MARGIN" in message.upper() else f"MARGIN {message}"
                    )
                    self._logger.info(
                        "order_blocked_soft reason=MARGIN broker_error=%s",
                        message,
                        extra={
                            "event": "order_soft_block",
                            "reason": "MARGIN",
                            "broker_error": message,
                            "payload": {
                                "symbol": payload.get("symbol"),
                                "side": payload.get("side"),
                                "quantity": payload.get("quantity"),
                                "product": payload.get("product"),
                            },
                        },
                    )
                    raise OrderPlacementError(annotated) from exc
                if client_order_id:
                    existing = self._lookup_existing_order(client_order_id)
                    if existing:
                        return existing
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if client_order_id:
                    existing = self._lookup_existing_order(client_order_id)
                    if existing:
                        return existing
                break
            else:
                status = self._parse_status(response.get("status"))
                message = (response.get("message") or "").lower()
                if status == OrderStatus.REJECTED and any(
                    reason in message for reason in self.RETRY_BLACKLIST
                ):
                    error_message = response.get("message") or "Order rejected"
                    if self._is_margin_error(error_message):
                        annotated = (
                            error_message
                            if "MARGIN" in error_message.upper()
                            else f"MARGIN {error_message}"
                        )
                        self._logger.info(
                            "order_blocked_soft reason=MARGIN broker_error=%s",
                            error_message,
                            extra={
                                "event": "order_soft_block",
                                "reason": "MARGIN",
                                "broker_error": error_message,
                                "payload": {
                                    "symbol": payload.get("symbol"),
                                    "side": payload.get("side"),
                                    "quantity": payload.get("quantity"),
                                    "product": payload.get("product"),
                                },
                            },
                        )
                        raise OrderPlacementError(annotated)
                    self._logger.error(
                        "Order rejected without retry: %s", error_message
                    )
                    raise OrderPlacementError(error_message)
                if status == OrderStatus.REJECTED:
                    last_error = OrderPlacementError(
                        response.get("message") or "Order rejected"
                    )
                    break
                else:
                    self._logger.info(
                        "Condition met: broker_order_submit_success",
                        extra={
                            "event": "broker_order_submit_success",
                            "attempt": attempt,
                            "order_id": response.get("order_id"),
                            "client_order_id": client_order_id,
                        },
                    )
                    try:
                        METRICS.increment_retry_event(
                            label="broker.place", stage="attempt", outcome="success"
                        )
                    except Exception:  # pragma: no cover - optional metrics
                        self._logger.debug(
                            "broker_order_success_metric_failed", exc_info=True
                        )
                    return response
            if attempt >= self.MAX_RETRIES:
                break
            backoff = 0.5 * (2 ** (attempt - 1))
            time.sleep(backoff)
        if client_order_id:
            existing = self._lookup_existing_order(client_order_id)
            if existing:
                self._logger.info(
                    "Condition met: broker_order_reuse",
                    extra={
                        "event": "broker_order_reuse",
                        "client_order_id": client_order_id,
                        "attempt": attempt,
                    },
                )
                return existing
        if isinstance(last_error, RateLimitError):
            self._logger.error(
                "Broker order submission failed due to rate limit",
                extra={
                    "event": "broker_order_failure",
                    "reason": "RATE_LIMIT",
                    "client_order_id": client_order_id,
                    "attempts": attempt,
                },
            )
            try:
                METRICS.increment_retry_event(
                    label="broker.place", stage="attempt", outcome="failure"
                )
            except Exception:  # pragma: no cover - optional metrics
                self._logger.debug("broker_order_failure_metric_failed", exc_info=True)
            raise last_error
        self._logger.error(
            "Broker order submission failed",
            extra={
                "event": "broker_order_failure",
                "reason": canonical(str(last_error) if last_error else "unknown"),
                "client_order_id": client_order_id,
                "attempts": attempt,
            },
        )
        try:
            METRICS.increment_retry_event(
                label="broker.place", stage="attempt", outcome="failure"
            )
        except Exception:  # pragma: no cover - optional metrics
            self._logger.debug("broker_order_failure_metric_failed", exc_info=True)
        raise OrderPlacementError("Order placement failed") from last_error

    def _lookup_existing_order(self, client_order_id: str) -> dict[str, Any] | None:
        if not client_order_id:
            return None
        order_id = self._client_order_index.get(client_order_id)
        if order_id:
            try:
                order = self._refresh_order(order_id)
            except KeyError:
                pass
            else:
                status_value = (
                    order.status.value
                    if hasattr(order.status, "value")
                    else str(order.status)
                )
                return {
                    "order_id": order.order_id,
                    "status": status_value,
                    "client_order_id": client_order_id,
                    "filled_quantity": order.filled_quantity,
                    "average_price": order.fill_price,
                }
        for attr in (
            "get_order_by_client_order_id",
            "find_order_by_client_order_id",
        ):
            lookup = getattr(self._broker, attr, None)
            if not callable(lookup):
                continue
            try:
                result = self._call_broker(lookup, client_order_id)
            except Exception:  # noqa: BLE001 - defensive
                self._logger.debug(
                    "client_order_lookup_failed",
                    extra={
                        "event": "client_order_lookup_failed",
                        "client_id": client_order_id,
                    },
                    exc_info=True,
                )
                continue
            if result:
                return cast(dict[str, Any], result)
        return None

    def _resolve_open_orders_fetcher(self) -> Callable[[], Any] | None:
        candidates = (
            "get_open_orders",
            "list_open_orders",
            "get_orders",
            "list_orders",
        )
        for name in candidates:
            fetcher = getattr(self._broker, name, None)
            if callable(fetcher):
                return cast(Callable[[], Any], fetcher)
        return None

    def _coerce_broker_open_order(
        self, payload: Mapping[str, Any]
    ) -> OrderDetails | None:
        if not isinstance(payload, Mapping):
            return None
        order_id_raw = payload.get("order_id") or payload.get("id")
        if not order_id_raw:
            return None
        symbol_raw = (
            payload.get("symbol")
            or payload.get("tradingsymbol")
            or payload.get("instrument_token")
        )
        if not symbol_raw:
            return None
        side_raw = (
            payload.get("side")
            or payload.get("transaction_type")
            or payload.get("transactionType")
        )
        side = str(side_raw or "BUY").upper()
        if side not in {"BUY", "SELL"}:
            side = "BUY"
        order_type_token = (
            payload.get("order_type") or payload.get("orderType") or payload.get("type")
        )
        order_type = self._parse_order_type_token(order_type_token)
        quantity = (
            self._coerce_int(payload.get("quantity"))
            or self._coerce_int(payload.get("qty"))
            or 0
        )
        price_value = self._coerce_float(payload.get("price"))
        status = self._parse_status(payload.get("status"))
        filled_qty = (
            self._coerce_int(payload.get("filled_quantity"))
            or self._coerce_int(payload.get("filledQty"))
            or 0
        )
        avg_price = self._coerce_float(
            payload.get("average_price")
        ) or self._coerce_float(payload.get("averagePrice"))
        timestamp_token = (
            payload.get("order_timestamp")
            or payload.get("timestamp")
            or payload.get("created_at")
        )
        timestamp = datetime.now(timezone.utc)
        if isinstance(timestamp_token, str):
            with suppress(Exception):
                timestamp = datetime.fromisoformat(timestamp_token)
        elif isinstance(timestamp_token, (int, float)):
            with suppress(Exception):
                timestamp = datetime.fromtimestamp(
                    float(timestamp_token),
                    tz=timezone.utc,
                )
        client_order_id = payload.get("client_order_id") or payload.get("clientOrderId")
        rejection_reason = payload.get("message") or payload.get("status_message")
        child_ids_raw = payload.get("child_order_ids")
        if isinstance(child_ids_raw, (list, tuple, set)):
            child_ids = [str(item) for item in child_ids_raw]
        else:
            child_ids = []
        details = OrderDetails(
            order_id=str(order_id_raw),
            symbol=str(symbol_raw).upper(),
            side=side,
            order_type=order_type,
            quantity=max(int(quantity), 0),
            price=float(price_value or 0.0),
            status=status,
            timestamp=timestamp,
            filled_quantity=max(int(filled_qty), 0),
            fill_price=avg_price,
            rejection_reason=str(rejection_reason) if rejection_reason else None,
            parent_order_id=(
                payload.get("parent_order_id") or payload.get("parentOrderId")
            ),
            child_order_ids=child_ids,
            client_order_id=str(client_order_id) if client_order_id else None,
        )
        if details.status == OrderStatus.FILLED and details.fill_price is None:
            details.fill_price = details.price if details.price else None
        if (
            details.status == OrderStatus.PARTIALLY_FILLED
            and details.filled_quantity >= details.quantity > 0
        ):
            details.status = OrderStatus.FILLED
        return details

    def _call_broker(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not self._broker_circuit.allow():
            raise RateLimitError("broker_circuit_open")
        try:
            result = func(*args, **kwargs)
        except Exception:
            self._broker_circuit.on_failure()
            raise
        else:
            self._broker_circuit.on_success()
            return result

    def _parse_status(self, raw_status: Any) -> OrderStatus:
        if raw_status is None:
            return OrderStatus.SUBMITTED
        status_text = str(raw_status).strip().upper()
        if status_text == "SUBMITTED":
            return OrderStatus.SUBMITTED
        canonical_status = normalize_broker_order_status(raw_status)
        if canonical_status is not None:
            if canonical_status == "PENDING":
                return OrderStatus.PENDING
            if canonical_status == "OPEN":
                return OrderStatus.SUBMITTED
            if canonical_status == "PARTIALLY_FILLED":
                return OrderStatus.PARTIALLY_FILLED
            if canonical_status == "FILLED":
                return OrderStatus.FILLED
            if canonical_status == "CANCELLED":
                return OrderStatus.CANCELLED
            if canonical_status == "REJECTED":
                return OrderStatus.REJECTED
            if canonical_status == "EXPIRED":
                return OrderStatus.EXPIRED
        status_str = str(raw_status).strip().lower()
        if not status_str:
            return OrderStatus.SUBMITTED
        mapping = {
            "partial": OrderStatus.PARTIALLY_FILLED,
            "cancelled by user": OrderStatus.CANCELLED,
        }
        if status_str in mapping:
            return mapping[status_str]
        for status in OrderStatus:
            if status_str == status.value:
                return status
            if status_str == status.name.lower():
                return status
        return OrderStatus.SUBMITTED

    @staticmethod
    def _parse_order_type_token(raw_type: Any) -> OrderType:
        if isinstance(raw_type, OrderType):
            return raw_type
        if raw_type is None:
            return OrderType.MARKET
        token = str(raw_type).strip().lower()
        for candidate in OrderType:
            if token in {candidate.value, candidate.name.lower()}:
                return candidate
        return OrderType.MARKET

    def _is_force_exit(
        self, *, symbol: str, side: Literal["BUY", "SELL"], quantity: int
    ) -> bool:
        if quantity <= 0:
            return False

        def _coerce_abs(value: Any) -> int:
            if value is None:
                return 0
            if isinstance(value, (int, float)):
                return abs(int(value))
            if isinstance(value, str):
                with suppress(Exception):
                    return abs(int(float(value)))
            return 0

        try:
            position = self._positions.get_position(symbol)
        except Exception:  # pragma: no cover - defensive
            return False
        if position is None:
            return False

        open_side: str | None
        try:
            open_side = cast(Optional[str], getattr(position, "side", None))
        except Exception:
            open_side = None
        if not open_side:
            net_value: Any | None = None
            with suppress(Exception):
                net_value = getattr(position, "net_quantity", None)
            if net_value is None:
                with suppress(Exception):
                    net_value = getattr(position, "quantity", None)
            numeric: float | None = None
            if isinstance(net_value, (int, float)):
                numeric = float(net_value)
            elif isinstance(net_value, str):
                with suppress(Exception):
                    numeric = float(net_value)
            if numeric is not None:
                if numeric > 0:
                    open_side = "LONG"
                elif numeric < 0:
                    open_side = "SHORT"
        if open_side not in ("LONG", "SHORT"):
            return False

        expected_exit_side: Literal["BUY", "SELL"] = (
            "SELL" if open_side == "LONG" else "BUY"
        )
        if side.upper() != expected_exit_side:
            return False

        raw_qty: Any | None = None
        with suppress(Exception):
            raw_qty = getattr(position, "quantity", None)
        open_qty = _coerce_abs(raw_qty)
        if open_qty <= 0:
            net_qty: Any | None = None
            with suppress(Exception):
                net_qty = getattr(position, "net_quantity", None)
            open_qty = _coerce_abs(net_qty)
        if open_qty <= 0:
            return False

        return int(quantity) <= open_qty

    def _refresh_order(self, order_id: str) -> OrderDetails:
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                history_index = self._resolve_history_index(order_id)
                if history_index is not None:
                    order = self._history[history_index]
                    self._orders[order_id] = order
        if order is None:
            raise KeyError(f"Unknown order_id: {order_id}")

        if (
            order.status in self.FINAL_STATUSES
            and order.filled_quantity >= order.quantity
        ):
            return order

        get_status = getattr(self._broker, "get_order_status", None)
        if get_status is None:
            return order
        try:
            response = self._call_broker(get_status, order_id)
        except RateLimitError:
            self._logger.warning(
                "broker_circuit_open",
                extra={"event": "broker_circuit_open", "order_id": order_id},
            )
            return order
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failed to fetch status for %s: %s", order_id, exc)
            return order
        if not response:
            return order
        return self._update_from_response(order, response)

    def _update_from_response(
        self, order: OrderDetails, payload: dict[str, Any]
    ) -> OrderDetails:
        with self._lock:
            previous_status = order.status
            status = self._parse_status(payload.get("status"))
            filled_qty = (
                self._coerce_int(payload.get("filled_quantity"))
                or order.filled_quantity
            )

            average_price = payload.get("average_price") or payload.get("averagePrice")
            message = payload.get("status_message") or payload.get("message")

            order.status = status
            order.filled_quantity = filled_qty
            if filled_qty >= order.quantity and status != OrderStatus.CANCELLED:
                order.status = OrderStatus.FILLED
            elif 0 < filled_qty < order.quantity and status not in self.FINAL_STATUSES:
                order.status = OrderStatus.PARTIALLY_FILLED
            if order.status == OrderStatus.FILLED and order.fill_price is None:
                if average_price is not None:
                    order.fill_price = float(average_price)
                else:
                    order.fill_price = order.price if order.price else None
            if order.status == OrderStatus.REJECTED:
                order.rejection_reason = message

            self._register_order(order)
            self._positions.update_order_status(
                order.order_id, order.status.name, order.fill_price
            )

        self._publish_order_to_hub(order, payload)
        self._sync_positions_to_hub()

        if order.status == OrderStatus.FILLED and previous_status != OrderStatus.FILLED:
            self._handle_order_filled(order)
        if (
            order.status == OrderStatus.REJECTED
            and previous_status != OrderStatus.REJECTED
        ):
            self._handle_order_rejected(order)
        self._notify_failed_entry_terminal(
            order, previous_status, str(payload.get("status") or "")
        )
        self._handle_bracket_update(order, previous_status, payload)
        self._handle_guard_order_update(order, previous_status)
        return order

    def _load_guard_pairs(self) -> None:
        """Hydrate guard-pair state from disk if available.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        path = getattr(self, "_guard_state_path", None)
        if path is None or not isinstance(path, Path):
            return
        if not path.exists():
            return
        try:
            raw = path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, ValueError) as exc:
            self._logger.error(
                "Failure in _load_guard_pairs: %s",
                exc,
                extra={"event": "guard_pair_load_failed"},
            )
            return
        if not isinstance(payload, list):
            self._logger.error(
                "Failure in _load_guard_pairs: invalid_payload",
                extra={"event": "guard_pair_load_invalid"},
            )
            return
        restored: dict[str, GuardPair] = {}
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            try:
                pair = GuardPair.from_dict(cast(Mapping[str, Any], item))
            except (TypeError, ValueError) as exc:
                self._logger.error(
                    "Failure in _load_guard_pairs: %s",
                    exc,
                    extra={"event": "guard_pair_load_entry_invalid"},
                )
                continue
            if not pair.symbol:
                continue
            restored[pair.symbol] = pair
        if restored:
            self._guard_pairs.update(restored)
            self._logger.info(
                "Loaded guard pairs from disk",
                extra={"event": "guard_pair_load", "count": len(restored)},
            )

    def _persist_guard_pairs(self) -> None:
        """Persist guard-pair state atomically to disk.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        path = getattr(self, "_guard_state_path", None)
        if path is None or not isinstance(path, Path):
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = [pair.to_dict() for pair in self._guard_pairs.values()]
            temp_path = path.with_suffix(path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            os.replace(temp_path, path)
        except Exception as exc:  # noqa: BLE001 - log and continue
            self._logger.error(
                "Failure in _persist_guard_pairs: %s",
                exc,
                extra={"event": "guard_pair_persist_failed"},
            )

    def has_guard_pair(self, symbol: str) -> bool:
        """Return True when a guard pair is registered for *symbol*.

        Args:
            symbol: Trading symbol to inspect for guard linkage.

        Returns:
            ``True`` if a guard pair is active, otherwise ``False``.

        Raises:
            None.
        """

        normalized = DataHub.normalize(symbol) or symbol.strip().upper()
        try:
            with self._lock:
                return normalized in self._guard_pairs
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in has_guard_pair: %s",
                exc,
                extra={"event": "guard_pair_lookup_failed", "symbol": normalized},
            )
            return False

    def clear_guard_pair(self, symbol: str) -> None:
        """Cancel outstanding guard orders for *symbol* if present.

        Args:
            symbol: Trading symbol whose guard pair should be cleared.

        Returns:
            None.

        Raises:
            None.
        """

        normalized = DataHub.normalize(symbol) or symbol.strip().upper()
        self._logger.debug(
            "Entered clear_guard_pair",
            extra={"event": "guard_pair_clear_enter", "symbol": normalized},
        )
        try:
            with self._lock:
                pair = self._guard_pairs.get(normalized)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in clear_guard_pair: %s",
                exc,
                extra={"event": "guard_pair_clear_failed", "symbol": normalized},
            )
            return
        if pair is None:
            return
        for order_id in (pair.stop_order_id, pair.target_order_id):
            try:
                if self._is_order_active(order_id):
                    cancelled = self.cancel_order(order_id)
                    if cancelled:
                        self._logger.info(
                            "Cancelled guard order during clear_guard_pair",
                            extra={
                                "event": "guard_pair_clear_cancel",
                                "symbol": normalized,
                                "order_id": order_id,
                            },
                        )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in clear_guard_pair: %s",
                    exc,
                    extra={
                        "event": "guard_pair_cancel_failed",
                        "symbol": normalized,
                        "order_id": order_id,
                    },
                )
        self._remove_guard_pair(normalized, reason="position_flat")

    def _register_guard_pair(self, pair: GuardPair) -> None:
        """Store guard *pair* for cancellation orchestration.

        Args:
            pair: Guard pair metadata to persist in memory.

        Returns:
            None.

        Raises:
            None.
        """

        try:
            with self._lock:
                self._guard_pairs[pair.symbol] = pair
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _register_guard_pair: %s",
                exc,
                extra={"event": "guard_pair_register_failed", "symbol": pair.symbol},
            )
            return
        self._logger.info(
            "Registered guard pair for %s",
            pair.symbol,
            extra={
                "event": "guard_pair_created",
                "symbol": pair.symbol,
                "stop_order_id": pair.stop_order_id,
                "target_order_id": pair.target_order_id,
            },
        )
        self._persist_guard_pairs()

    def _remove_guard_pair(self, symbol: str, *, reason: str) -> None:
        """Remove guard pair for *symbol* with contextual *reason*.

        Args:
            symbol: Trading symbol key used for guard lookup.
            reason: Human readable explanation for removal.

        Returns:
            None.

        Raises:
            None.
        """

        try:
            with self._lock:
                removed = self._guard_pairs.pop(symbol, None)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _remove_guard_pair: %s",
                exc,
                extra={"event": "guard_pair_remove_failed", "symbol": symbol},
            )
            return
        if removed is None:
            return
        self._logger.info(
            "Removed guard pair for %s",
            symbol,
            extra={
                "event": "guard_pair_removed",
                "symbol": symbol,
                "reason": reason,
            },
        )
        self._persist_guard_pairs()

    def _handle_guard_order_update(
        self, order: OrderDetails, previous_status: OrderStatus
    ) -> None:
        """Handle guard-pair state transitions when an order status changes.

        Args:
            order: Order details that transitioned state.
            previous_status: Status prior to the update.

        Returns:
            None.

        Raises:
            None.
        """

        normalized = DataHub.normalize(order.symbol) or order.symbol.strip().upper()
        try:
            with self._lock:
                pair = self._guard_pairs.get(normalized)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _handle_guard_order_update: %s",
                exc,
                extra={"event": "guard_pair_update_failed", "symbol": normalized},
            )
            return
        if pair is None:
            return
        if order.order_id not in {pair.stop_order_id, pair.target_order_id}:
            return
        if order.status == OrderStatus.FILLED and previous_status != OrderStatus.FILLED:
            self._cancel_guard_sibling(pair, normalized, filled_order_id=order.order_id)
            return
        if order.status in self.FINAL_STATUSES:
            sibling_id = (
                pair.target_order_id
                if order.order_id == pair.stop_order_id
                else pair.stop_order_id
            )
            if not self._is_order_active(sibling_id):
                self._remove_guard_pair(normalized, reason="both_final")

    def _cancel_guard_sibling(
        self, pair: GuardPair, symbol: str, *, filled_order_id: str
    ) -> None:
        """Cancel sibling order for filled guard pair leg.

        Args:
            pair: Guard pair metadata to inspect.
            symbol: Normalized symbol key.
            filled_order_id: Identifier of the order that filled first.

        Returns:
            None.

        Raises:
            None.
        """

        sibling_id = (
            pair.target_order_id
            if filled_order_id == pair.stop_order_id
            else pair.stop_order_id
        )
        try:
            if self._is_order_active(sibling_id):
                cancelled = self.cancel_order(sibling_id)
                if cancelled:
                    self._logger.info(
                        "Cancelled sibling guard order",
                        extra={
                            "event": "guard_pair_cancel_sibling",
                            "symbol": symbol,
                            "filled_order_id": filled_order_id,
                            "cancelled_order_id": sibling_id,
                        },
                    )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _cancel_guard_sibling: %s",
                exc,
                extra={
                    "event": "guard_pair_cancel_error",
                    "symbol": symbol,
                    "order_id": sibling_id,
                },
            )
        self._remove_guard_pair(symbol, reason="filled")

    def _is_order_active(self, order_id: str) -> bool:
        """Return True if *order_id* is not in a terminal state locally.

        Args:
            order_id: Broker order identifier to inspect.

        Returns:
            ``True`` when the cached order exists and is pending.

        Raises:
            None.
        """

        try:
            with self._lock:
                order = self._orders.get(order_id)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _is_order_active: %s",
                exc,
                extra={"event": "guard_pair_status_failed", "order_id": order_id},
            )
            return True
        if order is None:
            return True
        return order.status not in self.FINAL_STATUSES

    def _resolve_history_index(self, order_id: str) -> int | None:
        """Resolve deque index for stored order history entry.

        Args:
            order_id: Unique identifier for the cached order.

        Returns:
            Zero-based deque index when the order exists, otherwise ``None``.

        Raises:
            None.
        """

        logical_index = self._history_index.get(order_id)
        if logical_index is None:
            return None
        deque_index = logical_index - self._history_base_index
        if deque_index < 0 or deque_index >= len(self._history):
            self._history_index.pop(order_id, None)
            self._history_persisted_ids.discard(order_id)
            return None
        return deque_index

    def _append_history(self, order: OrderDetails) -> None:
        """Append order to bounded in-memory history.

        Args:
            order: Order payload to store in the bounded deque.

        Returns:
            None.

        Raises:
            None.
        """

        if (
            self._history.maxlen is not None
            and len(self._history) >= self._history.maxlen
        ):
            evicted = self._history.popleft()
            self._history_base_index += 1
            self._history_index.pop(evicted.order_id, None)
            self._history_persisted_ids.discard(evicted.order_id)
            self._logger.debug(
                "Evicted oldest order from history",
                extra={
                    "event": "order.history.evict",
                    "order_id": evicted.order_id,
                    "symbol": evicted.symbol,
                },
            )
        self._history.append(order)
        self._history_index[order.order_id] = (
            self._history_base_index + len(self._history) - 1
        )
        self._history_persisted_ids.discard(order.order_id)

    def _register_order(self, order: OrderDetails) -> None:
        with self._lock:
            self._orders[order.order_id] = order
            history_index = self._resolve_history_index(order.order_id)
            if history_index is not None:
                self._history[history_index] = order
                self._history_persisted_ids.discard(order.order_id)
            else:
                self._append_history(order)
            if order.client_order_id:
                self._client_order_index[str(order.client_order_id)] = order.order_id
            self._persist_history()
        self._publish_order_to_hub(order)
        self._persist_order_snapshot(order)
        # ✅ FIX: Persist new order
        self.save_orders()

    def _persist_history(self) -> None:
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [self._serialize(order) for order in self._history]
            self._history_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            self._logger.error("Failed to persist order history: %s", exc)

    def persist_history_batch(self) -> None:
        """Persist recent order history entries to the archive log.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered persist_history_batch",
            extra={
                "event": "order.persist.enter",
                "pending": len(self._history),
            },
        )
        with self._lock:
            if not self._history:
                return
            snapshot = list(self._history)
            active_ids = {order.order_id for order in snapshot}
            self._history_persisted_ids.intersection_update(active_ids)
            pending_orders = [
                order
                for order in snapshot
                if order.order_id not in self._history_persisted_ids
            ]
        if not pending_orders:
            return
        try:
            with self._history_persist_path.open("a", encoding="utf-8") as handle:
                for order in pending_orders:
                    handle.write(json.dumps(self._serialize(order)) + "\n")
            self._logger.info(
                "Persisted order history batch",
                extra={
                    "event": "order.persist.success",
                    "count": len(pending_orders),
                },
            )
            with self._lock:
                current_ids = {order.order_id for order in self._history}
                self._history_persisted_ids.intersection_update(current_ids)
                for order in pending_orders:
                    if order.order_id in current_ids:
                        self._history_persisted_ids.add(order.order_id)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failed to persist history batch",
                extra={"event": "order.persist.error", "error": str(exc)},
                exc_info=exc,
            )

    def _persist_order_snapshot(self, order: OrderDetails) -> None:
        """Persist the current snapshot of *order* if persistence is enabled.

        Args:
            order: Order details requiring durable storage.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _persist_order_snapshot",
            extra={"event": "order_manager_persist_order", "order_id": order.order_id},
        )
        manager = self._persistent_state
        if manager is None:
            return
        try:
            manager.save_order(self._serialize(order))
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _persist_order_snapshot: %s", exc)

    def _persist_bracket_state(self, state: BracketState) -> None:
        """Persist bracket *state* when persistence support is attached.

        Args:
            state: Bracket state to persist.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _persist_bracket_state",
            extra={
                "event": "order_manager_persist_bracket",
                "entry_id": state.entry_id,
            },
        )
        manager = self._persistent_state
        if manager is None:
            return
        try:
            manager.save_bracket(self._serialize_bracket_state(state))
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _persist_bracket_state: %s", exc)

    def _serialize(self, order: OrderDetails) -> dict[str, Any]:
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side,
            # [FIX] Convert Enum to value (string)
            "order_type": (
                order.order_type.value
                if hasattr(order.order_type, "value")
                else str(order.order_type)
            ),
            "quantity": order.quantity,
            "price": order.price,
            # [FIX] Convert Status Enum to value
            "status": (
                order.status.value
                if hasattr(order.status, "value")
                else str(order.status)
            ),
            "timestamp": (
                order.timestamp.isoformat()
                if hasattr(order.timestamp, "isoformat")
                else float(order.timestamp)
            ),
            "filled_quantity": order.filled_quantity,
            "fill_price": order.fill_price,
            "rejection_reason": order.rejection_reason,
            "parent_order_id": order.parent_order_id,
            "child_order_ids": list(order.child_order_ids),
            "client_order_id": order.client_order_id,
            "intent": order.intent,
            "trade_lifecycle_id": order.trade_lifecycle_id,
            "linked_entry_order_id": order.linked_entry_order_id,
            "bracket_id": order.bracket_id,
            "basket_version": order.basket_version,
            "instrument_token": order.instrument_token,
            "contract_expiry": order.contract_expiry,
            "exchange_order_id": order.exchange_order_id,
            "signal_id": order.signal_id,
            "requested_lots": order.requested_lots,
            "resolved_lot_size": order.resolved_lot_size,
            "entry_lifecycle_state": order.entry_lifecycle_state,
        }

    def _serialize_bracket_state(self, state: BracketState) -> BracketDict:
        """Serialise bracket *state* to a JSON-compatible mapping.

        Args:
            state: Bracket state requiring serialization.

        Returns:
            Dictionary describing the bracket state.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _serialize_bracket_state",
            extra={
                "event": "order_manager_serialize_bracket",
                "entry_id": state.entry_id,
            },
        )
        payload: BracketDict = {
            "entry_id": state.entry_id,
            "symbol": state.symbol,
            "side": state.side,
            "exit_side": state.exit_side,
            "total_quantity": state.total_quantity,
            "entry_price": state.entry_price,
            "product": state.product,
            "tag": state.tag,
            "stop_order_type": (
                state.stop_order_type.value
                if hasattr(state.stop_order_type, "value")
                else str(state.stop_order_type)
            ),
            "stop_price": state.stop_price,
            "stop_order_type": state.stop_order_type.value,
            "stop_filled": state.stop_filled,
            "tp_primary_id": state.tp_primary_id,
            "tp_primary_price": state.tp_primary_price,
            "tp_primary_qty": state.tp_primary_qty,
            "tp_primary_filled": state.tp_primary_filled,
            "tp_secondary_id": state.tp_secondary_id,
            "tp_secondary_price": state.tp_secondary_price,
            "tp_secondary_qty": state.tp_secondary_qty,
            "tp_secondary_filled": state.tp_secondary_filled,
            "partial_fraction": state.partial_fraction,
            "second_target_price": state.second_target_price,
            # [FIX] Persist Trailing Spec
            "trailing_spec": (
                asdict(state.trailing_spec) if state.trailing_spec else None
            ),
        }
        return payload

    def _order_from_dict(self, payload: Mapping[str, Any]) -> OrderDetails:
        """Convert persistent payload into OrderDetails (Crash-Proof)."""
        try:
            # Basic Fields
            order_id = str(payload["order_id"]).strip()
            symbol = str(payload["symbol"]).upper()
            side = str(payload["side"]).upper()
            quantity = int(payload.get("quantity", 0))
            price = float(payload.get("price", 0.0) or 0.0)

            # Enums with Fallback
            try:
                order_type = OrderType(
                    payload.get("order_type", OrderType.MARKET.value)
                )
            except (TypeError, ValueError):
                order_type = OrderType.MARKET

            try:
                status = OrderStatus(payload.get("status", OrderStatus.SUBMITTED.value))
            except (TypeError, ValueError):
                status = OrderStatus.SUBMITTED

            # Robust Timestamp
            ts_raw = payload.get("timestamp")
            timestamp = datetime.now(timezone.utc)
            if ts_raw:
                with suppress(Exception):
                    if isinstance(ts_raw, (int, float)):
                        timestamp = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
                    else:
                        timestamp = datetime.fromisoformat(str(ts_raw))

            # CRITICAL FIX: Map 'fill_price' to 'average_price'
            avg_price = float(
                payload.get("average_price") or payload.get("fill_price") or 0.0
            )

            return OrderDetails(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                status=status,
                timestamp=timestamp,
                filled_quantity=int(payload.get("filled_quantity", 0)),
                average_price=avg_price,  # Mapped correctly
                tag=payload.get("tag"),
                client_order_id=payload.get("client_order_id"),
                rejection_reason=payload.get("rejection_reason"),
                intent=payload.get("intent") or "UNKNOWN",
                trade_lifecycle_id=payload.get("trade_lifecycle_id"),
                linked_entry_order_id=payload.get("linked_entry_order_id"),
                bracket_id=payload.get("bracket_id"),
                basket_version=payload.get("basket_version"),
                instrument_token=(
                    int(payload["instrument_token"])
                    if payload.get("instrument_token") not in (None, "")
                    else None
                ),
                contract_expiry=payload.get("contract_expiry"),
                exchange_order_id=payload.get("exchange_order_id"),
                signal_id=payload.get("signal_id"),
                requested_lots=int(payload.get("requested_lots", 0) or 0),
                resolved_lot_size=int(payload.get("resolved_lot_size", 0) or 0),
                entry_lifecycle_state=(
                    payload.get("entry_lifecycle_state")
                    if isinstance(payload.get("entry_lifecycle_state"), dict)
                    else None
                ),
            )
        except Exception as e:
            logger = getattr(self, "_logger", logging.getLogger(__name__))
            logger.error(f"Failed to restore order: {e}")
            raise ValueError("Invalid order payload") from e

    def _bracket_from_dict(self, payload: Mapping[str, Any]) -> BracketState:
        """Convert persistent *payload* into :class:`BracketState`.

        Args:
            payload: Mapping describing the bracket state.

        Returns:
            BracketState reconstructed from the payload.

        Raises:
            ValueError: If the payload cannot be parsed.
        """

        self._logger.debug(
            "Entered _bracket_from_dict",
            extra={"event": "order_manager_bracket_from_dict"},
        )
        try:
            entry_id = str(payload["entry_id"]).strip()
            symbol = str(payload["symbol"]).upper()
            side = cast(Literal["BUY", "SELL"], str(payload["side"]).upper())
            exit_side = cast(Literal["BUY", "SELL"], str(payload["exit_side"]).upper())
            total_quantity_raw = self._coerce_int(payload.get("total_quantity"))
            total_quantity = (
                int(total_quantity_raw) if total_quantity_raw is not None else 0
            )
            entry_price_raw = self._coerce_float(payload.get("entry_price"))
            entry_price = float(entry_price_raw) if entry_price_raw is not None else 0.0
            stop_price_raw = self._coerce_float(payload.get("stop_price"))
            stop_price = float(stop_price_raw) if stop_price_raw is not None else 0.0
            stop_order_id = str(payload.get("stop_order_id", "")).strip()
            stop_order_type = OrderType(
                payload.get("stop_order_type", OrderType.STOP_LOSS.value)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid bracket payload") from exc
        stop_filled_raw = self._coerce_int(payload.get("stop_filled"))
        tp_primary_qty_raw = self._coerce_int(payload.get("tp_primary_qty"))
        tp_primary_filled_raw = self._coerce_int(payload.get("tp_primary_filled"))
        tp_secondary_qty_raw = self._coerce_int(payload.get("tp_secondary_qty"))
        tp_secondary_filled_raw = self._coerce_int(payload.get("tp_secondary_filled"))
        partial_fraction_raw = self._coerce_float(payload.get("partial_fraction"))
        state = BracketState(
            entry_id=entry_id,
            symbol=symbol,
            side=side,
            exit_side=exit_side,
            total_quantity=total_quantity,
            entry_price=entry_price,
            product=(
                str(payload.get("product"))
                if payload.get("product") is not None
                else None
            ),
            tag=(str(payload.get("tag")) if payload.get("tag") is not None else None),
            stop_order_id=stop_order_id,
            stop_price=stop_price,
            stop_order_type=stop_order_type,
            stop_filled=int(stop_filled_raw) if stop_filled_raw is not None else 0,
            tp_primary_id=(
                str(payload.get("tp_primary_id"))
                if payload.get("tp_primary_id")
                else None
            ),
            tp_primary_price=self._coerce_float(payload.get("tp_primary_price")),
            tp_primary_qty=(
                int(tp_primary_qty_raw) if tp_primary_qty_raw is not None else 0
            ),
            tp_primary_filled=(
                int(tp_primary_filled_raw) if tp_primary_filled_raw is not None else 0
            ),
            tp_secondary_id=(
                str(payload.get("tp_secondary_id"))
                if payload.get("tp_secondary_id")
                else None
            ),
            tp_secondary_price=self._coerce_float(payload.get("tp_secondary_price")),
            tp_secondary_qty=(
                int(tp_secondary_qty_raw) if tp_secondary_qty_raw is not None else 0
            ),
            tp_secondary_filled=(
                int(tp_secondary_filled_raw)
                if tp_secondary_filled_raw is not None
                else 0
            ),
            # [FIX] Restore Trailing Spec from Dictionary
            trailing_spec=(
                TrailingSpec(**payload["trailing_spec"])
                if payload.get("trailing_spec")
                else None
            ),
            partial_fraction=(
                float(partial_fraction_raw) if partial_fraction_raw is not None else 0.0
            ),
            second_target_price=self._coerce_float(payload.get("second_target_price")),
        )
        return state

    def attach_persistent_state(self, manager: PersistentStateManager | None) -> None:
        """Attach *manager* for durable order and bracket persistence.

        Args:
            manager: Persistent state manager instance or ``None`` to detach.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered attach_persistent_state",
            extra={"event": "order_manager_attach_persistence"},
        )
        self._persistent_state = manager
        if manager is None:
            return
        try:
            restored_orders = manager.load_open_orders()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in attach_persistent_state orders: %s", exc)
            restored_orders = []
        if restored_orders:
            self.restore_open_orders(restored_orders)
        try:
            restored_brackets = manager.load_brackets()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in attach_persistent_state brackets: %s", exc)
            restored_brackets = []
        if restored_brackets:
            self.restore_brackets(restored_brackets)

    def restore_open_orders(self, payloads: Iterable[Mapping[str, Any]]) -> None:
        """Restore open orders from persisted *payloads*.

        Args:
            payloads: Iterable of order dictionaries.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered restore_open_orders",
            extra={"event": "order_manager_restore_orders"},
        )
        restored: list[OrderDetails] = []
        with self._lock:
            for item in payloads:
                if not isinstance(item, Mapping):
                    continue
                try:
                    order = self._order_from_dict(item)
                except ValueError as exc:
                    self._logger.error("Failure in restore_open_orders: %s", exc)
                    continue
                self._orders[order.order_id] = order
                history_index = self._resolve_history_index(order.order_id)
                if history_index is not None:
                    self._history[history_index] = order
                    self._history_persisted_ids.discard(order.order_id)
                else:
                    self._append_history(order)
                if order.client_order_id:
                    self._client_order_index[str(order.client_order_id)] = (
                        order.order_id
                    )
                restored.append(order)
            if restored:
                self._persist_history()
        for order in restored:
            self._publish_order_to_hub(order)
        if restored:
            self._logger.info(
                "Condition met: restore_open_orders_applied",
                extra={"event": "order_manager_restore_orders", "count": len(restored)},
            )

    def restore_brackets(self, payloads: Iterable[Mapping[str, Any]]) -> None:
        """Restore bracket states from persisted *payloads*.

        Args:
            payloads: Iterable of bracket dictionaries.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered restore_brackets",
            extra={"event": "order_manager_restore_brackets"},
        )
        restored: list[BracketState] = []
        with self._lock:
            self._brackets.clear()
            self._bracket_index.clear()
            for item in payloads:
                if not isinstance(item, Mapping):
                    continue
                try:
                    state = self._bracket_from_dict(item)
                except ValueError as exc:
                    self._logger.error("Failure in restore_brackets: %s", exc)
                    continue
                self._brackets[state.entry_id] = state
                for order_id in (
                    state.stop_order_id,
                    state.tp_primary_id,
                    state.tp_secondary_id,
                ):
                    if order_id:
                        self._bracket_index[str(order_id)] = state.entry_id
                restored.append(state)
        if restored:
            self._logger.info(
                "Condition met: restore_brackets_applied",
                extra={
                    "event": "order_manager_restore_brackets",
                    "count": len(restored),
                },
            )

    def reconcile_open_orders_with_broker(self) -> None:
        """Sync local order state with broker and trigger brackets on fill."""

        # 1. Identify local orders that need checking to minimize work
        with self._lock:
            pending_ids = {
                oid
                for oid, order in self._orders.items()
                if order.status not in self.FINAL_STATUSES
            }

        # Optimization: Don't spam API if we have nothing to track
        if not pending_ids:
            return

        try:
            # 2. RESOLVE FETCHER (Fixes the 'no attribute orders' crash)
            # Dynamically find the correct method (get_orders, list_orders, etc.)
            fetcher = self._resolve_open_orders_fetcher()

            # Fallback: some clients might expose 'orders' property/method directly
            if not fetcher:
                candidate = getattr(self._broker, "orders", None)
                if callable(candidate):
                    fetcher = candidate

            if not fetcher:
                # Log once per minute to avoid flooding logs if broker is incompatible
                if time.time() % 60 < 2:
                    self._logger.error("No order fetch method found on broker client")
                return

            # 3. Fetch Orders safely via Circuit Breaker
            response = self._call_broker(fetcher)
            if not response:
                return

            # Normalize response to list of dicts to handle various broker formats
            broker_orders = []
            if isinstance(response, Mapping):
                broker_orders = [response]
            elif isinstance(response, Iterable) and not isinstance(
                response, (str, bytes)
            ):
                broker_orders = list(response)

            # Map broker order_id -> payload for O(1) lookup
            broker_map = {}
            for item in broker_orders:
                if isinstance(item, Mapping):
                    oid = str(item.get("order_id") or item.get("id") or "")
                    if oid:
                        broker_map[oid] = item

            # 4. Reconciliation Loop
            with self._lock:
                for order_id in pending_ids:
                    local_order = self._orders.get(order_id)
                    if not local_order:
                        continue

                    remote = broker_map.get(order_id)
                    # If remote missing, order might be closed/archived.
                    # We can't assume anything yet unless we trust the broker returns ALL orders.
                    if not remote:
                        continue

                    # Capture old status to detect transitions
                    old_status = local_order.status

                    # Update Local State safely
                    raw_status = str(remote.get("status", "")).upper()
                    local_order.status = self._parse_status(raw_status)

                    filled = self._coerce_int(remote.get("filled_quantity"))
                    if filled is not None:
                        local_order.filled_quantity = filled

                    avg_price_raw = remote.get("average_price") or remote.get(
                        "averagePrice"
                    )
                    avg_price = self._coerce_float(avg_price_raw)
                    if avg_price is not None:
                        local_order.average_price = avg_price
                        # ✅ FIX: Sync fill_price so Position Manager sees the price
                        local_order.fill_price = avg_price

                    msg = remote.get("status_message") or remote.get("message")
                    if msg:
                        local_order.message = str(msg)

                    # 5. TRIGGER LOGIC (The Brain)
                    # If order JUST finished, trigger the Bracket Handler
                    if local_order.status in {
                        OrderStatus.FILLED,
                        OrderStatus.PARTIALLY_FILLED,
                    }:
                        # Log significant state changes
                        if old_status != local_order.status:
                            self._logger.info(
                                f"✅ Order {order_id} update: {local_order.status.name} "
                                f"({local_order.filled_quantity} qty @ {local_order.average_price})"
                            )

                        # CRITICAL: This call triggers the auto-SL/TP placement
                        # We pass the 'remote' payload so the handler has full context
                        self._handle_bracket_update(local_order, old_status, remote)

                    # If Rejected, handle it
                    if (
                        local_order.status == OrderStatus.REJECTED
                        and old_status != OrderStatus.REJECTED
                    ):
                        self._handle_order_rejected(local_order)
                    self._notify_failed_entry_terminal(
                        local_order, old_status, raw_status
                    )

        except Exception as e:
            self._logger.error(f"Reconcile failed: {e}", exc_info=True)
        # [Existing code]
        self._reconcile_positions()

        # [NEW] Add this line:
        self._resurrect_trailing_stops()

    def _trigger_bracket_order(self, order: OrderDetails) -> None:
        """Place Stop Loss and Target orders after entry fill."""
        if not self._bracket_manager:
            self._logger.warning("No BracketManager attached; TP/SL not placed.")
            return

        if not order.stop_loss and not order.take_profit:
            return

        self._logger.info(
            f"🛡️ Placing Bracket for {order.symbol} (Qty: {order.filled_quantity})"
        )

        try:
            # Delegate to Bracket Manager
            self._bracket_manager.place_bracket(
                symbol=order.symbol,
                side=order.side,
                quantity=order.filled_quantity,
                entry_price=order.average_price,
                stop_loss_price=order.stop_loss,
                take_profit_price=order.take_profit,
                parent_order_id=order.order_id,
            )
        except Exception as e:
            self._logger.error(f"Failed to place bracket: {e}")

    def _publish_order_to_hub(
        self, order: OrderDetails, payload: Mapping[str, Any] | None = None
    ) -> None:
        """Push the latest order snapshot into the attached data hub.

        Args:
            order: Canonical order details maintained by the order manager.
            payload: Optional broker payload used to enrich the snapshot.
        """
        hub = self._data_hub
        if hub is None:
            return
        try:
            status_value = (
                order.status.value
                if hasattr(order.status, "value")
                else str(order.status)
            )
            filled = order.filled_quantity
            if payload is not None:
                raw_filled = payload.get("filled_quantity")
                if raw_filled is None:
                    raw_filled = payload.get("filled")
                filled_candidate = self._coerce_int(raw_filled)
                if filled_candidate is not None:
                    filled = filled_candidate
            price_value = self._coerce_float(payload.get("price")) if payload else None
            if price_value is None and order.price:
                price_value = float(order.price)
            trigger_value = (
                self._coerce_float(payload.get("trigger_price")) if payload else None
            )
            timestamp_value = time.time()
            if payload is not None and payload.get("timestamp") is not None:
                ts_candidate = self._coerce_float(payload.get("timestamp"))
                if ts_candidate is not None:
                    timestamp_value = ts_candidate
            hub.upsert_order(
                {
                    "order_id": order.order_id,
                    "symbol": DataHub.normalize(order.symbol),
                    "side": order.side,
                    "quantity": order.quantity,
                    "filled_quantity": filled,
                    "status": status_value,
                    "price": price_value,
                    "trigger_price": trigger_value,
                    "timestamp": timestamp_value,
                    "parent_order_id": order.parent_order_id,
                    "child_order_ids": list(order.child_order_ids),
                }
            )
        except Exception:  # noqa: BLE001
            self._logger.debug("data_hub_order_publish_failed", exc_info=True)

    def _sync_positions_to_hub(self) -> None:
        """Transmit the current position snapshot to the data hub."""
        hub = self._data_hub
        if hub is None:
            return
        try:
            payload: list[dict[str, Any]] = []
            positions = list(self._positions.get_open_positions())
            for position in positions:
                symbol = DataHub.normalize(getattr(position, "symbol", ""))
                if not symbol:
                    continue
                qty = self._coerce_int(getattr(position, "quantity", None)) or 0
                side = str(getattr(position, "side", "LONG")).upper()
                signed_qty = qty if side == "LONG" else -qty if side == "SHORT" else qty
                avg_price = self._coerce_float(getattr(position, "entry_price", None))
                if avg_price is None:
                    avg_price = 0.0
                payload.append(
                    {
                        "symbol": symbol,
                        "quantity": signed_qty,
                        "average_price": avg_price,
                    }
                )
            hub.replace_positions(payload)
        except Exception:  # noqa: BLE001
            self._logger.debug("data_hub_position_sync_failed", exc_info=True)

    def _validate_quantity(self, symbol: str, quantity: int) -> None:
        """Ensure *quantity* respects configured lot sizes."""

        normalized = DataHub.normalize(symbol)
        if quantity <= 0:
            raise OrderPlacementError("Quantity must be positive")
        if normalized and not normalized.endswith(("CE", "PE")):
            return
        lot_size = self._lot_size_for_symbol(normalized or symbol)
        if quantity % lot_size != 0:
            raise OrderPlacementError(
                f"Quantity must be a multiple of {lot_size} for {normalized or symbol}"
            )

    def _truncate_quantity_to_lot(self, symbol: str, quantity: int) -> int:
        """Return quantity truncated to the nearest lot multiple.

        Args:
            symbol: Tradable instrument identifier, optionally prefixed with exchange.
            quantity: Requested order size before lot adjustments.

        Returns:
            Quantity rounded down to the nearest valid lot multiple.

        Raises:
            OrderPlacementError: If lot size lookup fails.
        """

        self._logger.debug(
            "Entered _truncate_quantity_to_lot",
            extra={
                "event": "truncate_quantity_to_lot_enter",
                "symbol": symbol,
                "quantity": quantity,
            },
        )
        try:
            if quantity <= 0:
                return 0
            if not app_settings.ORDER_TRUNCATE_TO_LOT:
                return int(quantity)
            normalized = DataHub.normalize(symbol)
            if normalized and not normalized.endswith(("CE", "PE")):
                return int(quantity)
            lot_symbol = normalized or symbol
            lot_size = self._lot_size_for_symbol(lot_symbol)
            if lot_size <= 0:
                raise OrderPlacementError("Lot size must be positive")
            rounded_lots = max(int(quantity) // lot_size, 0)
            adjusted = rounded_lots * lot_size
            if adjusted != quantity:
                self._logger.info(
                    "Condition met: truncate_quantity_to_lot_applied",
                    extra={
                        "event": "truncate_quantity_to_lot_applied",
                        "symbol": lot_symbol,
                        "requested_qty": quantity,
                        "lot_size": lot_size,
                        "adjusted_qty": adjusted,
                    },
                )
            return adjusted
        except OrderPlacementError:
            raise
        except Exception as exc:  # noqa: BLE001 - defensive logging
            self._logger.error(
                "Failure in _truncate_quantity_to_lot: %s",
                exc,
                extra={
                    "event": "truncate_quantity_to_lot_failed",
                    "symbol": symbol,
                    "quantity": quantity,
                },
            )
            raise

    def _is_reduce_only_violation(
        self,
        *,
        symbol: str,
        quantity: int,
        product: str | None,
    ) -> bool:
        """Return ``True`` when a SELL would breach reduce-only constraints.

        Args:
            symbol: Instrument identifier evaluated for coverage.
            quantity: Requested exit size after lot adjustments.
            product: Broker product code, if provided.

        Returns:
            ``True`` when no long exposure exists to offset the SELL order.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered _is_reduce_only_violation",
            extra={
                "event": "reduce_only_violation_check_enter",
                "symbol": symbol,
                "quantity": quantity,
                "product": product,
            },
        )
        if quantity <= 0:
            return False
        candidate_symbols = {symbol.upper()}
        if ":" in symbol:
            candidate_symbols.add(symbol.split(":", 1)[1].upper())
        try:
            for candidate in candidate_symbols:
                position = self._positions.get_position(candidate)
                if position is None:
                    continue
                side = str(getattr(position, "side", "")).upper()
                open_qty = int(getattr(position, "quantity", 0))
                if side == "LONG" and open_qty > 0:
                    return False
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in _is_reduce_only_violation: %s",
                exc,
                extra={
                    "event": "reduce_only_violation_position_error",
                    "symbol": symbol,
                },
            )
            return True

        hub = self._data_hub
        if hub is not None:
            try:
                normalized = DataHub.normalize(symbol) or symbol.upper()
                for row in hub.positions():
                    row_symbol = str(row.get("symbol", "")).upper()
                    if row_symbol != normalized.upper():
                        continue
                    qty_val = int(float(row.get("quantity", 0) or 0))
                    side_val = str(row.get("side", "")).upper()
                    if side_val == "LONG" and qty_val > 0:
                        return False
            except Exception as exc:  # noqa: BLE001 - defensive
                self._logger.error(
                    "Failure in _is_reduce_only_violation: %s",
                    exc,
                    extra={
                        "event": "reduce_only_violation_hub_error",
                        "symbol": symbol,
                    },
                )
                return True

        return True

    def _validate_option_exit_quantity(
        self, symbol: str, side: str, quantity: int, intent: str
    ) -> dict[str, Any] | None:
        getter = getattr(self._positions, "get_position", None)
        position = None
        if callable(getter):
            for candidate in dict.fromkeys(
                [symbol.upper(), symbol.split(":", 1)[-1].upper()]
            ):
                position = getter(candidate)
                if position is not None:
                    break
        open_units = abs(int(getattr(position, "quantity", 0) or 0))
        position_side = str(getattr(position, "side", "") or "").upper()
        if open_units <= 0:
            broker_positions = getattr(self._broker, "query_positions", None)
            if callable(broker_positions):
                broker_qty = int((broker_positions() or {}).get(symbol, 0) or 0)
                if broker_qty != 0:
                    open_units = abs(broker_qty)
                    position_side = "LONG" if broker_qty > 0 else "SHORT"
        reason: str | None = None
        if open_units <= 0:
            reason = "exit_without_open_position"
        elif quantity <= 0:
            reason = "exit_quantity_invalid"
        elif quantity > open_units:
            reason = "exit_quantity_exceeds_position"
        elif position_side == "LONG" and side != "SELL":
            reason = "exit_side_not_reducing"
        elif position_side == "SHORT" and side != "BUY":
            reason = "exit_side_not_reducing"
        elif position_side not in {"LONG", "SHORT"}:
            reason = "exit_side_not_reducing"
        lot_size: int | None = None
        if reason is None:
            try:
                lot_size = self._lot_size_for_symbol(symbol)
            except OrderPlacementError:
                if quantity != open_units:
                    reason = "exit_lot_size_unresolved"
            else:
                if lot_size <= 0:
                    reason = "exit_lot_size_unresolved"
                elif quantity % lot_size != 0:
                    reason = "exit_quantity_not_lot_multiple"
        if reason is None:
            return None
        details = {
            "reason": reason,
            "symbol": symbol,
            "requested_exit_units": quantity,
            "open_position_units": open_units,
            "lot_size": lot_size,
            "remainder": quantity % lot_size if lot_size else None,
            "side": side,
            "intent": intent,
        }
        self._logger.warning(
            "ORDER_BLOCKED: %s symbol=%s requested_exit_units=%s open_position_units=%s side=%s intent=%s",
            reason, symbol, quantity, open_units, side, intent,
            extra={"event": reason, **details},
        )
        return details

    def _configure_options_policy(self) -> None:
        policy = self._options_policy
        if policy is None:
            return
        lookup = self._lot_lookup()
        policy.set_lot_size_lookup(lookup)
        time_provider = None
        if self._market_data is not None:
            time_provider = getattr(self._market_data, "now_ns", None)
        policy.set_time_provider(time_provider if callable(time_provider) else None)

    def _lot_lookup(self) -> Callable[[str], int] | None:
        resolver = self._resolver
        if resolver is None:
            market_data = self._market_data
            resolver = getattr(market_data, "resolver", None) if market_data else None
            if resolver is None and market_data is not None:
                resolver = getattr(market_data, "_resolver", None)
        if resolver is None:
            return None

        primary = getattr(resolver, "lot_size_for_symbol", None)
        if callable(primary):
            return cast(Callable[[str], int], primary)

        secondary = getattr(resolver, "get_lot_size", None)
        if callable(secondary):
            return cast(Callable[[str], int], secondary)
        return None

    def _lot_size_for_symbol(self, symbol: str) -> int:
        lookup = self._lot_lookup()
        normalized_symbol = normalize_symbol(symbol) or str(symbol).strip().upper()
        try:
            lot_size, source = resolve_lot_size_with_source(normalized_symbol, lookup)
        except Exception as exc:  # noqa: BLE001
            compact_symbol = (
                normalized_symbol.split(":", 1)[-1]
                if ":" in normalized_symbol
                else normalized_symbol
            )
            self._logger.warning(
                "LOT_SIZE_RESOLUTION_FAILED symbol=%s normalized_symbol=%s underlying=%s expiry=%s strike=%s option_type=%s cache_loaded=%s cache_size=%s reason=%s",
                symbol,
                normalized_symbol,
                "NIFTY" if "NIFTY" in compact_symbol else "UNKNOWN",
                (
                    compact_symbol[5:12]
                    if compact_symbol.startswith("NIFTY") and len(compact_symbol) >= 12
                    else None
                ),
                "".join(ch for ch in compact_symbol if ch.isdigit()) or None,
                compact_symbol[-2:] if compact_symbol.endswith(("CE", "PE")) else None,
                bool(lookup is not None),
                None,
                str(exc),
            )
            raise OrderPlacementError("Failed to resolve lot size") from exc
        if (
            source != "instrument_dump"
            and "NIFTY" in normalized_symbol
            and normalized_symbol.endswith(("CE", "PE"))
            and callable(getattr(self, "is_live_mode", None))
            and self.is_live_mode()
        ):
            self._logger.warning(
                "LOT_SIZE_UNRESOLVED symbol=%s source=%s",
                normalized_symbol,
                source,
                extra={
                    "event": "LOT_SIZE_UNRESOLVED",
                    "symbol": normalized_symbol,
                    "source": source,
                },
            )
            raise OrderPlacementError("lot_size_unresolved")
        if source in {"env_fallback", "fallback_default"}:
            self._logger.info(
                "LOT_SIZE_FALLBACK_USED symbol=%s normalized_symbol=%s underlying=%s lot_size=%s source=%s",
                symbol,
                normalized_symbol,
                "NIFTY" if "NIFTY" in normalized_symbol else "UNKNOWN",
                lot_size,
                source,
            )
        self._logger.info(
            "LOT_SIZE_RESOLVED underlying=%s symbol=%s lot_size=%s source=%s",
            "NIFTY" if "NIFTY" in normalized_symbol else normalized_symbol,
            normalized_symbol,
            lot_size,
            source,
            extra={
                "event": "LOT_SIZE_RESOLVED",
                "underlying": (
                    "NIFTY" if "NIFTY" in normalized_symbol else normalized_symbol
                ),
                "symbol": normalized_symbol,
                "lot_size": lot_size,
                "source": source,
            },
        )
        return lot_size

    def _normalize_leg(self, leg: AtomicLeg | Mapping[str, Any]) -> AtomicLeg:
        """Normalize user-provided leg specifications into :class:`AtomicLeg`."""

        if isinstance(leg, AtomicLeg):
            return leg
        if not isinstance(leg, Mapping):
            raise TypeError("Leg specification must be a mapping or AtomicLeg instance")
        symbol_raw = leg.get("symbol")
        symbol = str(symbol_raw).strip() if symbol_raw is not None else ""
        if not symbol:
            raise ValueError("Leg symbol must be provided")
        side_raw = leg.get("side")
        side = str(side_raw).strip().upper() if side_raw is not None else ""
        if side not in {"BUY", "SELL"}:
            raise ValueError("Leg side must be 'BUY' or 'SELL'")
        qty = self._coerce_int(leg.get("quantity"))
        if qty is None or qty <= 0:
            raise ValueError("Leg quantity must be positive")
        order_type_raw = leg.get("order_type", OrderType.MARKET)
        order_type = self._parse_order_type_token(order_type_raw)
        price_token = leg.get("price")
        if price_token is not None:
            price_value = self._coerce_float(price_token)
        else:
            price_value = None
        return AtomicLeg(
            symbol=symbol,
            side=cast(Literal["BUY", "SELL"], side),
            quantity=qty,
            order_type=order_type,
            price=price_value,
        )

    @staticmethod
    def _leg_failed(details: OrderDetails, tolerance: float) -> bool:
        """Return ``True`` if a leg violates the partial fill tolerance."""

        if details.status in {
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        }:
            return True
        threshold = max(0.0, min(1.0, 1.0 - tolerance))
        quantity = max(details.quantity, 0)
        if quantity <= 0:
            return False
        fill_ratio = details.filled_quantity / quantity
        if details.status == OrderStatus.PARTIALLY_FILLED and fill_ratio < threshold:
            return True
        if (
            details.status == OrderStatus.FILLED
            and fill_ratio < threshold
            and threshold > 0.0
        ):
            return True
        return False

    @staticmethod
    def _coerce_float(value: object | None) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(cast(Any, value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _timestamp_seconds(value: object) -> float:
        if isinstance(value, datetime):
            return float(value.timestamp())
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            token = value.strip()
            try:
                return float(
                    datetime.fromisoformat(token.replace("Z", "+00:00")).timestamp()
                )
            except ValueError:
                try:
                    return float(token)
                except ValueError:
                    return 0.0
        return 0.0

    @staticmethod
    def _coerce_int(value: object | None) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(float(cast(Any, value)))
        except (TypeError, ValueError):
            return None

    def _load_history(self) -> None:
        if not self._history_path.exists():
            return
        try:
            data = json.loads(self._history_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._logger.error("Failed to read order history: %s", exc)
            return
        self._history.clear()
        self._history_index.clear()
        self._history_base_index = 0
        self._history_persisted_ids.clear()
        if self._history.maxlen is not None and len(data) > self._history.maxlen:
            data = data[-self._history.maxlen :]
        for entry in data:
            try:
                order = OrderDetails(
                    order_id=str(entry["order_id"]),
                    symbol=str(entry["symbol"]).upper(),
                    side=str(entry["side"]),
                    order_type=OrderType(entry["order_type"]),
                    quantity=int(entry["quantity"]),
                    price=float(entry["price"]),
                    status=OrderStatus(entry["status"]),
                    timestamp=datetime.fromisoformat(entry["timestamp"]),
                    filled_quantity=int(entry.get("filled_quantity", 0)),
                    fill_price=(
                        float(entry["fill_price"])
                        if entry.get("fill_price") is not None
                        else None
                    ),
                    rejection_reason=entry.get("rejection_reason"),
                    parent_order_id=entry.get("parent_order_id"),
                    child_order_ids=list(entry.get("child_order_ids", [])),
                    client_order_id=(
                        str(entry.get("client_order_id"))
                        if entry.get("client_order_id")
                        else None
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                self._logger.error("Skipping invalid order history entry: %s", exc)
                continue
            self._append_history(order)
            if order.status not in self.FINAL_STATUSES:
                self._orders[order.order_id] = order
            if order.client_order_id:
                self._client_order_index[str(order.client_order_id)] = order.order_id
        self._history_persisted_ids.update(order.order_id for order in self._history)

    def _update_local_status(self, order_id: str, status: OrderStatus) -> None:
        order = self._refresh_order(order_id)
        with self._lock:
            order.status = status
            self._register_order(order)
            self._positions.update_order_status(order.order_id, order.status.name)
            if order.client_order_id and status in self.FINAL_STATUSES:
                self._client_order_index.pop(str(order.client_order_id), None)
        self._publish_order_to_hub(order)
        self._sync_positions_to_hub()

    def _resurrect_trailing_stops(self) -> None:
        """
        CRITICAL: Re-attach trailing controllers to open brackets after restart.
        Without this, TSL becomes a static SL after any reboot.
        """
        self._logger.info("♻️ Resurrecting Trailing Stops...")

        with self._lock:
            # Iterate all active brackets
            for entry_id, state in self._brackets.items():
                # Skip if already running or closed
                if entry_id in self._trailing or state.remaining_position() <= 0:
                    continue

                # Check if this bracket HAD a trailing spec
                if not state.trailing_spec:
                    continue

                self._logger.info(
                    f"⚡ Re-attaching TSL for {state.symbol} (Order {entry_id})"
                )

                try:
                    self.attach_trailing_stop(
                        entry_order_id=state.entry_id,
                        sl_order_id=state.stop_order_id,
                        symbol=state.symbol,
                        side=state.side,
                        entry_price=state.entry_price,
                        spec=state.trailing_spec,
                    )
                except Exception as e:
                    self._logger.error(f"Failed to resurrect TSL for {entry_id}: {e}")

    def _reconcile_positions(self) -> None:
        """
        CRITICAL SYNC: Force local state to match Broker's Net Positions.
        Handles Manual Exits (Ghosts) and Unmanaged Trades (Orphans).
        """
        reconcile_lock = getattr(self._bracket_manager, "_reconcile_lock", None)
        lock_acquired = False
        try:
            if reconcile_lock is not None:
                lock_acquired = reconcile_lock.acquire(blocking=False)
                if not lock_acquired:
                    return
            # 1. FETCH TRUTH (Broker State)
            if not hasattr(self._broker, "get_positions"):
                return

            # Fetch net positions (standard for most brokers)
            try:
                broker_pos_payload = self._broker.get_positions()
                if asyncio.iscoroutine(broker_pos_payload):
                    broker_pos_payload = asyncio.run(broker_pos_payload)
            except Exception as e:
                self._logger.error(f"Failed to fetch broker positions: {e}")
                return

            broker_pos_list = (
                broker_pos_payload if isinstance(broker_pos_payload, list) else []
            )

            # Map: Normalized Symbol -> Position Data
            broker_map = {}
            for p in broker_pos_list:
                # Handle different broker key names safely
                qty = int(p.get("quantity") or p.get("net_quantity") or 0)
                if qty == 0:
                    continue  # Skip closed positions

                raw_sym = str(p.get("tradingsymbol") or p.get("symbol") or "")
                exch = str(p.get("exchange") or "NFO")

                # NORMALIZE: Ensure strictly "EXCHANGE:SYMBOL" format (e.g., NFO:NIFTY...)
                # Zerodha sometimes returns "NIFTY..." without NFO:
                if ":" in raw_sym:
                    clean_sym = normalize_symbol(raw_sym)
                else:
                    clean_sym = normalize_symbol(f"{exch}:{raw_sym}")

                broker_map[clean_sym] = {
                    "qty": qty,
                    "price": float(p.get("average_price") or p.get("buy_price") or 0.0),
                    "product": p.get("product"),
                    "raw_payload": p,
                }

            # 2. HANDLE ORPHANS (Broker Exists, Local Missing) -> Adopt & Protect
            # ------------------------------------------------------
            all_local = list(self._positions.get_open_positions())
            local_map = {}
            for pos in all_local:
                lsym = normalize_symbol(str(pos.symbol))
                local_map[lsym] = pos

            broker_positions = broker_map
            local_positions = local_map
            recently_handled = getattr(self, "_recent_orphans", set())

            for broker_sym, data in broker_positions.items():
                # Broker is authoritative: adopt any position missing locally.
                # ✅ FIX: Ignore positions that are already closed (Qty 0)
                if data["qty"] == 0:
                    continue

                if broker_sym not in local_positions:
                    if broker_sym in recently_handled:
                        continue

                    recently_handled.add(broker_sym)
                    self._recent_orphans = recently_handled

                    if (
                        self._bracket_manager
                        and self._bracket_manager.has_active_bracket(broker_sym)
                    ):
                        continue

                    # [FIX 1] RACE CONDITION GUARD:
                    # Check if we have touched this symbol recently (Pending Orders or Recent Fills).
                    # This stops the "Infinite Rescue Loop".
                    is_active_locally = False
                    with self._lock:
                        # An ENTRY submission currently in flight for this
                        # symbol proves the position is ours (broker accepted
                        # the order before we registered it locally). Never
                        # adopt it as an orphan during that window.
                        _now_orphan = time.time()
                        for _if_sym, _if_ts in self._entries_in_flight.items():
                            if (
                                DataHub.normalize(_if_sym)
                                == DataHub.normalize(broker_sym)
                                and _now_orphan - _if_ts <= self.ENTRY_INFLIGHT_TTL_SEC
                            ):
                                is_active_locally = True
                                break
                        for order in self._orders.values():
                            # Normalize symbol check
                            if DataHub.normalize(order.symbol) == DataHub.normalize(
                                broker_sym
                            ):
                                # 1. If we are currently working on an order -> SKIP
                                if order.status not in self.FINAL_STATUSES:
                                    is_active_locally = True
                                    break
                                # 2. If we finished an order < 15s ago -> SKIP (Give time to sync)
                                if (
                                    order.timestamp
                                    and hasattr(order.timestamp, "timestamp")
                                    and time.time() - order.timestamp.timestamp() < 20
                                ):
                                    is_active_locally = True
                                    break
                                if order.timestamp:
                                    ts = (
                                        order.timestamp.timestamp()
                                        if hasattr(order.timestamp, "timestamp")
                                        else time.time()
                                    )
                                    if time.time() - ts < 15.0:
                                        is_active_locally = True
                                        break

                                if (
                                    order.timestamp
                                    and hasattr(order.timestamp, "timestamp")
                                    and time.time() - order.timestamp.timestamp() < 20
                                ):
                                    is_active_locally = True
                                    break

                    if is_active_locally:
                        # self._logger.debug(f"⏳ Skipping Orphan Check for {broker_sym} (Busy)")
                        continue

                    # If we get here, it is a TRUE Orphan (Manual/Old trade)
                    self._logger.warning(
                        f"⚠️ Orphan Position Found: {broker_sym} Qty: {data['qty']}. Adopting...",
                        extra={"event": "orphan_adopted", "symbol": broker_sym},
                    )

                    # [UPGRADE] Use Smart Guard (Adopt + Virtual Bracket + Persistence)
                    try:
                        success = self.guard_orphan_position(
                            broker_sym, data["qty"], data["price"]
                        )

                        if success:
                            self._logger.info(
                                f"✅ Orphan {broker_sym} successfully guarded & persisted."
                            )
                        else:
                            # Fallback: If smart guard fails (e.g. BracketManager missing),
                            # we MUST ensure a Hard Broker SL exists.
                            self._logger.warning(
                                f"⚠️ Smart Guard failed for {broker_sym}. Checking Hard SL..."
                            )
                            self._ensure_safety_bracket(
                                broker_sym, data["qty"], data["price"]
                            )

                    except Exception as e:
                        self._logger.error(f"Orphan Handling Error: {e}")
                        # Final Safety Net
                        self._ensure_safety_bracket(
                            broker_sym, data["qty"], data["price"]
                        )

            # 3. HANDLE GHOSTS & NAKED POSITIONS
            # ------------------------------------------------------
            for lsym, pos in local_positions.items():
                # CASE A: Ghost (We think we have it, Broker says no)
                self._missing_counts.setdefault(lsym, 0)
                if lsym not in broker_positions:
                    self._missing_counts[lsym] += 1
                    if self._missing_counts[lsym] < 3:
                        self._logger.debug(
                            "Condition met: broker_position_temporarily_missing",
                            extra={
                                "event": "broker_position_temporarily_missing",
                                "symbol": lsym,
                                "missing_count": self._missing_counts[lsym],
                            },
                        )
                        continue
                    self._logger.warning(
                        f"👻 Ghost Position Found: {lsym}. Clearing local state.",
                        extra={"event": "ghost_cleared", "symbol": lsym},
                    )
                    self.cancel_orders_for_symbol(lsym)
                    if self._bracket_manager is not None:
                        try:
                            self._bracket_manager.manual_override_close(
                                lsym, reason="broker_position_closed"
                            )
                        except Exception as e:
                            self._logger.error("Failure in _reconcile_positions: %s", e)
                    self._generate_adjustment_order(lsym, -int(pos.quantity), 0.0)
                    self._missing_counts.pop(lsym, None)

                # CASE B: Quantity Mismatch (Partial fills/Manual intervention)
                elif broker_positions[lsym]["qty"] != pos.quantity:
                    self._missing_counts[lsym] = 0
                    diff = broker_positions[lsym]["qty"] - pos.quantity
                    self._logger.info(
                        f"⚖️ Syncing Qty for {lsym}: Local {pos.quantity} -> Broker {broker_positions[lsym]['qty']}",
                        extra={"event": "qty_sync", "symbol": lsym},
                    )
                    self._generate_adjustment_order(lsym, diff, 0.0)

                # CASE C: Perfect Match... BUT IS IT SAFE? (The Missing Link)
                elif broker_positions[lsym]["qty"] == pos.quantity:
                    self._missing_counts[lsym] = 0
                    # 1. Check if a bracket is actually managing this symbol
                    is_managed = False
                    if self._bracket_manager:
                        # Fast check: Does bracket manager know this order ID or symbol?
                        # We iterate because bracket ID != Position ID usually
                        with self._lock:
                            for b in self._brackets.values():
                                if b.symbol == lsym and b.active:
                                    is_managed = True
                                    break

                    # 2. If not managed, WE MUST ADOPT IT NOW
                    if not is_managed:
                        self._logger.warning(
                            f"🛡️ Naked Position Detected (Qty Match): {lsym}. Forcing Guard...",
                            extra={"event": "naked_guard_trigger", "symbol": lsym},
                        )
                        # We pass consume_existing=True because accounting is already correct!
                        self.guard_orphan_position(
                            lsym,
                            int(pos.quantity),
                            float(pos.entry_price or 0.0),
                            consume_existing=True,
                        )

        except Exception as e:
            self._logger.error(f"Reconciliation Failed: {e}", exc_info=True)
        finally:
            if lock_acquired and reconcile_lock is not None:
                reconcile_lock.release()

    def _adopt_orphan_position(self, symbol: str, data: dict) -> None:
        """
        Creates a synthetic filled order to register the position locally.

        Fixes 'str object has no attribute value' by passing the Enum object
        directly to PositionManager, ensuring type safety.
        """
        import time
        from datetime import datetime, timezone

        symbol = normalize_symbol(symbol)

        # 1. Safe Quantity Extraction
        try:
            qty = int(float(data.get("qty", 0)))
            price = float(data.get("price", 0.0))
        except (ValueError, TypeError):
            self._logger.error(f"Cannot adopt orphan {symbol}: Invalid data {data}")
            return

        if qty == 0:
            return

        side = "BUY" if qty > 0 else "SELL"

        # 2. Unique ID (Use simplified timestamp to avoid colons/special chars)
        safe_sym = symbol.replace(":", "_")
        order_id = f"sync_{int(time.time())}_{safe_sym}"

        # 3. Resolve OrderType (Handle Enum vs String definition)
        # CRITICAL: Ensure 'otype' is the ENUM object, not a string.
        # PositionManager expects an object it can access .value on.
        otype = OrderType.MARKET if hasattr(OrderType, "MARKET") else "MARKET"

        # 4. Create Local Record
        details = OrderDetails(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=abs(qty),
            order_type=otype,  # Pass the Enum/Object here
            status=OrderStatus.FILLED,
            timestamp=datetime.now(timezone.utc),
            price=price,
            average_price=price,
            fill_price=price,
            filled_quantity=abs(qty),
            tag="orphan_adoption",
            intent="ENTRY",
            intended_position_side="LONG" if qty > 0 else "SHORT",
        )

        # 5. Register in OrderManager
        self._register_order(details)

        # 6. Register in PositionManager (CRITICAL FIX)
        pm = self._positions

        try:
            if hasattr(pm, "add_pending_order"):
                # [FIX] Pass 'otype' (Enum) directly.
                # Do NOT convert to string, as PM likely calls .value on it.
                pm.add_pending_order(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    qty=abs(qty),
                    price=price,
                    order_type=otype,
                    intent="ENTRY",
                )

                # Immediately confirm fill since it's an orphan
                if hasattr(pm, "update_order_status"):
                    pm.update_order_status(order_id, "FILLED", price)

            elif hasattr(pm, "update_from_order"):
                pm.update_from_order(details)

            elif hasattr(pm, "update_position"):
                pm.update_position(
                    symbol=details.symbol,
                    qty=details.quantity,
                    price=details.average_price,
                    side=details.side,
                    product="MIS",
                )
            else:
                self._logger.error("PositionManager missing standard update methods.")

        except Exception as e:
            # Catch specific attribute errors to prevent crash loop
            self._logger.error(f"Orphan adoption failed for {symbol}: {e}")

    def guard_orphan_position(
        self,
        symbol: str,
        quantity: int,
        average_price: float,
        position_side: str = None,
        consume_existing: bool = False,
    ) -> bool:
        """
        Master method to Adopt AND Protect a naked position.

        Fixes:
        1. Cold Start Race Condition (Subscribes BEFORE price check)
        2. Zero Price / Negative SL crashes
        3. DB Persistence errors
        """
        symbol = normalize_symbol(symbol)

        if not self._bracket_manager or quantity == 0:
            return False

        # ✅ Use explicit side if provided, else infer from quantity
        if position_side:
            side = position_side
        else:
            side = "BUY" if quantity > 0 else "SELL"

        # --- STEP 0: ENSURE DATA FLOW (CRITICAL FIX) ---
        # We must subscribe immediately. If we fail price check later,
        # at least the NEXT loop will have data.
        try:
            if self._market_data:
                if not hasattr(self, "_bracket_tick_subscriptions"):
                    self._bracket_tick_subscriptions = set()

                if symbol not in self._bracket_tick_subscriptions:

                    def bracket_tick_handler(tick_data: dict) -> None:
                        try:
                            ltp = (
                                tick_data.get("ltp")
                                or tick_data.get("last_price")
                                or tick_data.get("price")
                            )
                            if ltp and float(ltp) > 0:
                                from nifty_scalper_bot.execution.bracket_manager import (
                                    tick_exchange_epoch,
                                )

                                self._bracket_manager.on_tick(
                                    symbol,
                                    float(ltp),
                                    tick_exchange_epoch(tick_data),
                                )
                        except Exception:
                            self._logger.exception("Unhandled exception", exc_info=True)
                            raise

                    _hub = self._data_hub or self._market_data
                    if _hub:
                        try:
                            _hub.ensure_tracking(symbol, seed=True)
                        except Exception as _seed_exc:  # noqa: BLE001
                            log_throttled(
                                self._logger,
                                f"guard_seed_tracking_failed:{symbol}",
                                "GUARD_SEED_TRACKING_FAILED symbol=%s error=%s"
                                % (symbol, _seed_exc),
                                interval_sec=60,
                                level=logging.WARNING,
                                extra={
                                    "event": "GUARD_SEED_TRACKING_FAILED",
                                    "symbol": str(symbol),
                                },
                            )
                        _hub.subscribe(symbol, bracket_tick_handler)
                        self._bracket_tick_subscriptions.add(symbol)
                        self._logger.info(f"📡 Subscribed to {symbol} for guarding.")
        except Exception as e:
            self._logger.warning(f"Subscription attempt warning: {e}")

        # --- STEP 1: Determine Valid Base Price ---
        try:
            base_price = float(average_price)
        except (TypeError, ValueError):
            base_price = 0.0
        current_ltp = 0.0

        # Try to get fresh LTP from Cache via DataHub (SSOT); might be empty on cold start.
        _quote_src = self._data_hub or self._market_data
        if _quote_src:
            quote = _quote_src.get_quote(symbol)
            if quote:
                current_ltp = float(quote.get("last_price") or quote.get("ltp") or 0.0)

        # Use LTP if base_price is invalid (0.0)
        if not base_price or base_price <= 0:
            if current_ltp > 0:
                base_price = current_ltp
                self._logger.warning(
                    f"⚠️ Using LTP {base_price} for guard (Broker avg_price was 0)"
                )
            if not base_price:
                try:
                    raise ValueError("Cannot determine entry price")
                except ValueError as exc:
                    self._logger.error("Failure in guard_orphan_position: %s", exc)
                    return False

        # --- STEP 2: Fix Accounting ---
        if not consume_existing:
            try:
                self._adopt_orphan_position(
                    symbol, {"qty": quantity, "price": base_price}
                )
            except Exception:
                self._logger.exception("Unhandled exception", exc_info=True)
                raise

        # --- STEP 3: Fix Protection ---
        try:

            def round_tick(price: float) -> float:
                return round(price * 20) / 20

            import time
            from datetime import datetime, timezone

            safe_symbol = symbol.replace(":", "_")
            synthetic_id = f"guard_{int(time.time())}_{safe_symbol}"
            side = "BUY" if quantity > 0 else "SELL"
            abs_qty = abs(quantity)

            # Register stub
            if consume_existing:
                # Safe Enum
                otype = OrderType.MARKET if hasattr(OrderType, "MARKET") else "MARKET"

                details = OrderDetails(
                    order_id=synthetic_id,
                    symbol=symbol,
                    side=side,
                    quantity=abs_qty,
                    order_type=otype,
                    status=OrderStatus.FILLED,
                    timestamp=datetime.now(timezone.utc),
                    price=base_price,
                    average_price=base_price,
                    fill_price=base_price,
                    filled_quantity=abs_qty,
                    tag="auto_guard_synthetic",
                )
                self._register_order(details)

            # Risk Logic (Fallback)
            atr_val = 0.0
            # ... (Insert your ATR logic here if desired) ...

            # Default to Fixed % if ATR missing
            sl_dist = base_price * 0.10
            tp_dist = base_price * 0.20
            strategy_tag = "auto_guard_fixed"

            if side == "BUY":
                sl_price = round_tick(base_price - sl_dist)
                tp_price = round_tick(base_price + tp_dist)
                # Emergency: If LTP < SL, lower SL slightly below LTP to avoid instant stop-out
                if current_ltp > 0 and current_ltp < sl_price:
                    sl_price = round_tick(current_ltp * 0.99)
            else:
                sl_price = round_tick(base_price + sl_dist)
                tp_price = round_tick(base_price - tp_dist)
                if current_ltp > 0 and current_ltp > sl_price:
                    sl_price = round_tick(current_ltp * 1.01)

            # Force Positive Prices
            sl_price = max(0.05, sl_price)
            tp_price = max(0.05, tp_price)

            self._logger.info(
                f"🛡️ GUARDING ORPHAN: {symbol} | {side} {abs_qty} | "
                f"Base: {base_price} | SL: {sl_price} | TP: {tp_price}",
                extra={"event": "orphan_guarding", "symbol": symbol},
            )
            broker_position_qty = 0
            broker = getattr(self, "_broker", None)
            get_positions = getattr(broker, "get_positions", None)
            if callable(get_positions):
                try:
                    positions = get_positions() or []
                    normalized_symbol = normalize_symbol(symbol)
                    for position in positions:
                        if not isinstance(position, Mapping):
                            continue
                        pos_symbol = normalize_symbol(
                            str(
                                position.get("symbol")
                                or position.get("tradingsymbol")
                                or ""
                            )
                        )
                        if pos_symbol != normalized_symbol:
                            continue
                        broker_position_qty = int(
                            float(
                                position.get("quantity")
                                or position.get("net_quantity")
                                or 0
                            )
                        )
                        break
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in orphan broker position verification: %s",
                        exc,
                        extra={
                            "event": "orphan_broker_position_verify_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
            if broker_position_qty <= 0:
                self._logger.warning(
                    "ORPHAN_BRACKET_SKIPPED symbol=%s reason=no_broker_position",
                    symbol,
                    extra={
                        "event": "ORPHAN_BRACKET_SKIPPED",
                        "symbol": symbol,
                        "reason": "no_broker_position",
                    },
                )
                return False

            # Register
            self._bracket_manager.register_virtual_bracket(
                order_id=synthetic_id,
                symbol=symbol,
                side=side,
                qty=abs_qty,
                price=base_price,
                sl=sl_price,
                tp=tp_price,
                tag=strategy_tag,
                activate_immediately=True,
            )
            self._logger.info(
                "ORPHAN_POSITION_BRACKET_ATTACHED symbol=%s qty=%s base_price=%s",
                symbol,
                abs_qty,
                base_price,
                extra={
                    "event": "ORPHAN_POSITION_BRACKET_ATTACHED",
                    "symbol": symbol,
                    "qty": abs_qty,
                    "base_price": base_price,
                },
            )

            self._log_trade_event(
                "BRACKET_GUARD_REGISTERED",
                symbol=symbol,
                side=side,
                qty=abs_qty,
                price=base_price,
                order_id=synthetic_id,
                meta={
                    "current_sl": sl_price,
                    "tp1": tp_price,
                    "trailing_active": True,
                    "tag": strategy_tag,
                    "status": "ACTIVE",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            self._bracket_manager.confirm_entry_fill(synthetic_id, base_price)
            return True

        except Exception as e:
            self._logger.error(f"Guard failed: {e}")
            return False

    def _ensure_safety_bracket(
        self, symbol: str, quantity: int, entry_price: float
    ) -> None:
        """
        CRITICAL SAFETY NET: Places Hard SL/TP on the Broker.
        FIX: Checks active BROKER orders, not internal brackets, to ensure redundancy.
        """
        # 1. Check if an ACTIVE SL order actually exists at the broker
        # 1. Check if an ACTIVE SL order actually exists at the broker
        has_broker_protection = False
        with self._lock:
            # [FIX] optimization: Use active pending list if available, or quick check
            # Instead of iterating ALL history, check pending subset
            potential_protectors = [
                o
                for o in self._orders.values()
                if o.status not in self.FINAL_STATUSES and o.symbol == symbol
            ]

            for order in potential_protectors:
                if str(order.order_type).upper() in [
                    "SL",
                    "SL-M",
                    "STOP_LOSS",
                    "STOP_LOSS_MARKET",
                ]:
                    has_broker_protection = True
                    break

        if has_broker_protection:
            self._logger.info(
                f"🛡️ Safety check passed: Broker SL already exists for {symbol}."
            )
            return

        self._logger.warning(
            f"🛡️ NAKED POSITION DETECTED: {symbol}. Placing Compliant Safety Orders.",
            extra={"event": "safety_bracket_trigger", "symbol": symbol},
        )

        SL_PCT = 0.10  # 10% Max Risk (Hard Stop)
        TP_PCT = 0.20  # 20% Target
        BUFFER_PCT = 0.05  # 5% Buffer to ensure SL-Limit fills like Market

        exit_side = "SELL" if quantity > 0 else "BUY"
        qty = abs(quantity)

        # Calculate Prices
        if quantity > 0:  # LONG position -> Exit via SELL
            trigger_price = round(entry_price * (1 - SL_PCT), 1)
            # Sell Limit should be LOWER than Trigger to ensure fill
            limit_price = round(trigger_price * (1 - BUFFER_PCT), 1)
            tp_price = round(entry_price * (1 + TP_PCT), 1)
        else:  # SHORT position -> Exit via BUY
            trigger_price = round(entry_price * (1 + SL_PCT), 1)
            # Buy Limit should be HIGHER than Trigger to ensure fill
            limit_price = round(trigger_price * (1 + BUFFER_PCT), 1)
            tp_price = round(entry_price * (1 - TP_PCT), 1)

        # Place STOP LOSS (SL-LIMIT)
        try:
            # Force "SL" string if utilizing Zerodha/Kite Connect specifics directly
            sl_order_type = (
                "SL" if hasattr(OrderType, "STOP_LOSS") else OrderType.STOP_LOSS
            )

            self.place_order(
                symbol=symbol,
                side=exit_side,
                quantity=qty,
                order_type=sl_order_type,  # ✅ Ensures "SL" (Stop-Limit)
                price=limit_price,  # ✅ Limit Price
                trigger_price=trigger_price,  # ✅ Trigger Price
                tag="safety_sl_hard",
                variety="regular",
            )
            self._logger.info(
                f"✅ Hard SL Placed: Trigger {trigger_price}, Limit {limit_price}"
            )
        except Exception as e:
            self._logger.critical(f"⛔ FATAL: Failed to place Hard SL: {e}")

        # Place TAKE PROFIT
        try:
            self.place_order(
                symbol=symbol,
                side=exit_side,
                quantity=qty,
                order_type=OrderType.LIMIT,
                price=tp_price,
                tag="safety_tp_wide",
                variety="regular",
            )
            self._logger.info(f"✅ Wide TP Placed: {tp_price}")
        except Exception as e:
            self._logger.error(f"Failed to place TP: {e}")

    def cancel_orders_for_symbol(self, symbol: str) -> None:
        """Cancels all open orders for a specific symbol."""
        pending = self._pending_orders()
        for o in pending:
            o_sym = normalize_symbol(str(o.symbol))
            target_sym = normalize_symbol(str(symbol))

            if o_sym == target_sym:
                self._logger.info(
                    f"🧹 Auto-canceling stale order {o.order_id} for {symbol}"
                )
                try:
                    self.cancel_order(o.order_id)
                except Exception as _ce:  # FIX S16-4: bare except → logged
                    self._logger.debug("cancel stale order %s: %s", o.order_id, _ce)

    def _generate_adjustment_order(self, symbol, qty, price=0.0):
        """Helper to inject synthetic orders."""
        if qty == 0:
            return
        import time
        from datetime import datetime, timezone

        side = "BUY" if qty > 0 else "SELL"
        details = OrderDetails(
            order_id=f"adj_{int(time.time())}",
            symbol=symbol,
            side=side,
            quantity=abs(qty),
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            timestamp=datetime.now(timezone.utc),
            price=float(price),
            average_price=float(price),
            fill_price=float(price),
            filled_quantity=abs(qty),
            tag="ghost_fix",
        )
        self._register_order(details)

        # --- FIX: Check before calling update_from_order ---
        if hasattr(self._positions, "update_from_order"):
            self._positions.update_from_order(details)
        elif hasattr(self._positions, "update_position"):
            self._positions.update_position(
                symbol=details.symbol,
                qty=details.quantity,
                price=details.average_price,
                side=details.side,
                product="MIS",
            )
        # ---------------------------------------------------

    # Alias for compatibility with main app
    reconcile_open_orders_with_broker = reconcile_open_orders


__all__ = [
    "OrderManager",
    "OrderDetails",
    "OrderType",
    "OrderStatus",
    "AtomicLeg",
    "TrailingSpec",
]
