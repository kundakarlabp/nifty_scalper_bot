"""Order lifecycle management utilities."""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from contextlib import suppress
from dataclasses import asdict, dataclass, field
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
from nifty_scalper_bot.core.trading_switch import TradingSwitchState, trading_switch
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.persistent_state import (
    BracketDict,
    PersistentStateManager,
)
from nifty_scalper_bot.execution import exceptions as execution_exceptions
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
from nifty_scalper_bot.execution.position_manager import PositionManager
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
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.metrics import Counter, Gauge
from nifty_scalper_bot.utils.pricing import canonical_price_source
from nifty_scalper_bot.utils.rate_limiter import RateLimiter
from nifty_scalper_bot.utils.reasons import canonical

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


@dataclass(slots=True)
class OrderDetails:
    order_id: str
    symbol: str
    side: str
    order_type: OrderType
    quantity: int
    price: float
    status: OrderStatus
    timestamp: datetime
    filled_quantity: int = 0
    fill_price: float | None = None
    rejection_reason: str | None = None
    parent_order_id: str | None = None
    child_order_ids: list[str] = field(default_factory=list)
    client_order_id: str | None = None


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
    """Manage complete order lifecycle."""

    POLL_INTERVAL_SEC: float = 2.0
    MAX_RETRIES: int = 3
    RETRY_BLACKLIST: tuple[str, ...] = (
        "insufficient funds",
        "invalid symbol",
        "market closed",
    )
    BRACKET_ENTRY_TIMEOUT_SEC: float = 5.0
    FINAL_STATUSES: tuple[OrderStatus, ...] = (
        OrderStatus.CANCELLED,
        OrderStatus.FILLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    )

    def __init__(
        self,
        broker_client: BaseBrokerClient,
        position_manager: PositionManager,
        rate_limiter: RateLimiter,
        instrument_resolver: Any | None = None,
        history_path: str | Path | None = None,
    ):
        """Initialize with broker client and position manager."""

        self._broker = broker_client
        self._positions = position_manager
        self._limiter = rate_limiter
        self._logger = get_logger(__name__)
        self._broker_circuit = CircuitBreaker()
        self._history_path = Path(history_path or Path("data") / "order_history.json")
        self._orders: dict[str, OrderDetails] = {}
        self._history: deque[OrderDetails] = deque(maxlen=1000)
        self._history_index: dict[str, int] = {}
        self._history_base_index = 0
        self._history_persist_path = Path("data/order_history_archive.jsonl")
        self._history_persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._history_persisted_ids: set[str] = set()
        self._notifier: TelegramEnhancedNotifier | None = None
        self._bracket_manager: BracketManager | None = None
        self._lock = RLock()
        self._stop_event = Event()
        self._monitor_thread: Thread | None = None
        self._market_data: MarketDataManager | None = None
        self._data_hub: DataHub | None = None
        self._instrument_resolver = instrument_resolver
        self._resolver = instrument_resolver
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
        timeout_override = os.getenv("NSB__BRACKET_ENTRY_TIMEOUT_SEC")
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

    def set_market_data_manager(self, market_data_manager: MarketDataManager) -> None:
        """Inject the shared market data manager instance."""

        self._market_data = market_data_manager
        self._configure_options_policy()

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

    def set_broker_client(self, broker_client: Any) -> None:
        """Swap the underlying broker client used for routing orders."""

        self._broker = broker_client
        self._broker_circuit = CircuitBreaker()
        self._refresh_margin_engine()

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
                tag="PANIC_BUTTON"
            )

    def set_bracket_manager(self, bracket_manager: BracketManager | None) -> None:
        """Attach bracket manager responsible for OCO coordination."""

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

    def stop_trailing(self, entry_order_id: str) -> bool:
        """Stop and remove a trailing stop controller if it exists."""

        record = self._trailing.pop(entry_order_id, None)
        if record is None:
            return False
        controller, callback = record
        try:
            if self._market_data is not None:
                self._market_data.unsubscribe(controller.symbol, callback)
            self._trailing_journal.delete(controller.order_id)
        finally:
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
        mdm = self._market_data
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
        mdm = self._market_data
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
        mdm = self._market_data
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
    ) -> str | None:
        """Place an order with smart retry logic and risk checks.

        Returns:
            Order ID if successful, None otherwise.
        """
        start_time = time.monotonic()
        normalized_symbol = symbol.strip().upper()
        
        # 1. Trading Switch Guard
        if not trading_switch.can_trade_new():
            self._logger.warning(
                "Order blocked: Trading switch is OFF",
                extra={"symbol": normalized_symbol}
            )
            return None

        # 2. Risk Checks
        if check_risk and self._risk_manager:
            # [FIX] OrderSignal is now correctly imported
            signal = OrderSignal(
                symbol=normalized_symbol,
                side=side,
                quantity=quantity,
                price=price or 0.0,
                stop_loss=None,
                take_profit=None
            )
            # Check live/shadow status dynamically
            live = self._is_live_enabled() and not self._is_shadow_mode()
            allowed, reason = self._risk_manager.check_order(signal, live_enabled=live)
            
            if not allowed:
                self._logger.warning(
                    f"Risk Manager Blocked Order: {reason}",
                    extra={"symbol": normalized_symbol, "reason": reason}
                )
                if self.on_order_rejected:
                    self.on_order_rejected(normalized_symbol, reason)
                return None

        # 3. Execution with Retry Logic
        exchange = self._resolve_exchange(normalized_symbol)
        tradingsymbol = self._resolve_tradingsymbol(normalized_symbol)
        
        params = {
            "variety": variety,
            "exchange": exchange,
            "tradingsymbol": tradingsymbol,
            "transaction_type": side,
            "quantity": quantity,
            "product": product,
            "order_type": order_type.value,
            "validity": "DAY"
        }
        if price and price > 0:
            params["price"] = price
        if trigger_price and trigger_price > 0:
            params["trigger_price"] = trigger_price
        if tag:
            params["tag"] = tag

        self._logger.info(f"🚀 Sending Order: {side} {quantity} {normalized_symbol} @ {price or 'MKT'}")
        
        # [OPTIMIZATION] Retry loop for transient network errors
        attempts = 3
        last_exception = None
        
        for attempt in range(1, attempts + 1):
            try:
                # Blocking call to broker
                order_id = self._broker.place_order(**params)
                
                if order_id:
                    # 4. Register Order Locally
                    details = OrderDetails(
                        order_id=order_id,
                        symbol=normalized_symbol,
                        quantity=quantity,
                        order_type=order_type,
                        status=OrderStatus.PENDING,
                        price=price,
                        trigger_price=trigger_price,
                        tag=tag
                    )
                    self._register_order(details)
                    
                    # Metrics
                    duration = time.monotonic() - start_time
                    self._m_order_latency.observe(duration)
                    self._m_orders_placed.labels(side=side, type=order_type.value, status="submitted").inc()
                    
                    self._logger.info(
                        f"✅ Order Placed: {order_id} (Latency: {duration*1000:.2f}ms)",
                        extra={"order_id": order_id}
                    )
                    return order_id
                    
            except Exception as e:
                last_exception = e
                # Only retry network-like errors (timeouts, 503s), not logic errors
                error_str = str(e).lower()
                if "network" in error_str or "timeout" in error_str or "503" in error_str:
                    self._logger.warning(
                        f"Order retry {attempt}/{attempts}: {e}",
                        extra={"symbol": normalized_symbol}
                    )
                    time.sleep(0.1 * attempt) # Exponential backoff
                    continue
                else:
                    # Logic error (e.g. Insufficient Funds) - Break immediately
                    break

        self._logger.error(
            f"Broker execution failed after retries: {last_exception}", 
            exc_info=True,
            extra={"symbol": normalized_symbol}
        )
        return None

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

    def execute_bracket_trade(
        self,
        *,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        product: str | None = None,
        tag: str | None = None,
        trailing_spec: TrailingSpec | None = None,
        partial_profit_fraction: float = 0.0,
        second_target_price: float | None = None,
    ) -> tuple[str, str, str]:
        """Execute bracket with margin checks, trailing, and partial profits.

        Args:
            symbol: Instrument identifier for the trade.
            side: Trade direction (``BUY`` enters long, ``SELL`` enters short).
            quantity: Requested order quantity.
            entry_price: Limit entry price (<=0 treats as market).
            stop_loss: Protective stop-loss trigger price.
            take_profit: Initial take-profit price for the first exit leg.
            product: Optional broker product code.
            tag: Optional broker tag string for audit grouping.
            trailing_spec: Optional trailing stop specification.
            partial_profit_fraction: Fraction of quantity for the first target.
            second_target_price: Optional price for the follow-up target.

        Returns:
            Tuple containing entry, stop-loss, and first take-profit order IDs.

        Raises:
            OrderPlacementError: If placement fails after risk checks.
        """

        self._logger.debug(
            "Entered execute_bracket_trade",
            extra={
                "event": "execute_bracket_trade_enter",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            },
        )
        self._validate_quantity(symbol, quantity)
        if not self._ensure_trading_allowed(
            symbol=symbol, side=side, quantity=quantity
        ):
            self._logger.info(
                "Condition met: trading disabled for bracket",
                extra={"event": "bracket_blocked"},
            )
            return "", "", ""

        try:
            margin_ok, reason, meta = self._precheck_margin(
                symbol=symbol,
                side=side,
                quantity=quantity,
                product=product,
                price=entry_price,
                stop_loss=stop_loss,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in execute_bracket_trade: %s",
                exc,
                extra={"event": "execute_bracket_margin_failure", "symbol": symbol},
            )
            raise OrderPlacementError("Margin planning failed") from exc
        if not margin_ok:
            self._logger.info(
                "Condition met: margin block for bracket",
                extra={
                    "event": "order_blocked",
                    "reason": canonical(reason),
                    "needed": meta.get("needed"),
                    "available": meta.get("available"),
                },
            )
            return "", "", ""

        entry_order_type = (
            OrderType.MARKET
            if (entry_price is None or float(entry_price) <= 0.0)
            else OrderType.LIMIT
        )
        try:
            if entry_order_type == OrderType.MARKET:
                entry = self._execute_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    product=product,
                    tag=tag,
                )
            else:
                entry = self._place_single_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=entry_order_type,
                    price=entry_price,
                    product=product,
                    tag=tag,
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in execute_bracket_trade entry: %s",
                exc,
                extra={"event": "execute_bracket_entry_failed", "symbol": symbol},
            )
            raise

        try:
            filled_entry = self._await_entry_fill(
                entry,
                timeout=max(float(self.BRACKET_ENTRY_TIMEOUT_SEC), 0.5),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in execute_bracket_trade: %s",
                exc,
                extra={
                    "event": "execute_bracket_entry_wait_failed",
                    "symbol": symbol,
                },
            )
            raise OrderPlacementError("Entry fill confirmation failed") from exc

        if filled_entry is None or filled_entry.filled_quantity <= 0:
            self._logger.info(
                "Condition met: bracket entry unfilled",
                extra={
                    "event": "bracket_entry_unfilled",
                    "entry_id": entry.order_id,
                    "symbol": symbol,
                },
            )
            return "", "", ""

        entry = filled_entry
        filled_quantity = int(max(entry.filled_quantity, 0))
        if filled_quantity <= 0:
            self._logger.info(
                "Condition met: entry filled quantity zero",
                extra={
                    "event": "bracket_entry_zero_qty",
                    "entry_id": entry.order_id,
                    "symbol": symbol,
                },
            )
            return "", "", ""

        exit_side: Literal["BUY", "SELL"] = "SELL" if side == "BUY" else "BUY"
        effective_entry_price = (
            entry.fill_price or entry.price or float(entry_price or 0.0)
        )
        child_ids: list[str] = []
        stop_details: OrderDetails
        try:
            stop_details = self._place_single_order(
                symbol=symbol,
                side=exit_side,
                quantity=filled_quantity,
                order_type=OrderType.STOP_LOSS_MARKET,
                price=stop_loss,
                product=product,
                tag=tag,
                parent_order_id=entry.order_id,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.critical(
                "Failure in execute_bracket_trade stop: %s",
                exc,
                extra={"event": "execute_bracket_stop_failed", "symbol": symbol},
            )
            self._handle_failed_bracket_entry(
                entry_details=entry,
                exit_side=exit_side,
                product=product,
                tag=tag,
                original_exception=exc,
            )
            raise OrderPlacementError("Stop-loss placement failed") from exc
        child_ids.append(stop_details.order_id)

        fraction = float(partial_profit_fraction or 0.0)
        # [FIX] Lot-Aware Sizing
        lot_size = self._lot_size_for_symbol(symbol)
        
        tp_primary_qty = filled_quantity
        tp_secondary_qty = 0
        
        if 0 < fraction < 1:
            # [FIX] Lot-aware sizing to prevent broker rejection
            raw_target = int(filled_quantity * fraction)
            # Ensure we have a valid lot size using the proper resolver
            try:
                lot_size = self._lot_size_for_symbol(symbol)
            except Exception:
                lot_size = 1
            
            # Snap calculation to floor lot chunks (e.g. 37 -> 25 if lot is 25)
            lots_count = raw_target // lot_size
                        
            if lots_count == 0:
            # If fraction results in < 1 lot, force full exit at TP2
                tp_primary_qty = 0
            else:
                tp_primary_qty = lots_count * lot_size
                            
            tp_secondary_qty = filled_quantity - tp_primary_qty
                      
            # Failsafe: if primary became 0, move everything to secondary
            if tp_primary_qty == 0:
                tp_primary_qty = filled_quantity
                tp_secondary_qty = 0
        else:
            fraction = 0.0

        tp_details: OrderDetails | None = None
        if tp_primary_qty > 0:
            try:
                tp_details = self._place_single_order(
                    symbol=symbol,
                    side=exit_side,
                    quantity=tp_primary_qty,
                    order_type=OrderType.LIMIT,
                    price=take_profit,
                    product=product,
                    tag=tag,
                    parent_order_id=entry.order_id,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in execute_bracket_trade TP1: %s",
                    exc,
                    extra={"event": "execute_bracket_tp_failed", "symbol": symbol},
                )
                tp_details = None
        if tp_details is not None:
            child_ids.append(tp_details.order_id)

        with self._lock:
            entry.child_order_ids = list(child_ids)
            self._register_order(entry)

        state = BracketState(
            entry_id=entry.order_id,
            symbol=symbol,
            side=side,
            exit_side=exit_side,
            total_quantity=filled_quantity,
            entry_price=float(effective_entry_price),
            product=product,
            tag=tag,
            stop_order_id=stop_details.order_id,
            stop_price=float(stop_loss),
            stop_order_type=OrderType.STOP_LOSS_MARKET,
            tp_primary_id=tp_details.order_id if tp_details else None,
            tp_primary_price=float(take_profit),
            tp_primary_qty=tp_primary_qty if tp_details else 0,
            tp_secondary_qty=tp_secondary_qty,
            trailing_spec=trailing_spec,
            partial_fraction=fraction,
            second_target_price=second_target_price,
        )
        self._register_bracket_state(state)

        if self._bracket_manager is not None:
            try:
                self._bracket_manager.register_bracket(
                    entry_order_id=entry.order_id,
                    stop_loss_order_id=stop_details.order_id,
                    target_order_id=tp_details.order_id if tp_details else None,
                    tp2_order_id=None,
                    entry_quantity=filled_quantity,
                )
                self._logger.info(
                    "Bracket registered with external manager",
                    extra={
                        "event": "execute_bracket_trade_registered",
                        "entry_id": entry.order_id,
                        "stop_id": stop_details.order_id,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure registering bracket with manager: %s",
                    exc,
                    extra={
                        "event": "execute_bracket_trade_register_failed",
                        "entry_id": entry.order_id,
                    },
                    exc_info=exc,
                )

        if trailing_spec is not None:
            try:
                self.attach_trailing_stop(
                    entry_order_id=entry.order_id,
                    sl_order_id=stop_details.order_id,
                    symbol=symbol,
                    side=side,
                    entry_price=effective_entry_price,
                    spec=trailing_spec,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to attach trailing stop for %s: %s", symbol, exc
                )

        self._logger.info(
            "Condition met: bracket orders active",
            extra={
                "event": "execute_bracket_trade_success",
                "entry_id": entry.order_id,
                "stop_id": stop_details.order_id,
                "tp_id": tp_details.order_id if tp_details else None,
            },
        )
        return (
            entry.order_id,
            stop_details.order_id,
            tp_details.order_id if tp_details is not None else "",
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
    ) -> tuple[str, str, str]:
        """Place bracket order and return entry, stop-loss, and take-profit IDs.

        Args:
            symbol: Instrument identifier for the bracket order.
            side: Entry side (``BUY`` for long, ``SELL`` for short).
            quantity: Requested quantity for the entry leg.
            entry_price: Desired entry price (<=0 treated as market).
            stop_loss: Protective stop-loss trigger price.
            take_profit: Initial take-profit price.
            product: Optional broker product code.
            tag: Optional broker tag string.
            trailing_spec: Optional trailing stop specification.

        Returns:
            Tuple containing entry, stop-loss, and take-profit order identifiers.

        Raises:
            OrderPlacementError: If the bracket cannot be established.
        """

        return self.execute_bracket_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            product=product,
            tag=tag,
            trailing_spec=trailing_spec,
        )

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
                    extra={"entry_id": state.entry_id}
                )
                time.sleep(0.5)
        if not replacement:
            # CRITICAL: Failed to replace stop. Position is NAKED.
            # ACTION: Trigger Emergency Exit immediately.
            self._logger.critical(
                "CRITICAL: Failed to replace Stop Loss. Executing EMERGENCY EXIT.",
                extra={"event": "resize_stop_fatal_error", "entry_id": state.entry_id}
            )
            try:
                self._place_exit_order(
                    symbol=state.symbol,
                    side=state.exit_side,
                    quantity=new_quantity,
                    product=state.product,
                    tag="emergency_no_stop"
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
                state.trailing_spec = None # Mark as no longer trailing
                self._logger.error(
                    "Failure in _resize_stop_order trailing: %s. Stop is now STATIC.",
                    exc,
                    extra={"event": "resize_stop_trailing_failed", "entry_id": state.entry_id}
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
        """Process bracket state transitions for updated order.

        Args:
            order: Updated order snapshot from the broker.
            previous_status: Status before the latest update.
            payload: Raw broker payload for additional context.

        Returns:
            None.

        Raises:
            None.
        """

        entry_id = self._bracket_index.get(order.order_id)
        if not entry_id:
            return
        state = self._brackets.get(entry_id)
        if state is None:
            self._bracket_index.pop(order.order_id, None)
            return
        self._logger.debug(
            "Entered _handle_bracket_update",
            extra={
                "event": "handle_bracket_update",
                "entry_id": entry_id,
                "order_id": order.order_id,
                "status": order.status.value,
            },
        )
        if self._bracket_manager is not None:
            try:
                self._bracket_manager.handle_bracket_update(
                    order_id=order.order_id,
                    status=order.status.name,
                    filled_quantity=order.filled_quantity,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in external bracket update delegate: %s",
                    exc,
                    extra={
                        "event": "handle_bracket_update_delegate_failed",
                        "entry_id": entry_id,
                        "order_id": order.order_id,
                    },
                    exc_info=exc,
                )
        try:
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
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _handle_bracket_update: %s",
                exc,
                extra={"event": "handle_bracket_update_failed", "entry_id": entry_id},
            )

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
        """Cancel an order. Returns True if successful."""

        cancel = getattr(self._broker, "cancel_order", None)
        if cancel is None:
            raise NotImplementedError("Broker does not support order cancellation")
        response = self._call_broker(cancel, order_id)
        success = bool(response)
        if success:
            self._update_local_status(order_id, OrderStatus.CANCELLED)
        return success

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
            try:
                self._positions.update_order_status(
                    details.order_id, details.status.name, details.fill_price
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

    def modify_order(
        self,
        order_id: str,
        new_quantity: int | None = None,
        new_price: float | None = None,
    ) -> bool:
        """Modify an existing order."""

        modify = getattr(self._broker, "modify_order", None)
        if modify is None:
            raise NotImplementedError("Broker does not support order modifications")
        response = self._call_broker(
            modify, order_id, quantity=new_quantity, price=new_price
        )
        return bool(response)

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
                "status": order.status.value,
                "quantity": order.quantity,
                "filled_quantity": order.filled_quantity,
                "price": order.price,
                "fill_price": order.fill_price,
                "timestamp": order.timestamp.isoformat(),
                "rejection_reason": order.rejection_reason,
            }
            for order in orders
        ]

    def get_today_orders(self) -> list[OrderDetails]:
        """Get all orders placed today."""

        today = datetime.now(timezone.utc).date()
        with self._lock:
            return [order for order in self._history if order.timestamp.date() == today]

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

    def _monitor_orders(self) -> None:
        """Background thread that polls order status."""

        while not self._stop_event.wait(self.POLL_INTERVAL_SEC):
            pending = self._pending_orders()
            for order in pending:
                try:
                    self._refresh_order(order.order_id)
                except RateLimitError:
                    # Back off hard on rate limits
                    time.sleep(self.POLL_INTERVAL_SEC * 10)
                    break
                except Exception as exc:
                    # CRITICAL FIX: Check for fatal broker errors (401/403)
                    error_str = str(exc).lower()
                    if "401" in error_str or "403" in error_str or "access denied" in error_str:
                        self._logger.critical("FATAL: Broker session expired during monitoring. Stopping monitor.")
                        self._stop_event.set()
                        return
                    
                    self._logger.error("Order poll failed for %s: %s", order.order_id, exc)
                    time.sleep(1.0)

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

    # Internal helpers -------------------------------------------------

    def _pending_orders(self) -> list[OrderDetails]:
        with self._lock:
            return [
                order
                for order in self._orders.values()
                if order.status not in self.FINAL_STATUSES
            ]

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

    def _resolve_available_margin(self) -> tuple[float | None, str]:
        mdm = self._market_data
        if mdm is not None:
            try:
                mdm.refresh_margin_snapshot()
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _resolve_available_margin mdm refresh: %s",
                    exc,
                    extra={"event": "order_margin_mdm_refresh_error"},
                    exc_info=exc,
                )
            try:
                available = mdm.get_available_balance()
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in _resolve_available_margin mdm get: %s",
                    exc,
                    extra={"event": "order_margin_mdm_read_error"},
                    exc_info=exc,
                )
            else:
                if available is not None and available > 0:
                    return float(available), "mdm"
        risk_manager = self._risk_manager
        if risk_manager is not None:
            with suppress(Exception):
                balance = float(getattr(risk_manager, "current_balance", 0.0))
                if math.isfinite(balance) and balance > 0:
                    return balance, "risk"
        return None, "unknown"

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
        mdm = cast("MarketDataManager | None", self._market_data)

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
        """Assemble broker payload for a requested order placement.

        Args:
            symbol: Resolver key or broker tradingsymbol for the order.
            side: BUY or SELL side for the order.
            quantity: Total number of units to place.
            order_type: Desired order type (MARKET/LIMIT/etc.).
            price: Limit or trigger price depending on order type.
            product: Product code (MIS/NRML/etc.) if provided by caller.
            tag: Optional broker tag for analytics.
            parent_order_id: Parent identifier for multi-leg orders.
            client_order_id: Optional idempotency key supplied by caller.

        Returns:
            dict[str, Any]: Payload ready to send to the execution router.

        Raises:
            OrderPlacementError: If tokens remain mandatory but absent.
        """

        self._logger.debug(
            "Entered OrderManager._build_order_payload",
            extra={
                "event": "order_manager.build_order_payload",
                "symbol": symbol,
            },
        )
        order_type_value = (
            order_type.value if hasattr(order_type, "value") else order_type
        )
        resolver_metadata = self._resolve_order_metadata(symbol)
        tradingsymbol_meta = str(resolver_metadata.get("tradingsymbol") or "").strip()
        exchange_meta = resolver_metadata.get("exchange")
        instrument_token = resolver_metadata.get("instrument_token")
        if instrument_token is not None:
            try:
                instrument_token = int(instrument_token)
            except (TypeError, ValueError):
                instrument_token = None
        require_token = False
        try:
            runtime_settings = app_settings.get_settings()
            require_token = not bool(
                getattr(runtime_settings, "feature_order_without_token", True)
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _build_order_payload settings load: %s",
                exc,
                extra={"event": "order_manager.payload_settings_error"},
                exc_info=exc,
            )
        if require_token and instrument_token is None:
            self._logger.info(
                "Condition met: order_payload_missing_token",
                extra={
                    "event": "order_manager.payload_missing_token",
                    "symbol": symbol,
                },
            )
            raise OrderPlacementError("Instrument token required")
        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "quantity": int(quantity),
            "order_type": order_type_value,
            "product": product or "MIS",
        }
        exchange_hint = exchange_meta
        tradingsymbol_hint = tradingsymbol_meta
        if ":" in symbol:
            symbol_exchange, maybe_symbol = symbol.split(":", maxsplit=1)
            if not tradingsymbol_hint and maybe_symbol:
                tradingsymbol_hint = maybe_symbol
            if not exchange_hint and symbol_exchange:
                exchange_hint = symbol_exchange
        else:
            upper_symbol = str(symbol).upper()
            if not tradingsymbol_hint and upper_symbol.endswith(("FUT", "CE", "PE")):
                tradingsymbol_hint = symbol
        if tradingsymbol_hint:
            payload["tradingsymbol"] = tradingsymbol_hint
        payload["transaction_type"] = side

        normalized_type = str(order_type_value).lower()
        if price is not None:
            price_value = float(price)
            if normalized_type == OrderType.STOP_LOSS_MARKET.value:
                payload["trigger_price"] = price_value
            elif normalized_type == OrderType.STOP_LOSS.value:
                payload["price"] = price_value
                payload["trigger_price"] = price_value
            elif normalized_type != OrderType.MARKET.value:
                payload["price"] = price_value
        env_exch = os.getenv("INSTRUMENTS__TRADE_EXCHANGE")
        if env_exch:
            exchange_hint = env_exch
        if exchange_hint:
            payload["exchange"] = exchange_hint
        if tag:
            payload["tag"] = tag
        if parent_order_id is not None:
            payload["parent_order_id"] = parent_order_id
        if client_order_id:
            payload["client_order_id"] = client_order_id
        if instrument_token is not None:
            payload["instrument_token"] = instrument_token
        return payload

    def _submit_order_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
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
                response = self._call_broker(self._broker.place_order, payload)
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
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if client_order_id:
                    existing = self._lookup_existing_order(client_order_id)
                    if existing:
                        return existing
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
        status_str = str(raw_status).strip().lower()
        if not status_str:
            return OrderStatus.SUBMITTED
        mapping = {
            "open": OrderStatus.SUBMITTED,
            "put order request received": OrderStatus.SUBMITTED,
            "complete": OrderStatus.FILLED,
            "filled": OrderStatus.FILLED,
            "partial": OrderStatus.PARTIALLY_FILLED,
            "partially filled": OrderStatus.PARTIALLY_FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "cancelled by user": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
            "trigger pending": OrderStatus.PENDING,
            "pending": OrderStatus.PENDING,
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
            filled_qty = self._coerce_int(payload.get("filled_quantity")) or order.filled_quantity
            
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
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status.value,
            "timestamp": order.timestamp.isoformat(),
            "filled_quantity": order.filled_quantity,
            "fill_price": order.fill_price,
            "rejection_reason": order.rejection_reason,
            "parent_order_id": order.parent_order_id,
            "child_order_ids": list(order.child_order_ids),
            "client_order_id": order.client_order_id,
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
            "stop_order_id": state.stop_order_id,
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
        }
        return payload

    def _order_from_dict(self, payload: Mapping[str, Any]) -> OrderDetails:
        """Convert persistent *payload* into :class:`OrderDetails`.

        Args:
            payload: Mapping sourced from persistence layer.

        Returns:
            OrderDetails reconstructed from the payload.

        Raises:
            ValueError: If required fields are missing or invalid.
        """

        self._logger.debug(
            "Entered _order_from_dict",
            extra={"event": "order_manager_order_from_dict"},
        )
        try:
            order_id = str(payload["order_id"]).strip()
            symbol = str(payload["symbol"]).upper()
            side = str(payload["side"]).upper()
            order_type_raw = payload.get("order_type", OrderType.MARKET.value)
            status_raw = payload.get("status", OrderStatus.SUBMITTED.value)
            quantity = int(payload.get("quantity", 0))
            price = float(payload.get("price", 0.0))
            timestamp_raw = payload.get("timestamp")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid order payload") from exc
        if not order_id or not symbol:
            raise ValueError("Order payload missing identifiers")
        try:
            order_type = OrderType(order_type_raw)
        except Exception:  # noqa: BLE001 - fallback to market
            order_type = OrderType.MARKET
        try:
            status = OrderStatus(status_raw)
        except Exception:  # noqa: BLE001 - fallback to submitted
            status = OrderStatus.SUBMITTED
        try:
            timestamp = (
                datetime.fromisoformat(str(timestamp_raw))
                if timestamp_raw
                else datetime.now(timezone.utc)
            )
        except Exception:  # noqa: BLE001 - fallback to current time
            timestamp = datetime.now(timezone.utc)
        filled_quantity = int(payload.get("filled_quantity", 0) or 0)
        fill_price = payload.get("fill_price")
        if fill_price is not None:
            try:
                fill_price = float(fill_price)
            except (TypeError, ValueError):
                fill_price = None
        rejection_reason = payload.get("rejection_reason")
        parent_order_id = payload.get("parent_order_id")
        child_ids_raw = payload.get("child_order_ids", [])
        child_order_ids = [str(item) for item in child_ids_raw] if child_ids_raw else []
        client_order_id_raw = payload.get("client_order_id")
        client_order_id = (
            str(client_order_id_raw) if client_order_id_raw is not None else None
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
            filled_quantity=filled_quantity,
            fill_price=fill_price,
            rejection_reason=(
                str(rejection_reason) if rejection_reason is not None else None
            ),
            parent_order_id=(
                str(parent_order_id) if parent_order_id is not None else None
            ),
            child_order_ids=child_order_ids,
            client_order_id=client_order_id,
        )

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
            trailing_spec=None,
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
        """Reconcile persisted open orders with broker live state.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered reconcile_open_orders_with_broker",
            extra={"event": "order_manager_reconcile_orders"},
        )
        fetcher = self._resolve_open_orders_fetcher()
        if fetcher is None:
            self._logger.info(
                "Condition met: reconcile_open_orders_fetcher_missing",
                extra={"event": "order_manager_reconcile_skip"},
            )
            return
        try:
            response = self._call_broker(fetcher)
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in reconcile_open_orders_with_broker: %s", exc)
            return
        payloads: list[Mapping[str, Any]] = []
        if isinstance(response, Mapping):
            payloads.append(response)
        elif isinstance(response, Iterable) and not isinstance(response, (str, bytes)):
            payloads.extend(
                cast(Mapping[str, Any], item)
                for item in response
                if isinstance(item, Mapping)
            )
        broker_orders: dict[str, OrderDetails] = {}
        for item in payloads:
            detail = self._coerce_broker_open_order(item)
            if detail is None:
                continue
            broker_orders[detail.order_id] = detail
        open_ids = set(broker_orders.keys())
        changed: dict[str, OrderDetails] = {}
        with self._lock:
            for order_id, detail in broker_orders.items():
                existing = self._orders.get(order_id)
                if existing is None:
                    self._orders[order_id] = detail
                    self._append_history(detail)
                    if detail.client_order_id:
                        self._client_order_index[str(detail.client_order_id)] = order_id
                    changed[order_id] = detail
                else:
                    if existing.status != detail.status or (
                        detail.filled_quantity
                        and existing.filled_quantity != detail.filled_quantity
                    ):
                        existing.status = detail.status
                        existing.filled_quantity = detail.filled_quantity
                        existing.fill_price = detail.fill_price
                        self._history_persisted_ids.discard(order_id)
                        changed[order_id] = existing
            for order_id, existing in list(self._orders.items()):
                if existing.status in self.FINAL_STATUSES:
                    continue
                if order_id in open_ids:
                    continue
                existing.status = OrderStatus.FILLED
                if existing.filled_quantity <= 0:
                    existing.filled_quantity = existing.quantity
                changed[order_id] = existing
            stale_brackets = [
                entry_id
                for entry_id in list(self._brackets.keys())
                if entry_id not in open_ids
            ]
            if changed:
                self._persist_history()
        for order in changed.values():
            self._publish_order_to_hub(order)
            self._persist_order_snapshot(order)
        for entry_id in stale_brackets:
            self._cleanup_bracket_state(entry_id)
        if changed or stale_brackets:
            self._logger.info(
                "Condition met: reconcile_open_orders_applied",
                extra={
                    "event": "order_manager_reconcile_orders",
                    "orders_updated": len(changed),
                    "brackets_removed": len(stale_brackets),
                },
            )

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
        if not resolver: 
            return None
        if resolver is None:
            market_data = self._market_data
            resolver = getattr(market_data, "resolver", None) if market_data else None
            if resolver is None and market_data is not None:
                resolver = getattr(market_data, "_resolver", None)
        lookup = getattr(resolver, "lot_size_for_symbol", None)
        return lookup if callable(lookup) else None

    def _lot_size_for_symbol(self, symbol: str) -> int:
        lookup = self._lot_lookup()
        if lookup is None:
            raise OrderPlacementError("Instrument resolver with lot sizes required")
        try:
            resolved = lookup(symbol)
        except Exception as exc:  # noqa: BLE001 - surface resolver error
            self._logger.debug("lot_size_lookup_failed", exc_info=True)
            raise OrderPlacementError("Failed to resolve lot size") from exc
        if resolved is None:
            raise OrderPlacementError("No lot size available for symbol")
        try:
            lot_size = int(resolved)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise OrderPlacementError("Invalid lot size returned by resolver") from exc
        if lot_size <= 0:
            raise OrderPlacementError("Lot size must be positive")
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


__all__ = [
    "OrderManager",
    "OrderDetails",
    "OrderType",
    "OrderStatus",
    "AtomicLeg",
    "TrailingSpec",
]
