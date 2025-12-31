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
from nifty_scalper_bot.data.trade_store import TradeStore, TradeIntent
from nifty_scalper_bot.data.persistent_state import (
    BracketDict,
    PersistentStateManager,
)
from nifty_scalper_bot.execution import exceptions as execution_exceptions
from nifty_scalper_bot.execution.broker_rejects import BrokerReject
from nifty_scalper_bot.execution.execution_policy import ExecutionPolicy
from nifty_scalper_bot.execution.adaptive_trailing import AdaptiveTrailingController
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
    from nifty_scalper_bot.risk.risk_manager import RiskManager, OrderSignal
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
    pending_quantity: int = 0
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    tag: str | None = None
    parent_order_id: str | None = None
    child_order_ids: list[str] = field(default_factory=list)
    client_order_id: str | None = None
    rejection_reason: str | None = None

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
        indicator_engine: Any | None = None,
    ):
        """Initialize with broker client and position manager."""

        self._broker = broker_client
        self._positions = position_manager
        self._limiter = rate_limiter
        self.trade_store = TradeStore()
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
        # ✅ FIX: Restore state on startup
        self._load_orders()

    # ✅ ADDED: Missing _configure_options_policy method
    def _configure_options_policy(self) -> None:
        """
        Refresh options execution policy with current market data manager.
        Called on init and when components are updated.
        """
        if self._market_data:
            self._options_policy.set_market_data(self._market_data)
        
        # Also ensure resolver is linked if available
        if hasattr(self._options_policy, "set_instrument_resolver") and self._resolver:
             self._options_policy.set_instrument_resolver(self._resolver)

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
            self._tp_controllers[tp_order_id] = controller
            
            # Subscribe to ticks
            self._market_data.subscribe(symbol, controller.on_tick)
            self._logger.info(f"🚀 Dynamic TP attached to {tp_order_id}")
            
        except Exception as e:
            self._logger.error(f"Failed to attach Dynamic TP: {e}")

    def stop_dynamic_tp(self, tp_order_id: str) -> None:
        """Stop and remove a dynamic TP controller."""
        if not hasattr(self, "_tp_controllers"):
            return
            
        controller = self._tp_controllers.pop(tp_order_id, None)
        if controller:
            self._market_data.unsubscribe(controller.symbol, controller.on_tick)

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
                pass
        
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
                pass
        
        # 3. Fallback: Return as is (assuming it's already a tradingsymbol)
        return symbol

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
        # ✅ NEW: Idempotency Arguments
        signal_id: str | None = None,
        strategy_name: str = "manual",
    ) -> str | None:
        """
        Execute order with Idempotency, Safe Trading Window, Risk Gating, and Auto-Recovery.
        """
        import time
        import hashlib
        from datetime import datetime, timezone, time as dtime
        from zoneinfo import ZoneInfo
        from nifty_scalper_bot.core.trading_switch import trading_switch
        from nifty_scalper_bot.risk import OrderSignal
        
        # Lazy load TradeStore to avoid circular imports during init
        if not hasattr(self, "trade_store"):
            from nifty_scalper_bot.data.trade_store import TradeStore
            self.trade_store = TradeStore()
        from nifty_scalper_bot.data.trade_store import TradeIntent

        normalized_symbol = symbol.strip().upper()
        # ---------------------------------------------------------------------
        # 🛑 FIX 1: Smart Idempotency with Timeout
        # ---------------------------------------------------------------------
        with self._lock:
            current_time = time.time()
            # Check for any pending orders on this symbol same side
            pending_orders = [
                o for o in self._orders.values()
                if o.symbol == normalized_symbol 
                and o.side == side
                and o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
                # TIMEOUT SAFETY: Only block if order is fresh (< 45 seconds old)
                # This prevents getting stuck forever if an order is lost in limbo
                and (current_time - getattr(o.timestamp, "timestamp", lambda: 0)()) < 45
            ]
            
            if pending_orders:
                self._logger.warning(
                    f"🚫 BLOCKED: Fresh pending order exists for {normalized_symbol}. Ignored to prevent duplicate.",
                    extra={"event": "duplicate_block", "symbol": normalized_symbol}
                )
                return None

        # ---------------------------------------------------------------------
        # 1. IDEMPOTENCY CHECK (The Fix for Duplicate Trades)
        # ---------------------------------------------------------------------
        if signal_id and self.trade_store.exists_by_signal(signal_id):
            self._logger.warning(
                f"🛑 DUPLICATE BLOCKED: Signal {signal_id} already traded.",
                extra={"symbol": normalized_symbol, "event": "duplicate_block"}
            )
            return None

        # ---------------------------------------------------------------------
        # 2. TIME GUARD (Safe Window: 09:30 - 15:15 IST)
        # ---------------------------------------------------------------------
        if variety == "regular":
            try:
                ist = ZoneInfo("Asia/Kolkata")
                now = datetime.now(ist).time()
                safe_start = dtime(9, 30)
                safe_end = dtime(15, 15)
                market_open = dtime(9, 15)
                market_close = dtime(15, 30)

                if not (safe_start <= now <= safe_end):
                    reason = "Market Closed"
                    if market_open <= now < safe_start:
                        reason = f"Opening Volatility Buffer (Wait until {safe_start.strftime('%H:%M')})"
                    elif safe_end < now <= market_close:
                        reason = f"EOD Safety Cutoff (No trades after {safe_end.strftime('%H:%M')})"
                    
                    self._logger.warning(
                        f"🛑 Order Blocked: {reason}. Current Time: {now.strftime('%H:%M:%S')}",
                        extra={"symbol": normalized_symbol, "event": "time_guard_block"}
                    )
                    return None
            except Exception as e:
                self._logger.error(f"Time Guard Check Failed: {e}. Proceeding with caution.")

        # ---------------------------------------------------------------------
        # 3. TRADING SWITCH GUARD
        # ---------------------------------------------------------------------
        switch_instance = trading_switch() if callable(trading_switch) else trading_switch
        checker = getattr(switch_instance, "can_trade", getattr(switch_instance, "can_trade_new", None))
        
        if callable(checker) and not checker():
            self._logger.warning("Order blocked: Trading Switch is OFF", extra={"symbol": normalized_symbol})
            return None

        # ---------------------------------------------------------------------
        # 4. RISK MANAGER VALIDATION
        # ---------------------------------------------------------------------
        if check_risk and self._risk_manager:
            signal = OrderSignal(
                symbol=normalized_symbol, side=side, quantity=quantity,
                price=price or 0.0, stop_loss=stop_loss, take_profit=take_profit
            )
            is_live = False
            if hasattr(self, "_enable_live_getter") and self._enable_live_getter:
                is_live = self._enable_live_getter()
            elif hasattr(self, "_resolve_enable_live"):
                is_live = self._resolve_enable_live()
            
            allowed, reason = self._risk_manager.check_order(signal, live_enabled=is_live)
            if not allowed:
                self._logger.warning(f"Risk Block: {reason}", extra={"symbol": normalized_symbol, "event": "risk_block"})
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
        unique_client_id = f"bot_{signal_id[-12:]}" # Max 20 chars usually

        # Persist Intent to Disk
        intent = TradeIntent(
            trade_id=trade_id,
            symbol=normalized_symbol,
            signal_id=signal_id,
            strategy=strategy_name,
            side=side,
            qty=quantity,
            timestamp=time.time(),
            status="SUBMITTED"
        )
        self.trade_store.add_trade(intent)

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
                "STOPLOSSLIMIT": "SL"
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
        if final_order_type in {"SL", "SL-M"} and (price is None or price <= 0.0) and trigger_price:
            buffer_pct = 0.03  # 3% Buffer
            if normalized_side == "BUY":  # Short Exit -> Buy higher
                price = round(trigger_price * (1 + buffer_pct), 2)
            else:  # Long Exit -> Sell lower
                price = round(trigger_price * (1 - buffer_pct), 2)
            
            self._logger.info(
                f"🛡️ Converted SL-M to SL Limit. Trigger: {trigger_price}, Limit: {price}",
                extra={"event": "order.slm_to_sl_conversion", "trigger": trigger_price, "limit": price}
            )
            final_order_type = "SL"  # Force SL (not SL-M)

        self._logger.info(
            f"🚀 Sending Order: {normalized_side} {quantity} {normalized_symbol} ({final_order_type})",
            extra={"event": "order_sending", "symbol": normalized_symbol, "signal_id": signal_id}
        )

        # ---------------------------------------------------------------------
        # 7. EXECUTION LOOP (With Anti-Zombie Timeout)
        # ---------------------------------------------------------------------
        # Helper for threaded execution
        def _broker_call(kwargs):
            try:
                return self._broker.place_order(**kwargs)
            except Exception as exc:
                return exc

        # ✅ DEFINE CALL_ARGS OUTSIDE THE LOOP FIRST TO AVOID UnboundLocalError
        call_args = {
            "symbol": normalized_symbol, 
            "side": normalized_side,  # ✅ Already normalized String
            "quantity": quantity, 
            "product": product, 
            "order_type": final_order_type, # ✅ Already normalized String
            "price": price, 
            "trigger_price": trigger_price,
            "tag": tag, 
            "variety": variety, 
            "client_order_id": unique_client_id
        }

        for attempt in range(1, 4):
            # -----------------------------------------------------------------
            # ✅ FIX: Re-hydrate Enums to prevent Adapter Crash
            # The broker adapter expects Enum objects (e.g. OrderType.MARKET).
            # If we pass a string "MARKET", it crashes on .value access.
            # -----------------------------------------------------------------
            if isinstance(call_args["order_type"], str):
                ot_str = call_args["order_type"]
                if ot_str == "MARKET": call_args["order_type"] = OrderType.MARKET
                elif ot_str == "LIMIT": call_args["order_type"] = OrderType.LIMIT
                elif ot_str == "SL": call_args["order_type"] = OrderType.STOP_LOSS
                elif ot_str == "SL-M": call_args["order_type"] = OrderType.STOP_LOSS_MARKET
            try:
                # ✅ Run in thread with 3s timeout to prevent hanging
                result_holder = {"resp": None}
                
                def target():
                    result_holder["resp"] = _broker_call(call_args)

                # We use the 'Thread' class already imported at top of file
                t = Thread(target=target, name=f"ord_{unique_client_id}", daemon=True)
                t.start()
                t.join(timeout=3.0) # Strict 3s timeout

                if t.is_alive():
                    self._logger.critical(f"🚨 Broker API hung on attempt {attempt}! Timeout forced.")
                    raise TimeoutError("Broker API call timed out (3s)")

                response = result_holder["resp"]
                
                # Re-raise exceptions captured in thread
                if isinstance(response, Exception):
                    raise response

                # --- Success Logic ---
                order_id = response.get("order_id") if isinstance(response, dict) else str(response)
                
                if order_id:
                    # A. Update Trade Store
                    self.trade_store.update_status(trade_id, "FILLED", order_id)

                    # B. Register Order Locally
                    details = OrderDetails(
                        order_id=order_id, symbol=normalized_symbol, side=side,
                        order_type=order_type, quantity=quantity, price=float(price or 0.0),
                        status=OrderStatus.PENDING, timestamp=datetime.now(timezone.utc),
                        stop_loss=stop_loss, take_profit=take_profit, tag=tag,
                        average_price=0.0
                    )
                    self._register_order(details)

                    # C. Auto-Register Bracket
                    if self._bracket_manager and (stop_loss or take_profit):
                        # ... (Bracket logic same as before) ...
                        self._bracket_manager.register_virtual_bracket(
                            order_id=order_id,
                            symbol=normalized_symbol,
                            side=normalized_side, # Use string side
                            qty=quantity,
                            price=float(price or 0.0),
                            sl=float(stop_loss) if stop_loss else 0.0,
                            tp=float(take_profit) if take_profit else 0.0,
                            tag=tag or "auto",
                            activate_immediately=True
                        )
                        self._logger.info(f"🛡️ Auto-bracket registered for {order_id}")

                    # 🛑 FIX 3: Safe Instant Sync (0.5s Delay)
                    try:
                        time.sleep(0.5) # Wait for Broker Latency
                        if hasattr(self._broker, "get_order_status"):
                            status_update = self._broker.get_order_status(order_id)
                            if status_update:
                                self.on_order_update(status_update)
                    except Exception:
                        pass

                    return order_id
                    
            except Exception as e:
                msg = str(e).lower()
                # Fail Fast logic
                if any(x in msg for x in ["400", "invalid", "market closed", "bad request", "insufficient funds"]):
                    self._logger.critical(f"🛑 FATAL Payload Error: {e}", extra={"event": "fatal_order_error"})
                    self.trade_store.update_status(trade_id, "REJECTED_FATAL")
                    return None
                
                self._logger.warning(f"⚠️ Retry {attempt}/3 failed: {e}")
                time.sleep(0.5 * attempt)
                
        self._logger.error("❌ Order placement failed after retries.")
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
        # ✅ NEW: High-Profit Args (TP1 Scaling & ATR Trailing)
        tp1_price: float | None = None,
        tp1_qty: int | None = None,
        trailing_atr_mult: float | None = None,
    ) -> tuple[str, str, str]:
        """
        Places a PHYSICAL Entry and registers a VIRTUAL Bracket.
        World-Class Upgrade: Replaces legacy OCO with internal Sniper execution.
        """
        self._logger.info(
            f"🚀 Initiating Virtual Bracket: {symbol} {side} {quantity}",
            extra={"event": "virtual_bracket_init", "symbol": symbol}
        )

        # 1. Place the Entry Order (Ph