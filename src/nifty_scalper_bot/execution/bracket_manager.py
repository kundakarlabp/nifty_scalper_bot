"""
Thread-safe Bracket Manager with VIRTUAL Execution capabilities.
Production-Grade: Replaces legacy broker-side brackets with high-speed internal monitoring.
Enhanced with ATR Trailing, Multi-Target (TP1/TP2), Partial Scaling, and Orphan Sync.
"""
from __future__ import annotations

import threading
from threading import RLock
import time
import json
import os
from datetime import datetime, timezone 
import math 
from pathlib import Path
_THREADING_MODULE = threading
_RLOCK_CLASS = RLock
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)

from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.symbols import normalize_symbol
try:
    from nifty_scalper_bot.data.bracket_store import BracketStore
except ImportError:
    BracketStore = None

# --- NEW IMPORTS FOR WORLD-CLASS TRAILING ---
try:
    from nifty_scalper_bot.indicators.atr_provider import SafeATRProvider
    # Assuming TrailingSpec is defined in adaptive_trailing or we define a local fallback
    from nifty_scalper_bot.execution.adaptive_trailing import AdaptiveTrailingController, TrailingSpec
except ImportError:
    SafeATRProvider = None
    AdaptiveTrailingController = None
    # Fallback definition if import fails
    @dataclass
    class TrailingSpec:
        trail_by: float
        step: float
        activation: float

if TYPE_CHECKING:
    from nifty_scalper_bot.infra.metrics import MetricsCollector

# --------------------------------------------------------------------------
# METRICS INTEGRATION
# --------------------------------------------------------------------------
METRICS_AVAILABLE = False
METRICS = None

LOGGER = get_logger(__name__)


def _round_to_tick(price: float, tick_size: float = 0.05) -> float:
    """Round *price* to broker-compatible tick precision."""
    if tick_size <= 0:
        return round(price, 2)
    return round(round(float(price) / tick_size) * tick_size, 2)

def _normalize_bracket_side(side: str) -> str:
    """Normalize side to 'BUY'/'SELL' for consistent bracket comparisons.

    CRITICAL FIX (6 Feb 2026): attach_orphan_position stored 'LONG'/'SHORT'
    but _evaluate_exit_fast compared against 'BUY'/'SELL', causing SL to NEVER trigger.
    """
    s = side.strip().upper()
    if s in ("BUY", "LONG"):
        return "BUY"
    if s in ("SELL", "SHORT"):
        return "SELL"
    raise ValueError(f"Invalid bracket side: {side!r}. Must be BUY/LONG or SELL/SHORT.")

_FILLED_STATUSES = {"FILLED", "COMPLETE", "COMPLETED"}
_CANCELLED_STATUSES = {"CANCELLED", "REJECTED", "CANCELED"}


# --------------------------------------------------------------------------
# PROTOCOLS
# --------------------------------------------------------------------------
@runtime_checkable
class SupportsCancelOrder(Protocol):
    """Protocol representing broker cancel capability."""
    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> Any:
        ...

@runtime_checkable
class SupportsModifyOrder(Protocol):
    """Protocol representing broker order modification capability."""
    def modify_order(self, order_id: str, **kwargs: Any) -> Any:
        ...


# --------------------------------------------------------------------------
# DATA STRUCTURES
# --------------------------------------------------------------------------

@dataclass
class TargetLevel:
    """Represents a partial profit target level."""
    price: float
    quantity: int
    executed: bool = False
    name: str = "TP"


@dataclass
class BracketState:
    """
    State container for a managed trade exit.
    Held in memory; survives restarts if persisted via TradeStore.
    """
    entry_order_id: str
    symbol: str
    side: str          # Entry Side (BUY/SELL)
    quantity: int      # Original Quantity
    entry_price: float
    
    # Execution Triggers
    sl_trigger_price: float
    tp_trigger_price: float  # Final/Ultimate TP
    
    # Multi-Target & Scaling State (NEW)
    remaining_quantity: int = 0
    tp_levels: List[TargetLevel] = field(default_factory=list)
    
    # Trailing & Logic State (NEW)
    is_virtual: bool = True
    active: bool = True  # If False, waits for confirmation or is finished
    trailing_enabled: bool = True
    trailing_config: Dict[str, Any] = field(default_factory=dict) # e.g. {'mode': 'ATR', 'mult': 1.5}
    virtual_sl_id: str = "" # ID for the Adaptive Controller
    
    # Market Data Tracking (NEW)
    highest_ltp: float = 0.0  # High water mark since entry (for BUY)
    lowest_ltp: float = float('inf')  # Low water mark since entry (for SELL)
    last_ltp: float = 0.0 # Latest price seen
    previous_ltp: float = 0.0 # Previous tick for stop-loss cross detection
    
    # Metadata
    tag: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    exit_executed: bool = False
    # Tracks the most recently submitted (but not yet confirmed filled) exit order.
    # Set by _fire_exits_batch / _watchdog_exit_loop when the exit order is placed.
    # Cleared to None once the exit is confirmed FILLED, or used by
    # _handle_order_rejected / on_order_update(CANCELLED) to reactivate the bracket
    # when Zerodha rejects/cancels the exit order.
    pending_exit_order_id: str | None = None
    exit_in_progress: bool = False
    _atr_warning_logged: bool = False

    def __post_init__(self):
        # Auto-initialize state fields if not set
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity
        
        # Initialize High/Low water marks with entry price
        if self.highest_ltp == 0.0 or self.highest_ltp < self.entry_price:
            self.highest_ltp = self.entry_price
        
        if self.lowest_ltp == float('inf') or self.lowest_ltp > self.entry_price:
            self.lowest_ltp = self.entry_price

    # ✅ FIX: Add Serialization for Persistence
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for SQLite persistence."""
        return {
            "entry_order_id": self.entry_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "sl_trigger_price": self.sl_trigger_price,
            "tp_trigger_price": self.tp_trigger_price,
            "remaining_quantity": self.remaining_quantity,
            "tp_levels": [tp.to_dict() for tp in self.tp_levels],
            "is_virtual": self.is_virtual,
            "active": self.active,
            "trailing_enabled": self.trailing_enabled,
            "trailing_config": self.trailing_config,
            "highest_ltp": self.highest_ltp,
            "lowest_ltp": self.lowest_ltp,
            "tag": self.tag,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "exit_executed": self.exit_executed,
            "pending_exit_order_id": self.pending_exit_order_id,
            "previous_ltp": self.previous_ltp,
            "exit_in_progress": self.exit_in_progress,
            "atr_warning_logged": self._atr_warning_logged,
        }

    @property
    def stop_loss_price(self) -> float:
        """Backward-compatible alias for stop-loss trigger."""
        return self.sl_trigger_price

    @property
    def target_price(self) -> float:
        """Backward-compatible alias for take-profit trigger."""
        return self.tp_trigger_price

    def rehydrate_state_from_position(self, position: Any) -> None:
        """Rehydrate runtime state after reconnect from current position snapshot."""
        ltp_raw = getattr(position, "ltp", None)
        if ltp_raw is None:
            ltp_raw = getattr(position, "last_price", None)
        if ltp_raw is None:
            ltp_raw = self.last_ltp
        try:
            ltp = float(ltp_raw or 0.0)
        except (TypeError, ValueError):
            ltp = 0.0
        if ltp <= 0:
            return
        self.last_ltp = ltp
        if self.side == "BUY":
            self.highest_ltp = max(self.highest_ltp, ltp)
            self.sl_trigger_price = max(self.sl_trigger_price, 0.0)
        else:
            self.lowest_ltp = min(self.lowest_ltp, ltp)
            self.sl_trigger_price = max(self.sl_trigger_price, 0.0)
        self.updated_at = time.time()

# Mock Journal for Adaptive Controller (In-Memory)
class MockJournal:
    def set(self, key, value): pass
    def get(self, key): return None

class BracketManager:
    """
    The 'Sniper' Engine.
    Monitors LTP/ATR internally and fires immediate MARKET exits when levels are hit.
    Supports TP1/TP2 scaling, ATR-based Trailing, and Broker Sync.
    """

    def __init__(self, order_manager: Any, indicator_engine: Any = None, market_data: Any = None):
        """Initialize bracket manager runtime dependencies. Args: order_manager, indicator_engine, market_data; Returns: None; Raises: None."""
        self.order_manager = order_manager
        self._brackets: Dict[str, BracketState] = {}
        # Reverse Index: Map broker order IDs/Symbol to entry IDs
        self._order_to_entry: Dict[str, str] = {}
        self._symbol_map: Dict[str, List[str]] = {}  # Fast lookup: Symbol -> [Entry IDs]
        
        # --- ATR & Trailing Setup ---
        self._indicator_engine = indicator_engine
        self._market_data = market_data
        # ✅ FIX: Initialize Persistence Store
        self._store = None
        if BracketStore:
            try:
                self._store = BracketStore()
                LOGGER.info("✅ BracketStore initialized for persistence.")
            except Exception as e:
                LOGGER.error(f"❌ Failed to init BracketStore: {e}")
                
        self._atr_provider = None
        if SafeATRProvider and indicator_engine:
            # Initialize Safe Provider with 60s cache validity
            self._atr_provider = SafeATRProvider(indicator_engine, max_cache_age=60.0)
            
        # Store active trailing controllers: {entry_order_id: Controller}
        self._trailing_controllers: Dict[str, Any] = {}
        self._recent_ticks: Dict[str, deque] = {}
        self._max_tick_history = 20
        self._orphan_retry_count: Dict[str, int] = {}
        self._orphan_retry_last_attempt: Dict[str, float] = {}
        self._exit_executor: Callable[[str, int], Any] | None = None
        self._trailing_controller_factory: Callable[[BracketState], Any] | None = None
        # FIX S10-2: hook fired when bracket fully closes — lets runner clear orchestrator direction lock
        self._on_exit_complete_hook: Callable[[str], None] | None = None
    
        # Real-time Data Cache (Legacy Fallback)
        self._current_atr: Dict[str, float] = {}
        self._last_price_cache: Dict[str, float] = {}
        self._exit_cooldowns: Dict[str, float] = {}
        self._notifier: Callable[[str, Mapping[str, object] | None], None] | None = None
        self._trail_notify_at: Dict[str, float] = {}
        self._trail_notify_sl: Dict[str, float] = {}
        self._tick_error_logged: Dict[str, bool] = {}
        self._throttle_log_at: Dict[str, float] = {}
        self._lock = _RLOCK_CLASS()
        self._reconcile_lock = threading.Lock()
        self._running = True
        
        # Configuration
        self._auto_reduce_sl = True
        self._stale_cleanup_age = 86400  # 24 hours
        # Cache tiered-trailing thresholds once at startup — avoids os.getenv()
        # on every tick (100-300 ticks/sec × N active brackets = hot path).
        self._trail_tier1_pct = float(os.getenv("TRAIL_TIER1_PCT", "1.0"))
        self._trail_tier2_pct = float(os.getenv("TRAIL_TIER2_PCT", "2.0"))
        self._trail_tier3_pct = float(os.getenv("TRAIL_TIER3_PCT", "4.0"))
        self._trail_tier4_pct = float(os.getenv("TRAIL_TIER4_PCT", "6.0"))
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_exit_loop,
            name='bracket-watchdog',
            daemon=True,
        )
        self._watchdog_thread.start()

    def attach_exit_executor(self, executor: Callable[[str, int], Any] | None) -> None:
        """Attach an external market-exit executor. Args: executor; Returns: None; Raises: None."""
        self._exit_executor = executor

    def attach_on_exit_complete(self, hook: Callable[[str], None] | None) -> None:
        """Attach callback fired with symbol when bracket fully closes. Lets runner clear orchestrator direction lock."""
        self._on_exit_complete_hook = hook


    def attach_trailing_controller_factory(
        self, factory: Callable[[BracketState], Any] | None
    ) -> None:
        """Attach a trailing controller factory. Args: factory; Returns: None; Raises: None."""
        self._trailing_controller_factory = factory

    def _watchdog_exit_loop(self) -> None:
        """Run watchdog checks and force exits on hard SL breach. Args: none; Returns: none; Raises: none."""
        while self._running:
            try:
                pending: list[tuple[BracketState, dict[str, Any]]] = []
                with self._lock:
                    for bracket in self._brackets.values():
                        ltp = float(bracket.last_ltp or 0.0)
                        if (
                            bracket.exit_executed
                            or bracket.exit_in_progress
                            or bracket.remaining_quantity <= 0
                            or bracket.sl_trigger_price <= 0
                            or ltp <= 0
                        ):
                            continue
                        if bracket.side == 'BUY' and self._sl_crossed(bracket, ltp):
                            pending.append((bracket, {
                                'type': 'SL',
                                'price': ltp,
                                'qty': bracket.remaining_quantity,
                                'reason': 'WATCHDOG_HARD_SL',
                            }))
                        elif bracket.side == 'SELL' and self._sl_crossed(bracket, ltp):
                            pending.append((bracket, {
                                'type': 'SL',
                                'price': ltp,
                                'qty': bracket.remaining_quantity,
                                'reason': 'WATCHDOG_HARD_SL',
                            }))
                if pending:
                    self._log_throttled(
                        'critical',
                        'watchdog_exit_trigger',
                        3.0,
                        'WATCHDOG EXIT TRIGGER count=%s',
                        len(pending),
                    )
                    self._fire_exits_batch(pending)
                time.sleep(0.25)
            except Exception as e:
                LOGGER.error('Failure in _watchdog_exit_loop: %s', e)
                time.sleep(0.25)

    def _log_throttled(
        self,
        level: str,
        key: str,
        interval_sec: float,
        message: str,
        *args: object,
    ) -> None:
        """Emit a throttled log event. Args: level,key,interval_sec,message,args; Returns: none; Raises: none."""
        now = time.time()
        if now - self._throttle_log_at.get(key, 0.0) < interval_sec:
            return
        self._throttle_log_at[key] = now
        getattr(LOGGER, level, LOGGER.info)(message, *args)

    def shutdown(self) -> None:
        """Stop watchdog processing. Args: none; Returns: none; Raises: none."""
        self._running = False

    # --------------------------------------------------------------------------
    # 1. CORE API (Backward Compatible & Enhanced)
    # --------------------------------------------------------------------------

    def set_notifier(
        self, notifier: Callable[[str, Mapping[str, object] | None], None] | None
    ) -> None:
        """Attach notifier callback for bracket lifecycle updates.

        Args:
            notifier: Callback accepting (event, payload) or ``None`` to clear.

        Returns:
            None.

        Raises:
            None.
        """
        LOGGER.debug(
            'Entered set_notifier',
            extra={'event': 'bracket_manager_set_notifier_enter'},
        )
        try:
            self._notifier = notifier
            if notifier is None:
                LOGGER.info(
                    'Bracket notifier cleared',
                    extra={'event': 'bracket_manager_notifier_cleared'},
                )
            else:
                LOGGER.info(
                    'Bracket notifier attached',
                    extra={'event': 'bracket_manager_notifier_attached'},
                )
        except Exception as exc:
            LOGGER.error(
                'Failure in set_notifier: %s',
                exc,
                extra={'event': 'bracket_manager_notifier_failed'},
                exc_info=exc,
            )

    def _notify_event(
        self, event: str, payload: Mapping[str, object] | None = None
    ) -> None:
        """Emit a bracket lifecycle notification when configured.

        Args:
            event: Event name describing the lifecycle action.
            payload: Optional metadata payload for the notifier.

        Returns:
            None.

        Raises:
            None.
        """
        LOGGER.debug(
            'Entered _notify_event',
            extra={'event': 'bracket_manager_notify_enter', 'notify_event': event},
        )
        if self._notifier is None:
            return
        try:
            self._notifier(event, payload)
        except Exception as exc:
            LOGGER.error(
                'Failure in _notify_event: %s',
                exc,
                extra={'event': 'bracket_manager_notify_failed', 'notify_event': event},
                exc_info=exc,
            )

    def _should_notify_trail(self, entry_id: str, new_sl: float, old_sl: float) -> bool:
        """Decide if a trailing update should be notified.

        Args:
            entry_id: Bracket entry identifier for tracking cooldowns.
            new_sl: Proposed updated stop-loss level.
            old_sl: Previous stop-loss level before the update.

        Returns:
            ``True`` when a notification should be emitted.

        Raises:
            None.
        """
        try:
            now = time.time()
            last_ts = self._trail_notify_at.get(entry_id, 0.0)
            last_sl = self._trail_notify_sl.get(entry_id, old_sl)
            price_delta = abs(new_sl - last_sl)
            pct_delta = (price_delta / last_sl * 100.0) if last_sl > 0 else 100.0
            min_seconds = float(os.getenv('TRAIL_NOTIFY_COOLDOWN_SEC', '60'))
            min_pct = float(os.getenv('TRAIL_NOTIFY_MIN_PCT', '0.5'))
            if (now - last_ts) >= min_seconds or pct_delta >= min_pct:
                self._trail_notify_at[entry_id] = now
                self._trail_notify_sl[entry_id] = new_sl
                return True
            return False
        except Exception as exc:
            LOGGER.error(
                'Failure in _should_notify_trail: %s',
                exc,
                extra={
                    'event': 'bracket_manager_trail_notify_failed',
                    'entry_id': entry_id,
                },
                exc_info=exc,
            )
            return False

    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        stop_loss: float,
        take_profit: float,
        tag: str = "auto_bracket",
        entry_order_id: str | None = None,
        # --- NEW OPTIONAL ARGUMENTS ---
        tp1_price: float | None = None,
        tp1_qty: int | None = None,
        trailing_atr_mult: float | None = None
    ) -> str:
        """
        Legacy Bridge: Allows old code to call this method.
        Internally converts request to a Virtual Bracket.
        """
        if not entry_order_id:
            # Generate a synthetic ID if caller didn't provide one
            entry_order_id = f"virt_{int(time.time())}_{symbol}"

        self.register_virtual_bracket(
            order_id=entry_order_id,
            symbol=symbol,
            side=_normalize_bracket_side(side),
            qty=quantity,
            price=price,
            sl=stop_loss,
            tp=take_profit,
            tag=tag,
            tp1_price=tp1_price,
            tp1_qty=tp1_qty,
            trailing_atr_mult=trailing_atr_mult,
            activate_immediately=False # 🔒 Safety: Wait for fill or tick
        )

        LOGGER.info(f"🛡️ Bracket Registered: {entry_order_id} for {symbol}")
        return entry_order_id  # 🟢 FIX 2: Return the ID

    def register_virtual_bracket(
        self, 
        order_id: str, 
        symbol: str, 
        side: str, 
        qty: int, 
        price: float, 
        sl: float, 
        tp: float,
        tag: str = "virtual",
        # --- NEW OPTIONAL ARGUMENTS ---
        tp1_price: float | None = None,
        tp1_qty: int | None = None,
        trailing_atr_mult: float | None = None,
        activate_immediately: bool = True
    ) -> None:
        """Register a position for virtual bracket monitoring.

        Args:
            order_id: Entry order identifier for the bracket.
            symbol: Trading symbol for the bracket.
            side: Trade direction ("BUY" or "SELL").
            qty: Position size to monitor.
            price: Entry price used for bracket tracking.
            sl: Stop-loss trigger price.
            tp: Take-profit trigger price.
            tag: Optional strategy tag for auditing.
            tp1_price: Optional first profit target price.
            tp1_qty: Optional first profit target quantity.
            trailing_atr_mult: Optional ATR-based trailing multiplier.
            activate_immediately: Whether the bracket is active on registration.

        Returns:
            None.

        Raises:
            None.
        """
        symbol = normalize_symbol(symbol)
        with self._lock:
            # 1. Deduplication: Update existing if found
            if order_id in self._brackets:
                LOGGER.warning(f"Bracket {order_id} exists. Updating triggers.")
                existing = self._brackets[order_id]
                # Only update non-zero values to avoid overwriting with bad data
                if sl > 0: existing.sl_trigger_price = sl
                if tp > 0: existing.tp_trigger_price = tp
                # Reset quantity if re-registering (e.g. scale-in)
                existing.quantity = abs(qty)
                existing.remaining_quantity = abs(qty)
                self.save_state() # Persist updates
                return

            # 2. Setup Trailing Config
            t_config = {}
            if trailing_atr_mult:
                t_config = {'mode': 'ATR', 'mult': trailing_atr_mult}
            elif self._auto_reduce_sl:
                # Default logic if enabled but no explicit ATR
                t_config = {'mode': 'STANDARD'}

            # 3. Setup TP Levels (Partial Exits)
            targets = []
            if tp1_price and tp1_qty and tp1_qty < abs(qty):
                targets.append(TargetLevel(price=tp1_price, quantity=tp1_qty, name="TP1"))
                LOGGER.info(f"🔹 Configured TP1 for {symbol}: {tp1_price} (Qty: {tp1_qty})")

            # 4. ORPHAN SAFETY CHECK (CRITICAL)
            # If SL or TP are 0.0 (invalid/missing), we MUST NOT enable active exiting.
            # This prevents the bot from seeing LTP > 0 as "Target Hit".
            status_mode = "ACTIVE" if activate_immediately else "INACTIVE"
            if sl <= 0 and tp <= 0:
                LOGGER.warning(
                    f"⚠️ Bracket {symbol} has zero SL/TP (SL={sl}, TP={tp}). "
                    "Setting MONITORING_ONLY mode to prevent suicide exit."
                )
                # Force inactive so it monitors but doesn't fire
                status_mode = "MONITORING_ONLY"
                activate_immediately = False 

            # 5. Create State Object
            state = BracketState(
                entry_order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=abs(qty),
                remaining_quantity=abs(qty),
                entry_price=price,
                sl_trigger_price=sl,
                tp_trigger_price=tp, # This is effectively Final TP
                tp_levels=targets,
                is_virtual=True,
                active=activate_immediately,
                tag=tag,
                trailing_config=t_config,
                virtual_sl_id=f"vsl_{order_id}"
            )
            
            self._brackets[order_id] = state

            if self._trailing_controller_factory:
                if state.entry_order_id in self._trailing_controllers:
                    return
                controller = self._trailing_controller_factory(state)
                self._trailing_controllers[state.entry_order_id] = controller
            
            # 6. Populate Indices
            self._order_to_entry[order_id] = order_id
            if symbol not in self._symbol_map:
                self._symbol_map[symbol] = []
            self._symbol_map[symbol].append(order_id)
            # ✅ FIX: IMMEDIATE PERSISTENCE
            # Save to disk immediately so it survives a crash/restart
            if self._store:
                try:
                    self._store.save_bracket(state.to_dict())
                    LOGGER.info(f"💾 Bracket PERSISTED to DB: {symbol} (SL: {sl})")
                except Exception as e:
                    LOGGER.error(f"❌ Failed to persist bracket for {symbol}: {e}")
            
            # 7. Initialize Adaptive Controller (The "Brain")
            if (
                not self._trailing_controller_factory
                and trailing_atr_mult
                and self._atr_provider
                and AdaptiveTrailingController
            ):
                try:
                    spec = TrailingSpec(
                        trail_by=20.0, # Fallback
                        step=0.25,      # Tighter early trailing step
                        activation=0.3 # Earlier activation threshold
                    )
                    
                    # We pass a lambda that returns the latest price from the bracket state
                    ctrl = AdaptiveTrailingController(
                        symbol=symbol,
                        side="LONG" if side == "BUY" else "SHORT",
                        entry=price,
                        sl_order_id=state.virtual_sl_id,
                        variety="virtual",
                        spec=spec,
                        get_ltp=lambda s: state.last_ltp, 
                        modify_order=self._virtual_modify_sl,
                        atr_provider=self._atr_provider,
                        journal=MockJournal(),
                        atr_multiplier=trailing_atr_mult
                    )
                    self._trailing_controllers[order_id] = ctrl
                    LOGGER.info(f"🧠 Adaptive Controller Attached to {symbol} (Mult: {trailing_atr_mult}x)")
                    
                except Exception as e:
                    LOGGER.error(f"Failed to attach Adaptive Controller: {e}")

            trail_msg = f"| Trail={t_config.get('mode', 'None')}"
            LOGGER.info(
                f"🛡️ Bracket Registered for {symbol} (Qty: {qty}): "
                f"Entry={price} | SL={sl} | TP={tp} {trail_msg} | Mode={status_mode}"
            )
            self._notify_event(
                'BRACKET_REGISTERED',
                {
                    'symbol': symbol,
                    'side': side,
                    'qty': abs(qty),
                    'entry': round(price, 2),
                    'sl': round(sl, 2) if sl else 0.0,
                    'tp': round(tp, 2) if tp else 0.0,
                    'mode': status_mode,
                },
            )

            # 8. Record metric & Persist
            if METRICS_AVAILABLE and METRICS:
                try:
                    METRICS.brackets_created.inc()
                except Exception: pass
            
            # ✅ FIX: Persist immediately so we don't lose this if we crash now
            self.save_state()

    def confirm_entry_fill(self, order_id: str, fill_price: float) -> None:
        """Activate a bracket once entry fill is confirmed.

        Args:
            order_id: Entry order identifier for the bracket.
            fill_price: Confirmed fill price from the broker.

        Returns:
            None.

        Raises:
            None.
        """
        with self._lock:
            bracket = self._brackets.get(order_id)
            if not bracket:
                LOGGER.warning(f"⚠️ No bracket found for order {order_id}")
                return
            
            # Update entry price with actual fill
            if fill_price and fill_price > 0:
                old_price = bracket.entry_price
                bracket.entry_price = fill_price
                
                # Adjust SL/TP if they were calculated from expected price
                if old_price > 0 and old_price != fill_price:
                    price_diff_pct = (fill_price - old_price) / old_price
                    
                    # Adjust SL proportionally
                    if bracket.sl_trigger_price > 0:
                        bracket.sl_trigger_price = round(
                            bracket.sl_trigger_price * (1 + price_diff_pct), 2
                        )
                    
                    # Adjust TP proportionally
                    if bracket.tp_trigger_price > 0:
                        bracket.tp_trigger_price = round(
                            bracket.tp_trigger_price * (1 + price_diff_pct), 2
                        )
                    
                    LOGGER.info(
                        f"📊 Bracket adjusted for fill price: Entry={fill_price:.2f} "
                        f"SL={bracket.sl_trigger_price:.2f} TP={bracket.tp_trigger_price:.2f}"
                    )
            
            # ✅ ACTIVATE IMMEDIATELY
            bracket.active = True
            bracket.updated_at = time.time()
            
            # Initialize water marks
            bracket.highest_ltp = fill_price
            bracket.lowest_ltp = fill_price
            bracket.last_ltp = fill_price
            
            LOGGER.info(
                f"🎯 BRACKET ACTIVATED: {bracket.symbol} | "
                f"Entry={fill_price:.2f} | SL={bracket.sl_trigger_price:.2f} | "
                f"TP={bracket.tp_trigger_price:.2f}"
            )
            self._notify_event(
                'BRACKET_ACTIVATED',
                {
                    'symbol': bracket.symbol,
                    'side': bracket.side,
                    'entry': round(fill_price, 2),
                    'sl': round(bracket.sl_trigger_price, 2),
                    'tp': round(bracket.tp_trigger_price, 2),
                },
            )
            
            # Persist state
            try:
                self.save_state()
            except Exception:
                pass

    # --------------------------------------------------------------------------
    # 2. MARKET DATA INGESTION (NEW)
    # --------------------------------------------------------------------------

    def update_market_stats(self, symbol: str, atr: float = 0.0, volume: float = 0.0) -> None:
        """Feed external calculations (ATR) into the manager."""
        if atr > 0:
            # Update Legacy Cache
            self._current_atr[symbol] = atr
            # Feed Safe Provider if available
            if self._atr_provider and hasattr(self._atr_provider, 'feed_manual'):
                self._atr_provider.feed_manual(symbol, atr)

    def feed_atr_updates(self, symbol: str, atr: float) -> None:
        """Alias for update_market_stats."""
        self.update_market_stats(symbol, atr=atr)

    def _virtual_modify_sl(self, order_id: str, price: float) -> bool:
        """Callback for AdaptiveController to update Virtual SL."""
        # Find bracket by iterating (Safety lookup)
        target_bracket = None
        with self._lock:
            for b in self._brackets.values():
                if b.virtual_sl_id == order_id:
                    target_bracket = b
                    break
            
            if target_bracket:
                old_sl = target_bracket.sl_trigger_price
                target_bracket.sl_trigger_price = price
                target_bracket.updated_at = time.time()
                LOGGER.info(f"🔄 Dynamic ATR Trail: {target_bracket.symbol} SL {old_sl} -> {price}")
                # ✅ FIX: Persist trailing update
                self.save_state()
                return True
        return False

    # --------------------------------------------------------------------------
    # 3. EXECUTION LOGIC (The "Sniper")
    # --------------------------------------------------------------------------

    def on_tick(self, symbol: str, ltp: float) -> None:
        symbol = normalize_symbol(symbol)
        """
        Ultra-low-latency tick processing.
        
        ✅ WORLD-CLASS OPTIMIZATIONS:
        - Minimal lock scope (snapshot only)
        - No lock during exit evaluation
        - Batch exit firing
        - Atomic LTP updates
        """
        # ═══════════════════════════════════════════════════════════
        # FAST PATH: Early exit checks (no lock needed)
        # ═══════════════════════════════════════════════════════════
        if not self._brackets:
            return
        
        # Fast symbol lookup (dict access is atomic in Python)
        relevant_ids = self._symbol_map.get(symbol)
        if not relevant_ids:
            return
        
        if ltp <= 0:
            return
        
        # ═══════════════════════════════════════════════════════════
        # SNAPSHOT: Minimal lock scope - just get bracket references
        # ═══════════════════════════════════════════════════════════
        candidates = []
        with self._lock:
            for eid in relevant_ids:
                b = self._brackets.get(eid)
                # 🟢 FIX: Remove 'b.active' check so we can catch inactive ones
                if b and b.remaining_quantity > 0 and not b.exit_executed and not b.exit_in_progress:
                    candidates.append(b)
        
        if not candidates:
            return
        
        # ═══════════════════════════════════════════════════════════
        # TRACK TICK HISTORY (for momentum, outside lock)
        # ═══════════════════════════════════════════════════════════
        if not hasattr(self, '_recent_ticks'):
            self._recent_ticks = {}
        
        if symbol not in self._recent_ticks:
            from collections import deque
            self._recent_ticks[symbol] = deque(maxlen=20)
        self._recent_ticks[symbol].append(ltp)
        
        # ═══════════════════════════════════════════════════════════
        # EVALUATE: Check all brackets WITHOUT lock
        # ═══════════════════════════════════════════════════════════
        exits_to_fire = []
        
        for bracket in candidates:
            # 🟢 FAILSAFE: Auto-activate on first tick if stuck inactive
            if not bracket.active:
                if ltp > 0:
                    bracket.active = True
                    # Fix missing entry price (Market Orders)
                    if bracket.entry_price <= 0:
                        bracket.entry_price = ltp
                    
                    # Initialize Watermarks for Trailing
                    if bracket.side == "BUY": 
                        bracket.highest_ltp = ltp
                    else: 
                        bracket.lowest_ltp = ltp
                    
                    # ✅ CORRECTED ATTRIBUTE: Use 'entry_order_id', not 'order_id'
                    self._log_throttled('debug', f'failsafe_{bracket.entry_order_id}', 5.0, 'FAILSAFE_ACTIVATION entry_id=%s ltp=%s', bracket.entry_order_id, round(ltp, 2))
                else:
                    # If we can't activate (no price), skip this bracket
                    continue

            # Track previous tick for jump/cross stop-loss detection.
            bracket.previous_ltp = float(bracket.last_ltp or ltp)
            bracket.last_ltp = ltp
            
            # Update high/low water marks (atomic operations)
            if bracket.side == "BUY":
                if ltp > bracket.highest_ltp:
                    bracket.highest_ltp = ltp
            else:
                if ltp < bracket.lowest_ltp:
                    bracket.lowest_ltp = ltp
            
            # Fast SL check with cross detection (jump-safe) before trailing updates.
            pre_action = self._evaluate_exit_fast(bracket, ltp)
            if pre_action and pre_action.get('type') == 'SL':
                exits_to_fire.append((bracket, pre_action))
                continue
            
            # Check trailing controller (if attached)
            # Wrapped in try/except so trailing failures cannot block SL/TP eval
            entry_id = bracket.entry_order_id
            try:
                if entry_id in self._trailing_controllers:
                    ctrl = self._trailing_controllers[entry_id]
                    
                    # Inject ATR if available
                    current_atr = self._current_atr.get(symbol)
                    if current_atr and current_atr > 0 and hasattr(ctrl, 'update_atr'):
                        ctrl.update_atr(current_atr)
                    
                    # Run controller
                    ctrl.on_tick({"ltp": ltp})
                else:
                    # Use built-in adaptive trailing
                    self._apply_trailing_math(bracket)
            except Exception as _trail_exc:
                LOGGER.debug("Trailing logic failed for %s: %s (SL/TP eval continues)", bracket.symbol, _trail_exc, extra={"event": "trailing_error", "symbol": bracket.symbol})
            
            # ═══════════════════════════════════════════════════════
            # EVALUATE EXIT CONDITIONS (Pure logic, no side effects)
            # ═══════════════════════════════════════════════════════
            exit_action = self._evaluate_exit_fast(bracket, ltp)
            
            if exit_action:
                exits_to_fire.append((bracket, exit_action))
        
        # ═══════════════════════════════════════════════════════════
        # FIRE EXITS: Batch processing (takes lock once)
        # ═══════════════════════════════════════════════════════════
        if exits_to_fire:
            self._fire_exits_batch(exits_to_fire)


    def process_exit_checks(self, symbol: str, ltp: float) -> None:
        """Route tick to the single on-tick exit authority. Args: symbol, ltp; Returns: None; Raises: None."""
        LOGGER.debug('Entered process_exit_checks')
        try:
            self.on_tick(symbol, ltp)
        except Exception as e:
            LOGGER.error('Failure in process_exit_checks: %s', e)

    def _force_exit(self, bracket: BracketState) -> None:
        """Force bracket exit with confirmation and fallback. Args: bracket; Returns: None; Raises: None."""
        symbol = normalize_symbol(bracket.symbol)
        qty = bracket.remaining_quantity
        if not symbol or qty <= 0:
            return
        self._fire_exits_batch(
            [
                (
                    bracket,
                    {
                        'type': 'SL',
                        'price': bracket.last_ltp,
                        'qty': qty,
                        'reason': 'FORCED_SL_EXIT',
                    },
                )
            ]
        )

    def _sl_crossed(self, bracket: BracketState, ltp: float) -> bool:
        """Return True when a tick crosses stop-loss. Args: bracket, ltp; Returns: bool; Raises: none."""
        prev_ltp = float(bracket.previous_ltp or bracket.last_ltp or ltp)
        if bracket.side == 'BUY':
            return prev_ltp > bracket.sl_trigger_price and ltp <= bracket.sl_trigger_price
        return prev_ltp < bracket.sl_trigger_price and ltp >= bracket.sl_trigger_price

    def _evaluate_exit_fast(self, bracket: BracketState, ltp: float) -> dict | None:
        """
        Pure function to evaluate exit conditions.
        Returns exit action dict or None.
        
        ✅ NO LOCKS, NO SIDE EFFECTS - Just logic
        """
        # Check partial targets first (TP1, TP2, etc.)
        for target in bracket.tp_levels:
            if target.executed:
                continue
            
            triggered = False
            if bracket.side == "BUY" and ltp >= target.price:
                triggered = True
            elif bracket.side == "SELL" and ltp <= target.price:
                triggered = True
            
            if triggered:
                return {
                    "type": "PARTIAL_TP",
                    "target": target,
                    "price": ltp,
                    "qty": min(target.quantity, bracket.remaining_quantity),
                    "reason": f"{target.name} Hit ({ltp:.2f})"
                }
        
        # Hard TP breach must take precedence to guarantee immediate exit.
        if bracket.tp_trigger_price > 0:
            triggered = False
            if bracket.side == "BUY" and ltp >= bracket.tp_trigger_price:
                triggered = True
            elif bracket.side == "SELL" and ltp <= bracket.tp_trigger_price:
                triggered = True
            
            if triggered:
                return {
                    "type": "FINAL_TP",
                    "price": ltp,
                    "qty": bracket.remaining_quantity,
                    "reason": "HARD_TP_BREACH"
                }
        
        # Check SL (CRITICAL - always check last) using jump-safe cross detection.
        if bracket.sl_trigger_price > 0:
            prev_ltp = float(bracket.previous_ltp or ltp)
            triggered = False
            if bracket.side == "BUY":
                triggered = prev_ltp > bracket.sl_trigger_price and ltp <= bracket.sl_trigger_price
            elif bracket.side == "SELL":
                triggered = prev_ltp < bracket.sl_trigger_price and ltp >= bracket.sl_trigger_price

            if triggered:
                return {
                    "type": "SL",
                    "price": ltp,
                    "qty": bracket.remaining_quantity,
                    "reason": f"HARD_SL_BREACH prev={prev_ltp:.2f} curr={ltp:.2f} sl={bracket.sl_trigger_price:.2f}",
                }
        
        return None

    def _fire_exits_batch(self, exits: list) -> None:
        """Submit exits atomically with broker confirmation before state mutation. Args: exits; Returns: none; Raises: none."""
        now = time.time()
        approved: list[tuple[BracketState, dict[str, Any]]] = []

        with self._lock:
            for bracket, action in exits:
                if bracket.exit_executed or bracket.exit_in_progress:
                    continue
                action_type = str(action.get('type', '')).upper()
                is_protective_exit = action_type in {'SL', 'FINAL_TP', 'PARTIAL_TP'}
                if (not bracket.active and not is_protective_exit) or bracket.remaining_quantity <= 0:
                    continue
                last_attempt = self._exit_cooldowns.get(bracket.entry_order_id, 0.0)
                if now - last_attempt < 0.5:
                    continue
                bracket.exit_in_progress = True
                self._exit_cooldowns[bracket.entry_order_id] = now
                approved.append((bracket, action))

        for bracket, action in approved:
            symbol = normalize_symbol(bracket.symbol)
            qty = int(action.get('qty', 0))
            reason = str(action.get('reason', 'EXIT'))
            if not symbol or qty <= 0:
                with self._lock:
                    bracket.exit_in_progress = False
                continue

            exit_order_id: str | None = None
            confirmed = False

            try:
                LOGGER.info('EXIT TRIGGERED symbol=%s qty=%s reason=%s', symbol, qty, reason)
                for attempt in range(1, 4):
                    try:
                        if self._exit_executor is None:
                            confirmed = True
                            break
                        result = self._exit_executor(symbol, qty)
                        exit_order_id = result if isinstance(result, str) else None
                        if exit_order_id and hasattr(self.order_manager, 'wait_for_fill'):
                            confirmed = bool(self.order_manager.wait_for_fill(exit_order_id, timeout_sec=2.0))
                        else:
                            confirmed = result is not None
                        if confirmed:
                            break
                    except Exception as e:
                        LOGGER.error('Failure in _fire_exits_batch: %s', e)
                    time.sleep(0.2)

                if not confirmed:
                    confirmed = self._market_fallback_exit(
                        bracket=bracket,
                        qty=qty,
                        exit_side='SELL' if bracket.side == 'BUY' else 'BUY',
                        reason=reason,
                    )

                with self._lock:
                    bracket.pending_exit_order_id = exit_order_id
                    bracket.exit_in_progress = False
                    if confirmed:
                        bracket.exit_executed = True
                        bracket.active = False
                        bracket.remaining_quantity = max(0, bracket.remaining_quantity - qty)
                        bracket.updated_at = time.time()
                    else:
                        bracket.exit_executed = False
                        bracket.active = True
                        self._exit_cooldowns.pop(bracket.entry_order_id, None)

                if confirmed:
                    LOGGER.info('EXIT_EXECUTED symbol=%s qty=%s', symbol, qty)
                    # FIX S10-2: notify runner so orchestrator direction lock clears immediately
                    hook = self._on_exit_complete_hook
                    if hook is not None:
                        try:
                            hook(symbol)
                        except Exception:
                            pass
                else:
                    LOGGER.critical('EXIT_FAILED_AFTER_FALLBACK symbol=%s qty=%s', symbol, qty)
            except Exception as e:
                LOGGER.error('Failure in _fire_exits_batch: %s', e)
                with self._lock:
                    bracket.exit_in_progress = False
                    bracket.exit_executed = False
                    bracket.active = True
                    self._exit_cooldowns.pop(bracket.entry_order_id, None)


    def on_tick_event(self, tick: dict[str, Any]) -> None:
        """Args: tick; Returns: none; Raises: none."""
        try:
            symbol = str(tick.get("symbol") or "")
            ltp = tick.get("last_price") or tick.get("ltp")
            if not symbol or not isinstance(ltp, (int, float)):
                return
            self.on_tick(symbol, float(ltp))
        except Exception as e:
            LOGGER.error("Failure in BracketManager.on_tick_event: %s", e)

    def _execute_exit_order(
        self, 
        bracket: BracketState, 
        qty: int, 
        exit_side: str, 
        reason: str,
        is_partial: bool
    ) -> None:
        """
        Execute the actual exit order with slippage protection.
        Called outside the main lock for better concurrency.
        """
        # All protective exits are MARKET to guarantee deterministic execution.
        _order_type = "MARKET"
        _price = None
        
        # Place order via OrderManager
        order_id = self.order_manager.place_order(
            symbol=bracket.symbol,
            side=exit_side,
            quantity=qty,
            order_type=_order_type,
            price=_price,
            tag=f"exit_{reason[:3]}_{bracket.tag[:3] if bracket.tag else 'auto'}",
            check_risk=False,
            product="MIS"
        )
        
        if not order_id:
            raise RuntimeError(
                f"Exit order BLOCKED for {bracket.symbol} ({reason}). "
                f"place_order returned None — bracket state must be reverted."
            )
        
        LOGGER.info(
                f"✅ Exit order placed: {bracket.symbol} | order_id={order_id} | "
                f"type={_order_type} | reason={reason}"
        )
        
        # Cleanup if full exit
        if not is_partial and bracket.remaining_quantity <= 0:
            self.unregister_bracket(bracket.entry_order_id)

    @property
    def active_brackets(self) -> Dict[str, BracketState]:
        """Return symbol-indexed active bracket map for recovery hooks."""
        with self._lock:
            return {b.symbol: b for b in self._brackets.values() if b.active}


    def _check_and_fire(self, bracket: BracketState, ltp: float) -> None:
        """Evaluate logic for a single bracket: Exits, Partials, and Trailing."""
        
        # 1. UPDATE TRAILING STATE (High/Low Water Marks)
        self._process_trailing_logic(bracket, ltp)

        # 2. STOP LOSS CHECK
        if self._check_stop_loss(bracket, ltp):
            return  # SL hit, stop processing

        # 3. PARTIAL TARGETS (TP1)
        self._check_partial_targets(bracket, ltp)

        # 4. FINAL TARGET (TP2)
        self._check_final_target(bracket, ltp)

    def _process_trailing_logic(self, bracket: BracketState, ltp: float) -> None:
        """Updates High/Low marks and adjusts SL if Trailing is enabled."""
        
        # A. Update Water Marks
        if bracket.side == "BUY":
            if ltp > bracket.highest_ltp:
                bracket.highest_ltp = ltp
        else: # SELL
            if ltp < bracket.lowest_ltp:
                bracket.lowest_ltp = ltp

        # B. Delegate to World-Class Controller (If attached)
        if bracket.entry_order_id in self._trailing_controllers:
            ctrl = self._trailing_controllers[bracket.entry_order_id]
            # ✅ FIX: Inject dynamic ATR from feed
            current_atr = self._current_atr.get(bracket.symbol)
            if current_atr and hasattr(ctrl, "update_atr"):
                ctrl.update_atr(current_atr)
            # The controller calculates using ATR and calls _virtual_modify_sl if needed
            # We updated bracket.last_ltp in on_tick, so controller reads fresh data
            ctrl.on_tick(None) 
            return

        # C. Fallback Legacy Logic
        self._apply_trailing_math(bracket)

    def _apply_trailing_math(self, bracket: BracketState) -> None:
        """Apply adaptive trailing rules for the bracket.

        Args:
            bracket: Active bracket state with the latest LTP snapshot.

        Returns:
            None.

        Raises:
            None.
        """
        if not bracket.trailing_enabled:
            return
        
        ltp = bracket.last_ltp
        if not ltp or ltp <= 0:
            return
        
        entry = bracket.entry_price
        if entry <= 0:
            return
        
        current_sl = bracket.sl_trigger_price
        
        # Get ATR for trailing calculations.
        atr = self._get_current_atr(bracket.symbol)
        if atr <= 0:
            if not bracket._atr_warning_logged:
                LOGGER.warning(
                    "ATR unavailable — trailing fallback for %s",
                    bracket.symbol,
                    extra={"event": "trailing_atr_missing", "symbol": bracket.symbol},
                )
                bracket._atr_warning_logged = True
            # CRITICAL FIX: Use 2% of entry_price as ATR fallback.
            # NEVER use stop_loss_price (e.g. ₹120) as ATR — that causes
            # Tier 3/4 to compute: high_water - (120 * 1.5) = negative SL,
            # which disables the stop-loss entirely → unlimited loss exposure.
            atr = bracket.entry_price * 0.02 if bracket.entry_price > 0 else 1.0
        
        # Calculate profit metrics
        if bracket.side == "BUY":
            profit_pct = ((ltp - entry) / entry) * 100
            high_water = bracket.highest_ltp
            profit_points = ltp - entry
        else:
            profit_pct = ((entry - ltp) / entry) * 100
            high_water = bracket.lowest_ltp
            profit_points = entry - ltp
        
        # Calculate new SL based on tiered system
        new_sl = self._calculate_tiered_trailing_sl(
            bracket=bracket,
            ltp=ltp,
            profit_pct=profit_pct,
            high_water=high_water,
            atr=atr
        )
        
        if new_sl is None:
            return
        
        # Apply the new SL (only if it improves protection).
        # FIX: re-read current_sl INSIDE the lock — the watchdog thread can modify
        # sl_trigger_price between our stale read above and this write. Using the
        # stale value could write a LOWER SL than what the watchdog already set,
        # making protection WORSE on a trailing update.
        with self._lock:
            if bracket.side == "BUY":
                current_sl = bracket.sl_trigger_price  # authoritative read under lock
                if new_sl > current_sl:
                    old_sl = bracket.sl_trigger_price
                    bracket.sl_trigger_price = _round_to_tick(new_sl)
                    bracket.updated_at = time.time()
                    
                    LOGGER.debug(
                        f"📈 TRAIL UPDATE {bracket.symbol}: "
                        f"SL {old_sl:.2f} → {new_sl:.2f} | "
                        f"Profit: {profit_pct:.1f}% | LTP: {ltp:.2f}"
                    )
                    LOGGER.debug('TRAILING_SL_UPDATED symbol=%s sl=%s', bracket.symbol, round(new_sl, 2))
                    if self._should_notify_trail(
                        bracket.entry_order_id, bracket.sl_trigger_price, old_sl
                    ):
                        self._notify_event(
                            'BRACKET_TRAIL_UPDATE',
                            {
                                'symbol': bracket.symbol,
                                'side': bracket.side,
                                'sl': round(bracket.sl_trigger_price, 2),
                                'ltp': round(ltp, 2),
                                'profit_pct': round(profit_pct, 2),
                            },
                        )
            else:  # SELL
                current_sl = bracket.sl_trigger_price  # authoritative read under lock
                if new_sl < current_sl:
                    old_sl = bracket.sl_trigger_price
                    bracket.sl_trigger_price = _round_to_tick(new_sl)
                    bracket.updated_at = time.time()
                    
                    LOGGER.debug(
                        f"📉 TRAIL UPDATE {bracket.symbol}: "
                        f"SL {old_sl:.2f} → {new_sl:.2f} | "
                        f"Profit: {profit_pct:.1f}% | LTP: {ltp:.2f}"
                    )
                    LOGGER.debug('TRAILING_SL_UPDATED symbol=%s sl=%s', bracket.symbol, round(new_sl, 2))
                    if self._should_notify_trail(
                        bracket.entry_order_id, bracket.sl_trigger_price, old_sl
                    ):
                        self._notify_event(
                            'BRACKET_TRAIL_UPDATE',
                            {
                                'symbol': bracket.symbol,
                                'side': bracket.side,
                                'sl': round(bracket.sl_trigger_price, 2),
                                'ltp': round(ltp, 2),
                                'profit_pct': round(profit_pct, 2),
                            },
                        )

    def _calculate_tiered_trailing_sl(
        self,
        bracket: BracketState,
        ltp: float,
        profit_pct: float,
        high_water: float,
        atr: float
    ) -> float | None:
        """
        Calculate trailing SL using tiered protection system.
        
        TIERS:
        - Tier 0 (< 1%): No trailing, use original SL
        - Tier 1 (1-2%): Lock breakeven
        - Tier 2 (2-4%): Protect 40% of profit
        - Tier 3 (4-6%): Protect 50% of profit + ATR trail
        - Tier 4 (> 6%): Protect 60% of profit + tight ATR trail
        """
        entry = bracket.entry_price
        current_sl = bracket.sl_trigger_price
        
        # Use instance-cached thresholds (set at __init__) — not os.getenv on every tick.
        tier1_threshold = self._trail_tier1_pct
        tier2_threshold = self._trail_tier2_pct
        tier3_threshold = self._trail_tier3_pct
        tier4_threshold = self._trail_tier4_pct
        
        # ═══════════════════════════════════════════════════════════
        # TIER 0: NO PROFIT (< 1%) - Use original SL
        # ═══════════════════════════════════════════════════════════
        if profit_pct < tier1_threshold:
            return None  # No change
        
        # ═══════════════════════════════════════════════════════════
        # TIER 1: SMALL PROFIT (1-2%) - BREAKEVEN LOCK
        # ═══════════════════════════════════════════════════════════
        if tier1_threshold <= profit_pct < tier2_threshold:
            # Lock at breakeven (entry price)
            if bracket.side == "BUY":
                if entry > current_sl:
                    LOGGER.debug(f"🔒 Tier 1: Breakeven lock at {entry:.2f}")
                    return entry
            else:
                if entry < current_sl:
                    LOGGER.debug(f"🔒 Tier 1: Breakeven lock at {entry:.2f}")
                    return entry
            return None
        
        # ═══════════════════════════════════════════════════════════
        # TIER 2: MODERATE PROFIT (2-4%) - PROTECT 40%
        # ═══════════════════════════════════════════════════════════
        if tier2_threshold <= profit_pct < tier3_threshold:
            protection_pct = 0.40
            
            if bracket.side == "BUY":
                profit_amount = high_water - entry
                protected_sl = entry + (profit_amount * protection_pct)
                LOGGER.debug(f"🛡️ Tier 2: 40% protection at {protected_sl:.2f}")
                return protected_sl
            else:
                profit_amount = entry - high_water
                protected_sl = entry - (profit_amount * protection_pct)
                LOGGER.debug(f"🛡️ Tier 2: 40% protection at {protected_sl:.2f}")
                return protected_sl
        
        # ═══════════════════════════════════════════════════════════
        # TIER 3: GOOD PROFIT (4-6%) - PROTECT 50% + ATR TRAIL
        # ═══════════════════════════════════════════════════════════
        if tier3_threshold <= profit_pct < tier4_threshold:
            protection_pct = 0.50
            
            # Calculate momentum-adjusted ATR multiplier
            momentum = self._calculate_momentum(bracket.symbol)
            atr_mult = self._get_momentum_adjusted_atr_mult(momentum, base_mult=1.5)
            
            if bracket.side == "BUY":
                # Minimum: 50% profit protection
                profit_amount = high_water - entry
                min_sl = entry + (profit_amount * protection_pct)
                
                # ATR trail from high water
                if atr > 0:
                    atr_sl = high_water - (atr * atr_mult)
                    # Use whichever is HIGHER (more protective)
                    protected_sl = max(min_sl, atr_sl)
                else:
                    protected_sl = min_sl
                
                LOGGER.debug(f"📊 Tier 3: 50% + ATR trail at {protected_sl:.2f}")
                return protected_sl
            else:
                profit_amount = entry - high_water
                min_sl = entry - (profit_amount * protection_pct)
                
                if atr > 0:
                    atr_sl = high_water + (atr * atr_mult)
                    protected_sl = min(min_sl, atr_sl)
                else:
                    protected_sl = min_sl
                
                LOGGER.debug(f"📊 Tier 3: 50% + ATR trail at {protected_sl:.2f}")
                return protected_sl
        
        # ═══════════════════════════════════════════════════════════
        # TIER 4: EXCELLENT PROFIT (> 6%) - PROTECT 60% + TIGHT TRAIL
        # ═══════════════════════════════════════════════════════════
        if profit_pct >= tier4_threshold:
            protection_pct = 0.60
            
            # Tighter ATR multiplier for big winners
            momentum = self._calculate_momentum(bracket.symbol)
            atr_mult = self._get_momentum_adjusted_atr_mult(momentum, base_mult=1.0)
            
            if bracket.side == "BUY":
                profit_amount = high_water - entry
                min_sl = entry + (profit_amount * protection_pct)
                
                if atr > 0:
                    atr_sl = high_water - (atr * atr_mult)
                    protected_sl = max(min_sl, atr_sl)
                else:
                    protected_sl = min_sl
                
                LOGGER.info(f"🏆 Tier 4: 60% + tight trail at {protected_sl:.2f}")
                return protected_sl
            else:
                profit_amount = entry - high_water
                min_sl = entry - (profit_amount * protection_pct)
                
                if atr > 0:
                    atr_sl = high_water + (atr * atr_mult)
                    protected_sl = min(min_sl, atr_sl)
                else:
                    protected_sl = min_sl
                
                LOGGER.info(f"🏆 Tier 4: 60% + tight trail at {protected_sl:.2f}")
                return protected_sl
        
        return None

    def _get_current_atr(self, symbol: str) -> float:
        """Get current ATR from provider or cache."""
        # Try safe provider first
        if self._atr_provider:
            try:
                atr_value = None
                if hasattr(self._atr_provider, "get_current_atr"):
                    atr_value = self._atr_provider.get_current_atr(symbol)
                elif hasattr(self._atr_provider, "get_atr"):
                    snapshot = self._atr_provider.get_atr(symbol)
                    atr_value = getattr(snapshot, "value", snapshot)
                if atr_value is not None and float(atr_value) > 0:
                    return float(atr_value)
            except Exception:
                pass
        
        # Fallback to local cache with type-safe normalization.
        atr_raw = self._current_atr.get(symbol, 0.0)
        if isinstance(atr_raw, dict):
            atr_val = atr_raw.get("value")
        elif isinstance(atr_raw, (int, float)):
            atr_val = float(atr_raw)
        else:
            atr_val = None
        if not atr_val or atr_val <= 0:
            return 0.0
        return float(atr_val)

    def _calculate_momentum(self, symbol: str) -> float:
        """
        Calculate price momentum from recent ticks.
        Returns: Momentum as percentage change (positive = up, negative = down)
        """
        # Use last_ltp changes if available
        # This is a simplified momentum calculation
        # For production, you might want to track a deque of recent prices
        
        # Check if we have historical tick data
        if hasattr(self, '_recent_ticks') and symbol in self._recent_ticks:
            ticks = self._recent_ticks[symbol]
            if len(ticks) >= 5:
                first = ticks[-5]
                last = ticks[-1]
                if first > 0:
                    return ((last - first) / first) * 100
        
        # Fallback: use bracket's high/low water marks
        bracket = None
        with self._lock:
            for b in self._brackets.values():
                if b.symbol == symbol and b.active:
                    bracket = b
                    break
        
        if bracket:
            ltp = bracket.last_ltp
            entry = bracket.entry_price
            if entry > 0 and ltp > 0:
                # Simple momentum: current move from entry
                return ((ltp - entry) / entry) * 100
        
        return 0.0

    def _get_momentum_adjusted_atr_mult(self, momentum: float, base_mult: float) -> float:
        """
        Adjust ATR multiplier based on momentum.
        
        Strong momentum = tighter trail (capture the move)
        Weak momentum = looser trail (avoid noise)
        """
        abs_momentum = abs(momentum)
        
        if abs_momentum > 1.0:
            # Very strong momentum: tight trail
            return base_mult * 0.8
        elif abs_momentum > 0.5:
            # Strong momentum: slightly tighter
            return base_mult * 0.9
        elif abs_momentum > 0.2:
            # Moderate momentum: standard
            return base_mult
        else:
            # Weak/choppy: looser trail
            return base_mult * 1.3
            

    def _check_stop_loss(self, bracket: BracketState, ltp: float) -> bool:
        """Returns True if SL hit and exit fired (Safe against 0.0)."""
        triggered = False
        reason = ""

        # 🛑 SAFETY: Ignore if SL is 0.0 (Disabled/Unset)
        if bracket.sl_trigger_price <= 0:
            return False

        if bracket.side == "BUY":
            if ltp <= bracket.sl_trigger_price:
                triggered = True
                reason = f"SL Hit ({ltp} <= {bracket.sl_trigger_price:.2f})"
        else: # SELL
            if ltp >= bracket.sl_trigger_price:
                triggered = True
                reason = f"SL Hit ({ltp} >= {bracket.sl_trigger_price:.2f})"

        if triggered:
            LOGGER.warning(f"🛑 STOP LOSS TRIGGERED for {bracket.symbol} | {reason}")
            # Exit full remaining quantity
            self._execute_exit(bracket, bracket.remaining_quantity, reason, is_partial=False)
            return True
        return False

    def _check_partial_targets(self, bracket: BracketState, ltp: float) -> None:
        """Checks TP1/Intermediate levels."""
        for target in bracket.tp_levels:
            if target.executed:
                continue
            
            triggered = False
            if bracket.side == "BUY":
                if ltp >= target.price:
                    triggered = True
            else: # SELL
                if ltp <= target.price:
                    triggered = True
            
            if triggered:
                reason = f"{target.name} Hit ({ltp})"
                qty_to_close = min(target.quantity, bracket.remaining_quantity)
                
                # Execute Partial
                success = self._execute_exit(bracket, qty_to_close, reason, is_partial=True)
                
                if success:
                    target.executed = True
                    # AUTO-ADJUST: Move SL to Breakeven after TP1
                    if target.name == "TP1":
                        self._move_sl_to_breakeven(bracket)

    def _check_final_target(self, bracket: BracketState, ltp: float) -> None:
        """Checks Final TP (Safe against 0.0)."""
        # 🛑 SAFETY: Ignore if TP is 0.0 (Disabled/Unset)
        if bracket.tp_trigger_price <= 0:
            return

        triggered = False
        if bracket.side == "BUY":
            if ltp >= bracket.tp_trigger_price:
                triggered = True
        else: # SELL
            if ltp <= bracket.tp_trigger_price:
                triggered = True

        if triggered:
            reason = f"FINAL TP Hit ({ltp})"
            self._execute_exit(bracket, bracket.remaining_quantity, reason, is_partial=False)

    def _move_sl_to_breakeven(self, bracket: BracketState) -> None:
        """Moves SL to Entry Price (Cost)."""
        with self._lock:
            if bracket.side == "BUY":
                if bracket.entry_price > bracket.sl_trigger_price:
                    bracket.sl_trigger_price = bracket.entry_price
                    LOGGER.info(f"🔒 {bracket.symbol}: SL Moved to Breakeven ({bracket.entry_price})")
            else:
                if bracket.entry_price < bracket.sl_trigger_price:
                    bracket.sl_trigger_price = bracket.entry_price
                    LOGGER.info(f"🔒 {bracket.symbol}: SL Moved to Breakeven ({bracket.entry_price})")

    def _execute_exit(self, bracket: BracketState, qty: int, reason: str, is_partial: bool) -> bool:
        """
        Send exit order with SLIPPAGE PROTECTION.
        Uses aggressive LIMIT instead of MARKET.
        
        ✅ WORLD-CLASS OPTIMIZATION: Caps slippage at 2%
        """
        import os
        
        if qty <= 0:
            return False

        now = time.time()

        with self._lock:
            # 🛑 1. COOLDOWN CHECK (Prevent Rapid Firing)
            last_attempt = self._exit_cooldowns.get(bracket.entry_order_id, 0)
            cooldown_seconds = float(os.getenv("EXIT_COOLDOWN_SECONDS", "5.0"))
            
            if now - last_attempt < cooldown_seconds:
                LOGGER.debug(f"⏳ Exit Cooldown Active for {bracket.symbol}. Skipping.")
                return False

            # Double check if we still have quantity or are active
            if bracket.remaining_quantity <= 0 or not bracket.active:
                return False
            
            # Set Cooldown IMMEDIATELY
            self._exit_cooldowns[bracket.entry_order_id] = now
            
            # 🛑 2. STATE LOCKING (Update *Before* Order)
            bracket.remaining_quantity -= qty
            
            if not is_partial or bracket.remaining_quantity <= 0:
                bracket.active = False
            
            # Persist state (non-blocking if using async)
            try:
                self.save_state()
            except Exception:
                pass  # Don't let persistence failure block exit

        # ═══════════════════════════════════════════════════════════
        # ✅ WORLD-CLASS: SLIPPAGE-PROTECTED EXIT
        # ═══════════════════════════════════════════════════════════
        exit_side = "SELL" if bracket.side == "BUY" else "BUY"
        
        # Get current market price
        current_ltp = bracket.last_ltp
        if not current_ltp or current_ltp <= 0:
            # Fallback: Use entry price if LTP unavailable
            current_ltp = bracket.entry_price
        
        # Calculate LIMIT price with slippage buffer
        max_slippage_pct = float(os.getenv("EXIT_MAX_SLIPPAGE_PCT", "2.0"))
        
        if exit_side == "SELL":
            # SELL exit: Accept price up to 2% BELOW current
            limit_price = round(current_ltp * (1 - max_slippage_pct / 100), 2)
        else:
            # BUY exit (short cover): Accept price up to 2% ABOVE current
            limit_price = round(current_ltp * (1 + max_slippage_pct / 100), 2)
        
        # Ensure limit price is positive
        limit_price = max(limit_price, 0.05)
        
        LOGGER.warning(
            f"⚡ EXECUTING EXIT: {bracket.symbol} | {exit_side} {qty} | "
            f"LTP={current_ltp:.2f} | Limit={limit_price:.2f} | Reason={reason}"
        )

        try:
            # ✅ PRIMARY: Aggressive LIMIT order (should fill instantly)
            order_id = self.order_manager.place_order(
                symbol=bracket.symbol,
                side=exit_side,
                quantity=qty,
                order_type="LIMIT",           # ✅ LIMIT instead of MARKET
                price=limit_price,            # ✅ Protected price
                tag=f"exit_{reason[:3]}_{bracket.tag[:3] if bracket.tag else 'auto'}",
                check_risk=False,
                product="MIS"
            )
            
            if order_id:
                LOGGER.info(f"✅ Exit order placed: {order_id}")
                
                # Quick status check (non-blocking, 200ms)
                try:
                    time.sleep(0.2)
                    if hasattr(self.order_manager, '_broker'):
                        status = self.order_manager._broker.get_order_status(order_id)
                        if status:
                            status_str = str(status.get("status", "")).upper()
                            if status_str in {"COMPLETE", "FILLED"}:
                                LOGGER.info(f"✅ Exit filled immediately: {order_id}")
                            elif status_str == "REJECTED":
                                # Fallback to MARKET if LIMIT rejected
                                LOGGER.warning(f"⚠️ LIMIT rejected, trying MARKET fallback")
                                self._market_fallback_exit(bracket, qty, exit_side, reason)
                except Exception as e:
                    LOGGER.debug(f"Status check skipped: {e}")
            
            # Metrics
            if METRICS_AVAILABLE and METRICS:
                try:
                    METRICS.brackets_triggered.inc()
                except Exception:
                    pass
                
            # Cleanup if full exit
            if not is_partial and bracket.remaining_quantity <= 0:
                self.unregister_bracket(bracket.entry_order_id)

            return True

        except Exception as e:
            LOGGER.critical(f"🛑 EXIT FAILED for {bracket.symbol}: {e}", exc_info=True)
            
            # 🛑 3. REVERT STATE ON FAILURE
            with self._lock:
                bracket.remaining_quantity += qty
                if not is_partial:
                    bracket.active = True
                try:
                    self.save_state()
                except Exception:
                    pass
                
            # ✅ FALLBACK: Try MARKET order as last resort
            return self._market_fallback_exit(bracket, qty, exit_side, reason)

    def _verify_position_closed(self, symbol: str) -> bool:
        """Verify broker position closure. Args: symbol; Returns: bool; Raises: none."""
        try:
            broker = getattr(self.order_manager, '_broker', None)
            if broker is None:
                return True
            getter = getattr(broker, 'get_positions', None)
            if not callable(getter):
                return True
            positions = getter() or []
            normalized = normalize_symbol(symbol)
            for pos in positions:
                if not isinstance(pos, Mapping):
                    continue
                pos_symbol = normalize_symbol(str(pos.get('symbol') or pos.get('tradingsymbol') or ''))
                if pos_symbol != normalized:
                    continue
                qty = pos.get('quantity')
                if qty is None:
                    qty = pos.get('net_quantity')
                if abs(int(float(qty or 0))) > 0:
                    return False
            return True
        except Exception as e:
            LOGGER.error('Failure in _verify_position_closed: %s', e)
            return False

    def _market_fallback_exit(self, bracket: BracketState, qty: int, exit_side: str, reason: str) -> bool:
        """Force market exit after retries. Args: bracket,qty,exit_side,reason; Returns: bool; Raises: None."""
        try:
            LOGGER.warning('MARKET_FALLBACK_TRIGGER symbol=%s side=%s qty=%s', bracket.symbol, exit_side, qty)
            if hasattr(self.order_manager, 'cancel_orders_for_symbol'):
                try:
                    self.order_manager.cancel_orders_for_symbol(bracket.symbol)
                except Exception as e:
                    LOGGER.error('Failure in _market_fallback_exit: %s', e)

            order_id = self.order_manager.place_order(
                symbol=bracket.symbol,
                side=exit_side,
                quantity=qty,
                order_type='MARKET',
                tag=f'mkt_exit_{reason[:3]}',
                check_risk=False,
                product='MIS',
            )
            if not order_id:
                LOGGER.critical('MARKET_FALLBACK_REJECTED symbol=%s qty=%s', bracket.symbol, qty)
                return False

            if hasattr(self.order_manager, 'wait_for_fill'):
                filled = bool(self.order_manager.wait_for_fill(order_id, timeout_sec=3.0))
                if not filled:
                    LOGGER.critical('MARKET_FALLBACK_UNFILLED symbol=%s order_id=%s', bracket.symbol, order_id)
                    return False

            position_closed = self._verify_position_closed(bracket.symbol)
            if not position_closed:
                LOGGER.critical('MARKET_FALLBACK_POSITION_OPEN symbol=%s order_id=%s', bracket.symbol, order_id)
                return False

            LOGGER.info('MARKET_FALLBACK_EXECUTED symbol=%s order_id=%s', bracket.symbol, order_id)
            return True
        except Exception as e:
            LOGGER.critical('Failure in _market_fallback_exit: %s', e)
            return False


    # --------------------------------------------------------------------------
    # 4. SYNC & MANUAL INTERVENTION (World Class)
    # --------------------------------------------------------------------------

    def sync_manual_exit(self, symbol: str, quantity_left: int) -> None:
        """
        Called by OrderManager/PositionManager when position size changes externally.
        ORPHAN HANDLING: If qty goes to 0, kill all brackets for symbol.
        """
        if quantity_left <= 0:
            self.manual_override_close(symbol, reason="External/Manual Exit Detected")
        else:
            # Logic for partial manual exit can be added here
            # For now, we assume if some qty remains, we keep brackets active
            pass

    def manual_override_close(self, symbol: str, reason: str = "Manual Override") -> None:
        """Force close/remove all brackets for a symbol."""
        with self._lock:
            relevant_ids = self._symbol_map.get(symbol, [])
            if not relevant_ids:
                return
            
            count = 0
            for eid in list(relevant_ids):
                if eid in self._brackets:
                    # We strictly unregister, assuming the position is already gone/closing
                    self.unregister_bracket(eid)
                    count += 1
            
            if count > 0:
                LOGGER.info(f"🧹 Cleaned up {count} brackets for {symbol} due to: {reason}")

    def reconcile_with_broker(self, callback: Callable[[], Any]) -> Any:
        """Args: callback. Returns: callback result. Raises: Exception from callback."""
        with self._reconcile_lock:
            return callback()


    def sync_order_status(self, broker_order_id: str, status: str, filled_qty: int) -> None:
        """
        Detects if an Exit order initiated externally has filled.
        Used to keep internal state consistent.
        """
        if status not in _FILLED_STATUSES:
            return
            
        # This is a hook for future expansion where we map every broker order back to a bracket.
        # Currently handled via sync_manual_exit based on net position.
        pass

    # --------------------------------------------------------------------------
    # 5. DYNAMIC UPDATES (Trailing & Utils)
    # --------------------------------------------------------------------------

    def update_trailing_sl(self, symbol: str, new_sl: float) -> None:
        """Update SL monotonically for all active brackets on a symbol. Args: symbol,new_sl; Returns: None; Raises: None."""
        rounded_sl = _round_to_tick(new_sl)
        with self._lock:
            relevant_ids = self._symbol_map.get(symbol, [])
            if not relevant_ids:
                return

            for eid in relevant_ids:
                bracket = self._brackets.get(eid)
                if not bracket or not bracket.active or bracket.exit_in_progress:
                    continue

                old_sl = bracket.sl_trigger_price
                if bracket.side == 'BUY':
                    bracket.sl_trigger_price = max(old_sl, rounded_sl)
                else:
                    bracket.sl_trigger_price = min(old_sl, rounded_sl)

                if bracket.sl_trigger_price != old_sl:
                    bracket.updated_at = time.time()
                    LOGGER.debug('TRAILING_SL_UPDATED symbol=%s old=%s new=%s', symbol, round(old_sl, 2), round(bracket.sl_trigger_price, 2))

    # --------------------------------------------------------------------------
    # 6. HOUSEKEEPING & UTILS
    # --------------------------------------------------------------------------

    def is_symbol_managed(self, symbol: str) -> bool:
        """
        Check if symbol has active bracket protection.
        Used during position reconciliation to avoid duplicate brackets.
        """
        with self._lock:
            # 1. Fast check: symbol not in tracking map
            if symbol not in self._symbol_map:
                return False
            
            # 2. Deep check: Are any linked brackets actually active?
            # We iterate through all order IDs associated with this symbol
            entry_ids = self._symbol_map.get(symbol, [])
            for eid in entry_ids:
                bracket = self._brackets.get(eid)
                # It's managed if at least one bracket is Active and has Quantity remaining
                if bracket and bracket.active and bracket.remaining_quantity > 0:
                    return True
            
            return False

    def get_bracket(self, entry_id: str) -> Optional[BracketState]:
        with self._lock:
            return self._brackets.get(entry_id)

    def unregister_bracket(self, entry_id: str) -> None:
        """Remove a bracket from memory and indices."""
        with self._lock:
            if entry_id in self._brackets:
                bracket = self._brackets[entry_id]
                symbol = bracket.symbol
                
                # Cleanup Main Dict
                del self._brackets[entry_id]
                
                # Cleanup Symbol Map
                if symbol in self._symbol_map:
                    if entry_id in self._symbol_map[symbol]:
                        self._symbol_map[symbol].remove(entry_id)
                    if not self._symbol_map[symbol]:
                        del self._symbol_map[symbol]
                        
                # Cleanup Controller
                if entry_id in self._trailing_controllers:
                    del self._trailing_controllers[entry_id]

                # ✅ NEW: Cleanup Exit Cooldown
                if entry_id in self._exit_cooldowns:
                    del self._exit_cooldowns[entry_id]

                # ✅ FIX: Persist removal immediately
                self.save_state()

            # Cleanup reverse index (outside main check if orphaned)
            if entry_id in self._order_to_entry:
                del self._order_to_entry[entry_id]
                
    def cleanup_stale_brackets(self, max_age_seconds: int = 86400) -> int:
        """Remove old inactive brackets."""
        now = time.time()
        with self._lock:
            to_remove = [
                eid for eid, b in self._brackets.items()
                if (now - b.created_at) > max_age_seconds
            ]
            for eid in to_remove:
                self.unregister_bracket(eid)
            
            if to_remove:
                LOGGER.info(f"🧹 Cleaned up {len(to_remove)} stale brackets.")
            return len(to_remove)

    def get_stats(self) -> dict[str, Any]:
        """Diagnostic stats."""
        with self._lock:
            return {
                "active_brackets": len(self._brackets),
                "symbols_managed": len(self._symbol_map),
                "atr_tracked_symbols": len(self._current_atr),
                "adaptive_controllers": len(self._trailing_controllers)
            }
    # ----------------------------------------------------------------
    # 💾 PERSISTENCE LAYER (Add to BracketManager)
    # ----------------------------------------------------------------
    def _get_storage_path(self) -> Path:
        """Get storage path with DATA_DIR support for Railway.
        
        ✅ PRODUCTION FIX: Uses DATA_DIR env var with /tmp fallback.
        """
        import os
        
        data_dir = os.getenv("DATA_DIR", "data")
        path = Path(data_dir) / "virtual_brackets.json"
        
        # Test if we can write to this directory
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            test_file = path.parent / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
        except (PermissionError, OSError):
            # ✅ FIX: Fallback to /tmp
            path = Path(os.getenv("DATA_DIR", "data")) / "virtual_brackets.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(f"⚠️ Using /tmp fallback for brackets: {path}")
        
        return path

    def save_state(self) -> None:
        """Persist active brackets to disk ATOMICALLY with Enum handling.
        
        ✅ PRODUCTION FIX: Added Enum serialization and DATA_DIR support.
        """
        import uuid
        import os
        from enum import Enum
        from datetime import datetime, date
        from decimal import Decimal
        
        # ✅ FIX: Sanitize function for Enum types
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_sanitize(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value if hasattr(obj, 'value') else obj.name
            elif isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, Decimal):
                return float(obj)
            return obj
        
        data = {}
        with self._lock:
            for eid, b in self._brackets.items():
                data[eid] = _sanitize({
                    "entry_order_id": eid,
                    "symbol": b.symbol,
                    "quantity": b.quantity,
                    "remaining_quantity": b.remaining_quantity,
                    "entry_price": b.entry_price,
                    "side": b.side,
                    "sl_trigger_price": b.sl_trigger_price,
                    "tp_trigger_price": b.tp_trigger_price,
                    "trailing_enabled": b.trailing_enabled,
                    "trailing_config": b.trailing_config,
                    "active": b.active,
                    "created_at": b.created_at,
                    "updated_at": b.updated_at,
                    "highest_ltp": b.highest_ltp,
                    "lowest_ltp": b.lowest_ltp,
                    "last_ltp": b.last_ltp,
                    "tp_levels": [
                        {
                            "price": tl.price,
                            "quantity": tl.quantity,
                            "executed": tl.executed,
                            "name": tl.name
                        } for tl in b.tp_levels
                    ],
                    "tag": b.tag,
                    "exit_executed": b.exit_executed,
                    "atr_warning_logged": b._atr_warning_logged,
                })
        
        try:
            path = self._get_storage_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # ✅ FIX: Write to temp file first, then atomic replace
            tmp_path = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
            
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())  # Force write to physical disk
            
            # Atomic swap (Crash-safe)
            os.replace(tmp_path, path)
            LOGGER.debug(f"✅ Brackets saved to {path}")
            
        except Exception as e:
            LOGGER.error(f"Failed to save bracket state: {e}")

    def load_state(self) -> None:
        """Restore brackets from disk on startup."""
        path = self._get_storage_path()
        if not path.exists():
            LOGGER.info("No bracket state file found - starting fresh")
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            restored_count = 0
            with self._lock:
                for eid, d in data.items():
                    try:
                        # Restore TargetLevels
                        tp_levels = []
                        for tl_data in d.get("tp_levels", []):
                            tp_levels.append(TargetLevel(
                                price=tl_data["price"],
                                quantity=tl_data["quantity"],
                                executed=tl_data.get("executed", False),
                                name=tl_data.get("name", "TP")
                            ))
                        
                        # Reconstruct BracketState with CORRECT parameters
                        b = BracketState(
                            entry_order_id=eid,                                    # ✅ FIXED: was entry_id
                            symbol=d["symbol"],
                            quantity=d["quantity"],
                            entry_price=d["entry_price"],
                            side=d["side"],
                            sl_trigger_price=d["sl_trigger_price"],                # ✅ FIXED: was stop_loss
                            tp_trigger_price=d["tp_trigger_price"],                # ✅ FIXED: was take_profit
                            remaining_quantity=d.get("remaining_quantity", d["quantity"]),
                            tp_levels=tp_levels,
                            trailing_enabled=d.get("trailing_enabled", False),
                            trailing_config=d.get("trailing_config", {}),
                            created_at=d.get("created_at", time.time()),
                            updated_at=d.get("updated_at", time.time()),
                            highest_ltp=d.get("highest_ltp", d["entry_price"]),
                            lowest_ltp=d.get("lowest_ltp", d["entry_price"]),
                            last_ltp=d.get("last_ltp", d["entry_price"]),
                            previous_ltp=d.get("previous_ltp", d.get("last_ltp", d["entry_price"])),
                            tag=d.get("tag")
                        )
                        
                        # Set active state AFTER construction
                        b.active = d.get("active", True)
                        b.exit_executed = bool(d.get("exit_executed", False))
                        b.pending_exit_order_id = d.get("pending_exit_order_id")
                        b.exit_in_progress = bool(d.get("exit_in_progress", False))
                        b._atr_warning_logged = bool(d.get("atr_warning_logged", False))
                        b.virtual_sl_id = f"vsl_{eid}"
                        
                        # Register bracket
                        self._brackets[eid] = b
                        self._order_to_entry[eid] = eid
                        
                        # Update symbol map
                        if b.symbol not in self._symbol_map:
                            self._symbol_map[b.symbol] = []
                        self._symbol_map[b.symbol].append(eid)
                        
                        # Restore trailing controller if needed
                        if b.trailing_enabled and b.trailing_config.get("mode") == "ATR":
                            if self._atr_provider and AdaptiveTrailingController:
                                try:
                                    mult = b.trailing_config.get("mult", 1.5)
                                    spec = TrailingSpec(
                                        trail_by=20.0,
                                        step=0.25,
                                        activation=0.3
                                    )
                                    ctrl = AdaptiveTrailingController(
                                        symbol=b.symbol,
                                        side="LONG" if b.side == "BUY" else "SHORT",
                                        entry=b.entry_price,
                                        sl_order_id=b.virtual_sl_id,
                                        variety="virtual",
                                        spec=spec,
                                        get_ltp=lambda s: b.last_ltp,
                                        modify_order=self._virtual_modify_sl,
                                        atr_provider=self._atr_provider,
                                        journal=MockJournal(),
                                        atr_multiplier=mult
                                    )
                                    self._trailing_controllers[eid] = ctrl
                                    LOGGER.debug(f"🧠 Restored controller for {b.symbol}")
                                except Exception as e:
                                    LOGGER.warning(f"Failed to restore controller for {b.symbol}: {e}")
                        
                        restored_count += 1
                        
                    except Exception as e:
                        LOGGER.error(f"Failed to restore bracket {eid}: {e}")
                        continue
            
            LOGGER.info(f"♻️ Restored {restored_count} virtual brackets from disk")
            
            # ✅ CRITICAL: Resubscribe to market data
            self._resubscribe_restored_brackets()
            
        except Exception as e:
            LOGGER.error(f"Failed to load bracket state: {e}", exc_info=True)


    def _resubscribe_restored_brackets(self) -> None:
        """Resubscribe to market data for all restored brackets."""
        if not self._market_data:
            LOGGER.warning("Cannot resubscribe: MarketDataManager not available")
            return
        
        unique_symbols = set()
        with self._lock:
            for bracket in self._brackets.values():
                if bracket.active and bracket.remaining_quantity > 0:
                    unique_symbols.add(bracket.symbol)
        
        for symbol in unique_symbols:
            try:
                # Create callback closure for this symbol
                # NOTE: market_data.subscribe passes a tick dict, NOT (sym, ltp)
                def tick_callback(tick_data: dict, _sym=symbol) -> None:
                    try:
                        ltp = (
                            tick_data.get('ltp')
                            or tick_data.get('last_price')
                            or tick_data.get('price')
                        )
                        if ltp and float(ltp) > 0:
                            self.on_tick(_sym, float(ltp))
                    except Exception:
                        pass
                
                # Register with market data manager
                if hasattr(self._market_data, 'subscribe'):
                    self._market_data.subscribe(symbol, tick_callback)
                elif hasattr(self._market_data, 'register_callback'):
                    self._market_data.register_callback(symbol, tick_callback)
                
                LOGGER.info(f"🔔 Resubscribed {symbol} to market data feed")
                
            except Exception as e:
                LOGGER.error(f"Failed to resubscribe {symbol}: {e}")
        
        if unique_symbols:
            LOGGER.info(f"✅ Resubscribed {len(unique_symbols)} symbols to market data")

    def attach_orphan_position(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float
    ) -> str:
        """
        Wraps an existing naked position in a protective bracket.
        Called by Runner when 'ORPHAN GUARD' triggers.
        """
        symbol = normalize_symbol(symbol)
        if self._reconcile_lock.locked():
            return ""
        # ── FIX: use symbol-stable ID so repeated adoption attempts hit the
        # dedup path in register_virtual_bracket (which updates triggers on an
        # existing bracket) instead of creating a fresh bracket every second.
        # Timestamp-based oids bypassed dedup and filled _symbol_map with dead
        # brackets, causing is_symbol_managed() to always return False.
        oid = f"orphan_{symbol}"
        now = time.time()
        last_attempt = self._orphan_retry_last_attempt.get(symbol)
        if last_attempt is not None and (now - last_attempt) < 10:
            return oid
        if self._orphan_retry_count.get(symbol, 0) >= 3:
            LOGGER.error("Orphan adoption disabled after max retries: %s", symbol)
            return oid

        # 1. Dynamic ATR Calculation (if provider available)
        atr = max(entry_price * 0.005, 1.0)
        try:
            if self._atr_provider:
                calc_atr = None
                if hasattr(self._atr_provider, "get_current_atr"):
                    calc_atr = self._atr_provider.get_current_atr(symbol)
                elif hasattr(self._atr_provider, "get_atr"):
                    atr_snapshot = self._atr_provider.get_atr(symbol)
                    calc_atr = getattr(atr_snapshot, "value", atr_snapshot)
                if calc_atr and float(calc_atr) > 0:
                    atr = float(calc_atr)
        except Exception:
            self._orphan_retry_count[symbol] = self._orphan_retry_count.get(symbol, 0) + 1
            self._orphan_retry_last_attempt[symbol] = now
            LOGGER.exception("Failed orphan adoption")
            return oid

        # 2. Define Rescue Levels (1.5x Risk / 3.0x Reward)
        # Handle 'BUY'/'LONG' vs 'SELL'/'SHORT'
        is_long = _normalize_bracket_side(side) == 'BUY'
        
        if is_long:
            sl = entry_price - (atr * 0.7)
            tp = entry_price + (atr * 3.0)
        else:
            sl = entry_price + (atr * 0.7)
            tp = entry_price - (atr * 3.0)

        # 3. Register as IMMEDIATELY ACTIVE
        self.register_virtual_bracket(
            order_id=oid,
            symbol=symbol,
            side="BUY" if is_long else "SELL",
            qty=abs(qty),
            price=entry_price,
            sl=sl,
            tp=tp,
            tag="orphan_recovery",
            trailing_atr_mult=1.5,
            activate_immediately=True  # 🟢 Critical: It's already live
        )
        
        self._orphan_retry_count.pop(symbol, None)
        self._orphan_retry_last_attempt.pop(symbol, None)
        LOGGER.warning(
            f"🧯 ORPHAN ATTACHED: {symbol} | Entry={entry_price:.2f} | "
            f"SL={sl:.2f} | TP={tp:.2f} | ID={oid}"
        )
        return oid

    def create_bracket(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        quantity: int,
        strategy: str = "auto",
        order_id: str | None = None,
    ) -> str:
        """
        Create a virtual bracket with specified SL/TP levels.
        
        This is an alias method that provides the API expected by runner.py.
        Unlike attach_orphan_position(), this uses the SPECIFIED SL/TP values
        rather than auto-calculating them from ATR.
        
        Args:
            symbol: Trading symbol (e.g., "NIFTY2620324650CE")
            side: "LONG", "SHORT", "BUY", or "SELL"
            entry_price: Entry price for the position
            stop_loss: Stop loss price (must be > 0)
            take_profit: Take profit price (must be > 0)
            quantity: Position size (will use absolute value)
            strategy: Strategy name for tagging (default: "auto")
            order_id: Optional specific order ID (auto-generated if None)
            
        Returns:
            The order_id (entry_id) used for bracket registration.
            
        Example:
            >>> bm.create_bracket(
            ...     symbol="NIFTY2620324650CE",
            ...     side="LONG",
            ...     entry_price=165.0,
            ...     stop_loss=155.0,
            ...     take_profit=180.0,
            ...     quantity=65,
            ...     strategy="VWAP_Pro"
            ... )
        """
        import time
        
        # Normalize side to LONG/SHORT
        normalized_side = _normalize_bracket_side(side)
        if normalized_side == "BUY":
            normalized_side = "LONG"
        elif normalized_side == "SELL":
            normalized_side = "SHORT"
        
        # Validate side
        if normalized_side not in ("LONG", "SHORT"):
            LOGGER.warning(f"⚠️ create_bracket: Invalid side '{side}', defaulting to LONG")
            normalized_side = "LONG"
        
        # Auto-generate order_id if not provided
        if not order_id:
            safe_symbol = symbol.replace(":", "_")[:20]
            order_id = f"bracket_{int(time.time() * 1000)}_{safe_symbol}"
        
        # Validate SL/TP (use defaults if invalid)
        if stop_loss <= 0:
            LOGGER.warning(f"⚠️ create_bracket: Invalid SL={stop_loss}, using 5% default")
            stop_loss = entry_price * 0.95 if normalized_side == "BUY" else entry_price * 1.05
            
        if take_profit <= 0:
            LOGGER.warning(f"⚠️ create_bracket: Invalid TP={take_profit}, using 10% default")
            take_profit = entry_price * 1.10 if normalized_side == "BUY" else entry_price * 0.90
        
        # Register the bracket
        self.register_virtual_bracket(
            order_id=order_id,
            symbol=symbol,
            side=normalized_side,
            qty=abs(quantity),
            price=entry_price,
            sl=stop_loss,
            tp=take_profit,
            tag=strategy,
            trailing_atr_mult=1.5,  # Enable ATR-based trailing
            activate_immediately=True,  # Start monitoring immediately
        )
        
        LOGGER.info(
            f"✅ create_bracket: {symbol} | {normalized_side} | "
            f"Entry={entry_price:.2f} | SL={stop_loss:.2f} | TP={take_profit:.2f} | "
            f"ID={order_id}"
        )
        
        return order_id

    # ═══════════════════════════════════════════════════════════════════════════
    # ✅ FIX: Add get_bracket_by_symbol() for Symbol-Based Lookup
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_bracket_by_symbol(self, symbol: str) -> Optional[BracketState]:
        """
        Get the first active bracket for a given symbol.
        
        This is different from get_bracket(entry_id) which looks up by order_id.
        Use this when you have a symbol but not the order_id.
        
        Args:
            symbol: Trading symbol (e.g., "NIFTY2620324650CE")
            
        Returns:
            BracketState if found and active, None otherwise.
            
        Example:
            >>> bracket = bm.get_bracket_by_symbol("NIFTY2620324650CE")
            >>> if bracket:
            ...     print(f"SL: {bracket.sl_trigger_price}")
        """
        with self._lock:
            entry_ids = self._symbol_map.get(symbol, [])
            
            # First pass: Look for active brackets
            for entry_id in entry_ids:
                bracket = self._brackets.get(entry_id)
                if bracket and bracket.active and bracket.remaining_quantity > 0:
                    return bracket
            
            # Second pass: Look for any bracket (even inactive)
            for entry_id in entry_ids:
                bracket = self._brackets.get(entry_id)
                if bracket and bracket.remaining_quantity > 0:
                    return bracket
            
            return None

    def reactivate_bracket_after_rejected_exit(
        self, symbol: str, rejected_order_id: str, reason: str
    ) -> bool:
        """Reactivate a bracket whose exit order was REJECTED or CANCELLED by the broker.

        Called by OrderManager._handle_order_rejected / on_order_update(CANCELLED)
        when Zerodha rejects or cancels an exit order.  Without this, the bracket
        stays permanently dead and the open position has zero SL protection.

        Args:
            symbol: Normalised trading symbol.
            rejected_order_id: The order_id that was rejected/cancelled.
            reason: Human-readable rejection reason for logging.

        Returns:
            True if a bracket was found and reactivated; False otherwise.
        """
        sym = normalize_symbol(symbol)
        with self._lock:
            entry_ids = self._symbol_map.get(sym, [])
            for entry_id in entry_ids:
                bracket = self._brackets.get(entry_id)
                if bracket is None:
                    continue
                if bracket.remaining_quantity <= 0:
                    continue
                # Match by pending_exit_order_id when possible; fall back to any
                # bracket marked exit_executed that still has remaining quantity.
                is_match = (
                    bracket.pending_exit_order_id == rejected_order_id
                    or (bracket.exit_executed and bracket.pending_exit_order_id is None)
                )
                if not is_match:
                    continue

                # Reactivate for retry on next tick / watchdog cycle.
                bracket.exit_executed = False
                bracket.active = True
                bracket.pending_exit_order_id = None
                # Clear exit cooldown so the very next tick can retry.
                self._exit_cooldowns.pop(entry_id, None)
                LOGGER.critical(
                    "🔁 EXIT ORDER %s REJECTED/CANCELLED for %s (qty=%s, reason=%s) "
                    "— bracket REACTIVATED for retry.",
                    rejected_order_id, sym, bracket.remaining_quantity, reason,
                )
                return True
        return False

    def get_all_brackets_for_symbol(self, symbol: str) -> List[BracketState]:
        """
        Get all brackets (active and inactive) for a given symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of BracketState objects (may be empty).
        """
        with self._lock:
            entry_ids = self._symbol_map.get(symbol, [])
            brackets = []
            for entry_id in entry_ids:
                bracket = self._brackets.get(entry_id)
                if bracket:
                    brackets.append(bracket)
            return brackets

    def has_active_bracket(self, symbol: str) -> bool:
        """
        Quick check if a symbol has any active bracket.
        
        This is an alias for is_symbol_managed() for clarity.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if symbol has at least one active bracket with remaining quantity.
        """
        return self.is_symbol_managed(symbol)
