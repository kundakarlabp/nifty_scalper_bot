"""
Order Processor: Dedicated Asynchronous State Machine for Order Lifecycle.
Production-Grade: Handles Risk Checks, Thread Safety, Position Awareness, and Exit Priority.
"""

import asyncio
from datetime import datetime, timezone
import os
from typing import Any, Dict, Literal, Mapping, cast

from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType
from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.risk.risk_manager import OrderSignal, RiskManager
from nifty_scalper_bot.utils.logging import get_logger

# --- Order Execution States ---
INTENT = "INTENT"  # Signal accepted, execution pending

LOGGER = get_logger(__name__)


class OrderProcessor:
    """
    Orchestrates order execution with strict Risk Management, Concurrency Control,
    and Position Awareness.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        safe_order_manager: OrderManager,
        risk_manager: RiskManager,
        position_manager: PositionManager,
        data_hub: Any,
    ) -> None:
        """Initialize the order processor. Args: message_bus, safe_order_manager, risk_manager, position_manager, data_hub. Returns: None. Raises: None."""
        self.bus = message_bus
        self.executor = safe_order_manager
        self.risk_manager = risk_manager
        self.pos_manager = position_manager
        self.data_hub = data_hub
        self._running = False

        # 1. Thread Safety Lock
        self._lock = asyncio.Lock()

        # 2. State Tracking
        self._active_trades: Dict[str, str] = {}
        self._last_signal_time: Dict[str, float] = {}
        self._max_active_option_trades = int(
            os.getenv("MAX_ACTIVE_OPTION_TRADES", "0") or 0
        )

        # 3. Validation: Ensure OrderManager supports brackets
        # This prevents silent naked trading if the underlying manager is outdated
        try:
            if not hasattr(self.executor, "place_bracket_order"):
                LOGGER.critical(
                    "FATAL: OrderManager does not support 'place_bracket_order'. Brackets will fail."
                )
                # We don't raise here to allow startup, but this is a critical configuration error.
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderProcessor.__init__: %s",
                exc,
                extra={"event": "order_processor_init_error"},
                exc_info=exc,
            )

    async def start(self) -> None:
        """Start the order processor message listener. Args: None. Returns: None. Raises: None."""
        LOGGER.info("OrderProcessor: Starting...")
        self._running = True
        try:
            self.bus.subscribe(MessageType.SIGNAL, self.on_strategy_signal)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderProcessor.start subscribe: %s",
                exc,
                extra={"event": "order_processor_subscribe_error"},
                exc_info=exc,
            )
        # Ensure executor monitoring is active
        try:
            self.executor.start_monitoring()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderProcessor.start: %s",
                exc,
                extra={"event": "order_processor_start_error"},
                exc_info=exc,
            )

    async def stop(self) -> None:
        """Stop the order processor. Args: None. Returns: None. Raises: None."""
        LOGGER.info("OrderProcessor: Stopping...")
        self._running = False
        try:
            self.executor.stop_monitoring()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderProcessor.stop: %s",
                exc,
                extra={"event": "order_processor_stop_error"},
                exc_info=exc,
            )

    def _resolve_trade_price(self, symbol: str, price: float) -> float:
        """Resolve a usable trade price. Args: symbol, price. Returns: float. Raises: None."""
        if price > 0:
            return price
        if not self.data_hub or not hasattr(self.data_hub, "get_quote"):
            return 0.0
        try:
            quote = self.data_hub.get_quote(symbol, allow_pull=True)
            if not isinstance(quote, dict):
                return 0.0
            ltp = quote.get("ltp") or quote.get("last_price")
            if isinstance(ltp, (int, float)) and ltp > 0:
                return float(ltp)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in OrderProcessor._resolve_trade_price: %s",
                exc,
                extra={
                    "event": "order_processor_price_resolve_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )
        return 0.0

    def _resolve_atr(self, symbol: str, metadata: Mapping[str, Any] | None) -> float:
        """Resolve ATR for a symbol. Args: symbol, metadata. Returns: float. Raises: None."""
        LOGGER.debug(
            'Entered OrderProcessor._resolve_atr',
            extra={'event': 'order_processor_resolve_atr', 'symbol': symbol},
        )
        default_atr = 10.0
        symbol_upper = symbol.upper()
        if 'NIFTY' in symbol_upper and 'BANK' not in symbol_upper:
            default_atr = 50.0
        elif 'BANKNIFTY' in symbol_upper or 'FINNIFTY' in symbol_upper:
            default_atr = 150.0
        try:
            if metadata:
                for key in ('atr', 'avg_true_range', 'atr_value'):
                    value = metadata.get(key)
                    if value is not None:
                        return float(value)
            if self.data_hub is not None:
                indicator_fn = getattr(self.data_hub, 'get_indicator', None)
                if callable(indicator_fn):
                    atr_value = indicator_fn(symbol, 'atr')
                    if atr_value is not None:
                        return float(atr_value)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                'Failure in OrderProcessor._resolve_atr: %s',
                exc,
                extra={'event': 'order_processor_resolve_atr_error', 'symbol': symbol},
                exc_info=exc,
            )
            return default_atr
        LOGGER.info(
            'Condition met: default_atr_applied',
            extra={
                'event': 'order_processor_default_atr',
                'symbol': symbol,
                'atr': default_atr,
            },
        )
        return default_atr

    def _resolve_regime(self, symbol: str, metadata: Mapping[str, Any] | None) -> str:
        """Resolve market regime for a symbol. Args: symbol, metadata. Returns: str. Raises: None."""
        LOGGER.debug(
            'Entered OrderProcessor._resolve_regime',
            extra={'event': 'order_processor_resolve_regime', 'symbol': symbol},
        )
        try:
            if metadata:
                for key in ('regime', 'market_regime', 'trend_state', 'market_state'):
                    value = metadata.get(key)
                    if value:
                        return str(value)
            if self.data_hub is not None:
                indicator_fn = getattr(self.data_hub, 'get_indicator', None)
                if callable(indicator_fn):
                    regime_value = indicator_fn(symbol, 'regime')
                    if regime_value:
                        return str(regime_value)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                'Failure in OrderProcessor._resolve_regime: %s',
                exc,
                extra={
                    'event': 'order_processor_resolve_regime_error',
                    'symbol': symbol,
                },
                exc_info=exc,
            )
            return 'UNKNOWN'
        LOGGER.info(
            'Condition met: default_regime_applied',
            extra={
                'event': 'order_processor_default_regime',
                'symbol': symbol,
                'regime': 'UNKNOWN',
            },
        )
        return 'UNKNOWN'

    def _build_protection_levels(
        self,
        symbol: str,
        side: str,
        price: float,
        stop_loss: float,
        take_profit: float,
        metadata: Mapping[str, Any] | None,
    ) -> tuple[float, float, float | None, str]:
        """Build protection levels. Args: symbol, side, price, stop_loss, take_profit, metadata. Returns: tuple. Raises: None."""
        LOGGER.debug(
            'Entered OrderProcessor._build_protection_levels',
            extra={'event': 'order_processor_protection_enter', 'symbol': symbol},
        )
        try:
            if price <= 0:
                return stop_loss, take_profit, None, 'UNKNOWN'

            side = side.upper()
            atr = self._resolve_atr(symbol, metadata)
            regime = self._resolve_regime(symbol, metadata)
            regime_upper = regime.upper()
            trailing_atr_mult = None
            if metadata:
                trailing_atr_mult = metadata.get('trailing_atr_mult') or metadata.get(
                    'trail_atr_mult'
                )
            if trailing_atr_mult is not None:
                try:
                    trailing_atr_mult = float(trailing_atr_mult)
                except (TypeError, ValueError) as exc:
                    LOGGER.warning(
                        'Condition met: trailing_mult_invalid',
                        extra={
                            'event': 'order_processor_trailing_mult_invalid',
                            'symbol': symbol,
                            'value': trailing_atr_mult,
                            'error': str(exc),
                        },
                    )
                    trailing_atr_mult = None
            default_sl_pct = float(os.getenv('DEFAULT_SL_PCT', '1.0'))

            sl_mult = 1.2
            tp_mult = 2.0
            trail_mult = 1.5
            if any(
                key in regime_upper for key in ('TREND', 'BREAKOUT', 'MOMENTUM')
            ):
                sl_mult = 1.3
                tp_mult = 3.0
                trail_mult = 1.2
            elif any(key in regime_upper for key in ('RANGE', 'CHOP', 'MEAN')):
                sl_mult = 1.0
                tp_mult = 1.8
                trail_mult = 1.8
            elif 'VOL' in regime_upper or 'SPIKE' in regime_upper:
                sl_mult = 1.5
                tp_mult = 2.5
                trail_mult = 1.4

            base_distance = atr if atr > 0 else price * (default_sl_pct / 100.0)
            if base_distance <= 0:
                base_distance = max(price * 0.005, 1.0)

            resolved_sl = stop_loss
            resolved_tp = take_profit
            used_fallback = False
            if resolved_sl <= 0:
                used_fallback = True
                if side == 'BUY':
                    resolved_sl = price - (base_distance * sl_mult)
                else:
                    resolved_sl = price + (base_distance * sl_mult)

            if resolved_tp <= 0:
                used_fallback = True
                if side == 'BUY':
                    resolved_tp = price + (base_distance * tp_mult)
                else:
                    resolved_tp = price - (base_distance * tp_mult)

            if side == 'BUY' and resolved_sl >= price:
                resolved_sl = price - base_distance
                used_fallback = True
            if side == 'SELL' and resolved_sl <= price:
                resolved_sl = price + base_distance
                used_fallback = True

            if side == 'BUY' and resolved_tp <= price:
                resolved_tp = price + (base_distance * tp_mult)
                used_fallback = True
            if side == 'SELL' and resolved_tp >= price:
                resolved_tp = price - (base_distance * tp_mult)
                used_fallback = True

            if trailing_atr_mult is None:
                trailing_atr_mult = trail_mult
            if trailing_atr_mult is not None and trailing_atr_mult <= 0:
                trailing_atr_mult = None

            if used_fallback:
                LOGGER.info(
                    'Condition met: protection_defaults_applied',
                    extra={
                        'event': 'order_processor_defaults_applied',
                        'symbol': symbol,
                        'regime': regime,
                        'atr': atr,
                        'stop_loss': resolved_sl,
                        'take_profit': resolved_tp,
                        'trailing_atr_mult': trailing_atr_mult,
                    },
                )
            return resolved_sl, resolved_tp, trailing_atr_mult, regime
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                'Failure in OrderProcessor._build_protection_levels: %s',
                exc,
                extra={
                    'event': 'order_processor_protection_error',
                    'symbol': symbol,
                },
                exc_info=exc,
            )
            return stop_loss, take_profit, None, 'UNKNOWN'

    async def on_strategy_signal(self, message: Message) -> None:
        """Handle incoming strategy signals with enforced bracket logic. Args: message. Returns: None. Raises: None."""
        if not self._running:
            return

        LOGGER.debug(
            "Entered OrderProcessor.on_strategy_signal",
            extra={"event": "order_processor_signal_enter"},
        )
        signal: dict[str, Any] = message.data
        symbol = signal.get("symbol")
        side = signal.get("side")
        qty = signal.get("quantity")
        price = float(signal.get("price", 0.0) or 0.0)

        # ✅ FIX 1: Extract Protection Data & Strategy Name
        stop_loss = float(signal.get("stop_loss") or 0.0)
        take_profit = float(signal.get("target") or signal.get("take_profit") or 0.0)
        strategy_name = str(signal.get("strategy_name") or "strategy_auto")
        metadata = signal.get("metadata") or {}
        trailing_spec = signal.get("trailing_spec") or metadata.get("trailing_spec")

        if not symbol or not side or not qty:
            LOGGER.error("Invalid Signal: %s", signal)
            return
        if side not in ("BUY", "SELL"):
            LOGGER.error(
                "Condition met: invalid_signal_side",
                extra={"event": "order_processor_invalid_side", "side": side},
            )
            return
        price = self._resolve_trade_price(symbol, price)
        stop_loss, take_profit, trailing_atr_mult, regime = (
            self._build_protection_levels(
                symbol=str(symbol),
                side=str(side),
                price=float(price),
                stop_loss=float(stop_loss),
                take_profit=float(take_profit),
                metadata=metadata,
            )
        )
        if trailing_spec is None and trailing_atr_mult:
            trailing_spec = {'kind': 'atr', 'mult': trailing_atr_mult}
        LOGGER.info(
            'Condition met: protection_resolved',
            extra={
                'event': 'order_processor_protection_resolved',
                'symbol': symbol,
                'regime': regime,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_atr_mult': trailing_atr_mult,
            },
        )

        # ================= OPTION ENTRY LIMIT (ENV CONTROLLED) =================
        if self._max_active_option_trades > 0 and symbol.startswith("NFO:"):
            # exits must NEVER be blocked
            position = self.pos_manager.get_position(symbol)
            is_exit_check = False
            if position and position.quantity != 0:
                if (position.side == "LONG" and side == "SELL") or (
                    position.side == "SHORT" and side == "BUY"
                ):
                    is_exit_check = True

            if not is_exit_check:
                try:
                    active_positions = self.pos_manager.get_all_positions()
                except Exception:
                    active_positions = []

                active_option_count = sum(
                    1
                    for pos in active_positions
                    if pos.symbol.startswith("NFO:") and pos.quantity != 0
                )

                if active_option_count >= self._max_active_option_trades:
                    LOGGER.warning(
                        "⛔ OPTION ENTRY BLOCKED (ENV LIMIT)",
                        extra={
                            "event": "option_trade_limit_block",
                            "limit": self._max_active_option_trades,
                            "active": active_option_count,
                            "incoming_symbol": symbol,
                            "strategy": strategy_name,
                        },
                    )
                    return
        # ================= END OPTION ENTRY LIMIT =============================

        # 1. Concurrency Check (Debounce)
        key = f"{symbol}"
        if key in self._active_trades:
            LOGGER.warning("⚠️ Skipping Signal %s: Active Order in Process", symbol)
            return

        self._active_trades[key] = INTENT

        try:
            # 2. Position Awareness (Exit vs Entry)
            position = self.pos_manager.get_position(symbol)
            is_exit = False

            if position and position.quantity != 0:
                # Simple logic: if side differs, it's an exit/reduction
                if (position.side == "LONG" and side == "SELL") or (
                    position.side == "SHORT" and side == "BUY"
                ):
                    is_exit = True
                    LOGGER.info("🔻 Signal Identified as EXIT for %s", symbol)

            # 3. Risk Check (Skip for exits to allow closing)
            if not is_exit:
                if price <= 0:
                    LOGGER.error(
                        "Condition met: price_missing_for_risk_check",
                        extra={
                            "event": "order_processor_price_missing",
                            "symbol": symbol,
                        },
                    )
                    return
                risk_signal = OrderSignal(
                    symbol=str(symbol),
                    side=cast(Literal["BUY", "SELL"], side),
                    quantity=int(qty),
                    price=float(price),
                    stop_loss=float(stop_loss) if stop_loss > 0 else None,
                    take_profit=float(take_profit) if take_profit > 0 else None,
                )
                risk_ok, risk_msg = self.risk_manager.check_order(
                    risk_signal,
                    live_enabled=True,
                )
                if not risk_ok:
                    LOGGER.warning("⛔ Risk Reject %s: %s", symbol, risk_msg)
                    # Release lock immediately on rejection
                    self._active_trades.pop(key, None)
                    return

            # 4. Determine Order Type
            order_type = OrderType.MARKET
            if price > 0:
                order_type = OrderType.LIMIT

            LOGGER.info(
                "🚀 Executing: %s %s %s @ %s (Is Exit: %s)",
                side,
                qty,
                symbol,
                price or "MKT",
                is_exit,
            )

            # 5. Execution Logic with FORCED BRACKETS
            broker_order_id = None

            # ✅ FIX 2: FORCE BRACKET EXECUTION FOR ENTRIES
            # If it's an ENTRY and we have valid Stop/Target, use place_bracket_order
            if not is_exit and stop_loss > 0 and take_profit > 0:

                # Double check capability to avoid crash
                if hasattr(self.executor, "place_bracket_order"):
                    LOGGER.info(
                        "🛡️ BRACKET ORDER SUBMITTED | %s | SL=%s | TP=%s",
                        symbol,
                        stop_loss,
                        take_profit,
                    )

                    broker_order_id = await asyncio.to_thread(
                        self.executor.place_bracket_order,
                        symbol=symbol,
                        side=side,
                        quantity=qty,
                        entry_price=price if order_type == OrderType.LIMIT else None,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        trailing_atr_mult=trailing_atr_mult,
                        tag=strategy_name,
                        trailing_spec=trailing_spec,
                    )
                else:
                    # Critical fallback if OrderManager is outdated, but logs the orphan risk
                    LOGGER.error(
                        "❌ OrderManager missing 'place_bracket_order'. Placing ORPHAN trade."
                    )
                    broker_order_id = await asyncio.to_thread(
                        self.executor.place_order,
                        symbol=symbol,
                        side=side,
                        quantity=qty,
                        order_type=order_type,
                        price=price,
                        tag=strategy_name,
                    )
            else:
                # Standard Execution for Exits or Naked Entries (if missing SL/TP)
                if not is_exit:
                    LOGGER.warning(
                        "⚠️ Executing NAKED ENTRY for %s (Missing SL/TP in signal)",
                        symbol,
                    )

                broker_order_id = await asyncio.to_thread(
                    self.executor.place_order,
                    symbol=symbol,
                    side=side,
                    quantity=qty,
                    order_type=order_type,
                    price=price,
                    tag=strategy_name,
                )

            # ✅ FIX 3: Update message with strategy source
            entry_id = broker_order_id
            stop_id = None
            target_id = None
            if isinstance(broker_order_id, tuple) and len(broker_order_id) == 3:
                entry_id, stop_id, target_id = broker_order_id
            await self.bus.publish(
                Message(
                    type=MessageType.ORDER_UPDATE,
                    timestamp=datetime.now(timezone.utc),
                    data={
                        "order_id": entry_id,
                        "symbol": symbol,
                        "status": "SUBMITTED",
                        "price": price,
                        "side": side,
                        "strategy": strategy_name,
                        "stop_order_id": stop_id,
                        "target_order_id": target_id,
                        "trailing_spec": trailing_spec,
                    },
                    source="order_processor",
                )
            )

        except Exception as exc:  # noqa: BLE001
            LOGGER.error("❌ Order Execution Failed: %s", exc, exc_info=exc)

            await self.bus.publish(
                Message(
                    type=MessageType.ORDER_UPDATE,
                    timestamp=datetime.now(timezone.utc),
                    data={"status": "REJECTED", "symbol": symbol, "error": str(exc)},
                    source="order_processor",
                )
            )
        finally:
            # ✅ FIX 3 (Refined): Release lock AFTER protection logic completes.
            # This ensures we don't accept a new signal until the bracket is effectively registered
            # (since place_bracket_order is synchronous-inside-thread).
            self._active_trades.pop(key, None)
