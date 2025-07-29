# src/execution/order_executor.py
"""
Handles order placement, GTT (Good-Till-Triggered) SL/TP setup,
trailing SL logic, and state management for the Nifty Scalper Bot.
Integrates with the Zerodha KiteConnect API.
"""
import logging
import time
import threading
from typing import Dict, List, Optional, Any
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

# Import configuration using the Config class for consistency
from config import Config

logger = logging.getLogger(__name__)

class OrderExecutor:
    """
    Manages the execution of trades including entry orders, stop-loss (SL),
    take-profit (TP), and trailing stop-loss logic using KiteConnect GTT orders.
    """
    def __init__(self, kite: KiteConnect):
        """
        Initializes the OrderExecutor.

        Args:
            kite (KiteConnect): An authenticated KiteConnect instance.

        Raises:
            ValueError: If the kite instance is None.
        """
        if not kite:
            raise ValueError("KiteConnect instance is required")
        self.kite: KiteConnect = kite
        # Stores information about active trading positions
        # Key: entry_order_id (str) -> Value: position details (dict)
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        # Stores information about active GTT orders
        # Key: gtt_order_id (str) -> Value: GTT details (dict)
        self.gtt_orders: Dict[str, Dict[str, Any]] = {}
        # For tracking last trail check times (if needed for rate limiting)
        self._last_trail_check: Dict[str, float] = {}
        # Ensures thread-safe updates to internal state dictionaries
        self.lock = threading.Lock()

    def place_entry_order(self,
                          symbol: str,
                          exchange: str,
                          transaction_type: str,
                          quantity: int,
                          product: Optional[str] = None,
                          order_type: Optional[str] = None) -> Optional[str]:
        """
        Places the initial market or limit entry order.

        Args:
            symbol (str): Trading symbol (e.g., 'NIFTY2351018000CE').
            exchange (str): Exchange (e.g., 'NSE', 'NFO').
            transaction_type (str): Order side ('BUY' or 'SELL').
            quantity (int): Number of units to trade.
            product (str, optional): Product code ('MIS', 'NRML', etc.).
                                    Defaults to Config.DEFAULT_PRODUCT.
            order_type (str, optional): Order type ('MARKET', 'LIMIT').
                                       Defaults to Config.DEFAULT_ORDER_TYPE.

        Returns:
            Optional[str]: The order ID if successful, None otherwise.
        """
        # Use Config defaults if not explicitly provided
        product_to_use = product if product is not None else Config.DEFAULT_PRODUCT
        order_type_to_use = order_type if order_type is not None else Config.DEFAULT_ORDER_TYPE
        validity_to_use = Config.DEFAULT_VALIDITY # Always use configured validity

        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product_to_use,
                order_type=order_type_to_use,
                validity=validity_to_use
                # price=limit_price, # Add if supporting limit entry orders explicitly
            )
            logger.info(
                f"✅ Entry order placed: Symbol={symbol}, "
                f"Type={transaction_type}, Qty={quantity}, Product={product_to_use}, "
                f"OrderType={order_type_to_use}, OrderID={order_id}"
            )
            return order_id
        except KiteException as e:
            logger.error(f"❌ Kite API error placing entry order: {e.message} (Code: {e.code})")
        except Exception as e:
            logger.error(f"❌ Unexpected error placing entry order for {symbol}: {e}", exc_info=True)
        return None

    def setup_gtt_orders(self,
                         entry_order_id: str,
                         entry_price: float,
                         stop_loss_price: float,
                         target_price: float,
                         symbol: str,
                         exchange: str,
                         quantity: int,
                         transaction_type: str,
                         product: Optional[str] = None) -> bool:
        """
        Sets up GTT orders for Stop-Loss and Take-Profit after an entry order is filled.

        Args:
            entry_order_id (str): The ID of the filled entry order.
            entry_price (float): The actual filled entry price.
            stop_loss_price (float): The stop-loss trigger price.
            target_price (float): The take-profit target price.
            symbol (str): Trading symbol.
            exchange (str): Exchange.
            quantity (int): Quantity traded.
            transaction_type (str): Original entry transaction type ('BUY'/'SELL').
            product (str, optional): Product code. Defaults to Config.DEFAULT_PRODUCT.

        Returns:
            bool: True if both GTTs are placed successfully, False otherwise.
        """
        product_to_use = product if product is not None else Config.DEFAULT_PRODUCT

        try:
            # Determine the reverse transaction type for exit orders
            reverse_transaction = (
                self.kite.TRANSACTION_TYPE_SELL
                if transaction_type == self.kite.TRANSACTION_TYPE_BUY
                else self.kite.TRANSACTION_TYPE_BUY
            )

            # --- Prepare GTT parameters for Stop-Loss ---
            sl_trigger_price = stop_loss_price
            sl_limit_price = stop_loss_price # Using trigger price as limit for SL
            sl_params = {
                "exchange": exchange,
                "tradingsymbol": symbol,
                "transaction_type": reverse_transaction,
                "quantity": quantity,
                "product": product_to_use,
                "order_type": self.kite.ORDER_TYPE_SL, # Stop-loss order type
                "price": sl_limit_price, # Limit price for the SL order
                "trigger_price": sl_trigger_price # Trigger price for the GTT
            }

            # --- Prepare GTT parameters for Take-Profit ---
            tp_trigger_price = target_price
            tp_limit_price = target_price # Using target price as limit for TP
            tp_params = {
                "exchange": exchange,
                "tradingsymbol": symbol,
                "transaction_type": reverse_transaction,
                "quantity": quantity,
                "product": product_to_use,
                "order_type": self.kite.ORDER_TYPE_LIMIT, # Limit order type for TP
                "price": tp_limit_price, # Limit price for the TP order
                "trigger_price": tp_trigger_price # Trigger price for the GTT
            }

            # --- Place GTT for Stop-Loss ---
            gtt_sl_id = self.kite.place_gtt(
                trigger_type=self.kite.GTT_TYPE_SINGLE,
                tradingsymbol=symbol,
                exchange=exchange,
                trigger_values=[sl_trigger_price], # List for single leg GTT
                last_price=entry_price, # Last traded price for validation
                orders=[sl_params] # List of order parameters
            )
            logger.info(f"✅ GTT Stop-Loss placed: GTT_ID={gtt_sl_id}, Trigger={sl_trigger_price}")

            # --- Place GTT for Take-Profit ---
            gtt_tp_id = self.kite.place_gtt(
                trigger_type=self.kite.GTT_TYPE_SINGLE,
                tradingsymbol=symbol,
                exchange=exchange,
                trigger_values=[tp_trigger_price],
                last_price=entry_price,
                orders=[tp_params]
            )
            logger.info(f"✅ GTT Take-Profit placed: GTT_ID={gtt_tp_id}, Trigger={tp_trigger_price}")

            # --- Update Internal State (Thread-Safe) ---
            with self.lock:
                # Store GTT SL info
                self.gtt_orders[str(gtt_sl_id)] = {
                    "parent_order_id": entry_order_id,
                    "type": "SL",
                    "original_trigger_price": sl_trigger_price, # Store for reference
                    "trigger_price": sl_trigger_price,
                    "limit_price": sl_limit_price,
                    "status": "active"
                }
                # Store GTT TP info
                self.gtt_orders[str(gtt_tp_id)] = {
                    "parent_order_id": entry_order_id,
                    "type": "TP",
                    "original_trigger_price": tp_trigger_price,
                    "trigger_price": tp_trigger_price,
                    "limit_price": tp_limit_price,
                    "status": "active"
                }
                # Store position info
                self.active_positions[entry_order_id] = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "quantity": quantity,
                    "entry_transaction_type": transaction_type,
                    "entry_price": entry_price,
                    "sl_price": stop_loss_price,
                    "tp_price": target_price,
                    "gtt_sl_id": str(gtt_sl_id),
                    "gtt_tp_id": str(gtt_tp_id),
                    "status": "open"
                }

            return True

        except KiteException as e:
            logger.error(f"❌ Kite API error placing GTT orders: {e.message} (Code: {e.code})")
        except Exception as e:
            logger.error(f"❌ Unexpected error placing GTT orders for {entry_order_id}: {e}", exc_info=True)
        return False

    def trail_stop_loss(self,
                        entry_order_id: str,
                        new_sl_price: float,
                        current_market_price: float) -> bool:
        """
        Modifies the existing Stop-Loss GTT order by cancelling the old one
        and placing a new GTT at a better price (trailing).

        Args:
            entry_order_id (str): The ID of the original entry order.
            new_sl_price (float): The new, improved stop-loss price.
            current_market_price (float): Current market price (for GTT validation).

        Returns:
            bool: True if the SL was trailed successfully, False otherwise.
        """
        try:
            with self.lock:
                # 1. Validate position exists
                if entry_order_id not in self.active_positions:
                    logger.warning(f"⚠️ Cannot trail SL: Unknown position {entry_order_id}")
                    return False

                position = self.active_positions[entry_order_id]
                old_gtt_sl_id = position.get("gtt_sl_id")
                old_gtt_info = self.gtt_orders.get(old_gtt_sl_id)

                # 2. Validate GTT SL exists and is active
                if not old_gtt_info or old_gtt_info.get("status") != "active":
                    logger.warning(f"⚠️ SL GTT not active or found for order {entry_order_id}")
                    return False

                # 3. Validate new SL price is beneficial
                entry_type = position["entry_transaction_type"]
                old_sl_price = old_gtt_info["trigger_price"]
                # For a BUY, new SL should be HIGHER. For a SELL, new SL should be LOWER.
                is_better_sl = (
                    (entry_type == self.kite.TRANSACTION_TYPE_BUY and new_sl_price > old_sl_price) or
                    (entry_type == self.kite.TRANSACTION_TYPE_SELL and new_sl_price < old_sl_price)
                )
                if not is_better_sl:
                    logger.debug(
                        f"ℹ️ SL trail skipped: New SL ({new_sl_price}) not better than "
                        f"Old SL ({old_sl_price}) for {entry_type} position {entry_order_id}"
                    )
                    return False

                # 4. Cancel the old GTT SL
                try:
                    self.kite.delete_gtt(old_gtt_sl_id)
                    logger.info(f"✅ Old SL GTT canceled: {old_gtt_sl_id}")
                    old_gtt_info["status"] = "cancelled"
                except KiteException as e:
                    if e.code == 404:
                        # GTT already triggered or was cancelled by another process
                        logger.info(f"ℹ️ Old SL GTT {old_gtt_sl_id} already triggered/cancelled.")
                        old_gtt_info["status"] = "triggered_or_cancelled"
                        # If triggered, the position is likely closed. Prevent further trailing.
                        return False
                    else:
                        logger.error(f"❌ Error canceling old SL GTT {old_gtt_sl_id}: {e.message}")
                        return False # Fail if we can't cancel the old one
                except Exception as e:
                    logger.error(f"❌ Unexpected error canceling old SL GTT {old_gtt_sl_id}: {e}", exc_info=True)
                    return False

            # 5. Place the new GTT SL (outside lock to minimize Kite API call time)
            reverse_transaction = (
                self.kite.TRANSACTION_TYPE_SELL
                if entry_type == self.kite.TRANSACTION_TYPE_BUY
                else self.kite.TRANSACTION_TYPE_BUY
            )
            new_sl_trigger_price = new_sl_price
            new_sl_limit_price = new_sl_price # Adjust if needed for limit orders

            new_sl_params = {
                "exchange": position["exchange"],
                "tradingsymbol": position["symbol"],
                "transaction_type": reverse_transaction,
                "quantity": position["quantity"],
                "product": position.get("product", product_to_use), # Use stored or default product
                "order_type": self.kite.ORDER_TYPE_SL,
                "price": new_sl_limit_price,
                "trigger_price": new_sl_trigger_price
            }

            try:
                new_gtt_sl_id = self.kite.place_gtt(
                    trigger_type=self.kite.GTT_TYPE_SINGLE,
                    tradingsymbol=position["symbol"],
                    exchange=position["exchange"],
                    trigger_values=[new_sl_trigger_price],
                    last_price=current_market_price, # Important for Kite's validation
                    orders=[new_sl_params]
                )
                logger.info(f"✅ New SL GTT placed: GTT_ID={new_gtt_sl_id}, Trigger={new_sl_trigger_price}")

                # 6. Update internal tracking (re-acquire lock)
                with self.lock:
                    self.gtt_orders[str(new_gtt_sl_id)] = {
                        "parent_order_id": entry_order_id,
                        "type": "SL",
                        "original_trigger_price": old_gtt_info["original_trigger_price"],
                        "trigger_price": new_sl_trigger_price,
                        "limit_price": new_sl_limit_price,
                        "status": "active"
                    }
                    position["gtt_sl_id"] = str(new_gtt_sl_id)
                    position["sl_price"] = new_sl_price # Update tracked SL price
                return True

            except KiteException as e:
                logger.error(f"❌ Kite API error placing new SL GTT: {e.message} (Code: {e.code})")
                # Note: The old GTT was cancelled, but the new one failed.
                # This leaves the position without a SL GTT. This is a critical state.
                # Consider adding logic to alert or handle this scenario.
            except Exception as e:
                logger.error(f"❌ Unexpected error placing new SL GTT: {e}", exc_info=True)
                # Same critical state as above KiteException.

        except Exception as e:
            logger.error(f"❌ Error in trail_stop_loss for {entry_order_id}: {e}", exc_info=True)
        return False # Return False if any step failed

    def cancel_all_orders_for_position(self, entry_order_id: str) -> bool:
        """
        Cancels both the Stop-Loss and Take-Profit GTT orders associated
        with a specific position.

        Args:
            entry_order_id (str): The ID of the entry order for the position.

        Returns:
            bool: True if cancellation was initiated for all relevant GTTs,
                  False if there was an error or the position was not found.
                  Note: A return of True does not guarantee the orders were
                  successfully cancelled by the exchange, only that the
                  cancellation request was sent.
        """
        success = True # Optimistic initial state
        try:
            with self.lock:
                if entry_order_id not in self.active_positions:
                    logger.warning(f"⚠️ Position not found for cancellation: {entry_order_id}")
                    return False # Or True, depending on desired semantics for "not found"

                position = self.active_positions[entry_order_id]
                gtt_sl_id = position.get("gtt_sl_id")
                gtt_tp_id = position.get("gtt_tp_id")

                # Iterate through GTT IDs and attempt cancellation
                for gtt_id in [gtt_sl_id, gtt_tp_id]:
                    if gtt_id and gtt_id in self.gtt_orders:
                        gtt_info = self.gtt_orders[gtt_id]
                        # Only try to cancel if it's marked as active
                        if gtt_info.get("status") == "active":
                            try:
                                self.kite.delete_gtt(gtt_id)
                                logger.info(f"✅ GTT order cancellation requested: {gtt_id}")
                                gtt_info["status"] = "cancelled"
                            except KiteException as e:
                                if e.code == 404:
                                    logger.info(f"ℹ️ GTT {gtt_id} already triggered or cancelled.")
                                    gtt_info["status"] = "triggered_or_cancelled"
                                    # This is not an error from our perspective for the cancel action
                                else:
                                    logger.error(f"❌ Error canceling GTT {gtt_id}: {e.message}")
                                    success = False # Mark overall failure if any Kite error (other than 404)
                            except Exception as e:
                                logger.error(f"❌ Unexpected error canceling GTT {gtt_id}: {e}", exc_info=True)
                                success = False # Mark overall failure on unexpected errors

            # Update position status outside the inner lock operations but still within the main lock context
            with self.lock:
                 if entry_order_id in self.active_positions:
                    self.active_positions[entry_order_id]["status"] = "orders_cancelled"

            return success

        except Exception as e:
            logger.error(f"❌ Error canceling orders for position {entry_order_id}: {e}", exc_info=True)
            return False

    def update_position_on_exit(self, entry_order_id: str, exit_reason: str = "unknown"):
        """
        Updates the internal state to mark a position as closed.

        Args:
            entry_order_id (str): The ID of the entry order.
            exit_reason (str): Reason for exit (e.g., 'SL_hit', 'TP_hit', 'manual').
        """
        with self.lock:
            if entry_order_id in self.active_positions:
                self.active_positions[entry_order_id]["status"] = f"closed_{exit_reason}"
                logger.info(f"✅ Position {entry_order_id} marked as closed. Reason: {exit_reason}")
            else:
                logger.warning(f"⚠️ Attempted to close unknown position: {entry_order_id}")

    # --- State Query Methods ---

    def get_position_status(self, entry_order_id: str) -> Optional[Dict[str, Any]]:
        """
        Gets the current status information for a specific position.

        Args:
            entry_order_id (str): The ID of the entry order.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing position details,
                                    or None if the position is not found.
        """
        # Returning a direct reference from a locked dict is safe for reads
        # if all modifications are within the lock. For absolute safety,
        # returning a copy is preferred if the caller might modify the result.
        return self.active_positions.get(entry_order_id)

    def get_gtt_status(self, gtt_id: str) -> Optional[Dict[str, Any]]:
        """
        Gets the current status information for a specific GTT order.

        Args:
            gtt_id (str): The GTT order ID.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing GTT details,
                                    or None if the GTT is not found.
        """
        return self.gtt_orders.get(gtt_id)

    def get_all_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Gets a copy of all currently tracked active positions.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary of all active positions.
        """
        with self.lock:
            return self.active_positions.copy()

    def get_all_gtt_orders(self) -> Dict[str, Dict[str, Any]]:
        """
        Gets a copy of all currently tracked GTT orders.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary of all GTT orders.
        """
        with self.lock:
            return self.gtt_orders.copy()

# Example usage concept (within RealTimeTrader context)
# This would typically be used inside _handle_trading_signal in RealTimeTrader
#
# def _handle_trading_signal(self, token, signal, position_details):
#     # ... (risk management to get position_details)
#
#     if self.execution_enabled and self.order_executor:
#         # --- IMPORTANT: Map token to symbol/exchange ---
#         # You MUST replace this placeholder logic with actual data from your
#         # strike selection process (e.g., main.py or a lookup table).
#         # Example placeholder mapping (NEEDS REAL DATA):
#         token_to_instrument = {
#             256265: {"symbol": "NIFTY 50", "exchange": "NSE"}, # Nifty Index
#             # Add your selected option tokens here, e.g., from strike selector
#             # 123456: {"symbol": "NIFTY23APR18000CE", "exchange": "NFO"},
#         }
#         instrument_info = token_to_instrument.get(token)
#         if not instrument_info:
#             logger.error(f"❌ Cannot execute: No symbol/exchange mapping for token {token}")
#             return
#
#         symbol = instrument_info["symbol"]
#         exchange = instrument_info["exchange"]
#         transaction_type = signal['signal'] # Assuming 'BUY' or 'SELL'
#
#         # 1. Place Entry Order
#         entry_order_id = self.order_executor.place_entry_order(
#             symbol=symbol,
#             exchange=exchange,
#             transaction_type=transaction_type,
#             quantity=position_details['quantity']
#             # product and order_type can use defaults
#         )
#
#         if entry_order_id:
#             # 2. Wait for order fill confirmation (simplified)
#             # In a real scenario, you'd poll kite.order_history or listen via WebSocket OMS
#             time.sleep(2) # Placeholder wait
#             # Get filled price (simplified, use actual order history)
#             filled_price = signal['entry_price'] # Placeholder
#
#             # 3. Setup GTT Orders
#             success = self.order_executor.setup_gtt_orders(
#                 entry_order_id=entry_order_id,
#                 entry_price=filled_price,
#                 stop_loss_price=signal['stop_loss'],
#                 target_price=signal['target'],
#                 symbol=symbol,
#                 exchange=exchange,
#                 quantity=position_details['quantity'],
#                 transaction_type=transaction_type
#             )
#             if success:
#                 logger.info("✅ Entry and GTTs placed successfully")
#                 # Update risk manager
#                 self.risk_manager.update_position_status(is_open=True)
#             else:
#                 logger.error("❌ Failed to place GTT orders")
#                 # Handle failure (e.g., cancel entry order if it didn't fill as expected)
#         else:
#             logger.error("❌ Failed to place entry order")
#     else:
#         logger.info("⚠️ Execution is disabled or OrderExecutor not available (simulated).")