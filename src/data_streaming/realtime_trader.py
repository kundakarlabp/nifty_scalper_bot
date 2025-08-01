"""
Simplified realâ€‘time trading engine.

This module glues together the strategy, risk manager, order executor and
Telegram controller.  In contrast to the original project which relied
on the Zerodha WebSocket feed and complex asynchronous callbacks, this
refactored version exposes a straightforward interface for starting and
stopping the bot, processing new OHLCV data and producing status
updates.  It is intended both for live trading (when integrated with
broker data) and for unit testing.

Usage example::

    from src.data_streaming.realtime_trader import RealTimeTrader
    from src.utils.indicators import indicator_utils
    import pandas as pd

    trader = RealTimeTrader()
    trader.start()  # enable trading and polling
    while live_data_available:
        df = get_latest_dataframe()
        trader.process_bar(df)

    trader.stop()

The ``TelegramController`` polls in a background thread when the
trader is started.  Use Telegram commands ``/start``, ``/stop``,
``/status`` and ``/summary`` to control the bot remotely.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import pandas as pd

# Import modules from the root package.  Relative imports are avoided to
# simplify execution when the code resides outside of a package.  All
# dependencies reside in the same directory.
from config import Config
from scalping_strategy import EnhancedScalpingStrategy
from position_sizing import PositionSizing
from order_executor import OrderExecutor
from telegram_controller import TelegramController

logger = logging.getLogger(__name__)


class RealTimeTrader:
    """Core orchestrator for trading operations."""

    def __init__(self) -> None:
        """Initialise the trading engine and all subordinate components."""
        # Trading state
        self.is_trading: bool = False
        self.daily_pnl: float = 0.0
        self.trades: List[Dict[str, Any]] = []
        # Determine whether we are in live or simulated mode based on config.
        self.live_mode: bool = False

        # Initialise strategy with adaptive parameters.  Pass an explicit
        # ``min_score_threshold`` derived from the configuration to avoid
        # inadvertently tying it to the confidence threshold.
        self.strategy = EnhancedScalpingStrategy(
            base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
            base_target_points=Config.BASE_TARGET_POINTS,
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
            min_score_threshold=int(Config.MIN_SIGNAL_SCORE),
        )
        # Risk manager (position sizing and drawdown control)
        self.risk_manager = PositionSizing()
        # Order executor: decide between simulated and live based on config
        self.order_executor = self._init_order_executor()
        # Telegram controller with callbacks
        self.telegram_controller = TelegramController(
            status_callback=self.get_status,
            control_callback=self._handle_control,
            summary_callback=self.get_summary,
        )
        # Polling thread (controlled via start()/stop())
        self._polling_thread: Optional[threading.Thread] = None

    def _init_order_executor(self) -> OrderExecutor:
        """Instantiate the order executor in live or simulated mode.

        If ``Config.ENABLE_LIVE_TRADING`` is true and a valid KiteConnect
        instance can be created, the executor will operate in live mode
        using the broker API.  Otherwise a simulated executor is returned.
        """
        # If live trading is disabled, always use simulation
        if not Config.ENABLE_LIVE_TRADING:
            logger.info("Live trading disabled. Using simulated order executor.")
            self.live_mode = False
            return OrderExecutor()
        # Attempt to instantiate KiteConnect for live trading
        try:
            from kiteconnect import KiteConnect  # type: ignore

            api_key = Config.ZERODHA_API_KEY
            access_token = Config.KITE_ACCESS_TOKEN
            if not api_key or not access_token:
                raise ValueError("Missing Zerodha API credentials.")
            kite = KiteConnect(api_key=api_key)
            # Set the access token.  If this fails an exception will be raised.
            kite.set_access_token(access_token)
            self.live_mode = True
            logger.info("Live order executor initialised.")
            return OrderExecutor(kite=kite)
        except Exception as exc:
            logger.error(
                "Failed to initialise live trading. Falling back to simulation: %s", exc, exc_info=True
            )
            self.live_mode = False
            return OrderExecutor()

    # --- Public API ---
    def start(self) -> bool:
        """Enable trading and start the Telegram polling thread."""
        if self.is_trading:
            logger.info("Trader is already running.")
            return True
        self.is_trading = True
        # Start Telegram polling in background
        self._start_polling()
        # Send a Telegram notification that trading has begun
        try:
            self.telegram_controller.send_realtime_session_alert("START")
        except Exception:
            pass
        logger.info("Trading started.")
        return True

    def stop(self) -> bool:
        """Disable trading and stop polling."""
        if not self.is_trading:
            logger.info("Trader is not running.")
            return True
        self.is_trading = False
        self._stop_polling()
        # Send a Telegram notification that trading has stopped
        try:
            self.telegram_controller.send_realtime_session_alert("STOP")
        except Exception:
            pass
        logger.info("Trading stopped.")
        return True

    def process_bar(self, ohlc: pd.DataFrame) -> None:
        """Evaluate a new bar of OHLC data and execute trades if warranted.

        The caller must supply a DataFrame containing at least ``close``
        prices.  The method uses the last row of the DataFrame as the
        current price.  If trading is disabled or the DataFrame has
        insufficient rows, the function does nothing.
        """
        # Skip processing if the bot is inactive or data is insufficient
        if not self.is_trading:
            return
        if ohlc is None or len(ohlc) < 30:
            return
        try:
            # Enforce session time filter: only generate trades within the
            # configured start/end window.  If no timestamp is present on
            # the DataFrame index this check is skipped.
            try:
                ts = ohlc.index[-1]
                # Convert to string for comparison (HH:MM)
                current_time_str = ts.strftime("%H:%M")
                if (
                    Config.TIME_FILTER_START
                    and Config.TIME_FILTER_END
                    and (
                        current_time_str < Config.TIME_FILTER_START
                        or current_time_str > Config.TIME_FILTER_END
                    )
                ):
                    # Outside trading hours
                    return
            except Exception:
                # If index is not datetime or formatting fails ignore
                pass
            current_price = float(ohlc.iloc[-1]["close"])
            # Generate a potential signal from the strategy
            signal = self.strategy.generate_signal(ohlc, current_price)
            # Validate the signal based on confidence threshold
            if not signal:
                return
            confidence = float(signal.get("confidence", 0.0))
            if confidence < Config.CONFIDENCE_THRESHOLD:
                return
            # Calculate position size.  Market volatility can be passed through
            # the signal dictionary if provided by the strategy in the future.
            position = self.risk_manager.calculate_position_size(
                entry_price=signal.get("entry_price", current_price),
                stop_loss=signal.get("stop_loss", current_price),
                signal_confidence=confidence,
                market_volatility=signal.get("market_volatility", 0.0),
            )
            if not position or position.get("quantity", 0) <= 0:
                return
            # Send a Telegram alert before placing the order.  Use an
            # incrementing token based on the trade count for display.
            token = len(self.trades) + 1
            self.telegram_controller.send_signal_alert(token, signal, position)
            # Place the entry order.  In live mode this communicates with
            # KiteConnect; in simulation a UUID is returned.
            transaction_type = signal.get("signal") or signal.get("direction")
            order_id = self.order_executor.place_entry_order(
                symbol="NIFTY50",
                exchange="NFO",
                transaction_type=transaction_type,
                quantity=position["quantity"],
            )
            if not order_id:
                return
            # Set up corresponding GTT orders for stop loss and target
            self.order_executor.setup_gtt_orders(
                entry_order_id=order_id,
                entry_price=signal.get("entry_price", current_price),
                stop_loss_price=signal.get("stop_loss", current_price),
                target_price=signal.get("target", current_price),
                symbol="NIFTY50",
                exchange="NFO",
                quantity=position["quantity"],
                transaction_type=transaction_type,
            )
            # Record the trade for later summarisation.  No realised PnL yet.
            self.trades.append(
                {
                    "order_id": order_id,
                    "direction": transaction_type,
                    "quantity": position["quantity"],
                    "entry_price": signal.get("entry_price", current_price),
                    "stop_loss": signal.get("stop_loss", current_price),
                    "target": signal.get("target", current_price),
                    "confidence": confidence,
                }
            )
        except Exception as exc:
            logger.error("Error processing bar: %s", exc, exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the bot's current state."""
        status: Dict[str, Any] = {
            "is_trading": self.is_trading,
            "open_orders": len(self.order_executor.get_active_orders()),
            "trades_today": len(self.trades),
            "live_mode": self.live_mode,
        }
        # Merge in risk metrics such as equity and drawdown
        status.update(self.risk_manager.get_risk_status())
        return status

    def get_summary(self) -> str:
        """Produce a humanâ€‘readable summary of daily trades and P&L."""
        lines = [f"Total trades: {len(self.trades)}", f"PNL: {self.daily_pnl:.2f}"]
        for trade in self.trades:
            lines.append(
                f"{trade['direction']} {trade['quantity']} @ {trade['entry_price']:.2f} (SL {trade['stop_loss']:.2f}, TP {trade['target']:.2f})"
            )
        return "\n".join(lines)

    # --- Internal methods ---
    def _start_polling(self) -> None:
        if self._polling_thread and self._polling_thread.is_alive():
            return
        self.telegram_controller.send_startup_alert()
        self._polling_thread = threading.Thread(target=self.telegram_controller.start_polling, daemon=True)
        self._polling_thread.start()

    def _stop_polling(self) -> None:
        self.telegram_controller.stop_polling()
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=2)
        self._polling_thread = None

    def _handle_control(self, command: str) -> bool:
        """Handle start/stop commands from Telegram."""
        if command == "start":
            return self.start()
        if command == "stop":
            return self.stop()
        # Handle mode switching commands from the Telegram controller
        if command == "mode_live":
            return self._set_live_mode(True)
        if command == "mode_shadow":
            return self._set_live_mode(False)
        logger.warning("Unknown control command: %s", command)
        return False

    def _set_live_mode(self, enable: bool) -> bool:
        """Switch between live and simulated trading modes.

        When enabling live mode a new ``OrderExecutor`` is created with a
        ``KiteConnect`` instance.  Disabling live mode switches back to
        simulation.  The method returns ``True`` if the change took effect.
        """
        desired = bool(enable)
        if desired == self.live_mode:
            # No change required
            return True
        # Stop trading if currently running.  Mode cannot be changed midâ€‘trade
        if self.is_trading:
            logger.info("Cannot change mode while trading is active. Stop trading first.")
            return False
        # Reâ€‘initialise the order executor according to the desired mode
        if desired:
            # Attempt to create a live executor
            try:
                from kiteconnect import KiteConnect  # type: ignore

                api_key = Config.ZERODHA_API_KEY
                access_token = Config.KITE_ACCESS_TOKEN
                if not api_key or not access_token:
                    raise ValueError("Missing Zerodha API credentials.")
                kite = KiteConnect(api_key=api_key)
                kite.set_access_token(access_token)
                self.order_executor = OrderExecutor(kite=kite)
                self.live_mode = True
                logger.info("Switched to live trading mode.")
                return True
            except Exception as exc:
                logger.error(
                    "Failed to enable live mode. Remaining in simulation: %s", exc, exc_info=True
                )
                return False
        else:
            # Switch to simulated mode
            self.order_executor = OrderExecutor()
            self.live_mode = False
            logger.info("Switched to shadow/simulated trading mode.")
            return True