"""
Complete Real-Time Options Trading System
A comprehensive automated trading system for Indian options market
"""

import logging
import threading
import atexit
import signal
import sys
from typing import Any, Dict, List, Optional
import pandas as pd
from datetime import datetime
import time
import csv

# -------------------------
# Safe imports with fallbacks
# -------------------------
try:
    from src.config import Config
except ImportError:
    logging.warning("Could not import Config, using default configuration")
    class Config:  # minimal, keep names you already use
        ENABLE_LIVE_TRADING = False
        BASE_STOP_LOSS_POINTS = 50
        BASE_TARGET_POINTS = 100
        CONFIDENCE_THRESHOLD = 7.0
        MIN_SIGNAL_SCORE = 6
        SPOT_SYMBOL = "NSE:NIFTY 50"
        NIFTY_LOT_SIZE = 75
        DATA_LOOKBACK_MINUTES = 30
        STRIKE_RANGE = 4
        TIME_FILTER_START = "09:15"
        TIME_FILTER_END = "15:25"
        STRIKE_SELECTION_TYPE = "OTM"
        OPTIONS_STOP_LOSS_PCT = 20.0
        OPTIONS_TARGET_PCT = 50.0
        MAX_DAILY_OPTIONS_TRADES = 5
        MAX_POSITION_VALUE = 50000
        MIN_SIGNAL_CONFIDENCE = 6.0
        TRADE_SYMBOL = "NIFTY50"
        TRADE_EXCHANGE = "NFO"
        ZERODHA_API_KEY = ""
        KITE_ACCESS_TOKEN = ""

try:
    from src.strategies.scalping_strategy import EnhancedScalpingStrategy
except ImportError:
    logging.warning("Could not import EnhancedScalpingStrategy, using a tiny stub")
    class EnhancedScalpingStrategy:
        def __init__(self, **kwargs):
            pass
        def generate_signal(self, ohlc: pd.DataFrame, current_price: float) -> Optional[Dict[str, Any]]:
            # Return None = no signal
            return None

try:
    from src.risk.position_sizing import PositionSizing
except ImportError:
    logging.warning("Could not import PositionSizing, using a minimal fallback")
    class PositionSizing:
        def __init__(self, *args, **kwargs): ...
        def calculate_position_size(
            self,
            entry_price: float,
            stop_loss: float,
            signal_confidence: float,
            market_volatility: float,
            lot_size: int = 75
        ) -> Optional[Dict[str, Any]]:
            if entry_price <= 0 or stop_loss <= 0:
                return None
            if signal_confidence < getattr(Config, 'MIN_SIGNAL_CONFIDENCE', 6.0):
                return None
            # Keep it simple: 1 lot default
            return {"quantity": 1}

try:
    from src.execution.order_executor import OrderExecutor
except ImportError:
    logging.warning("Could not import OrderExecutor, using a no-op fallback")
    class OrderExecutor:
        def __init__(self, *args, **kwargs): ...
        def place_entry_order(self, symbol: str, exchange: str, transaction_type: str, quantity: int) -> Optional[str]:
            # Simulate an order id
            return f"SIM-{int(time.time())}"
        def setup_gtt_orders(
            self,
            entry_order_id: str,
            entry_price: float,
            stop_loss_price: float,
            target_price: float,
            symbol: str,
            exchange: str,
            quantity: int,
            transaction_type: str,
        ) -> None:
            return
        def get_active_orders(self) -> List[Dict[str, Any]]:
            return []

try:
    from src.notifications.telegram_controller import TelegramController
except ImportError:
    logging.warning("Could not import TelegramController, using a safe stub")
    class TelegramController:
        def __init__(self, *args, **kwargs): ...
        def send_signal_alert(self, token: int, signal: Dict[str, Any], position: Dict[str, Any]) -> None:
            return
        def send_message(self, text: str, parse_mode: Optional[str] = None) -> None:
            return

logger = logging.getLogger(__name__)


class RealTimeTrader:
    """
    Core trading engine for real-time scalping strategy.
    Manages trading state, signal processing, order execution, and Telegram control.
    """

    def __init__(self) -> None:
        self.is_trading: bool = False
        self.daily_pnl: float = 0.0
        self.trades: List[Dict[str, Any]] = []
        self.live_mode: bool = getattr(Config, "ENABLE_LIVE_TRADING", False)

        # External components
        self.strategy = EnhancedScalpingStrategy(
            base_stop_loss_points=getattr(Config, "BASE_STOP_LOSS_POINTS", 50),
            base_target_points=getattr(Config, "BASE_TARGET_POINTS", 100),
            confidence_threshold=getattr(Config, "CONFIDENCE_THRESHOLD", 7.0),
        )
        self.risk_manager = PositionSizing()
        self.order_executor = OrderExecutor()

        # ---- TelegramController now requires callbacks; wire them here ----
        self.telegram_controller = TelegramController(
            status_callback=self.get_status,
            control_callback=self._control_command,
            summary_callback=self.get_summary
        )

        # Instruments cache timestamp used in get_status()
        self._instruments_cache_timestamp: float = time.time()

        # Graceful shutdown hooks (optional)
        atexit.register(self._on_exit)
        try:
            signal.signal(signal.SIGINT, self._on_signal)
            signal.signal(signal.SIGTERM, self._on_signal)
        except Exception:
            pass  # Some environments don't allow setting signals

        self._lock = threading.RLock()

    # ---------- Control callback for Telegram ----------

    def _control_command(self, command: str) -> str:
        """
        Handle Telegram control actions. Expected commands:
        'start', 'stop', 'status', 'summary'
        """
        cmd = (command or "").strip().lower()
        if cmd == "start":
            self.start_trading()
            return "🟢 Trading ENABLED."
        if cmd == "stop":
            self.stop_trading()
            return "🛑 Trading DISABLED."
        if cmd == "status":
            return str(self.get_status())
        if cmd == "summary":
            return self.get_summary()
        return f"Unknown command: {command}. Try: start | stop | status | summary"

    # ---------- Internal helpers ----------

    def _passes_time_filter(self, ts: datetime) -> bool:
        """Enforce intraday time window (e.g., 09:15–15:25)."""
        try:
            start_s = getattr(Config, "TIME_FILTER_START", "09:15")
            end_s = getattr(Config, "TIME_FILTER_END", "15:25")
            start = datetime.strptime(start_s, "%H:%M").time()
            end = datetime.strptime(end_s, "%H:%M").time()
            t = ts.time()
            wd = ts.weekday()  # 0=Mon ... 6=Sun
            return (0 <= wd <= 4) and (start <= t <= end)
        except Exception as e:
            logger.debug(f"Time filter error (allowing by default): {e}")
            return True

    def _calculate_options_stop_loss(self, current_price: float, strike_info: Dict[str, Any]) -> float:
        """Percentage-based SL for options; adjusted for moneyness."""
        try:
            stop_loss_pct = float(getattr(Config, 'OPTIONS_STOP_LOSS_PCT', 20.0))
            if strike_info.get('is_otm', False):
                stop_loss_pct *= 1.2
            elif strike_info.get('is_itm', False):
                stop_loss_pct *= 0.8
            return max(0.05, current_price * (1 - stop_loss_pct / 100))
        except Exception as e:
            logger.error(f"Error calculating options stop loss: {e}")
            return max(0.05, current_price * 0.8)

    def _calculate_options_target(self, current_price: float, strike_info: Dict[str, Any]) -> float:
        """Percentage-based target for options; adjusted for ATM/OTM."""
        try:
            target_pct = float(getattr(Config, 'OPTIONS_TARGET_PCT', 50.0))
            if strike_info.get('is_atm', False):
                target_pct *= 1.2
            elif strike_info.get('is_otm', False):
                target_pct *= 1.5
            return current_price * (1 + target_pct / 100)
        except Exception as e:
            logger.error(f"Error calculating options target: {e}")
            return current_price * 1.5

    def _calculate_options_position_size(
        self,
        signal: Dict[str, Any],
        strike_info: Dict[str, Any],
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Position sizing with simple options-specific adjustments."""
        try:
            base_position = self.risk_manager.calculate_position_size(
                entry_price=float(signal.get("entry_price", current_price)),
                stop_loss=float(signal.get("stop_loss", current_price)),
                signal_confidence=float(signal.get("confidence", 0.0)),
                market_volatility=float(signal.get("market_volatility", 0.0)),
                lot_size=int(getattr(Config, 'NIFTY_LOT_SIZE', 75)),
            )
            if not base_position:
                return None

            quantity = int(base_position.get('quantity', 0))
            if quantity <= 0:
                return None

            if strike_info.get('is_otm', False):
                quantity = max(1, int(quantity * 0.7))
            elif strike_info.get('is_atm', False):
                quantity = max(1, int(quantity * 1.1))

            base_position['quantity'] = int(quantity)
            return base_position
        except Exception as e:
            logger.error(f"Error calculating options position size: {e}")
            return None

    def _passes_risk_checks(self, signal: Dict[str, Any], position: Dict[str, Any]) -> bool:
        """Basic sanity checks before placing orders."""
        try:
            if int(position.get('quantity', 0)) <= 0:
                return False

            max_daily_trades = int(getattr(Config, 'MAX_DAILY_OPTIONS_TRADES', 5))
            if len(self.trades) >= max_daily_trades:
                logger.warning(f"Daily trade limit reached: {len(self.trades)}/{max_daily_trades}")
                return False

            max_position_value = float(getattr(Config, 'MAX_POSITION_VALUE', 50000))
            lot = int(getattr(Config, 'NIFTY_LOT_SIZE', 75))
            entry_price = float(signal.get("entry_price", 0.0))
            position_value = position['quantity'] * entry_price * lot
            if position_value > max_position_value:
                logger.warning(f"Position value {position_value:.2f} exceeds limit {max_position_value:.2f}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error in risk checks: {e}")
            return False

    def _execute_options_trade(
        self,
        symbol: str,
        signal: Dict[str, Any],
        position: Dict[str, Any],
        strike_info: Dict[str, Any]
    ) -> bool:
        """Execute options trade → alert, place entry, setup GTT, record."""
        try:
            token = len(self.trades) + 1
            try:
                self.telegram_controller.send_signal_alert(token, signal, position)
            except Exception as e:
                logger.warning(f"Failed to send signal alert: {e}")

            order_transaction_type = signal.get("signal") or signal.get("direction") or "BUY"
            qty = int(position["quantity"])
            logger.info(f"Placing {order_transaction_type} order for {symbol}, Qty: {qty}")

            order_id = self.order_executor.place_entry_order(
                symbol=symbol,
                exchange="NFO",
                transaction_type=order_transaction_type,
                quantity=qty,
            )
            if not order_id:
                logger.error(f"Failed to place entry order for {symbol}")
                return False

            try:
                self.order_executor.setup_gtt_orders(
                    entry_order_id=order_id,
                    entry_price=float(signal.get("entry_price", 0)),
                    stop_loss_price=float(signal.get("stop_loss", 0)),
                    target_price=float(signal.get("target", 0)),
                    symbol=symbol,
                    exchange="NFO",
                    quantity=qty,
                    transaction_type=order_transaction_type,
                )
            except Exception as e:
                logger.warning(f"Failed to setup GTT orders: {e}")

            trade_record = {
                "order_id": str(order_id),
                "symbol": symbol,
                "direction": order_transaction_type,
                "quantity": qty,
                "entry_price": float(signal.get("entry_price", 0)),
                "stop_loss": float(signal.get("stop_loss", 0)),
                "target": float(signal.get("target", 0)),
                "confidence": float(signal.get("confidence", 0.0)),
                "strike_info": dict(strike_info or {}),
                "timestamp": datetime.now(),
                "strategy_type": signal.get("strategy_type", "unknown")
            }
            self.trades.append(trade_record)
            logger.info(f"✅ Options trade recorded: {order_transaction_type} {qty}x {symbol} @ {trade_record['entry_price']:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error executing options trade for {symbol}: {e}", exc_info=True)
            return False

    # ---------- Public methods (backward compatible) ----------

    def process_bar(self, ohlc: pd.DataFrame) -> None:
        """
        Original bar processing method for non-options strategies.
        Keeps existing structure but hardened against None/shape/index errors.
        """
        logger.debug(f"process_bar called. Trading active: {self.is_trading}, OHLC points: {len(ohlc) if getattr(ohlc, 'shape', [0])[0] else 'None'}")
        if not self.is_trading:
            return
        if ohlc is None or len(ohlc) < 30:
            return
        try:
            if not isinstance(ohlc.index, pd.DatetimeIndex):
                logger.error("OHLC data must have DatetimeIndex.")
                return

            ts = ohlc.index[-1]
            if not self._passes_time_filter(ts):
                return

            current_price = float(ohlc.iloc[-1]["close"])
            if not self.strategy:
                logger.warning("No strategy available")
                return

            signal = self.strategy.generate_signal(ohlc, current_price)
            if not signal:
                return

            signal_confidence = float(signal.get("confidence", 0.0))
            if signal_confidence < float(getattr(Config, 'CONFIDENCE_THRESHOLD', 7.0)):
                return

            position = self.risk_manager.calculate_position_size(
                entry_price=float(signal.get("entry_price", current_price)),
                stop_loss=float(signal.get("stop_loss", current_price)),
                signal_confidence=signal_confidence,
                market_volatility=float(signal.get("market_volatility", 0.0)),
            )
            if not position or int(position.get("quantity", 0)) <= 0:
                return

            token = len(self.trades) + 1
            try:
                self.telegram_controller.send_signal_alert(token, signal, position)
            except Exception as e:
                logger.warning(f"Failed to send signal alert: {e}")

            transaction_type = signal.get("signal") or signal.get("direction")
            if not transaction_type:
                logger.warning("Missing signal direction.")
                return

            symbol = getattr(Config, "TRADE_SYMBOL", "NIFTY50")
            exchange = getattr(Config, "TRADE_EXCHANGE", "NFO")

            order_id = self.order_executor.place_entry_order(
                symbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=int(position["quantity"]),
            )
            if not order_id:
                return

            try:
                self.order_executor.setup_gtt_orders(
                    entry_order_id=order_id,
                    entry_price=float(signal.get("entry_price", current_price)),
                    stop_loss_price=float(signal.get("stop_loss", current_price)),
                    target_price=float(signal.get("target", current_price)),
                    symbol=symbol,
                    exchange=exchange,
                    quantity=int(position["quantity"]),
                    transaction_type=transaction_type,
                )
            except Exception as e:
                logger.warning(f"Failed to setup GTT orders: {e}")

            self.trades.append({
                "order_id": str(order_id),
                "symbol": symbol,
                "direction": transaction_type,
                "quantity": int(position["quantity"]),
                "entry_price": float(signal.get("entry_price", current_price)),
                "stop_loss": float(signal.get("stop_loss", current_price)),
                "target": float(signal.get("target", current_price)),
                "confidence": float(signal.get("confidence", 0.0)),
                "timestamp": datetime.now(),
                "strategy_type": signal.get("strategy_type", "unknown"),
            })
            logger.info(f"✅ Trade recorded: {transaction_type} {int(position['quantity'])} @ {signal.get('entry_price', current_price)}")
        except Exception as exc:
            logger.error("Error processing bar: %s", exc, exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """Enhanced status reporting; robust if executors are stubs."""
        try:
            active_orders = 0
            try:
                active_orders = len(self.order_executor.get_active_orders())
            except Exception as e:
                logger.debug(f"Could not get active orders: {e}")

            cache_age = max(0.0, (time.time() - float(getattr(self, "_instruments_cache_timestamp", time.time()))) / 60.0)

            status: Dict[str, Any] = {
                "is_trading": self.is_trading,
                "open_orders": active_orders,
                "trades_today": len(self.trades),
                "live_mode": self.live_mode,
                "cache_age_minutes": round(cache_age, 2),
                "total_pnl": float(self.daily_pnl),
                "last_update": datetime.now().strftime("%H:%M:%S"),
            }
            try:
                if hasattr(self.risk_manager, "get_risk_status"):
                    status.update(self.risk_manager.get_risk_status())
            except Exception as e:
                logger.debug(f"Could not get risk status: {e}")

            return status
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}

    def get_summary(self) -> str:
        """Daily summary; safe if fields missing."""
        try:
            lines = [
                "📊 <b>Daily Options Trading Summary</b>",
                f"🔁 <b>Total trades:</b> {len(self.trades)}",
                f"💰 <b>PnL:</b> ₹{self.daily_pnl:.2f}",
                f"📈 <b>Mode:</b> {'🟢 LIVE' if self.live_mode else '🛡️ SHADOW'}",
                "────────────────────"
            ]

            ce_trades = [t for t in self.trades if t.get('strike_info', {}).get('type') == 'CE']
            pe_trades = [t for t in self.trades if t.get('strike_info', {}).get('type') == 'PE']

            if ce_trades:
                lines.append(f"📈 <b>CE Trades:</b> {len(ce_trades)}")
            if pe_trades:
                lines.append(f"📉 <b>PE Trades:</b> {len(pe_trades)}")

            lines.append("────────────────────")

            recent_trades = self.trades[-5:] if len(self.trades) > 5 else self.trades
            for trade in recent_trades:
                symbol = trade.get('symbol', 'N/A')
                strike_info = trade.get('strike_info', {})
                opt_type = strike_info.get('type', 'UNK')
                strike = strike_info.get('strike', 'N/A')
                entry_price = float(trade.get('entry_price', 0.0))
                sl = float(trade.get('stop_loss', 0.0))
                tp = float(trade.get('target', 0.0))
                qty = int(trade.get('quantity', 0))
                direction = trade.get('direction', '—')
                lines.append(
                    f"{opt_type} {strike} {direction} {qty} @ ₹{entry_price:.2f} (SL ₹{sl:.2f}, TP ₹{tp:.2f}) | {symbol}"
                )

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"📊 Summary Error: {str(e)}"

    def get_trading_statistics(self) -> Dict[str, Any]:
        """Basic stats (PnL breakdown would need fills/MTM; left as zero-safe)."""
        try:
            if not self.trades:
                return {"message": "No trades executed today"}

            total_trades = len(self.trades)
            ce_trades = [t for t in self.trades if t.get('strike_info', {}).get('type') == 'CE']
            pe_trades = [t for t in self.trades if t.get('strike_info', {}).get('type') == 'PE']

            winning_trades = 0  # Needs real PnL
            losing_trades = 0   # Needs real PnL

            strategy_stats: Dict[str, int] = {}
            for t in self.trades:
                strategy = t.get('strategy_type', 'unknown')
                strategy_stats[strategy] = strategy_stats.get(strategy, 0) + 1

            avg_conf = sum(float(t.get('confidence', 0)) for t in self.trades) / total_trades if total_trades else 0.0

            return {
                "total_trades": total_trades,
                "ce_trades": len(ce_trades),
                "pe_trades": len(pe_trades),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0.0,
                "strategy_breakdown": strategy_stats,
                "daily_pnl": float(self.daily_pnl),
                "average_confidence": round(avg_conf, 2),
                "last_trade_time": self.trades[-1].get('timestamp', 'N/A') if self.trades else 'N/A'
            }
        except Exception as e:
            logger.error(f"Error getting trading statistics: {e}")
            return {"error": str(e)}

    def export_trades_to_csv(self, filename: Optional[str] = None) -> str:
        """Export today's trades to CSV."""
        try:
            if not filename:
                filename = f"trades_{datetime.now().strftime('%Y%m%d')}.csv"

            fieldnames = [
                'timestamp', 'order_id', 'symbol', 'direction', 'quantity',
                'entry_price', 'stop_loss', 'target', 'confidence',
                'strategy_type', 'option_type', 'strike', 'expiry'
            ]
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for trade in self.trades:
                    strike_info = trade.get('strike_info', {})
                    writer.writerow({
                        'timestamp': trade.get('timestamp', ''),
                        'order_id': trade.get('order_id', ''),
                        'symbol': trade.get('symbol', ''),
                        'direction': trade.get('direction', ''),
                        'quantity': trade.get('quantity', 0),
                        'entry_price': trade.get('entry_price', 0.0),
                        'stop_loss': trade.get('stop_loss', 0.0),
                        'target': trade.get('target', 0.0),
                        'confidence': trade.get('confidence', 0.0),
                        'strategy_type': trade.get('strategy_type', ''),
                        'option_type': strike_info.get('type', ''),
                        'strike': strike_info.get('strike', ''),
                        'expiry': strike_info.get('expiry', ''),
                    })
            return filename
        except Exception as e:
            logger.error(f"Error exporting trades: {e}")
            return ""

    # ---------- Lifecycle ----------

    def start_trading(self) -> None:
        with self._lock:
            self.is_trading = True
            logger.info("Trading ENABLED")

    def stop_trading(self) -> None:
        with self._lock:
            self.is_trading = False
            logger.info("Trading DISABLED")

    def _on_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, stopping trading.")
        self.stop_trading()
        sys.exit(0)

    def _on_exit(self):
        try:
            logger.info("Shutting down RealTimeTrader.")
        except Exception:
            pass
