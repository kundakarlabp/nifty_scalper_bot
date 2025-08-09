"""
Complete Real-Time Options Trading System
A comprehensive automated trading system for Indian options market
"""

import logging
import threading
import atexit
import signal
import sys
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import schedule
from datetime import datetime, timedelta
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

# Safe imports with fallbacks
try:
    from src.config import Config
except ImportError:
    logging.warning("Could not import Config, using default configuration")
    class Config:
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
        ZERODHA_API_KEY = ""
        KITE_ACCESS_TOKEN = ""

try:
    from src.strategies.scalping_strategy import EnhancedScalpingStrategy
except ImportError:
    logging.warning("Could not import EnhancedScalpingStrategy, using dummy class")
    class EnhancedScalpingStrategy:
        def __init__(self, **kwargs):
            pass
        def generate_signal(self, ohlc, current_price):
            return     def _calculate_options_stop_loss(self, current_price: float, strike_info: Dict[str, Any]) -> float:
        """Calculate options-specific stop loss"""
        try:
            # Options stop loss should be percentage-based due to high volatility
            stop_loss_pct = getattr(Config, 'OPTIONS_STOP_LOSS_PCT', 20.0)  # 20% default
            
            # Adjust based on option type and moneyness
            if strike_info.get('is_otm', False):
                stop_loss_pct *= 1.2  # OTM options need wider stops
            elif strike_info.get('is_itm', False):
                stop_loss_pct *= 0.8  # ITM options can have tighter stops
                
            return current_price * (1 - stop_loss_pct / 100)
            
        except Exception as e:
            logger.error(f"Error calculating options stop loss: {e}")
            return current_price * 0.8  # 20% default

    def _calculate_options_target(self, current_price: float, strike_info: Dict[str, Any]) -> float:
        """Calculate options-specific target"""
        try:
            # Options targets should account for time decay and volatility
            target_pct = getattr(Config, 'OPTIONS_TARGET_PCT', 50.0)  # 50% default
            
            # Adjust based on option characteristics
            if strike_info.get('is_atm', False):
                target_pct *= 1.2  # ATM options have good profit potential
            elif strike_info.get('is_otm', False):
                target_pct *= 1.5  # OTM options can give higher returns
                
            return current_price * (1 + target_pct / 100)
            
        except Exception as e:
            logger.error(f"Error calculating options target: {e}")
            return current_price * 1.5  # 50% default

    def _calculate_options_position_size(self, signal: Dict[str, Any], strike_info: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        """Calculate position size specific to options trading"""
        try:
            # Use the existing risk manager but with options-specific adjustments
            base_position = self.risk_manager.calculate_position_size(
                entry_price=signal.get("entry_price", current_price),
                stop_loss=signal.get("stop_loss", current_price),
                signal_confidence=signal.get("confidence", 0.0),
                market_volatility=signal.get("market_volatility", 0.0),
                lot_size=getattr(Config, 'NIFTY_LOT_SIZE', 75)
            )
            
            if not base_position:
                return None
                
            # Options-specific adjustments
            quantity = base_position.get('quantity', 0)
            
            # Reduce quantity for OTM options (higher risk)
            if strike_info.get('is_otm', False):
                quantity = max(1, int(quantity * 0.7))
            
            # Increase quantity for ATM options (balanced risk/reward)
            elif strike_info.get('is_atm', False):
                quantity = int(quantity * 1.1)
                
            base_position['quantity'] = quantity
            return base_position
            
        except Exception as e:
            logger.error(f"Error calculating options position size: {e}")
            return None

    def _passes_risk_checks(self, signal: Dict[str, Any], position: Dict[str, Any], strike_info: Dict[str, Any]) -> bool:
        """Enhanced risk management checks for options"""
        try:
            # Basic quantity check
            if position.get('quantity', 0) <= 0:
                return False
                
            # Maximum daily trades check
            max_daily_trades = getattr(Config, 'MAX_DAILY_OPTIONS_TRADES', 5)
            if len(self.trades) >= max_daily_trades:
                logger.warning(f"Daily trade limit reached: {len(self.trades)}/{max_daily_trades}")
                return False
                
            # Maximum position size check
            max_position_value = getattr(Config, 'MAX_POSITION_VALUE', 50000)
            position_value = position['quantity'] * signal.get('entry_price', 0) * getattr(Config, 'NIFTY_LOT_SIZE', 75)
            
            if position_value > max_position_value:
                logger.warning(f"Position value {position_value} exceeds limit {max_position_value}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in risk checks: {e}")
            return False

    def _execute_options_trade(self, symbol: str, signal: Dict[str, Any], position: Dict[str, Any], strike_info: Dict[str, Any]) -> bool:
        """Execute the complete options trade workflow"""
        try:
            # Send telegram alert
            token = len(self.trades) + 1
            try:
                self.telegram_controller.send_signal_alert(token, signal, position)
            except Exception as e:
                logger.warning(f"Failed to send signal alert: {e}")

            # Place entry order
            order_transaction_type = signal.get("signal", "BUY")
            
            logger.info(f"Placing {order_transaction_type} order for {symbol}, Qty: {position['quantity']}")
            
            order_id = self.order_executor.place_entry_order(
                symbol=symbol,
                exchange="NFO",
                transaction_type=order_transaction_type,
                quantity=position["quantity"],
            )
            
            if not order_id:
                logger.error(f"Failed to place entry order for {symbol}")
                return False

            # Setup GTT orders
            logger.debug(f"Setting up GTT orders for {symbol}")
            try:
                self.order_executor.setup_gtt_orders(
                    entry_order_id=order_id,
                    entry_price=signal.get("entry_price", 0),
                    stop_loss_price=signal.get("stop_loss", 0),
                    target_price=signal.get("target", 0),
                    symbol=symbol,
                    exchange="NFO",
                    quantity=position["quantity"],
                    transaction_type=order_transaction_type,
                )
            except Exception as e:
                logger.warning(f"Failed to setup GTT orders: {e}")

            # Record trade
            trade_record = {
                "order_id": order_id,
                "symbol": symbol,
                "direction": order_transaction_type,
                "quantity": position["quantity"],
                "entry_price": signal.get("entry_price", 0),
                "stop_loss": signal.get("stop_loss", 0),
                "target": signal.get("target", 0),
                "confidence": signal.get("confidence", 0.0),
                "strike_info": strike_info,
                "timestamp": datetime.now(),
                "strategy_type": signal.get("strategy_type", "unknown")
            }
            
            self.trades.append(trade_record)
            
            logger.info(f"✅ Options trade recorded: {order_transaction_type} {position['quantity']}x {symbol} @ {signal.get('entry_price', 0)}")
            return True

        except Exception as e:
            logger.error(f"Error executing options trade for {symbol}: {e}", exc_info=True)
            return False

    # Keep the original process_bar method for backward compatibility
    def process_bar(self, ohlc: pd.DataFrame) -> None:
        """Original bar processing method for non-options strategies"""
        logger.debug(f"process_bar called. Trading active: {self.is_trading}, OHLC data points: {len(ohlc) if ohlc is not None else 'None'}")
        if not self.is_trading:
            logger.debug("process_bar: Trading not active, returning.")
            return
        if ohlc is None or len(ohlc) < 30:
            logger.debug("Insufficient data to process bar (less than 30 points).")
            return
        try:
            if not isinstance(ohlc.index, pd.DatetimeIndex):
                logger.error("OHLC data must have DatetimeIndex.")
                return
            ts = ohlc.index[-1]
            
            if not self._passes_time_filter(ts):
                logger.debug("Time filter rejected bar")
                return
                
            current_price = float(ohlc.iloc[-1]["close"])
            logger.debug(f"Current bar timestamp: {ts}, price: {current_price}")
            
            if not self.strategy:
                logger.warning("No strategy available")
                return
                
            signal = self.strategy.generate_signal(ohlc, current_price)
            logger.debug(f"Strategy returned signal: {signal}")
            if not signal:
                logger.debug("No signal generated by strategy.")
                return
                
            signal_confidence = float(signal.get("confidence", 0.0))
            logger.debug(f"Signal confidence: {signal_confidence}, Threshold: {getattr(Config, 'CONFIDENCE_THRESHOLD', 7.0)}")
            if signal_confidence < getattr(Config, 'CONFIDENCE_THRESHOLD', 7.0):
                logger.debug("Signal confidence below threshold, discarding.")
                return
                
            position = self.risk_manager.calculate_position_size(
                entry_price=signal.get("entry_price", current_price),
                stop_loss=signal.get("stop_loss", current_price),
                signal_confidence=signal.get("confidence", 0.0),
                market_volatility=signal.get("market_volatility", 0.0),
            )
            logger.debug(f"Position sizing returned: {position}")
            if not position or position.get("quantity", 0) <= 0:
                logger.debug("Position sizing failed or quantity is zero/negative.")
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
            logger.debug(f"Attempting to place entry order. Symbol: {symbol}, Exchange: {exchange}, Type: {transaction_type}, Qty: {position['quantity']}")
            
            order_id = self.order_executor.place_entry_order(
                symbol=symbol,
                exchange=exchange,
                transaction_type=transaction_type,
                quantity=position["quantity"],
            )
            if not order_id:
                logger.warning("Failed to place entry order.")
                return
                
            logger.debug("Attempting to setup GTT orders...")
            try:
                self.order_executor.setup_gtt_orders(
                    entry_order_id=order_id,
                    entry_price=signal.get("entry_price", current_price),
                    stop_loss_price=signal.get("stop_loss", current_price),
                    target_price=signal.get("target", current_price),
                    symbol=symbol,
                    exchange=exchange,
                    quantity=position["quantity"],
                    transaction_type=transaction_type,
                )
            except Exception as e:
                logger.warning(f"Failed to setup GTT orders: {e}")
            
            self.trades.append({
                "order_id": order_id,
                "direction": transaction_type,
                "quantity": position["quantity"],
                "entry_price": signal.get("entry_price", current_price),
                "stop_loss": signal.get("stop_loss", current_price),
                "target": signal.get("target", current_price),
                "confidence": signal.get("confidence", 0.0),
            })
            logger.info(f"✅ Trade recorded: {transaction_type} {position['quantity']} @ {signal.get('entry_price', current_price)}")
        except Exception as exc:
            logger.error("Error processing bar: %s", exc, exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """Enhanced status reporting"""
        try:
            active_orders = 0
            try:
                active_orders = len(self.order_executor.get_active_orders())
            except Exception as e:
                logger.debug(f"Could not get active orders: {e}")
                
            status: Dict[str, Any] = {
                "is_trading": self.is_trading,
                "open_orders": active_orders,
                "trades_today": len(self.trades),
                "live_mode": self.live_mode,
                "cache_age_minutes": (time.time() - self._instruments_cache_timestamp) / 60,
                "total_pnl": self.daily_pnl,
                "last_update": datetime.now().strftime("%H:%M:%S")
            }
            
            # Add risk manager status if available
            try:
                status.update(self.risk_manager.get_risk_status())
            except Exception as e:
                logger.debug(f"Could not get risk status: {e}")
                
            return status
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}

    def get_summary(self) -> str:
        """Enhanced daily summary with options-specific information"""
        try:
            lines = [
                f"📊 <b>Daily Options Trading Summary</b>",
                f"🔁 <b>Total trades:</b> {len(self.trades)}",
                f"💰 <b>PnL:</b> ₹{self.daily_pnl:.2f}",
                f"📈 <b>Mode:</b> {'🟢 LIVE' if self.live_mode else '🛡️ SHADOW'}",
                "────────────────────"
            ]
            
            # Group trades by option type
            ce_trades = [t for t in self.trades if t.get('strike_info', {}).get('type') == 'CE']
            pe_trades = [t for t in self.trades if t.get('strike_info', {}).get('type') == 'PE']
            
            if ce_trades:
                lines.append(f"📈 <b>CE Trades:</b> {len(ce_trades)}")
            if pe_trades:
                lines.append(f"📉 <b>PE Trades:</b> {len(pe_trades)}")
                
            lines.append("────────────────────")
            
            # Show recent trades
            recent_trades = self.trades[-5:] if len(self.trades) > 5 else self.trades
            for trade in recent_trades:
                symbol = trade.get('symbol', 'N/A')
                strike_info = trade.get('strike_info', {})
                opt_type = strike_info.get('type', 'UNK')
                strike = strike_info.get('strike', 'N/A')
                
                lines.append(
                    f"{opt_type} {strike} {trade['direction']} {trade['quantity']} @ ₹{trade['entry_price']:.2f} "
                    f"(SL ₹{trade['stop_loss']:.2f}, TP ₹{trade['target']:.2f})"
                )
            
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"📊 Summary Error: {str(e)}"

    # Additional Utility Methods for Enhanced Functionality
    def get_trading_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trading statistics"""
        try:
            if not self.trades:
                return {"message": "No trades executed today"}
            
            # Basic stats
            total_trades = len(self.trades)
            ce_trades = [t for t in self.trades if t.get('strike_info', {}).get('type') == 'CE']
            pe_trades = [t for t in self.trades if t.get('strike_info', {}).get('type') == 'PE']
            
            # Calculate success metrics (placeholder - would need actual P&L data)
            winning_trades = 0  # Would need to check actual positions
            losing_trades = 0   # Would need to check actual positions
            
            # Strategy breakdown
            strategy_stats = {}
            for trade in self.trades:
                strategy = trade.get('strategy_type', 'unknown')
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = 0
                strategy_stats[strategy] += 1
            
            stats = {
                "total_trades": total_trades,
                "ce_trades": len(ce_trades),
                "pe_trades": len(pe_trades),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                "strategy_breakdown": strategy_stats,
                "daily_pnl": self.daily_pnl,
                "average_confidence": sum(t.get('confidence', 0) for t in self.trades) / total_trades if total_trades > 0 else 0,
                "last_trade_time": self.trades[-1].get('timestamp', 'N/A') if self.trades else 'N/A'
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting trading statistics: {e}")
            return {"error": str(e)}

    def export_trades_to_csv(self, filename: Optional[str] = None) -> str:
        """Export trades to CSV file"""
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
                    row = {
                        'timestamp': trade.get('timestamp', ''),
                        'order_id': trade.get('order_id', ''),
                        'symbol': trade.get('symbol', ''),
                        'direction': trade.get('direction', ''),
                        'quantity': trade.get('quantity', 0),
                        'entry_price': trade.get('entry_price', 0),
                        'stop_loss': trade.get('stop_loss', 0),
                        'target': trade.get('target', 0),
                        'confidence': trade.get('confidence', 0),
                        'strategy_type': trade.get('strategy_type', 'unknown'),
                        'option_type': strike_info.get('type', ''),
                        'strike': strike_info.get('strike', ''),
                        'expiry': strike_info.get('expiry', '')
                    }
                    writer.writerow(row)
            
            logger.info(f"Trades exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting trades: {e}")
            return ""

    def reset_daily_data(self) -> None:
        """Reset daily trading data (typically called at start of new trading day)"""
        try:
            logger.info("Resetting daily trading data...")
            
            # Export current day's trades before reset
            if self.trades:
                export_filename = self.export_trades_to_csv()
                logger.info(f"Previous day trades exported to {export_filename}")
            
            # Reset counters
            self.trades.clear()
            self.daily_pnl = 0.0
            
            # Clear caches
            self._atm_cache.clear()
            
            # Force refresh instruments cache
            self._refresh_instruments_cache(force=True)
            
            logger.info("✅ Daily data reset completed")
            
            # Send notification
            self._safe_send_message("🔄 Daily data reset completed. Ready for new trading session.")
                
        except Exception as e:
            logger.error(f"Error resetting daily data: {e}")

    def emergency_stop_all(self) -> bool:
        """Emergency stop all trading activities and cancel pending orders"""
        try:
            logger.warning("🚨 EMERGENCY STOP INITIATED")
            
            # Stop trading
            self.is_trading = False
            
            # Cancel all pending orders
            cancelled_orders = 0
            try:
                cancelled_orders = self.order_executor.cancel_all_orders()
                logger.info(f"Cancelled {cancelled_orders} pending orders")
            except Exception as e:
                logger.error(f"Error cancelling orders: {e}")
            
            # Send emergency notification
            try:
                emergency_msg = f"""
🚨 **EMERGENCY STOP ACTIVATED**
⏰ Time: {datetime.now().strftime('%H:%M:%S')}
🛑 Trading: STOPPED
📋 Orders Cancelled: {cancelled_orders}
💼 Trades Today: {len(self.trades)}
💰 Current P&L: ₹{self.daily_pnl:.2f}
                """
                self._safe_send_message(emergency_msg, parse_mode="Markdown")
            except Exception as e:
                logger.error(f"Failed to send emergency notification: {e}")
            
            logger.warning("🚨 Emergency stop completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
            return False

    def get_position_summary(self) -> Dict[str, Any]:
        """Get current position summary"""
        try:
            try:
                positions = self.order_executor.get_positions()
            except Exception as e:
                logger.debug(f"Could not get positions: {e}")
                return {"message": "Position data not available"}
            
            summary = {
                "total_positions": len(positions) if positions else 0,
                "net_quantity": 0,
                "unrealized_pnl": 0,
                "realized_pnl": self.daily_pnl,
                "positions": []
            }
            
            if positions:
                for pos in positions:
                    if pos.get('quantity', 0) != 0:  # Only active positions
                        pos_data = {
                            "symbol": pos.get('tradingsymbol', 'N/A'),
                            "quantity": pos.get('quantity', 0),
                            "average_price": pos.get('average_price', 0),
                            "last_price": pos.get('last_price', 0),
                            "pnl": pos.get('pnl', 0)
                        }
                        summary["positions"].append(pos_data)
                        summary["net_quantity"] += pos.get('quantity', 0)
                        summary["unrealized_pnl"] += pos.get('pnl', 0)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting position summary: {e}")
            return {"error": str(e)}

    def schedule_end_of_day_tasks(self) -> None:
        """Schedule end of day maintenance tasks"""
        try:
            # Schedule at 3:35 PM (after market close)
            schedule.every().day.at("15:35").do(self._end_of_day_tasks)
            logger.info("End of day tasks scheduled for 15:35")
        except Exception as e:
            logger.error(f"Error scheduling end of day tasks: {e}")

    def _end_of_day_tasks(self) -> None:
        """Execute end of day maintenance tasks"""
        try:
            logger.info("Starting end of day tasks...")
            
            # Stop trading if still active
            if self.is_trading:
                self.stop()
            
            # Get final statistics
            stats = self.get_trading_statistics()
            
            # Export trades
            export_filename = self.export_trades_to_csv()
            
            # Send daily summary
            try:
                eod_summary = f"""
📊 **END OF DAY SUMMARY**
📅 Date: {datetime.now().strftime('%Y-%m-%d')}
🔄 Total Trades: {stats.get('total_trades', 0)}
📈 CE Trades: {stats.get('ce_trades', 0)}
📉 PE Trades: {stats.get('pe_trades', 0)}
💰 Daily P&L: ₹{self.daily_pnl:.2f}
🎯 Avg Confidence: {stats.get('average_confidence', 0):.1f}
📋 Exported to: {export_filename}
                """
                self._safe_send_message(eod_summary, parse_mode="Markdown")
            except Exception as e:
                logger.error(f"Failed to send EOD summary: {e}")
            
            logger.info("✅ End of day tasks completed")
            
        except Exception as e:
            logger.error(f"Error in end of day tasks: {e}")

    # Safe utility methods for telegram communication
    def _safe_send_message(self, message: str, **kwargs) -> None:
        """Safely send telegram message with error handling"""
        try:
            if self.telegram_controller:
                self.telegram_controller.send_message(message, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to send telegram message: {e}")

    def _safe_send_alert(self, action: str) -> None:
        """Safely send telegram alert with error handling"""
        try:
            if self.telegram_controller:
                self.telegram_controller.send_realtime_session_alert(action)
        except Exception as e:
            logger.warning(f"Failed to send telegram alert: {e}")

    def __repr__(self) -> str:
        return (f"<RealTimeTrader is_trading={self.is_trading} "
                f"live_mode={self.live_mode} trades_today={len(self.trades)} "
                f"cache_age={int((time.time() - self._instruments_cache_timestamp)/60)}min>")

    def __del__(self):
        """Cleanup on object destruction"""
        try:
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
        except Exception:
            pass


# Enhanced Main Execution
def main():
    """Enhanced main execution with proper setup and error handling"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('realtime_trader.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("🚀 Starting RealTime Options Trader...")
    
    try:
        # Initialize trader
        trader = RealTimeTrader()
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            trader.emergency_stop_all()
            trader.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Schedule end of day tasks
        trader.schedule_end_of_day_tasks()
        
        # Run initial health check
        trader._run_health_check()
        
        # Send startup notification
        startup_msg = f"""
🚀 **REALTIME TRADER STARTED**
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 Mode: {'🟢 LIVE' if trader.live_mode else '🛡️ SHADOW'}
🔄 Status: {'✅ ACTIVE' if trader.is_trading else '⏹️ STANDBY'}
📱 Telegram: ✅ CONNECTED
        """
        trader._safe_send_message(startup_msg, parse_mode="Markdown")
        
        logger.info("✅ RealTime Trader initialized successfully")
        
        # Main loop
        while True:
            try:
                # Run scheduled tasks
                schedule.run_pending()
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)  # Wait before retrying
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        logger.info("Shutting down RealTime Trader...")


if __name__ == "__main__":
    main()

try:
    from src.risk.position_sizing import PositionSizing
except ImportError:
    logging.warning("Could not import PositionSizing, using dummy class")
    class PositionSizing:
        def __init__(self):
            pass
        def calculate_position_size(self, **kwargs):
            return {"quantity": 1}
        def get_risk_status(self):
            return {}

try:
    from src.execution.order_executor import OrderExecutor
except ImportError:
    logging.warning("Could not import OrderExecutor, using dummy class")
    class OrderExecutor:
        def __init__(self, kite=None):
            self.kite = kite
        def place_entry_order(self, **kwargs):
            return "DUMMY_ORDER_ID"
        def setup_gtt_orders(self, **kwargs):
            pass
        def get_active_orders(self):
            return []
        def get_positions(self):
            return []
        def cancel_all_orders(self):
            return 0

try:
    from src.notifications.telegram_controller import TelegramController
except ImportError:
    logging.warning("Could not import TelegramController, using dummy class")
    class TelegramController:
        def __init__(self, **kwargs):
            pass
        def send_message(self, message, **kwargs):
            pass
        def send_realtime_session_alert(self, action):
            pass
        def send_startup_alert(self):
            pass
        def send_signal_alert(self, token, signal, position):
            pass
        def start_polling(self):
            pass
        def stop_polling(self):
            pass

try:
    from src.utils.strike_selector import (_get_spot_ltp_symbol, get_instrument_tokens, 
                                         get_next_expiry_date, health_check)
except ImportError:
    logging.warning("Could not import strike_selector functions, using dummy functions")
    def _get_spot_ltp_symbol():
        return "NSE:NIFTY 50"
    def get_instrument_tokens(**kwargs):
        return None
    def get_next_expiry_date(kite_instance):
        return ""
    def health_check(kite_instance):
        return {"overall_status": "ERROR", "message": "Strike selector not available"}

logger = logging.getLogger(__name__)

class RealTimeTrader:
    def __init__(self) -> None:
        self.is_trading: bool = False
        self.daily_pnl: float = 0.0
        self.trades: List[Dict[str, Any]] = []
        self.live_mode: bool = getattr(Config, 'ENABLE_LIVE_TRADING', False)

        # Enhanced Instrument Caching for Rate Limiting
        self._nfo_instruments_cache: Optional[List[Dict]] = None
        self._nse_instruments_cache: Optional[List[Dict]] = None
        self._instruments_cache_timestamp: float = 0
        self._INSTRUMENT_CACHE_DURATION: int = 300  # 5 minutes
        self._cache_lock = threading.RLock()

        # Performance optimization: Pre-calculate commonly used values
        self._atm_cache: Dict[str, Tuple[int, float]] = {}
        self._ATM_CACHE_DURATION: int = 30  # 30 seconds
        
        # Thread pool for parallel data fetching
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="OptionsDataWorker")

        # Initialize components safely
        self._init_components()
        self._polling_thread: Optional[threading.Thread] = None

        # Setup
        self._start_polling()
        self._setup_smart_scheduling()
        
        atexit.register(self.shutdown)
        logger.info("RealTimeTrader initialized and ready to receive commands.")

    def _init_components(self) -> None:
        """Initialize all components with error handling"""
        try:
            # Strategy initialization
            self.strategy = EnhancedScalpingStrategy(
                base_stop_loss_points=getattr(Config, 'BASE_STOP_LOSS_POINTS', 50),
                base_target_points=getattr(Config, 'BASE_TARGET_POINTS', 100),
                confidence_threshold=getattr(Config, 'CONFIDENCE_THRESHOLD', 7.0),
                min_score_threshold=int(getattr(Config, 'MIN_SIGNAL_SCORE', 6)),
            )
        except Exception as e:
            logger.warning(f"Failed to initialize strategy: {e}")
            self.strategy = None
            
        # Risk manager
        try:
            self.risk_manager = PositionSizing()
        except Exception as e:
            logger.warning(f"Failed to initialize risk manager: {e}")
            self.risk_manager = PositionSizing()  # Use dummy
            
        # Order executor
        self.order_executor = self._init_order_executor()
        
        # Telegram controller
        try:
            self.telegram_controller = TelegramController(
                status_callback=self.get_status,
                control_callback=self._handle_control,
                summary_callback=self.get_summary,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Telegram controller: {e}")
            self.telegram_controller = TelegramController()  # Use dummy

    def _setup_smart_scheduling(self) -> None:
        """Setup intelligent scheduling based on market hours and volatility"""
        try:
            schedule.every(30).seconds.do(self._smart_fetch_and_process)
            logger.info("Scheduled smart_fetch_and_process to run every 30 seconds during market hours.")
        except Exception as e:
            logger.error(f"Error setting up scheduling: {e}")

    def _smart_fetch_and_process(self) -> None:
        """Intelligent data fetching based on current market conditions"""
        try:
            current_time = datetime.now()
            hour = current_time.hour
            
            # Skip during non-market hours (basic check)
            if hour < 9 or hour > 15:
                logger.debug("Outside market hours, skipping data fetch.")
                return
                
            # Call the main processing function
            self.fetch_and_process_data()
        except Exception as e:
            logger.error(f"Error in smart fetch and process: {e}")

    def _init_order_executor(self) -> OrderExecutor:
        if not self.live_mode:
            logger.info("Live trading disabled. Using simulated order executor.")
            return OrderExecutor()
        try:
            from kiteconnect import KiteConnect
            api_key = getattr(Config, 'ZERODHA_API_KEY', '')
            access_token = getattr(Config, 'KITE_ACCESS_TOKEN', '')
            if not api_key or not access_token:
                raise ValueError("API key or access token missing")
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            logger.info("✅ Live order executor initialized with Kite Connect.")
            return OrderExecutor(kite=kite)
        except Exception as exc:
            logger.error(f"Failed to initialize live trading. Falling back to simulation: {exc}")
            self.live_mode = False
            return OrderExecutor()

    def start(self) -> bool:
        """Start trading operations"""
        if self.is_trading:
            logger.info("Trader already running.")
            self._safe_send_message("🛑 Trader already running.")
            return True
        self.is_trading = True
        try:
            self._safe_send_alert("START")
            logger.info("✅ Trading started.")
        except Exception as exc:
            logger.warning(f"Failed to send START alert: {exc}")
        return True

    def stop(self) -> bool:
        """Stop trading operations"""
        if not self.is_trading:
            logger.info("Trader is not running.")
            self._safe_send_message("🛑 Trader is already stopped.")
            return True
        self.is_trading = False
        try:
            self._safe_send_alert("STOP")
            logger.info("🛑 Trading stopped. Telegram polling remains active.")
        except Exception as exc:
            logger.warning(f"Failed to send STOP alert: {exc}")
        return True

    def _handle_control(self, command: str, arg: str = "") -> bool:
        """Handle control commands from Telegram"""
        command = command.strip().lower()
        arg = arg.strip().lower() if arg else ""
        logger.info(f"Received command: /{command} {arg}")
        
        try:
            if command == "start":
                return self.start()
            elif command == "stop":
                return self.stop()
            elif command == "mode":
                if arg not in ["live", "shadow"]:
                    logger.warning(f"Invalid mode argument: {arg}")
                    self._safe_send_message("⚠️ Usage: `/mode live` or `/mode shadow`", parse_mode="Markdown")
                    return False
                return self._set_live_mode(arg)
            elif command == "refresh":
                return self._force_refresh_cache()
            elif command == "status":
                return self._send_detailed_status()
            elif command == "health":
                return self._run_health_check()
            elif command == "emergency":
                return self.emergency_stop_all()
            else:
                logger.warning(f"Unknown control command: {command}")
                self._safe_send_message(f"❌ Unknown command: `{command}`", parse_mode="Markdown")
                return False
        except Exception as e:
            logger.error(f"Error handling control command: {e}")
            return False

    def _force_refresh_cache(self) -> bool:
        """Force refresh instrument cache"""
        try:
            self._refresh_instruments_cache(force=True)
            self._safe_send_message("🔄 Instrument cache refreshed successfully.")
            return True
        except Exception as e:
            logger.error(f"Error refreshing cache: {e}")
            return False

    def _send_detailed_status(self) -> bool:
        """Send detailed status information"""
        try:
            status = self.get_status()
            cache_age = (time.time() - self._instruments_cache_timestamp) / 60
            
            status_msg = f"""
📊 **Detailed Status**
🔄 Trading: {'✅ Active' if status['is_trading'] else '❌ Stopped'}
🎯 Mode: {'🟢 LIVE' if self.live_mode else '🛡️ SHADOW'}
📈 Open Orders: {status.get('open_orders', 0)}
💼 Trades Today: {status['trades_today']}
🕐 Cache Age: {cache_age:.1f} min
📊 Daily PnL: ₹{self.daily_pnl:.2f}
            """
            self._safe_send_message(status_msg, parse_mode="Markdown")
            return True
        except Exception as e:
            logger.error(f"Error sending detailed status: {e}")
            return False

    def _run_health_check(self) -> bool:
        """Run and report health check"""
        try:
            if hasattr(self.order_executor, 'kite') and self.order_executor.kite:
                health_result = health_check(self.order_executor.kite)
            else:
                health_result = {"overall_status": "ERROR", "message": "No Kite instance available"}
            
            health_msg = f"""
🏥 **System Health Check**
📊 Overall: {health_result.get('overall_status', 'UNKNOWN')}
🔄 Trading: {'✅ ACTIVE' if self.is_trading else '⏹️ STOPPED'}
🎯 Mode: {'🟢 LIVE' if self.live_mode else '🛡️ SHADOW'}
💼 Trades: {len(self.trades)}
📱 Telegram: {'✅ ACTIVE' if self._polling_thread and self._polling_thread.is_alive() else '❌ INACTIVE'}
            """
            
            if health_result.get('overall_status') in ['CRITICAL', 'DEGRADED']:
                health_msg += f"\n⚠️ Issues detected: {health_result.get('summary', 'Check logs')}"
            
            self._safe_send_message(health_msg, parse_mode="Markdown")
            return True
        except Exception as e:
            logger.error(f"Error running health check: {e}")
            return False

    def _set_live_mode(self, mode: str) -> bool:
        """Switch between live and shadow mode"""
        desired_live = (mode == "live")
        if desired_live == self.live_mode:
            current_mode = "LIVE" if self.live_mode else "SHADOW"
            logger.info(f"Already in {current_mode} mode.")
            self._safe_send_message(f"🟢 Already in *{current_mode}* mode.", parse_mode="Markdown")
            return True
        if self.is_trading:
            logger.warning("Cannot change mode while trading is active. Stop trading first.")
            self._safe_send_message("🛑 Cannot change mode while trading. Use `/stop` first.", parse_mode="Markdown")
            return False
        if desired_live:
            try:
                from kiteconnect import KiteConnect
                kite = KiteConnect(api_key=getattr(Config, 'ZERODHA_API_KEY', ''))
                kite.set_access_token(getattr(Config, 'KITE_ACCESS_TOKEN', ''))
                self.order_executor = OrderExecutor(kite=kite)
                self.live_mode = True
                logger.info("🟢 Switched to LIVE mode.")
                self._refresh_instruments_cache(force=True)
                self._safe_send_message("🚀 Switched to *LIVE* trading mode.", parse_mode="Markdown")
                return True
            except Exception as exc:
                logger.error(f"Failed to switch to LIVE mode: {exc}")
                self._safe_send_message(
                    f"❌ Failed to switch to LIVE mode: `{str(exc)[:100]}...` Reverted to SHADOW mode.", 
                    parse_mode="Markdown"
                )
                self.live_mode = False
                self.order_executor = OrderExecutor()
                return False
        else:
            self.order_executor = OrderExecutor()
            self.live_mode = False
            logger.info("🛡️ Switched to SHADOW (simulation) mode.")
            self._safe_send_message("🛡️ Switched to *SHADOW* (simulation) mode.", parse_mode="Markdown")
            return True

    def _start_polling(self) -> None:
        """Start Telegram polling thread"""
        if self._polling_thread and self._polling_thread.is_alive():
            logger.debug("Polling thread already running.")
            return
        if not self.telegram_controller:
            logger.warning("No telegram controller available for polling")
            return
        try:
            self.telegram_controller.send_startup_alert()
        except Exception as e:
            logger.warning(f"Failed to send startup alert: {e}")
        try:
            self._polling_thread = threading.Thread(
                target=self.telegram_controller.start_polling,
                daemon=True
            )
            self._polling_thread.start()
            logger.info("✅ Telegram polling started (daemon).")
        except Exception as e:
            logger.error(f"Failed to start polling thread: {e}")

    def _stop_polling(self) -> None:
        """Stop Telegram polling thread"""
        logger.info("🛑 Stopping Telegram polling (app shutdown)...")
        if self.telegram_controller:
            try:
                self.telegram_controller.stop_polling()
            except Exception as e:
                logger.warning(f"Error stopping telegram polling: {e}")
        if self._polling_thread and self._polling_thread.is_alive():
            if threading.current_thread() != self._polling_thread:
                self._polling_thread.join(timeout=3)
        self._polling_thread = None

    def shutdown(self) -> None:
        """Shutdown the trader gracefully"""
        if not self.is_trading and (not self._polling_thread or not self._polling_thread.is_alive()):
            return
        logger.info("👋 Shutting down RealTimeTrader...")
        self.stop()
        self._stop_polling()
        
        if self._executor:
            try:
                self._executor.shutdown(wait=True, timeout=5)
            except Exception as e:
                logger.warning(f"Error shutting down executor: {e}")
                
        logger.info("✅ RealTimeTrader shutdown complete.")

    def _refresh_instruments_cache(self, force: bool = False) -> None:
        """Thread-safe instrument cache refresh with enhanced error handling"""
        with self._cache_lock:
            if not self.order_executor or not hasattr(self.order_executor, 'kite') or not self.order_executor.kite:
                logger.warning("Cannot refresh cache, Kite instance not available.")
                self._nfo_instruments_cache = []
                self._nse_instruments_cache = []
                return

            current_time = time.time()
            needs_refresh = (
                force or
                self._nfo_instruments_cache is None or
                self._nse_instruments_cache is None or
                (current_time - self._instruments_cache_timestamp) > self._INSTRUMENT_CACHE_DURATION
            )

            if needs_refresh:
                try:
                    logger.debug("Refreshing instruments cache...")
                    
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        nfo_future = executor.submit(self.order_executor.kite.instruments, "NFO")
                        nse_future = executor.submit(self.order_executor.kite.instruments, "NSE")
                        
                        self._nfo_instruments_cache = nfo_future.result(timeout=10)
                        self._nse_instruments_cache = nse_future.result(timeout=10)
                    
                    logger.debug(f"Cached {len(self._nfo_instruments_cache)} NFO instruments.")
                    logger.debug(f"Cached {len(self._nse_instruments_cache)} NSE instruments.")
                    self._instruments_cache_timestamp = current_time
                    logger.info("✅ Instruments cache refreshed.")
                    
                except Exception as e:
                    logger.error(f"Failed to refresh instruments cache: {e}")
                    if self._nfo_instruments_cache is None:
                        self._nfo_instruments_cache = []
                    if self._nse_instruments_cache is None:
                        self._nse_instruments_cache = []

    def _get_cached_instruments(self) -> Tuple[List[Dict], List[Dict]]:
        """Thread-safe access to cached instruments"""
        with self._cache_lock:
            return self._nfo_instruments_cache or [], self._nse_instruments_cache or []

    @lru_cache(maxsize=128)
    def _get_cached_atm_strike(self, spot_price: float, timestamp_bucket: int) -> int:
        """Cached ATM strike calculation to avoid repeated computation"""
        return round(spot_price / 50) * 50

    def fetch_and_process_data(self) -> None:
        """
        Optimized data fetching with parallel processing and better error handling
        """
        if not self.is_trading:
            logger.debug("fetch_and_process_data: Trading not active, skipping.")
            return

        start_time_processing = time.time()
        
        try:
            if not hasattr(self.order_executor, 'kite') or not self.order_executor.kite:
                logger.error("KiteConnect instance not found. Is live mode enabled?")
                return

            # Refresh instruments cache with timeout
            self._refresh_instruments_cache()
            cached_nfo, cached_nse = self._get_cached_instruments()
            
            if not cached_nfo and not cached_nse:
                logger.error("Instrument cache is empty. Cannot proceed.")
                return

            # Parallel spot price and instrument token fetching
            with ThreadPoolExecutor(max_workers=3) as executor:
                spot_future = executor.submit(self._fetch_spot_price)
                instruments_future = executor.submit(
                    self._get_instruments_data,
                    cached_nfo,
                    cached_nse
                )
                
                spot_price = spot_future.result(timeout=5)
                instruments_data = instruments_future.result(timeout=5)

            if not spot_price or not instruments_data:
                logger.error("Failed to fetch essential data (spot price or instruments)")
                return

            atm_strike = instruments_data['atm_strike']
            expiry = instruments_data['expiry']
            spot_token = instruments_data.get('spot_token')

            logger.info(f"ATM Strike: {atm_strike}, Expiry: {expiry}, Spot: {spot_price}")

            # Optimized timeframe calculation
            end_time = datetime.now()
            lookback_minutes = getattr(Config, 'DATA_LOOKBACK_MINUTES', 30)
            start_time_data = end_time - timedelta(minutes=lookback_minutes)

            # Parallel data fetching for spot and options
            spot_df, options_data = self._fetch_all_data_parallel(
                spot_token, atm_strike, start_time_data, end_time, cached_nfo, cached_nse
            )

            # Process options data
            if options_data:
                selected_strikes_info = self._analyze_options_data_optimized(spot_price, options_data)
            else:
                selected_strikes_info = self._get_fallback_strikes(atm_strike, cached_nfo, cached_nse)

            if not selected_strikes_info:
                logger.warning("No strikes selected for processing")
                return

            # Process selected strikes
            self._process_selected_strikes(selected_strikes_info, options_data, spot_df)

            processing_time = time.time() - start_time_processing
            logger.debug(f"Data fetch and process completed in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Error in fetch_and_process_data: {e}", exc_info=True)

    def _fetch_spot_price(self) -> Optional[float]:
        """Fetch spot price with error handling"""
        try:
            spot_symbol_ltp = _get_spot_ltp_symbol()
            ltp_data = self.order_executor.kite.ltp([spot_symbol_ltp])
            spot_price = ltp_data.get(spot_symbol_ltp, {}).get('last_price')
            
            if spot_price is None:
                logger.error(f"Failed to fetch spot price for {spot_symbol_ltp}")
                return None
                
            logger.debug(f"Successfully fetched spot price: {spot_price}")
            return float(spot_price)
            
        except Exception as e:
            logger.error(f"Exception fetching spot price: {e}")
            return None

    def _get_instruments_data(self, cached_nfo: List[Dict], cached_nse: List[Dict]) -> Optional[Dict]:
        """Get instrument tokens data"""
        try:
            instruments_data = get_instrument_tokens(
                symbol=getattr(Config, 'SPOT_SYMBOL', 'NIFTY'),
                kite_instance=self.order_executor.kite,
                cached_nfo_instruments=cached_nfo,
                cached_nse_instruments=cached_nse
            )
            return instruments_data
        except Exception as e:
            logger.error(f"Error getting instruments data: {e}")
            return None

    def _fetch_all_data_parallel(self, spot_token: Optional[int], atm_strike: int, 
                               start_time: datetime, end_time: datetime,
                               cached_nfo: List[Dict], cached_nse: List[Dict]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Fetch all required data in parallel"""
        spot_df = pd.DataFrame()
        options_data = {}

        fetch_tasks = []
        
        # Add spot data fetch task
        if spot_token:
            fetch_tasks.append(('spot', spot_token, 'SPOT'))

        # Add options data fetch tasks
        try:
            strike_range = getattr(Config, 'STRIKE_RANGE', 4)
            for offset in range(-strike_range, strike_range + 1):
                temp_instruments = get_instrument_tokens(
                    symbol=getattr(Config, 'SPOT_SYMBOL', 'NIFTY'),
                    offset=offset,
                    kite_instance=self.order_executor.kite,
                    cached_nfo_instruments=cached_nfo,
                    cached_nse_instruments=cached_nse
                )
                
                if temp_instruments:
                    for opt_type, token_key, symbol_key in [('CE', 'ce_token', 'ce_symbol'), ('PE', 'pe_token', 'pe_symbol')]:
                        token = temp_instruments.get(token_key)
                        symbol = temp_instruments.get(symbol_key)
                        if token and symbol:
                            fetch_tasks.append((symbol, token, opt_type))
        except Exception as e:
            logger.error(f"Error preparing fetch tasks: {e}")

        # Execute all fetch tasks in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_symbol = {
                executor.submit(self._fetch_historical_data, token, start_time, end_time): symbol
                for symbol, token, _ in fetch_tasks
            }

            for future in as_completed(future_to_symbol, timeout=15):
                symbol = future_to_symbol[future]
                try:
                    hist_data = future.result()
                    if hist_data is not None:
                        if symbol == 'SPOT':
                            spot_df = hist_data
                        else:
                            options_data[symbol] = hist_data
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")

        return spot_df, options_data

    def _fetch_historical_data(self, token: int, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Fetch historical data for a single instrument"""
        try:
            hist_data = self.order_executor.kite.historical_data(
                instrument_token=token,
                from_date=start_time,
                to_date=end_time,
                interval="minute"
            )
            
            if hist_data:
                df = pd.DataFrame(hist_data)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                return df
            return None
            
        except Exception as e:
            logger.error(f"Error fetching historical data for token {token}: {e}")
            return None

    def _analyze_options_data_optimized(self, spot_price: float, options_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Optimized options data analysis with parallel processing and caching
        """
        if not options_data:
            return []

        timestamp_bucket = int(time.time() // 30)
        atm_strike = self._get_cached_atm_strike(spot_price, timestamp_bucket)
        
        selected_strikes = []
        
        def analyze_option_type(opt_type: str) -> List[Dict[str, Any]]:
            type_strikes = []
            relevant_data = {k: v for k, v in options_data.items() if opt_type in k and not v.empty}
            
            if not relevant_data:
                return type_strikes

            for symbol, df in relevant_data.items():
                try:
                    opt_index = symbol.find(opt_type)
                    if opt_index == -1:
                        continue
                    
                    strike_str = symbol[opt_index-5:opt_index]
                    strike = int(strike_str)
                    
                    if len(df) < 1:
                        continue

                    last_row = df.iloc[-1]
                    prev_row = df.iloc[-2] if len(df) > 1 else last_row

                    oi_change = last_row.get('oi', 0) - prev_row.get('oi', 0)
                    delta_approx = self._calculate_delta_approximation(opt_type, strike, atm_strike, spot_price)

                    type_strikes.append({
                        'symbol': symbol,
                        'strike': strike,
                        'type': opt_type,
                        'ltp': last_row.get('last_price', 0),
                        'oi': last_row.get('oi', 0),
                        'oi_change': oi_change,
                        'delta': delta_approx,
                        'is_atm': strike == atm_strike,
                        'is_otm': (strike > atm_strike) if opt_type == 'CE' else (strike < atm_strike),
                        'is_itm': (strike < atm_strike) if opt_type == 'CE' else (strike > atm_strike)
                    })

                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not extract strike from symbol {symbol}: {e}")
                    continue

            type_strikes.sort(key=lambda x: x['oi_change'], reverse=True)
            return self._select_best_strikes(type_strikes, opt_type)

        # Process both types in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            ce_future = executor.submit(analyze_option_type, 'CE')
            pe_future = executor.submit(analyze_option_type, 'PE')
            
            selected_strikes.extend(ce_future.result())
            selected_strikes.extend(pe_future.result())

        return selected_strikes

    def _calculate_delta_approximation(self, opt_type: str, strike: int, atm_strike: int, spot_price: float) -> float:
        """Fast delta approximation for option selection"""
        try:
            moneyness = (spot_price - strike) / spot_price if spot_price > 0 else 0
            
            if opt_type == 'CE':
                if strike == atm_strike:
                    return 0.5
                elif strike < atm_strike:
                    return 0.5 + min(0.3, abs(moneyness) * 2)
                else:
                    return max(0.1, 0.5 - abs(moneyness) * 2)
            else:
                if strike == atm_strike:
                    return -0.5
                elif strike > atm_strike:
                    return -0.5 - min(0.3, abs(moneyness) * 2)
                else:
                    return max(-0.9, -0.5 + abs(moneyness) * 2)
        except Exception as e:
            logger.error(f"Error calculating delta approximation: {e}")
            return 0.5 if opt_type == 'CE' else -0.5

    def _select_best_strikes(self, strikes: List[Dict[str, Any]], opt_type: str) -> List[Dict[str, Any]]:
        """Select the best strikes based on strategy configuration"""
        if not strikes:
            return []

        selected = []
        
        atm_option = next((s for s in strikes if s['is_atm']), None)
        if atm_option:
            selected.append(atm_option)
            logger.debug(f"Selected ATM {opt_type}: {atm_option['symbol']}")

        strategy_type = getattr(Config, 'STRIKE_SELECTION_TYPE', 'OTM')
        
        if strategy_type == "ITM":
            best_other = next((s for s in strikes if s['is_itm'] and s not in selected), None)
        elif strategy_type == "OTM":
            best_other = next((s for s in strikes if s['is_otm'] and s not in selected), None)
        else:
            best_other = next((s for s in strikes if s not in selected), None)

        if best_other:
            selected.append(best_other)
            logger.debug(f"Selected {strategy_type} {opt_type}: {best_other['symbol']} (OI Change: {best_other['oi_change']})")

        return selected

    def _get_fallback_strikes(self, atm_strike: int, cached_nfo: List[Dict], cached_nse: List[Dict]) -> List[Dict[str, Any]]:
        """Optimized fallback strike selection"""
        logger.warning("Using fallback strike selection")
        fallback_strikes = []
        
        try:
            for offset in [0, 1, -1]:
                fb_strike = atm_strike + (offset * 50)
                fb_instruments = get_instrument_tokens(
                    symbol=getattr(Config, 'SPOT_SYMBOL', 'NIFTY'),
                    offset=offset,
                    kite_instance=self.order_executor.kite,
                    cached_nfo_instruments=cached_nfo,
                    cached_nse_instruments=cached_nse
                )
                
                if fb_instruments:
                    for opt_type, symbol_key in [('CE', 'ce_symbol'), ('PE', 'pe_symbol')]:
                        symbol = fb_instruments.get(symbol_key)
                        if symbol:
                            fallback_strikes.append({
                                'symbol': symbol,
                                'strike': fb_strike,
                                'type': opt_type,
                                'ltp': 0,
                                'oi': 0,
                                'oi_change': 0,
                                'delta': 0.5 if opt_type == 'CE' else -0.5,
                                'is_atm': offset == 0,
                                'is_otm': (offset > 0) if opt_type == 'CE' else (offset < 0),
                                'is_itm': (offset < 0) if opt_type == 'CE' else (offset > 0)
                            })
        except Exception as e:
            logger.error(f"Error in fallback strikes: {e}")
        
        return fallback_strikes

    def _process_selected_strikes(self, selected_strikes: List[Dict[str, Any]], 
                                options_data: Dict[str, pd.DataFrame], spot_df: pd.DataFrame) -> None:
        """Process selected strikes with parallel execution"""
        if not selected_strikes:
            return

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for strike_info in selected_strikes:
                symbol = strike_info['symbol']
                df = options_data.get(symbol)
                
                if df is not None and not df.empty:
                    future = executor.submit(self.process_options_bar, symbol, df, spot_df, strike_info)
                    futures.append(future)
                else:
                    future = executor.submit(self._process_ltp_based, symbol, spot_df, strike_info)
                    futures.append(future)
            
            for future in as_completed(futures, timeout=10):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing strike: {e}")

    def _process_ltp_based(self, symbol: str, spot_df: pd.DataFrame, strike_info: Dict[str, Any]) -> None:
        """Process strike using LTP data when historical data is unavailable"""
        try:
            ltp_data = self.order_executor.kite.ltp([f"NFO:{symbol}"])
            ltp_price = ltp_data.get(f"NFO:{symbol}", {}).get('last_price', 0)
            
            if ltp_price > 0:
                dummy_df = pd.DataFrame([{
                    'date': pd.Timestamp.now(),
                    'last_price': ltp_price,
                    'oi': strike_info.get('oi', 0),
                    'volume': 0
                }]).set_index('date')
                
                self.process_options_bar(symbol, dummy_df, spot_df, strike_info)
            else:
                logger.warning(f"Could not fetch valid LTP for {symbol}")
                
        except Exception as e:
            logger.error(f"Error in LTP-based processing for {symbol}: {e}")

    def process_options_bar(self, symbol: str, ohlc: pd.DataFrame, spot_ohlc: pd.DataFrame, strike_info: Dict[str, Any]) -> None:
        """
        Optimized options bar processing with enhanced error handling
        """
        if not self.is_trading:
            logger.debug(f"process_options_bar: Trading not active for {symbol}")
            return

        try:
            current_time = datetime.now()
            if not self._is_trading_hours(current_time):
                logger.debug(f"Outside trading hours for {symbol}")
                return

            current_price = self._get_current_price(ohlc, strike_info)
            if current_price <= 0:
                logger.warning(f"Invalid current price for {symbol}: {current_price}")
                return

            if not ohlc.empty:
                ts = ohlc.index[-1]
                if not self._passes_time_filter(ts):
                    logger.debug(f"Time filter rejected bar for {symbol}")
                    return
            else:
                ts = pd.Timestamp.now()

            logger.debug(f"Processing {symbol} at {ts}, price: {current_price}")

            # Enhanced signal generation
            signal = self._generate_enhanced_options_signal(ohlc, spot_ohlc, strike_info, current_price)
            
            if not signal:
                logger.debug(f"No signal generated for {symbol}")
                return

            # Optimized confidence check
            signal_confidence = float(signal.get("confidence", 0.0))
            confidence_threshold = getattr(Config, 'CONFIDENCE_THRESHOLD', 7.0)
            
            if signal_confidence < confidence_threshold:
                logger.debug(f"Signal confidence {signal_confidence} below threshold {confidence_threshold} for {symbol}")
                return

            # Enhanced position sizing with options-specific logic
            position = self._calculate_options_position_size(signal, strike_info, current_price)
            if not position or position.get("quantity", 0) <= 0:
                logger.debug(f"Invalid position sizing for {symbol}")
                return

            # Risk management checks
            if not self._passes_risk_checks(signal, position, strike_info):
                logger.warning(f"Risk checks failed for {symbol}")
                return

            # Execute trade workflow
            success = self._execute_options_trade(symbol, signal, position, strike_info)
            if success:
                logger.info(f"✅ Successfully executed options trade for {symbol}")
            else:
                logger.warning(f"❌ Failed to execute options trade for {symbol}")

        except Exception as exc:
            logger.error(f"Error processing options bar for {symbol}: {exc}", exc_info=True)

    def _is_trading_hours(self, current_time: datetime) -> bool:
        """Check if current time is within trading hours"""
        hour = current_time.hour
        minute = current_time.minute
        
        # Market hours: 9:15 AM to 3:30 PM
        if hour < 9 or (hour == 9 and minute < 15):
            return False
        if hour > 15 or (hour == 15 and minute > 30):
            return False
        
        return True

    def _get_current_price(self, ohlc: pd.DataFrame, strike_info: Dict[str, Any]) -> float:
        """Get current price with multiple fallback mechanisms"""
        if not ohlc.empty:
            return float(ohlc.iloc[-1].get("last_price", 0))
        
        # Fallback to strike_info
        ltp = strike_info.get('ltp', 0)
        if ltp > 0:
            return float(ltp)
            
        logger.warning("Could not determine current price from any source")
        return 0.0

    def _passes_time_filter(self, timestamp: pd.Timestamp) -> bool:
        """Enhanced time filter check"""
        time_filter_start = getattr(Config, 'TIME_FILTER_START', None)
        time_filter_end = getattr(Config, 'TIME_FILTER_END', None)
        
        if not time_filter_start or not time_filter_end:
            return True
            
        current_time_str = timestamp.strftime("%H:%M")
        return time_filter_start <= current_time_str <= time_filter_end

    def _generate_enhanced_options_signal(self, ohlc: pd.DataFrame, spot_ohlc: pd.DataFrame, 
                                        strike_info: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        """
        Enhanced options signal generation with multiple strategies
        """
        try:
            signal_dict = {
                "signal": None,
                "entry_price": current_price,
                "stop_loss": None,
                "target": None,
                "confidence": 0.0,
                "market_volatility": 0.0,
                "strategy_type": "options_momentum"
            }

            # Multi-factor signal generation
            confidence_score = 0.0
            
            # 1. Options price momentum analysis
            momentum_signal = self._analyze_options_momentum(ohlc, strike_info)
            if momentum_signal:
                confidence_score += momentum_signal['confidence']
                if not signal_dict["signal"]:
                    signal_dict.update(momentum_signal)

            # 2. Spot-options correlation analysis
            correlation_signal = self._analyze_spot_options_correlation(spot_ohlc, strike_info, current_price)
            if correlation_signal:
                confidence_score += correlation_signal['confidence']
                if correlation_signal['confidence'] > momentum_signal.get('confidence', 0):
                    signal_dict.update(correlation_signal)

            # 3. Options Greeks-based signals (simplified)
            greeks_signal = self._analyze_options_greeks(strike_info, current_price)
            if greeks_signal:
                confidence_score += greeks_signal['confidence']

            # 4. Volume and OI analysis
            volume_signal = self._analyze_volume_oi(ohlc, strike_info)
            if volume_signal:
                confidence_score *= volume_signal['multiplier']

            # Final confidence adjustment
            signal_dict["confidence"] = min(confidence_score, 10.0)  # Cap at 10

            # Only return signal if above minimum threshold
            min_confidence = getattr(Config, 'MIN_SIGNAL_CONFIDENCE', 6.0)
            if signal_dict["confidence"] >= min_confidence and signal_dict["signal"]:
                # Set risk parameters
                signal_dict["stop_loss"] = self._calculate_options_stop_loss(current_price, strike_info)
                signal_dict["target"] = self._calculate_options_target(current_price, strike_info)
                return signal_dict

            return None

        except Exception as e:
            logger.error(f"Error generating enhanced options signal: {e}")
            return None

    def _analyze_options_momentum(self, ohlc: pd.DataFrame, strike_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze options price momentum"""
        if ohlc.empty or len(ohlc) < 3:
            return None

        try:
            # Calculate momentum indicators
            recent_prices = ohlc['last_price'].tail(5).values
            if len(recent_prices) < 2:
                return None

            # Price change momentum
            price_change_pct = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
            
            # Volume momentum (if available)
            volume_momentum = 1.0
            if 'volume' in ohlc.columns:
                recent_volumes = ohlc['volume'].tail(3)
                avg_volume = recent_volumes.mean()
                current_volume = recent_volumes.iloc[-1]
                volume_momentum = min(current_volume / avg_volume, 3.0) if avg_volume > 0 else 1.0

            # Determine signal strength
            confidence = 0.0
            signal_direction = None

            # Bullish momentum
            if price_change_pct > 2.0:  # 2% price increase
                signal_direction = "BUY"
                confidence = min(price_change_pct * volume_momentum, 5.0)

            # Bearish momentum (for selling options)
            elif price_change_pct < -1.5:  # 1.5% price decrease
                # For options, we typically buy on dips if it's a good setup
                if strike_info.get('is_otm', False):  # OTM options on dips can be opportunities
                    signal_direction = "BUY"
                    confidence = min(abs(price_change_pct) * volume_momentum * 0.8, 4.0)

            if signal_direction and confidence > 0:
                return {
                    "signal": signal_direction,
                    "confidence": confidence,
                    "strategy_type": "momentum"
                }

            return None

        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return None

    def _analyze_spot_options_correlation(self, spot_ohlc: pd.DataFrame, strike_info: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        """Analyze spot-options price correlation"""
        if spot_ohlc.empty or len(spot_ohlc) < 5:
            return None

        try:
            # Calculate spot momentum
            spot_prices = spot_ohlc['close'].tail(5).values
            spot_change_pct = (spot_prices[-1] - spot_prices[0]) / spot_prices[0] * 100

            # Analyze based on option type and moneyness
            opt_type = strike_info.get('type', '')
            is_atm = strike_info.get('is_atm', False)
            is_itm = strike_info.get('is_itm', False)
            is_otm = strike_info.get('is_otm', False)

            confidence = 0.0
            signal_direction = None

            # CE analysis
            if opt_type == 'CE':
                if spot_change_pct > 0.3:  # Spot moving up
                    if is_atm or is_otm:
                        signal_direction = "BUY"
                        confidence = min(spot_change_pct * 2.0, 6.0)
                    elif is_itm:
                        signal_direction = "BUY"
                        confidence = min(spot_change_pct * 1.5, 5.0)

            # PE analysis
            elif opt_type == 'PE':
                if spot_change_pct < -0.3:  # Spot moving down
                    if is_atm or is_otm:
                        signal_direction = "BUY"
                        confidence = min(abs(spot_change_pct) * 2.0, 6.0)
                    elif is_itm:
                        signal_direction = "BUY"
                        confidence = min(abs(spot_change_pct) * 1.5, 5.0)

            if signal_direction and confidence > 0:
                return {
                    "signal": signal_direction,
                    "confidence": confidence,
                    "strategy_type": "correlation"
                }

            return None

        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return None

    def _analyze_options_greeks(self, strike_info: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        """Simplified Greeks analysis"""
        try:
            delta = strike_info.get('delta', 0)
            opt_type = strike_info.get('type', '')
            
            confidence_boost = 0.0
            
            # Delta-based analysis
            if opt_type == 'CE':
                if 0.4 <= abs(delta) <= 0.7:  # Sweet spot for CE delta
                    confidence_boost = 1.5
                elif abs(delta) > 0.7:  # High delta, more sensitive
                    confidence_boost = 1.0
            elif opt_type == 'PE':
                if 0.4 <= abs(delta) <= 0.7:  # Sweet spot for PE delta
                    confidence_boost = 1.5
                elif abs(delta) > 0.7:  # High delta, more sensitive
                    confidence_boost = 1.0

            return {"confidence": confidence_boost} if confidence_boost > 0 else None

        except Exception as e:
            logger.error(f"Error in Greeks analysis: {e}")
            return None

    def _analyze_volume_oi(self, ohlc: pd.DataFrame, strike_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze volume and open interest"""
        try:
            multiplier = 1.0
            
            # Volume analysis (if available)
            if not ohlc.empty and 'volume' in ohlc.columns:
                recent_volume = ohlc['volume'].tail(3).sum()
                if recent_volume > 0:
                    multiplier *= 1.2
                    
            # OI analysis
            oi_change = strike_info.get('oi_change', 0)
            if oi_change > 0:
                multiplier *= 1.3
            elif oi_change < 0:
                multiplier *= 0.9

            return {"multiplier": multiplier}

        except Exception as e:
            logger.error(f"Error in volume/OI analysis: {e}")
            return {"multiplier": 1.0}