#!/usr/bin/env python3
"""
Telegram Bot Module for Nifty Scalper Bot v2.0
Enhanced with proper market timing and auto-trading controls
"""
import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from config import Config
from utils import (
    is_market_open,
    get_market_status,
    time_until_market_open,
    get_market_session_info,
    format_currency,
    format_percentage,
    format_trade_duration,
    calculate_performance_metrics
)

logger = logging.getLogger(__name__)

class TelegramBot:
    """Enhanced Telegram bot for trading commands and notifications"""

    def __init__(self, trading_bot_instance=None):
        self.trading_bot = trading_bot_instance
        self.app: Optional[Application] = None
        self.is_running = False
        self._background_tasks = set() # To keep references to tasks
        self._stop_event = asyncio.Event() # Event to signal stop

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command with enhanced status"""
        try:
            market_info = get_market_session_info()
            start_message = f"""🚀 *Nifty Scalper Bot v2.0 Started!*

*⚙️ Current Status:*
• Mode: 💰 LIVE TRADING
• Auto-trading: {'✅ ON' if getattr(self.trading_bot, 'auto_trade', False) else '❌ OFF'}
• Market: {get_market_status()}
• Time: {market_info.get('current_time', 'Unknown')} IST

Use /help to see available commands.
"""
            await update.message.reply_text(start_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in start command: {e}")
            await update.message.reply_text("❌ Error starting bot. Please try again.")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """*🤖 Nifty Scalper Bot Commands:*

/start - Start the bot and show status
/status - Show detailed bot status
/config - Show configuration details
/performance - Show performance metrics
/positions - Show current open positions
/start_trading - Enable auto-trading
/stop_trading - Disable auto-trading
/exit_position - Manually close current position
/help - Show this help message
"""
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command with comprehensive dashboard"""
        try:
            if not self.trading_bot:
                await update.message.reply_text("❌ Bot is not connected to trading engine.")
                return

            market_info = get_market_session_info()
            market_status = get_market_status()

            # Get trading bot status
            auto_trade_status = "✅ ON" if getattr(self.trading_bot, 'auto_trade', False) else "❌ OFF"
            risk_manager = getattr(self.trading_bot, 'risk_manager', None)
            current_position = getattr(self.trading_bot, 'current_position', None)

            # Market timing info
            market_extra = ""
            if not is_market_open():
                time_until_open = time_until_market_open()
                if time_until_open:
                    hours, remainder = divmod(int(time_until_open.total_seconds()), 3600)
                    minutes, _ = divmod(remainder, 60)
                    market_extra = f" (Opens in {hours}h {minutes}m)"

            # Circuit breaker info
            circuit_breaker_info = "🟢 *Circuit Breaker:* Inactive"
            if getattr(risk_manager, 'circuit_breaker_active', False):
                remaining_time = getattr(risk_manager, 'circuit_breaker_until', None)
                if remaining_time:
                    mins = max(0, int((remaining_time - datetime.now()).total_seconds() / 60))
                    circuit_breaker_info = f"🚨 *Circuit Breaker:* Active ({mins}m remaining)"

            # Position info
            position_text = "💤 No active trades"
            if current_position:
                direction = current_position.get('direction', 'N/A')
                entry_price = current_position.get('entry_price', 0)
                qty = current_position.get('quantity', 0)
                entry_time = current_position.get('entry_time', '')

                unrealized_pnl = ""
                try:
                    md = self.trading_bot.get_market_data()
                    if md and 'ltp' in md:
                        cp = md['ltp']
                        pnl = (cp - entry_price) * qty if direction == 'BUY' else (entry_price - cp) * qty
                        unrealized_pnl = f" | P&L: {format_currency(pnl)}"
                except:
                    pass

                position_text = f"🔥 *{direction}* {qty} @ ₹{entry_price:.2f}{unrealized_pnl}"

            # Performance info
            todays_pnl = getattr(risk_manager, 'todays_pnl', 0) if risk_manager else 0
            daily_trades = getattr(risk_manager, 'daily_trades', 0) if risk_manager else 0
            current_balance = getattr(risk_manager, 'current_balance', 0) if risk_manager else 0

            pnl_emoji = "📈" if todays_pnl >= 0 else "📉"
            pnl_color = "+" if todays_pnl >= 0 else ""

            status_message = f"""*🔄 Bot Status Dashboard*

*💼 Trading Status:*
• *Mode:* 💰 LIVE TRADING
• *Auto-trading:* {auto_trade_status}
• *Market:* {market_status}{market_extra}
• *Time:* {market_info.get('current_time', 'Unknown')} IST
{circuit_breaker_info}

*📊 Today's Performance:*
• *Balance:* {format_currency(current_balance)}
• *P&L:* {pnl_emoji} {pnl_color}{format_currency(todays_pnl)}
• *Trades:* {daily_trades}

*📍 Positions:*
{position_text}
"""
            await update.message.reply_text(status_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await update.message.reply_text("❌ Error getting status. Please try again.")

    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command with detailed configuration"""
        try:
            market_session = get_market_session_info()

            config_message = f"""*⚙️ Bot Configuration*

*📈 Trading Parameters:*
• *Signal Threshold:* {Config.SIGNAL_THRESHOLD}
• *Risk Per Trade:* {format_percentage(Config.RISK_PER_TRADE_PCT * 100)}
• *Max Daily Trades:* {Config.MAX_DAILY_TRADES}
• *Default Lots:* {Config.DEFAULT_LOTS}
• *Max Daily Loss:* {format_percentage(Config.MAX_DAILY_LOSS_PCT * 100)}

*🕒 Market Timings (IST):*
• *Market Hours:* {Config.MARKET_START_HOUR}:{Config.MARKET_START_MINUTE:02d} AM - {Config.MARKET_END_HOUR}:{Config.MARKET_END_MINUTE:02d} PM
• *Trading Days:* Monday to Friday
• *Current Day:* {market_session.get('day_of_week', 'Unknown')}

*🛡️ Risk Management:*
• *Max Consecutive Losses:* {Config.MAX_CONSECUTIVE_LOSSES}
• *Circuit Breaker Pause:* {Config.CIRCUIT_BREAKER_PAUSE_MINUTES} minutes
• *Position Sizing:* Dynamic (Risk-based)

*📱 Telegram Settings:*
• *Notifications:* ✅ Enabled
• *Commands:* ✅ Active
"""
            await update.message.reply_text(config_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in config command: {e}")
            await update.message.reply_text("❌ Error getting configuration. Please try again.")

    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command with metrics"""
        try:
            if not self.trading_bot:
                await update.message.reply_text("❌ Bot is not connected to trading engine.")
                return

            risk_manager = getattr(self.trading_bot, 'risk_manager', None)
            if not risk_manager:
                await update.message.reply_text("❌ Risk manager not available.")
                return

            # Calculate performance metrics
            metrics = calculate_performance_metrics(
                risk_manager.current_balance,
                risk_manager.initial_balance,
                risk_manager.todays_pnl
            )

            perf_message = f"""*🏆 Performance Metrics*

*💰 Capital:*
• *Current Balance:* {format_currency(risk_manager.current_balance)}
• *Initial Capital:* {format_currency(risk_manager.initial_balance)}
• *Net P&L:* {format_currency(risk_manager.current_balance - risk_manager.initial_balance)}

*📊 Today's Trading:*
• *P&L:* {format_currency(risk_manager.todays_pnl)}
• *Trades Executed:* {risk_manager.daily_trades}
• *Win Rate:* {format_percentage(metrics.get('win_rate', 0))}
• *Profit Factor:* {metrics.get('profit_factor', 0):.2f}

*🛡️ Risk Metrics:*
• *Consecutive Losses:* {risk_manager.consecutive_losses}
• *Circuit Breaker:* {'Active' if risk_manager.circuit_breaker_active else 'Inactive'}
"""
            await update.message.reply_text(perf_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in performance command: {e}")
            await update.message.reply_text("❌ Error getting performance metrics. Please try again.")

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        try:
            if not self.trading_bot:
                await update.message.reply_text("❌ Bot is not connected to trading engine.")
                return

            current_position = getattr(self.trading_bot, 'current_position', None)

            if not current_position:
                await update.message.reply_text("📭 *No Open Positions*\nAll positions are closed.", parse_mode='Markdown')
                return

            direction = current_position.get('direction', 'N/A')
            entry_price = current_position.get('entry_price', 0)
            quantity = current_position.get('quantity', 0)
            stop_loss = current_position.get('stop_loss', 0)
            target = current_position.get('target', 0)
            entry_time = current_position.get('entry_time', 'N/A')

            # Try to get current market price for unrealized P&L
            unrealized_pnl = ""
            try:
                md = self.trading_bot.get_market_data()
                if md and 'ltp' in md:
                    cp = md['ltp']
                    pnl = (cp - entry_price) * quantity if direction == 'BUY' else (entry_price - cp) * quantity
                    unrealized_pnl = f"\n• *Unrealized P&L:* {format_currency(pnl)}"
            except:
                pass

            position_message = f"""*📍 Current Position*

• *Direction:* {direction}
• *Quantity:* {quantity}
• *Entry Price:* ₹{entry_price:.2f}
• *Stop Loss:* ₹{stop_loss:.2f}
• *Target:* ₹{target:.2f}
• *Entry Time:* {entry_time}{unrealized_pnl}
"""
            await update.message.reply_text(position_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in positions command: {e}")
            await update.message.reply_text("❌ Error getting position details. Please try again.")

    async def start_trading_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start_trading command"""
        try:
            if self.trading_bot:
                self.trading_bot.auto_trade = True
                message = "✅ *Auto-trading STARTED*\n• New trades will be executed automatically\n• Existing positions will be managed\n• Use /stop_trading to pause"
            else:
                message = "❌ Bot is not connected to trading engine."
            await update.message.reply_text(message, parse_mode='Markdown')
            logger.info(f"Start trading command executed by user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error in start trading command: {e}")
            await update.message.reply_text("❌ Error starting trading. Please try again.")

    async def stop_trading_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE): # Renamed from stop_command
        """Handle /stop_trading command with confirmation"""
        try:
            if self.trading_bot:
                self.trading_bot.auto_trade = False
                current_position = getattr(self.trading_bot, 'current_position', None)
                position_warning = ""
                if current_position:
                    direction = current_position.get('direction', 'N/A')
                    entry_price = current_position.get('entry_price', 0)
                    position_warning = f"\n⚠️ *Warning:* Active {direction} position @ ₹{entry_price:.2f} will continue running!"

                message = f"""🛑 *Auto-trading STOPPED*
• New trades will not be executed
• Existing positions will be monitored
• Use /start_trading to re-enable auto-trading
• Use /exit_position to close current position{position_warning}
Bot remains active for monitoring and manual commands."""
            else:
                message = "❌ Bot is not connected to trading engine."
            await update.message.reply_text(message, parse_mode='Markdown')
            logger.info(f"Stop trading command executed by user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error in stop trading command: {e}")
            await update.message.reply_text("❌ Error stopping bot. Please try again.")

    async def exit_position_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /exit_position command"""
        try:
            if not self.trading_bot:
                await update.message.reply_text("❌ Bot is not connected to trading engine.")
                return

            current_position = getattr(self.trading_bot, 'current_position', None)
            if not current_position:
                await update.message.reply_text("📭 *No Open Positions*\nNothing to close.", parse_mode='Markdown')
                return

            # Close position
            success = self.trading_bot.close_position('manual_exit')
            if success:
                await update.message.reply_text("✅ *Position Closed*\nPosition has been successfully closed.", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ *Error*\nFailed to close position. Please check logs.", parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in exit position command: {e}")
            await update.message.reply_text("❌ Error exiting position. Please try again.")

    def notify_trade_entry(self, trade_ Dict[str, Any]):
        """Notify about trade entry"""
        if not self.is_running or not self.app:
            return

        message = f"""🟢 *Trade Entry*

• *Direction:* {trade_data.get('direction', 'N/A')}
• *Price:* ₹{trade_data.get('entry_price', 0):.2f}
• *Quantity:* {trade_data.get('quantity', 0)}
• *Stop Loss:* ₹{trade_data.get('stop_loss', 0):.2f}
• *Target:* ₹{trade_data.get('target', 0):.2f}
• *Time:* {trade_data.get('timestamp', 'N/A')}
"""
        # Schedule the coroutine to be run
        if self.app:
             task = asyncio.create_task(self._send_message_to_all(message))
             self._background_tasks.add(task)
             task.add_done_callback(self._background_tasks.discard)

    def notify_trade_exit(self, trade_ Dict[str, Any]):
        """Notify about trade exit"""
        if not self.is_running or not self.app:
            return

        message = f"""🔴 *Trade Exit*

• *Direction:* {trade_data.get('direction', 'N/A')}
• *Exit Price:* ₹{trade_data.get('exit_price', 0):.2f}
• *P&L:* {format_currency(trade_data.get('pnl', 0))}
• *Duration:* {trade_data.get('duration', 'N/A')}
"""
        # Schedule the coroutine to be run
        if self.app:
            task = asyncio.create_task(self._send_message_to_all(message))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    def notify_circuit_breaker(self, consecutive_losses: int, pause_minutes: int):
        """Notify about circuit breaker activation"""
        if not self.is_running or not self.app:
            return

        message = f"""🚨 *Circuit Breaker Activated!*

• *Consecutive Losses:* {consecutive_losses}
• *Pause Duration:* {pause_minutes} minutes
• *Trading will resume at:* {datetime.now() + timedelta(minutes=pause_minutes):%H:%M:%S} IST

Bot is temporarily paused to prevent further losses.
"""
        # Schedule the coroutine to be run
        if self.app:
            task = asyncio.create_task(self._send_message_to_all(message))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    # Methods to fix the original errors and the new threading issue
    async def start_bot(self):
        """Start the Telegram bot"""
        try:
            if not Config.TELEGRAM_BOT_TOKEN:
                logger.warning("Telegram bot token not configured")
                return

            self.app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()

            # Register command handlers
            self.app.add_handler(CommandHandler("start", self.start_command))
            self.app.add_handler(CommandHandler("help", self.help_command))
            self.app.add_handler(CommandHandler("status", self.status_command))
            self.app.add_handler(CommandHandler("config", self.config_command))
            self.app.add_handler(CommandHandler("performance", self.performance_command))
            self.app.add_handler(CommandHandler("positions", self.positions_command))
            self.app.add_handler(CommandHandler("start_trading", self.start_trading_command))
            self.app.add_handler(CommandHandler("stop_trading", self.stop_trading_command)) # Updated name
            self.app.add_handler(CommandHandler("exit_position", self.exit_position_command))

            self.is_running = True
            logger.info("Telegram bot started")
            # Send startup notification once bot is ready
            startup_msg = f"🚀 *Nifty Scalper Bot v2.0 Started!*\n" \
                          f"• *Market Status:* {get_market_status()}\n" \
                          f"• *Auto-trading:* {'✅ ON' if getattr(self.trading_bot, 'auto_trade', False) else '❌ OFF'}\n" \
                          f"• *Balance:* ₹{getattr(getattr(self.trading_bot, 'risk_manager', None), 'current_balance', 0):,.2f}\n" \
                          f"• *Mode:* 💰 LIVE TRADING"
            # Send the startup message
            if self.app:
                await self._send_message_to_all(startup_msg)

            # Start polling (without signal handling issues)
            await self.app.updater.start_polling()
            logger.info("Telegram bot polling started")

            # Wait until stop is requested
            await self._stop_event.wait()
            logger.info("Stop event received, stopping Telegram bot...")

            # Stop the updater gracefully
            await self.app.updater.stop()
            logger.info("Telegram bot updater stopped")
            self.is_running = False

        except Exception as e:
            logger.error(f"Error starting Telegram bot: {e}", exc_info=True)
            self.is_running = False

    async def stop_bot(self):
        """Signal the Telegram bot to stop"""
        if self.is_running:
            logger.info("Setting stop event for Telegram bot")
            self._stop_event.set() # Signal the bot loop to stop
        else:
            logger.info("Telegram bot is not running, nothing to stop")

    async def _send_message_to_all(self, message: str):
        """Internal method to send message (placeholder)"""
        # This is a placeholder. In a real implementation, you'd track user chat IDs
        # and send messages to them.
        logger.info(f"Broadcasting message: {message}")
        # Example of how you might send to specific users (requires storing chat IDs):
        # for chat_id in self.registered_chat_ids:
        #     try:
        #         await self.app.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
        #     except Exception as e:
        #         logger.error(f"Failed to send message to {chat_id}: {e}")

# Example usage (if run as script)
if __name__ == "__main__":
    # This would typically be run from the main bot
    pass
