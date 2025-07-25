#!/usr/bin/env python3
"""
Telegram Bot Module for Nifty Scalper Bot v2.0
Enhanced with proper market timing and auto-trading controls
"""
import logging
import asyncio
from typing import Optional, Dict, Any, Set
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
    format_percentage
)

logger = logging.getLogger(__name__)

class TelegramBot:
    """Enhanced Telegram bot for trading commands and notifications"""

    def __init__(self, trading_bot_instance=None):
        self.trading_bot = trading_bot_instance
        self.app: Optional[Application] = None
        self.is_running = False
        self._background_tasks: Set[asyncio.Task] = set()
        self._stop_event = asyncio.Event()
        # Store chat IDs to send notifications to users
        self.registered_chat_ids: Set[int] = set()

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command, register user for notifications, and show status."""
        try:
            # Register the user's chat ID for notifications
            if update.effective_chat:
                self.registered_chat_ids.add(update.effective_chat.id)
                logger.info(f"Registered chat_id: {update.effective_chat.id} for notifications.")

            market_info = get_market_session_info()
            auto_trade_status = '✅ ON' if getattr(self.trading_bot, 'auto_trade', False) else '❌ OFF'
            
            start_message = f"""🚀 *Nifty Scalper Bot v2.0 Started!*

*⚙️ Current Status:*
• Mode: 💰 LIVE TRADING
• Auto-trading: {auto_trade_status}
• Market: {get_market_status()}
• Time: {market_info.get("current_time", "Unknown")} IST

Notifications have been enabled for this chat. Use /help to see all commands.
"""
            await update.message.reply_text(start_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in start command: {e}", exc_info=True)
            await update.message.reply_text("❌ Error processing /start command. Please try again.")

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
            auto_trade_status = "✅ ON" if getattr(self.trading_bot, 'auto_trade', False) else "❌ OFF"
            risk_manager = getattr(self.trading_bot, 'risk_manager', None)
            current_position = getattr(self.trading_bot, 'current_position', None)

            market_extra = ""
            if not is_market_open():
                time_until_open = time_until_market_open()
                if time_until_open:
                    hours, rem = divmod(int(time_until_open.total_seconds()), 3600)
                    minutes, _ = divmod(rem, 60)
                    market_extra = f" (Opens in {hours}h {minutes}m)"

            circuit_breaker_info = "🟢 *Circuit Breaker:* Inactive"
            if risk_manager and getattr(risk_manager, 'circuit_breaker_active', False):
                until = getattr(risk_manager, 'circuit_breaker_until', None)
                if until:
                    mins = max(0, int((until - datetime.now()).total_seconds() / 60))
                    circuit_breaker_info = f"🚨 *Circuit Breaker:* Active ({mins}m remaining)"

            position_text = "💤 No active trades"
            if current_position:
                direction = current_position.get('direction', 'N/A')
                entry_price = current_position.get('entry_price', 0)
                qty = current_position.get('quantity', 0)
                unrealized_pnl = ""
                try:
                    md = self.trading_bot.get_market_data()
                    if md and 'ltp' in md:
                        cp = md['ltp']
                        pnl = (cp - entry_price) * qty if direction == 'BUY' else (entry_price - cp) * qty
                        unrealized_pnl = f" | P&L: {format_currency(pnl)}"
                except Exception as e:
                    logger.warning(f"Could not fetch market data for P&L: {e}")

                position_text = f"🔥 *{direction}* {qty} @ ₹{entry_price:.2f}{unrealized_pnl}"

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
• *Time:* {market_info.get("current_time", "Unknown")} IST
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
            logger.error(f"Error in status command: {e}", exc_info=True)
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
• *Current Day:* {market_session.get("day_of_week", "Unknown")}

*🛡️ Risk Management:*
• *Max Consecutive Losses:* {Config.MAX_CONSECUTIVE_LOSSES}
• *Circuit Breaker Pause:* {Config.CIRCUIT_BREAKER_PAUSE_MINUTES} minutes
• *Position Sizing:* Dynamic (Risk-based)
"""
            await update.message.reply_text(config_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in config command: {e}", exc_info=True)
            await update.message.reply_text("❌ Error getting configuration.")

    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command with metrics"""
        try:
            risk_manager = getattr(self.trading_bot, 'risk_manager', None)
            if not risk_manager:
                await update.message.reply_text("❌ Risk manager not available.")
                return

            metrics = calculate_performance_metrics(
                risk_manager.current_balance,
                risk_manager.initial_balance,
                risk_manager.todays_pnl
            )
            circuit_breaker_status = 'Active' if risk_manager.circuit_breaker_active else 'Inactive'
            perf_message = f"""*🏆 Performance Metrics*

*💰 Capital:*
• *Current Balance:* {format_currency(risk_manager.current_balance)}
• *Initial Capital:* {format_currency(risk_manager.initial_balance)}
• *Net P&L:* {format_currency(risk_manager.current_balance - risk_manager.initial_balance)}

*📊 Today's Trading:*
• *P&L:* {format_currency(risk_manager.todays_pnl)}
• *Trades Executed:* {risk_manager.daily_trades}
• *Win Rate:* {format_percentage(metrics.get("win_rate", 0))}
• *Profit Factor:* {metrics.get("profit_factor", 0):.2f}

*🛡️ Risk Metrics:*
• *Consecutive Losses:* {risk_manager.consecutive_losses}
• *Circuit Breaker:* {circuit_breaker_status}
"""
            await update.message.reply_text(perf_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in performance command: {e}", exc_info=True)
            await update.message.reply_text("❌ Error getting performance metrics.")

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        try:
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
            unrealized_pnl_text = ""
            try:
                md = self.trading_bot.get_market_data()
                if md and 'ltp' in md:
                    cp = md['ltp']
                    pnl = (cp - entry_price) * quantity if direction == 'BUY' else (entry_price - cp) * quantity
                    unrealized_pnl_text = f"\n• *Unrealized P&L:* {format_currency(pnl)}"
            except Exception as e:
                logger.warning(f"Could not fetch market data for P&L: {e}")

            position_message = f"""*📍 Current Position*
• *Direction:* {direction}
• *Quantity:* {quantity}
• *Entry Price:* ₹{entry_price:.2f}
• *Stop Loss:* ₹{stop_loss:.2f}
• *Target:* ₹{target:.2f}
• *Entry Time:* {entry_time}{unrealized_pnl_text}
"""
            await update.message.reply_text(position_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in positions command: {e}", exc_info=True)
            await update.message.reply_text("❌ Error getting position details.")

    async def start_trading_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start_trading command"""
        if self.trading_bot:
            self.trading_bot.auto_trade = True
            logger.info(f"Auto-trading enabled by user {update.effective_user.id}")
            await update.message.reply_text("✅ *Auto-trading STARTED*\nNew trades will be executed automatically.", parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ Bot is not connected to trading engine.", parse_mode='Markdown')

    async def stop_trading_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop_trading command"""
        if self.trading_bot:
            self.trading_bot.auto_trade = False
            logger.info(f"Auto-trading disabled by user {update.effective_user.id}")
            await update.message.reply_text("🛑 *Auto-trading STOPPED*\nNew trades will not be executed.", parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ Bot is not connected to trading engine.", parse_mode='Markdown')

    async def exit_position_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /exit_position command"""
        try:
            if not self.trading_bot:
                await update.message.reply_text("❌ Bot is not connected to trading engine.")
                return
            if not getattr(self.trading_bot, 'current_position', None):
                await update.message.reply_text("📭 *No Open Positions*\nNothing to close.", parse_mode='Markdown')
                return

            await update.message.reply_text("⏳ Closing position, please wait...", parse_mode='Markdown')
            # Ensure close_position in your main bot is an async function
            success = await self.trading_bot.close_position('manual_exit')
            
            if success:
                await update.message.reply_text("✅ *Position Closed Successfully*", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ *Error*\nFailed to close position. Please check logs.", parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in exit position command: {e}", exc_info=True)
            await update.message.reply_text("❌ An error occurred while exiting the position.")

    def _schedule_notification(self, message: str):
        """Helper to schedule the sending of a notification."""
        if self.is_running and self.app:
            task = asyncio.create_task(self._send_message_to_all(message))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    def notify_trade_entry(self, trade_data: Dict[str, Any]):
        """Notify about trade entry"""
        message = f"""🟢 *Trade Entry*
• *Direction:* {trade_data.get("direction", "N/A")}
• *Price:* {format_currency(trade_data.get("entry_price", 0))}
• *Quantity:* {trade_data.get("quantity", 0)}
• *Stop Loss:* {format_currency(trade_data.get("stop_loss", 0))}
• *Target:* {format_currency(trade_data.get("target", 0))}
"""
        self._schedule_notification(message)

    def notify_trade_exit(self, trade_data: Dict[str, Any]):
        """Notify about trade exit"""
        pnl = trade_data.get('pnl', 0)
        pnl_emoji = "✅" if pnl >= 0 else "🔻"
        message = f"""🔴 *Trade Exit*
• *Direction:* {trade_data.get("direction", "N/A")}
• *Exit Price:* {format_currency(trade_data.get("exit_price", 0))}
• *P&L:* {pnl_emoji} {format_currency(pnl)}
"""
        self._schedule_notification(message)

    def notify_circuit_breaker(self, consecutive_losses: int, pause_minutes: int):
        """Notify about circuit breaker activation"""
        resume_time = (datetime.now() + timedelta(minutes=pause_minutes)).strftime('%H:%M:%S')
        message = f"""🚨 *Circuit Breaker Activated!*
• *Consecutive Losses:* {consecutive_losses}
• *Pause Duration:* {pause_minutes} minutes
• *Trading will resume at:* {resume_time} IST
"""
        self._schedule_notification(message)

    async def _send_message_to_all(self, message: str):
        """Internal method to send a message to all registered chat IDs."""
        if not self.app: return
        logger.info(f"Broadcasting message to {len(self.registered_chat_ids)} chats.")
        for chat_id in self.registered_chat_ids:
            try:
                await self.app.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Failed to send message to chat_id {chat_id}: {e}")

    async def start_bot(self):
        """Initializes and starts the Telegram bot, including polling."""
        try:
            if not Config.TELEGRAM_BOT_TOKEN:
                logger.warning("Telegram bot token not configured. Bot will not start.")
                return

            self.app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
            
            handlers = [
                CommandHandler("start", self.start_command), CommandHandler("help", self.help_command),
                CommandHandler("status", self.status_command), CommandHandler("config", self.config_command),
                CommandHandler("performance", self.performance_command), CommandHandler("positions", self.positions_command),
                CommandHandler("start_trading", self.start_trading_command), CommandHandler("stop_trading", self.stop_trading_command),
                CommandHandler("exit_position", self.exit_position_command),
            ]
            self.app.add_handlers(handlers)

            await self.app.initialize()
            await self.app.updater.start_polling()
            await asyncio.sleep(1) # Ensure bot is connected before sending startup message

            self.is_running = True
            logger.info("Telegram bot started and polling.")

            risk_manager = getattr(self.trading_bot, 'risk_manager', None)
            balance = getattr(risk_manager, 'current_balance', 0) if risk_manager else 0
            auto_trade_status = '✅ ON' if getattr(self.trading_bot, 'auto_trade', False) else '❌ OFF'
            startup_msg = (
                f"🚀 *Nifty Scalper Bot v2.0 Online!*\n"
                f"• *Market Status:* {get_market_status()}\n"
                f"• *Auto-trading:* {auto_trade_status}\n"
                f"• *Balance:* {format_currency(balance)}"
            )
            await self._send_message_to_all(startup_msg)

            await self._stop_event.wait()

        except Exception as e:
            logger.error(f"Fatal error starting Telegram bot: {e}", exc_info=True)
        finally:
            if self.app and self.app.updater:
                await self.app.updater.stop()
            self.is_running = False
            logger.info("Telegram bot has been shut down.")

    async def stop_bot(self):
        """Signals the Telegram bot to stop gracefully."""
        if self.is_running:
            logger.info("Stopping Telegram bot...")
            self._stop_event.set()
        else:
            logger.info("Telegram bot is not running.")
