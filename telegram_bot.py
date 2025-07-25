#!/usr/bin/env python3
"""
Telegram Bot Module for Nifty Scalper Bot v4.0
Aligned with the new consolidated configuration and advanced trading logic.
"""
import asyncio
import logging
from typing import Optional, Dict, Any, Set
from datetime import datetime, timedelta
import pytz

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ALIGNED: Imports the final Config class and necessary utils
from config import Config
from utils import (
    is_market_open,
    get_market_status,
    time_until_market_open,
    get_market_session_info,
    format_currency,
    format_percentage,
    calculate_performance_metrics # Assuming this is in your utils
)

logger = logging.getLogger(__name__)
IST = pytz.timezone('Asia/Kolkata')

class TelegramBot:
    """Enhanced Telegram bot for trading commands and notifications"""

    def __init__(self, trading_bot_instance=None):
        self.trading_bot = trading_bot_instance
        self.app: Optional[Application] = None
        self.is_running = False
        self._background_tasks: Set[asyncio.Task] = set()
        self._stop_event = asyncio.Event()
        self.registered_chat_ids: Set[int] = set()

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command, register user for notifications, and show status."""
        try:
            if update.effective_chat:
                self.registered_chat_ids.add(update.effective_chat.id)
                logger.info(f"Registered chat_id: {update.effective_chat.id} for notifications.")

            market_info = get_market_session_info()
            auto_trade_status = '✅ ON' if not Config.DRY_RUN else '❌ OFF (Dry Run)'
            
            start_message = f"""🚀 *Nifty Scalper Bot v4.0 Started!*

*⚙️ Current Status:*
• *Mode:* {'💰 LIVE TRADING' if not Config.DRY_RUN else '🔬 DRY RUN'}
• *Auto-trading:* {auto_trade_status}
• *Market:* {get_market_status()}
• *Time:* {market_info.get("current_time", "Unknown")} IST

Notifications have been enabled for this chat. Use /help to see all commands.
"""
            await update.message.reply_text(start_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in start command: {e}", exc_info=True)
            await update.message.reply_text("❌ Error processing /start command.")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """*🤖 Nifty Scalper Bot Commands:*

/start - Start the bot and show status
/status - Show detailed bot status
/config - Show the bot's current trading configuration
/performance - Show today's performance metrics
/positions - Show current open positions or GTT orders
/start_trading - Enable auto-trading (if in live mode)
/stop_trading - Disable auto-trading (enter dry run mode)
/exit_position - Manually cancel the current GTT order
/help - Show this help message
"""
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command with a comprehensive dashboard."""
        try:
            if not self.trading_bot:
                await update.message.reply_text("❌ Bot is not connected to the trading engine.")
                return

            risk_manager = self.trading_bot.risk_manager
            current_position = self.trading_bot.current_position
            market_info = get_market_session_info()
            market_status = get_market_status()
            auto_trade_status = "✅ ON" if self.trading_bot.auto_trade else "❌ OFF"

            market_extra = ""
            if not is_market_open():
                time_left = time_until_market_open()
                if time_left.total_seconds() > 0:
                    hours, rem = divmod(int(time_left.total_seconds()), 3600)
                    minutes, _ = divmod(rem, 60)
                    market_extra = f" (Opens in {hours}h {minutes}m)"

            circuit_breaker_info = "🟢 *Circuit Breaker:* Inactive"
            if risk_manager.circuit_breaker_active:
                until = risk_manager.circuit_breaker_until
                if until:
                    mins = max(0, int((until - datetime.now(IST)).total_seconds() / 60))
                    circuit_breaker_info = f"🚨 *Circuit Breaker:* Active ({mins}m remaining)"

            position_text = "💤 No active trades"
            if current_position:
                instrument = current_position.get('instrument', 'N/A')
                qty = current_position.get('quantity', 0)
                entry = current_position.get('entry_price', 0)
                target = current_position.get('target', 0)
                sl = current_position.get('stop_loss', 0)
                position_text = f"🔥 *Active GTT Order*\n• *Instrument:* {instrument}\n• *Qty:* {qty}\n• *Entry:* {format_currency(entry)}\n• *Target:* {format_currency(target)}\n• *Stop-Loss:* {format_currency(sl)}"

            pnl_emoji = "📈" if risk_manager.todays_pnl >= 0 else "📉"
            
            status_message = f"""*🔄 Bot Status Dashboard*

*💼 Trading Status:*
• *Mode:* {'💰 LIVE' if not Config.DRY_RUN else '🔬 DRY RUN'}
• *Auto-trading:* {auto_trade_status}
• *Market:* {market_status}{market_extra}
{circuit_breaker_info}

*📊 Today's Performance:*
• *Balance:* {format_currency(risk_manager.current_balance)}
• *P&L:* {pnl_emoji} {format_currency(risk_manager.todays_pnl)}
• *Trades:* {risk_manager.daily_trades}

*📍 Current Position:*
{position_text}
"""
            await update.message.reply_text(status_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in status command: {e}", exc_info=True)
            await update.message.reply_text("❌ Error getting status.")

    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /config command with the new, relevant configuration.
        This function is now aligned with the final config.py.
        """
        try:
            rr_ratio = Config.ATR_TP_MULT / Config.ATR_SL_MULT if Config.ATR_SL_MULT > 0 else 0
            config_message = f"""*⚙️ Bot Configuration Overview*

*Capital & Risk:*
• *Initial Capital:* {format_currency(Config.INITIAL_CAPITAL)}
• *Risk Per Trade:* {format_percentage(Config.RISK_PER_TRADE_PCT * 100)}
• *Max Daily Loss:* {format_percentage(Config.MAX_DAILY_LOSS_PCT * 100)}
• *Max Daily Trades:* {Config.MAX_DAILY_TRADES}
• *Consecutive Loss Limit:* {Config.MAX_CONSECUTIVE_LOSSES} trades

*Trade Execution:*
• *ATR SL Multiplier:* {Config.ATR_SL_MULT}x
• *ATR TP Multiplier:* {Config.ATR_TP_MULT}x (RR Ratio: 1:{rr_ratio:.1f})
• *Nifty Lot Size:* {Config.NIFTY_LOT_SIZE}

*Technical Indicators:*
• *Fast/Slow EMA:* {Config.EMA_FAST}/{Config.EMA_SLOW}
• *RSI Period:* {Config.RSI_PERIOD}
• *ATR Period:* {Config.ATR_PERIOD}

*Operational:*
• *Dry Run Mode:* {'✅ ON' if Config.DRY_RUN else '❌ OFF'}
• *Tick Interval:* {Config.TICK_INTERVAL_SECONDS} seconds
"""
            await update.message.reply_text(config_message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error in config command: {e}", exc_info=True)
            await update.message.reply_text("❌ Error getting configuration.")

    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show today's performance metrics."""
        # Example: Replace these lines with your real implementation using your trading bot's risk manager/data
        if not self.trading_bot or not hasattr(self.trading_bot, "risk_manager"):
            await update.message.reply_text("❌ Performance data is currently unavailable.")
            return

        risk_manager = self.trading_bot.risk_manager
        msg = f"""*Today's Performance Metrics:*
• *Trades:* {risk_manager.daily_trades}
• *Gross P&L:* {format_currency(risk_manager.todays_pnl)}
• *Net P&L (after costs):* {format_currency(getattr(risk_manager, "todays_net_pnl", risk_manager.todays_pnl))}
• *Win Rate:* {getattr(risk_manager, "win_rate", 'N/A')}%"""
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current open positions or GTT orders."""
        if not self.trading_bot or not self.trading_bot.current_position:
            await update.message.reply_text("📭 No open positions or GTT orders at the moment.", parse_mode='Markdown')
            return

        pos = self.trading_bot.current_position
        msg = (
            f"*Open GTT Order:*\n"
            f"• *Instrument:* {pos.get('instrument', 'N/A')}\n"
            f"• *Qty:* {pos.get('quantity', 0)}\n"
            f"• *Entry:* {format_currency(pos.get('entry_price', 0))}\n"
            f"• *Target:* {format_currency(pos.get('target', 0))}\n"
            f"• *Stop-loss:* {format_currency(pos.get('stop_loss', 0))}"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def start_trading_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enables auto-trading if not in dry run mode."""
        if Config.DRY_RUN:
            await update.message.reply_text("⚠️ Cannot start trading. Bot is in Dry Run mode.", parse_mode='Markdown')
            return
        if self.trading_bot:
            self.trading_bot.auto_trade = True
            logger.info(f"Auto-trading enabled by user {update.effective_user.id}")
            await update.message.reply_text("✅ *Auto-trading STARTED*. The bot will now execute live trades.", parse_mode='Markdown')

    async def stop_trading_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Disables auto-trading."""
        if self.trading_bot:
            self.trading_bot.auto_trade = False
            logger.info(f"Auto-trading disabled by user {update.effective_user.id}")
            await update.message.reply_text("🛑 *Auto-trading STOPPED*. The bot will not execute new trades.", parse_mode='Markdown')

    async def exit_position_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manually cancels the current open GTT order."""
        try:
            if not self.trading_bot or not self.trading_bot.current_position:
                await update.message.reply_text("📭 *No Open Positions/GTT orders to exit.*", parse_mode='Markdown')
                return

            await update.message.reply_text("⏳ Attempting to cancel GTT order, please wait...", parse_mode='Markdown')
            
            success = await self.trading_bot.manual_exit_position()
            
            if success:
                await update.message.reply_text("✅ *GTT Order Cancelled Successfully*", parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ *Error*\nFailed to cancel GTT order. Please check logs.", parse_mode='Markdown')
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
        """Notify about a new GTT order being placed."""
        message = f"""🟢 *New GTT Order Placed*
• *Instrument:* {trade_data.get("instrument", "N/A")}
• *Qty:* {trade_data.get("quantity", 0)}
• *Entry Price:* {format_currency(trade_data.get("entry_price", 0))}
• *Target:* {format_currency(trade_data.get("target", 0))}
• *Stop-Loss:* {format_currency(trade_data.get("stop_loss", 0))}
"""
        self._schedule_notification(message)

    def notify_trade_exit(self, trade_data: Dict[str, Any]):
        """Notify about a trade being closed."""
        pnl = trade_data.get('pnl', 0)
        pnl_emoji = "✅" if pnl >= 0 else "🔻"
        message = f"""🔴 *Trade Closed*
• *Instrument:* {trade_data.get("instrument", "N/A")}
• *Exit Price:* {format_currency(trade_data.get("exit_price", 0))}
• *P&L:* {pnl_emoji} {format_currency(pnl)}
• *Reason:* {trade_data.get("reason", "N/A")}
"""
        self._schedule_notification(message)

    def notify_circuit_breaker(self, consecutive_losses: int, pause_minutes: int):
        """Notify about circuit breaker activation."""
        resume_time = (datetime.now(IST) + timedelta(minutes=pause_minutes)).strftime('%H:%M:%S')
        message = f"""🚨 *Circuit Breaker Activated!*
• *Consecutive Losses:* {consecutive_losses}
• *Trading Paused For:* {pause_minutes} minutes
• *Trading will resume at:* {resume_time} IST
"""
        self._schedule_notification(message)

    async def _send_message_to_all(self, message: str):
        """Internal method to send a message to all registered chat IDs."""
        if not self.app: return
        for chat_id in self.registered_chat_ids:
            try:
                await self.app.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Failed to send message to chat_id {chat_id}: {e}")

    async def start_bot(self):
        """Initializes and starts the Telegram bot."""
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
            await asyncio.sleep(1)

            self.is_running = True
            logger.info("Telegram bot started and polling.")
            
            # Send a startup message to the admin if specified
            if Config.TELEGRAM_ADMIN_CHAT_ID:
                self.registered_chat_ids.add(Config.TELEGRAM_ADMIN_CHAT_ID)
                await self._send_message_to_all(f"🚀 *Bot is online and running.* (Mode: {'Live' if not Config.DRY_RUN else 'Dry Run'})")

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
