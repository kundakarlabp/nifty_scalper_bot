import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from config import Config
from utils import is_market_open, get_market_status, time_until_market_open

logger = logging.getLogger(__name__)

class TelegramBot:
    """Telegram bot for trading commands and notifications"""

    def __init__(self, trading_bot_instance=None):
        self.trading_bot = trading_bot_instance
        self.app: Optional[Application] = None
        self.is_running = False

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        status = get_market_status()
        mode = '✅ ON' if self.trading_bot and self.trading_bot.auto_trade else '❌ OFF'
        start_message = (
            "🚀 *Nifty Scalper Bot v2.0 Started!*\n\n"
            "⚙️ *Configuration:*\n"
            f"• Mode: 💰 LIVE TRADING\n"
            f"• Auto-trading: {mode}\n"
            f"• Market: {status}\n\n"
            "Use /help to see all available commands."
        )
        await update.message.reply_text(start_message, parse_mode='Markdown')

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        if self.trading_bot:
            self.trading_bot.auto_trade = False
            message = (
                "🛑 Auto-trading *STOPPED*\n\n"
                "The bot will no longer execute new trades."
            )
        else:
            message = "Bot is not connected to trading engine."
        await update.message.reply_text(message, parse_mode='Markdown')

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not self.trading_bot:
            await update.message.reply_text("Bot is not connected to trading engine.")
            return

        # Fetch metrics
        balance = getattr(self.trading_bot.risk_manager, 'current_balance', 0.0)
        todays_pnl = getattr(self.trading_bot.risk_manager, 'todays_pnl', 0.0)
        trade_history = getattr(self.trading_bot, 'trade_history', [])
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')
        today_trades = sum(1 for t in trade_history if t.get('entry_time', '').startswith(today_str))

        auto_status = '✅ ON' if self.trading_bot.auto_trade else '❌ OFF'
        market = get_market_status()
        next_open = '' if is_market_open() else f"\n• *Next Open:* {time_until_market_open()}"

        # Active position
        pos = getattr(self.trading_bot, 'current_position', None)
        if pos:
            direction = pos.get('direction', 'N/A')
            entry = pos.get('entry_price', 0)
            position_text = f"🔥 *Active Trade:* {direction} @ ₹{entry:.2f}"
        else:
            position_text = "💤 No active trades."

        status_message = (
            "*🔄 Bot Status:*\n\n"
            f"• *Mode:* 💰 LIVE TRADING\n"
            f"• *Auto-trading:* {auto_status}\n"
            f"• *Market:* {market}{next_open}\n"
            f"• *Today's trades:* {today_trades}/{Config.MAX_DAILY_TRADES}\n"
            f"• *Today's P&L:* ₹{todays_pnl:,.2f}\n"
            f"• *Balance:* ₹{balance:,.2f}\n\n"
            f"{position_text}"
        )
        await update.message.reply_text(status_message, parse_mode='Markdown')

    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command"""
        loss_pct = Config.MAX_DAILY_LOSS_PCT * 100
        start_hr = Config.MARKET_START_HOUR
        start_min = Config.MARKET_START_MINUTE
        end_hr = Config.MARKET_END_HOUR
        end_min = Config.MARKET_END_MINUTE
        config_message = (
            "*⚙️ Bot Configuration:*\n"
            f"• *Signal Threshold:* {Config.SIGNAL_THRESHOLD}\n"
            f"• *Max Daily Trades:* {Config.MAX_DAILY_TRADES}\n"
            f"• *Max Daily Loss:* {loss_pct:.1f}%\n"
            f"• *Market Hours:* {start_hr:02d}:{start_min:02d} - {end_hr:02d}:{end_min:02d} IST\n"
            f"• *Telegram Mode:* Polling"
        )
        await update.message.reply_text(config_message, parse_mode='Markdown')

    async def position_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /position command"""
        if not self.trading_bot:
            await update.message.reply_text("Bot is not connected to trading engine.")
            return

        pos = getattr(self.trading_bot, 'current_position', None)
        if not pos:
            await update.message.reply_text("📭 No open positions")
            return

        msg = (
            "📊 *Current Position*\n\n"
            f"🔹 *Direction:* {pos.get('direction', 'N/A')}\n"
            f"💲 *Entry Price:* ₹{pos.get('entry_price', 0):.2f}\n"
            f"📦 *Quantity:* {pos.get('quantity', 0)}\n"
            f"🛑 *Stop Loss:* ₹{pos.get('stop_loss', 0):.2f}\n"
            f"🎯 *Target:* ₹{pos.get('target', 0):.2f}\n"
            f"⏰ *Entry Time:* {pos.get('entry_time', 'N/A')}"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def exit_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /exit command"""
        if not self.trading_bot:
            await update.message.reply_text("Bot is not connected to trading engine.")
            return

        if not getattr(self.trading_bot, 'current_position', None):
            await update.message.reply_text("📭 No open positions to exit")
            return

        try:
            success = await self._force_exit_position()
            text = "✅ Position closed successfully!" if success else "❌ Failed to close position. Please check manually."
            await update.message.reply_text(text)
        except Exception as e:
            logger.error(f"Error in exit command: {e}")
            await update.message.reply_text(f"❌ Error closing position: {e}")

    async def _force_exit_position(self) -> bool:
        """Force exit current position"""
        pos = getattr(self.trading_bot, 'current_position', None)
        if not pos or not hasattr(self.trading_bot, 'kite_client'):
            return False

        direction = pos['direction']
        opposite = 'SELL' if direction == 'BUY' else 'BUY'
        try:
            resp = self.trading_bot.kite_client.place_order(
                symbol=Config.UNDERLYING_SYMBOL,
                transaction_type=opposite,
                quantity=pos['quantity'],
                order_type="MARKET"
            )
            if resp:
                self.trading_bot.current_position = None
                return True
        except Exception as e:
            logger.error(f"Error force exiting position: {e}")
        return False

    async def trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command - show recent trades"""
        if not self.trading_bot:
            await update.message.reply_text("Bot is not connected to trading engine.")
            return

        history = getattr(self.trading_bot, 'trade_history', [])
        if not history:
            await update.message.reply_text("📭 No trades executed yet")
            return

        recent = history[-5:]
        msg = "📈 *Recent Trades*\n\n"
        for i, t in enumerate(recent, 1):
            pnl = t.get('pnl', 0)
            icon = '✅' if pnl > 0 else '❌'
            msg += (
                f"{icon} *Trade {i}*\n"
                f"Direction: {t.get('direction', 'N/A')}\n"
                f"Entry: ₹{t.get('entry_price', 0):.2f}\n"
                f"Exit: ₹{t.get('exit_price', 0):.2f}\n"
                f"P&L: ₹{pnl:.2f}\n"
                f"Time: {t.get('entry_time', 'N/A')}\n\n"
            )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = (
            "*📱 Available Commands:*\n\n"
            "*🔄 Trading:*\n"
            "/exit - Exit current trade\n"
            "/stop - Stop auto-trading\n"
            "/start - Re-enable auto-trading\n\n"
            "*📊 Monitoring:*\n"
            "/status - Current status\n"
            "/config - Show bot configuration\n"
            "/position - Show current open position\n"
            "/trades - Show recent trades\n"
            "/help - Show this help\n\n"
            "*🕐 Market Hours:* 9:15 AM - 3:30 PM IST\n"
            "*📅 Trading Days:* Monday to Friday"
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def send_notification(self, message: str, parse_mode: str = 'Markdown'):
        """Send notification to configured chat"""
        if not self.app or not Config.TELEGRAM_CHAT_ID:
            return
        try:
            await self.app.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    def setup_handlers(self):
        """Setup command handlers"""
        if not self.app:
            return
        for cmd, func in [
            ('start', self.start_command),
            ('stop', self.stop_command),
            ('status', self.status_command),
            ('position', self.position_command),
            ('exit', self.exit_command),
            ('trades', self.trades_command),
            ('help', self.help_command),
            ('config', self.config_command),
        ]:
            self.app.add_handler(CommandHandler(cmd, func))
        logger.info("Telegram handlers setup complete")

    async def start_bot(self):
        """Start the Telegram bot"""
        if not Config.TELEGRAM_BOT_TOKEN:
            logger.warning("Telegram bot token not provided - skipping Telegram bot")
            return
        self.app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        self.setup_handlers()
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        self.is_running = True
        logger.info("Telegram bot started successfully")

    async def stop_bot(self):
        """Stop the Telegram bot"""
        if self.app and self.is_running:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            self.is_running = False
            logger.info("Telegram bot stopped")

    def notify_trade_entry(self, trade_data: Dict[str, Any]):
        """Send trade entry notification"""
        message = (
            "🚀 *Trade Entry Alert*\n\n"
            f"📊 *Direction:* {trade_data.get('direction', 'N/A')}\n"
            f"💲 *Entry Price:* ₹{trade_data.get('entry_price', 0):.2f}\n"
            f"📦 *Quantity:* {trade_data.get('quantity', 0)}\n"
            f"🛑 *Stop Loss:* ₹{trade_data.get('stop_loss', 0):.2f}\n"
            f"🎯 *Target:* ₹{trade_data.get('target', 0):.2f}\n"
            f"⏰ *Time:* {trade_data.get('timestamp', 'Now')}"
        )
        asyncio.create_task(self.send_notification(message))

    def notify_trade_exit(self, trade_data: Dict[str, Any]):
        """Send trade exit notification"""
        pnl = trade_data.get('pnl', 0)
        emoji = '✅ Profit' if pnl > 0 else '❌ Loss'
        message = (
            f"{emoji} *Trade Exit Alert*\n\n"
            f"📊 *Direction:* {trade_data.get('direction', 'N/A')}\n"
            f"💲 *Exit Price:* ₹{trade_data.get('exit_price', 0):.2f}\n"
            f"💰 *P&L:* ₹{pnl:.2f}\n"
            f"⏰ *Duration:* {trade_data.get('duration', 'N/A')}"
        )
        asyncio.create_task(self.send_notification(message))

    def notify_circuit_breaker(self, loss_streak: int, pause_time: int):
        """Send circuit breaker notification"""
        message = (
            "🚨 *Circuit Breaker Activated*\n\n"
            f"📉 *Consecutive Losses:* {loss_streak}\n"
            f"⏱️ *Trading Paused For:* {pause_time} minutes\n"
            "🛑 *Auto-trading will resume automatically*\n\n"
            "Please review your strategy!"
        )
        asyncio.create_task(self.send_notification(message))
