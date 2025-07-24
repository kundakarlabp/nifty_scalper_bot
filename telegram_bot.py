import logging
import asyncio
from typing import Optional, Dict, Any
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from config import Config

logger = logging.getLogger(__name__)

class TelegramBot:
    """Telegram bot for trading commands and notifications"""
    
    def __init__(self, trading_bot_instance=None):
        self.trading_bot = trading_bot_instance
        self.app = None
        self.is_running = False
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        start_message = """
🚀 *Nifty Scalper Bot v2.0 Started!*

*⚙️ Configuration:*
• Mode: 💰 LIVE TRADING
• Auto-trading: ✅ ON
• Market: 🟢 OPEN

Use /help to see all available commands.
"""
        await update.message.reply_text(start_message, parse_mode='Markdown')
    
    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command"""
        if self.trading_bot:
            self.trading_bot.auto_trade = False
            message = "🛑 Auto-trading *STOPPED*\n\nThe bot will no longer execute new trades."
        else:
            message = "Bot is not connected to trading engine."
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not self.trading_bot:
            await update.message.reply_text("Bot is not connected to trading engine.")
            return
        
        # Get live data from the bot
        balance = getattr(self.trading_bot.risk_manager, 'current_balance', 0)
        todays_pnl = getattr(self.trading_bot.risk_manager, 'todays_pnl', 0)
        current_position = getattr(self.trading_bot, 'current_position', None)
        trade_history = getattr(self.trading_bot, 'trade_history', [])
        today_trades = len([t for t in trade_history if t.get('entry_time', '').startswith('2024')]) # Simple date check
        auto_trade_status = '✅ ON' if getattr(self.trading_bot, 'auto_trade', False) else '❌ OFF'
        
        # Determine position status text
        position_text = "💤 No active trades."
        if current_position:
            position_text = f"🔥 *Active Trade:* {current_position.get('direction', 'N/A')}"

        # Format the new status message
        status_message = f"""
*🔄 Bot Status:*

• *Mode:* 💰 LIVE TRADING
• *Auto-trading:* {auto_trade_status}
• *Market:* 🟢 OPEN
• *Today's trades:* {today_trades}/{Config.MAX_DAILY_TRADES}
• *Today's P&L:* ₹{todays_pnl:,.2f}

{position_text}
"""
        await update.message.reply_text(status_message, parse_mode='Markdown')

    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command"""
        config_message = f"""
*⚙️ Bot Configuration:*
• *Signal Threshold:* {Config.SIGNAL_THRESHOLD}
• *Max Daily Trades:* {Config.MAX_DAILY_TRADES}
• *Max Daily Loss:* ₹{Config.MAX_DAILY_LOSS_PCT * 1000}
• *Market Hours:* {Config.MARKET_START_HOUR}:{Config.MARKET_START_MINUTE:02d} AM - {Config.MARKET_END_HOUR - 12}:{Config.MARKET_END_MINUTE:02d} PM IST
• *Telegram Mode:* Polling
"""
        await update.message.reply_text(config_message, parse_mode='Markdown')

    async def position_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /position command"""
        if not self.trading_bot:
            await update.message.reply_text("Bot is not connected to trading engine.")
            return
        
        current_position = getattr(self.trading_bot, 'current_position', None)
        
        if not current_position:
            await update.message.reply_text("📭 No open positions")
            return
        
        position_message = f"""
📊 *Current Position*

🔹 *Direction:* {current_position.get('direction', 'N/A')}
💲 *Entry Price:* ₹{current_position.get('entry_price', 0):.2f}
📦 *Quantity:* {current_position.get('quantity', 0)}
🛑 *Stop Loss:* ₹{current_position.get('stop', 0):.2f}
🎯 *Target:* ₹{current_position.get('target', 0):.2f}
⏰ *Entry Time:* {current_position.get('entry_time', 'N/A')}
        """
        
        await update.message.reply_text(position_message, parse_mode='Markdown')
    
    async def exit_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /exit command"""
        if not self.trading_bot:
            await update.message.reply_text("Bot is not connected to trading engine.")
            return
        
        current_position = getattr(self.trading_bot, 'current_position', None)
        
        if not current_position:
            await update.message.reply_text("📭 No open positions to exit")
            return
        
        try:
            # Force exit current position
            success = await self._force_exit_position()
            
            if success:
                await update.message.reply_text("✅ Position closed successfully!")
            else:
                await update.message.reply_text("❌ Failed to close position. Please check manually.")
                
        except Exception as e:
            logger.error(f"Error in exit command: {e}")
            await update.message.reply_text(f"❌ Error closing position: {str(e)}")
    
    async def _force_exit_position(self) -> bool:
        """Force exit current position"""
        if not self.trading_bot or not hasattr(self.trading_bot, 'current_position'):
            return False
        
        current_position = self.trading_bot.current_position
        if not current_position:
            return False
        
        try:
            # Get opposite direction
            direction = current_position['direction']
            opposite_direction = 'SELL' if direction == 'BUY' else 'BUY'
            
            # Place market order to close position
            if hasattr(self.trading_bot, 'kite_client'):
                order_id = self.trading_bot.kite_client.place_order(
                    symbol=Config.UNDERLYING_SYMBOL,
                    transaction_type=opposite_direction,
                    quantity=current_position['quantity'],
                    order_type="MARKET"
                )
                
                if order_id:
                    # Clear current position
                    self.trading_bot.current_position = None
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error force exiting position: {e}")
            return False
    
    async def trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command - show recent trades"""
        if not self.trading_bot:
            await update.message.reply_text("Bot is not connected to trading engine.")
            return
        
        trade_history = getattr(self.trading_bot, 'trade_history', [])
        
        if not trade_history:
            await update.message.reply_text("📭 No trades executed yet")
            return
        
        # Show last 5 trades
        recent_trades = trade_history[-5:]
        
        trades_message = "📈 *Recent Trades*\n\n"
        
        for i, trade in enumerate(recent_trades, 1):
            pnl = trade.get('pnl', 0)
            emoji = "✅" if pnl > 0 else "❌"
            
            trades_message += f"""
{emoji} *Trade {i}*
Direction: {trade.get('direction', 'N/A')}
Entry: ₹{trade.get('entry_price', 0):.2f}
Exit: ₹{trade.get('exit_price', 0):.2f}
P&L: ₹{pnl:.2f}
Time: {trade.get('entry_time', 'N/A')}

"""
        
        await update.message.reply_text(trades_message, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_message = """
*📱 Available Commands:*

*🔄 Trading:*
/exit - Exit current trade
/stop - Stop auto-trading
/start - Re-enable auto-trading

*📊 Monitoring:*
/status - Current status
/config - Show bot configuration
/position - Show current open position
/trades - Show recent trades
/help - Show this help

*🕐 Market Hours:* 9:15 AM - 3:30 PM IST
*📅 Trading Days:* Monday to Friday
"""
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
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
        
        handlers = [
            CommandHandler("start", self.start_command),
            CommandHandler("stop", self.stop_command),
            CommandHandler("status", self.status_command),
            CommandHandler("position", self.position_command),
            CommandHandler("exit", self.exit_command),
            CommandHandler("trades", self.trades_command),
            CommandHandler("help", self.help_command),
            CommandHandler("config", self.config_command), # Added the new config command
        ]
        
        for handler in handlers:
            self.app.add_handler(handler)
        
        logger.info("Telegram handlers setup complete")
    
    async def start_bot(self):
        """Start the Telegram bot"""
        if not Config.TELEGRAM_BOT_TOKEN:
            logger.warning("Telegram bot token not provided - skipping Telegram bot")
            return
        
        try:
            # Create application
            self.app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
            
            # Setup handlers
            self.setup_handlers()
            
            # Start the bot
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling(drop_pending_updates=True)
            
            self.is_running = True
            logger.info("Telegram bot started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")
    
    async def stop_bot(self):
        """Stop the Telegram bot"""
        if self.app and self.is_running:
            try:
                await self.app.updater.stop()
                await self.app.stop()
                await self.app.shutdown()
                self.is_running = False
                logger.info("Telegram bot stopped")
            except Exception as e:
                logger.error(f"Error stopping Telegram bot: {e}")
    
    def notify_trade_entry(self, trade_data: Dict[str, Any]):
        """Send trade entry notification"""
        message = f"""
🚀 *Trade Entry Alert*

📊 *Direction:* {trade_data.get('direction', 'N/A')}
💲 *Entry Price:* ₹{trade_data.get('entry_price', 0):.2f}
📦 *Quantity:* {trade_data.get('quantity', 0)}
🛑 *Stop Loss:* ₹{trade_data.get('stop_loss', 0):.2f}
🎯 *Target:* ₹{trade_data.get('target', 0):.2f}
⏰ *Time:* {trade_data.get('timestamp', 'Now')}
        """
        
        asyncio.create_task(self.send_notification(message))
    
    def notify_trade_exit(self, trade_data: Dict[str, Any]):
        """Send trade exit notification"""
        pnl = trade_data.get('pnl', 0)
        emoji = "✅ Profit" if pnl > 0 else "❌ Loss"
        
        message = f"""
{emoji} *Trade Exit Alert*

📊 *Direction:* {trade_data.get('direction', 'N/A')}
💲 *Exit Price:* ₹{trade_data.get('exit_price', 0):.2f}
💰 *P&L:* ₹{pnl:.2f}
⏰ *Duration:* {trade_data.get('duration', 'N/A')}
        """
        
        asyncio.create_task(self.send_notification(message))
    
    def notify_circuit_breaker(self, loss_streak: int, pause_time: int):
        """Send circuit breaker notification"""
        message = f"""
🚨 *Circuit Breaker Activated*

📉 *Consecutive Losses:* {loss_streak}
⏱️ *Trading Paused For:* {pause_time} minutes
🛑 *Auto-trading will resume automatically*

Please review your strategy!
        """
        
        asyncio.create_task(self.send_notification(message))
