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
        if self.trading_bot:
            self.trading_bot.auto_trade = True
            message = "🚀 Auto-trading *STARTED*\n\nThe bot will now execute trades automatically based on signals."
        else:
            message = "Bot is not connected to trading engine."
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
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
        
        # Get current status
        balance = getattr(self.trading_bot.risk_manager, 'current_balance', 0)
        peak_balance = getattr(self.trading_bot.risk_manager, 'peak_balance', 0)
        loss_streak = getattr(self.trading_bot.risk_manager, 'loss_streak', 0)
        circuit_breaker = getattr(self.trading_bot.risk_manager, 'circuit_breaker_active', False)
        
        # Calculate drawdown
        drawdown = 0
        if peak_balance > 0:
            drawdown = ((peak_balance - balance) / peak_balance) * 100
        
        # Get current position
        current_position = getattr(self.trading_bot, 'current_position', None)
        
        # Get today's trades
        trade_history = getattr(self.trading_bot, 'trade_history', [])
        today_trades = len([t for t in trade_history if t.get('entry_time', '').startswith('2024')])  # Simple date check
        
        status_message = f"""
📊 *Trading Bot Status*

💰 *Balance:* ₹{balance:,.2f}
📈 *Peak Balance:* ₹{peak_balance:,.2f}
📉 *Drawdown:* {drawdown:.2f}%

🔄 *Auto Trade:* {'✅ ON' if getattr(self.trading_bot, 'auto_trade', False) else '❌ OFF'}
🚨 *Circuit Breaker:* {'🔴 ACTIVE' if circuit_breaker else '🟢 INACTIVE'}
📊 *Loss Streak:* {loss_streak}

📋 *Current Position:* {current_position['direction'] if current_position else 'None'}
📝 *Today\'s Trades:* {today_trades}

⏰ *Last Update:* Just now
        """
        
        await update.message.reply_text(status_message, parse_mode='Markdown')
    
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
🤖 *Nifty Scalper Bot Commands*

/start - Start auto-trading
/stop - Stop auto-trading  
/status - Show bot status
/position - Show current position
/exit - Force close current position
/trades - Show recent trades
/help - Show this help message

⚠️ *Important Notes:*
• Always monitor your positions
• Use /stop before market close
• Check /status regularly
• Keep sufficient margin

📞 *Support:* Contact your administrator for issues
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