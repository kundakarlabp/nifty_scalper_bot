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
        self.app = None
        self.is_running = False

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command with enhanced status"""
        try:
            market_info = get_market_session_info()
            start_message = f"""
🚀 *Nifty Scalper Bot v2.0 Started!* 

*⚙️ Current Status:* 
• Mode: 💰 LIVE TRADING 
• Auto-trading: {'✅ ON' if getattr(self.trading_bot, 'auto_trade', False) else '❌ OFF'} 
• Market: {market_info.get('market_status', 'Unknown')} 
• Current Time: {market_info.get('current_time', 'Unknown')} IST 

*📱 Quick Commands:* 
/status - View detailed status 
/help - Show all commands 

{f"• *Next Open:* {market_info.get('time_until_open')}" if not market_info.get('is_market_open') else ""} 

Ready to trade! 🎯
"""
            if self.trading_bot:
                self.trading_bot.auto_trade = True
            await update.message.reply_text(start_message, parse_mode='Markdown')
            logger.info(f"Start command executed by user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error in start command: {e}")
            await update.message.reply_text("❌ Error starting bot. Please try again.")

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command with confirmation"""
        try:
            if self.trading_bot:
                self.trading_bot.auto_trade = False
                current_position = getattr(self.trading_bot, 'current_position', None)
                position_warning = ""
                if current_position:
                    direction = current_position.get('direction', 'N/A')
                    entry_price = current_position.get('entry_price', 0)
                    position_warning = f"\n⚠️ *Warning:* Active {direction} position @ ₹{entry_price:.2f} will continue running!"
                message = f"""
🛑 *Auto-trading STOPPED* 

• New trades will not be executed 
• Existing positions will be monitored 
• Use /start to re-enable auto-trading 
• Use /exit to close current position{position_warning} 

Bot remains active for monitoring and manual commands.
"""
            else:
                message = "❌ Bot is not connected to trading engine."
            await update.message.reply_text(message, parse_mode='Markdown')
            logger.info(f"Stop command executed by user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error in stop command: {e}")
            await update.message.reply_text("❌ Error stopping auto-trading. Please try again.")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command with comprehensive information"""
        try:
            if not self.trading_bot:
                await update.message.reply_text("❌ Bot is not connected to trading engine.")
                return
            market_info = get_market_session_info()
            balance = getattr(self.trading_bot.risk_manager, 'current_balance', 0)
            todays_pnl = getattr(self.trading_bot.risk_manager, 'todays_pnl', 0)
            current_position = getattr(self.trading_bot, 'current_position', None)
            trade_history = getattr(self.trading_bot, 'trade_history', [])
            today = datetime.now().strftime('%Y-%m-%d')
            today_trades = len([t for t in trade_history if t.get('entry_time', '').startswith(today)])
            auto_trade_status = '✅ ON' if getattr(self.trading_bot, 'auto_trade', False) else '❌ OFF'
            market_status = market_info.get('market_status', 'Unknown')
            market_extra = ""
            if not market_info.get('is_market_open'):
                next_open = market_info.get('time_until_open')
                if next_open:
                    market_extra = f"\n• *Next Open:* {next_open}"
            risk_manager = self.trading_bot.risk_manager
            circuit_breaker_info = ""
            if getattr(risk_manager, 'circuit_breaker_active', False):
                remaining_time = getattr(risk_manager, 'circuit_breaker_until', None)
                if remaining_time:
                    mins = max(0, int((remaining_time - datetime.now()).total_seconds()/60))
                    circuit_breaker_info = f"\n🚨 *Circuit Breaker:* Active ({mins}m remaining)"
            position_text = "💤 No active trades"
            if current_position:
                direction = current_position.get('direction','N/A')
                entry_price = current_position.get('entry_price',0)
                qty = current_position.get('quantity',0)
                entry_time = current_position.get('entry_time','')
                unrealized_pnl = ""
                try:
                    md = self.trading_bot.get_market_data()
                    if md and 'ltp' in md:
                        cp = md['ltp']
                        pnl = (cp-entry_price)*qty if direction=='BUY' else (entry_price-cp)*qty
                        unrealized_pnl = f" | P&L: {format_currency(pnl)}"
                except: pass
                position_text = f"🔥 *{direction}* {qty} @ ₹{entry_price:.2f}{unrealized_pnl}"
            pnl_emoji = "📈" if todays_pnl>=0 else "📉"
            pnl_color = "+" if todays_pnl>=0 else ""
            status_message = f"""
*🔄 Bot Status Dashboard* 

*💼 Trading Status:* 
• *Mode:* 💰 LIVE TRADING 
• *Auto-trading:* {auto_trade_status} 
• *Market:* {market_status}{market_extra} 
• *Time:* {market_info.get('current_time','Unknown')} IST{circuit_breaker_info} 

*📊 Today's Performance:* 
• *Trades:* {today_trades}/{Config.MAX_DAILY_TRADES} 
• *P&L:* {pnl_emoji} {pnl_color}{format_currency(todays_pnl)} 
• *Balance:* {format_currency(balance)} 

*📈 Current Position:* 
{position_text} 

*🎯 Risk Limits:* 
• *Max Daily Loss:* {format_currency(Config.MAX_DAILY_LOSS_PCT * balance)} 
• *Consecutive Losses:* {getattr(risk_manager,'consecutive_losses',0)}/{Config.MAX_CONSECUTIVE_LOSSES} 

Last updated: {datetime.now().strftime('%H:%M:%S')} IST
"""
            await update.message.reply_text(status_message, parse_mode='Markdown')
            logger.info(f"Status command executed by user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await update.message.reply_text("❌ Error getting status. Please try again.")

    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /config command with detailed configuration"""
        try:
            market_session = get_market_session_info()
            config_message = f"""
*⚙️ Bot Configuration* 

*📈 Trading Parameters:* 
• *Signal Threshold:* {Config.SIGNAL_THRESHOLD} 
• *Risk Per Trade:* {format_percentage(Config.RISK_PER_TRADE_PCT * 100)} 
• *Max Daily Trades:* {Config.MAX_DAILY_TRADES} 
• *Max Daily Loss:* {format_percentage(Config.MAX_DAILY_LOSS_PCT * 100)} 

*🕒 Market Timings (IST):* 
• *Market Hours:* {Config.MARKET_START_HOUR}:{Config.MARKET_START_MINUTE:02d} AM - {Config.MARKET_END_HOUR}:{Config.MARKET_END_MINUTE:02d} PM 
• *Trading Days:* Monday to Friday 
• *Current Day:* {market_session.get('day_of_week','Unknown')} 

*🛡️ Risk Management:* 
• *Max Consecutive Losses:* {Config.MAX_CONSECUTIVE_LOSSES} 
• *Circuit Breaker Pause:* {Config.CIRCUIT_BREAKER_PAUSE_MINUTES} minutes 
• *Position Sizing:* Dynamic (Risk-based) 

*📱 Telegram Settings:* 
• *Notifications:* ✅ Enabled 
• *Commands:* ✅ Active 
• *Auto-restart:* ✅ Enabled 

*🎯 Trading Symbol:* {Config.UNDERLYING_SYMBOL}
"""
            await update.message.reply_text(config_message, parse_mode='Markdown')
            logger.info(f"Config command executed by user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error in config command: {e}")
            await update.message.reply_text("❌ Error getting configuration. Please try again.")

    async def position_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /position command with detailed position info"""
        try:
            if not self.trading_bot:
                await update.message.reply_text("❌ Bot is not connected to trading engine.")
                return
            current_position = getattr(self.trading_bot,'current_position',None)
            if not current_position:
                await update.message.reply_text("📭 *No Open Positions*\n\nAll positions are closed.",parse_mode='Markdown')
                return
            direction= current_position.get('direction','N/A')
            entry_price=current_position.get('entry_price',0)
            quantity=current_position.get('quantity',0)
            stop_loss=current_position.get('stop_loss',0)
            target=current_position.get('target',0)
            entry_time=current_position.get('entry_time','N/A')
            symbol=current_position.get('symbol',Config.UNDERLYING_SYMBOL)
            try:
                entry_dt=datetime.strptime(entry_time,'%Y-%m-%d %H:%M:%S')
                duration=format_trade_duration(entry_dt)
            except:
                duration="Unknown"
            current_price=0
            unrealized_pnl=0
            pnl_text=""
            try:
                md=self.trading_bot.get_market_data()
                if md and 'ltp' in md:
                    current_price=md['ltp']
                    unrealized_pnl=(current_price-entry_price)*quantity if direction=='BUY' else (entry_price-current_price)*quantity
                    pnl_emoji="📈" if unrealized_pnl>=0 else "📉"
                    pnl_sign="+" if unrealized_pnl>=0 else ""
                    pnl_text=f"\n💰 *Unrealized P&L:* {pnl_emoji} {pnl_sign}{format_currency(unrealized_pnl)}"
            except Exception as e:
                logger.error(f"Error calculating unrealized P&L: {e}")
            risk_amt=abs(entry_price-stop_loss)*quantity if stop_loss else 0
            reward_amt=abs(target-entry_price)*quantity if target else 0
            rr_ratio=reward_amt/risk_amt if risk_amt>0 else 0
            position_message=f"""
📊 *Current Position Details* 

*🔹 Position Info:* 
• *Symbol:* {symbol} 
• *Direction:* {direction} 
• *Quantity:* {quantity:,} 
• *Entry Price:* ₹{entry_price:.2f} 
• *Current Price:* ₹{current_price:.2f} 

*🎯 Levels:* 
• *Stop Loss:* ₹{stop_loss:.2f} ({format_currency(-risk_amt)} risk) 
• *Target:* ₹{target:.2f} ({format_currency(reward_amt)} reward) 
• *Risk:Reward:* 1:{rr_ratio:.2f} 

*⏰ Timing:* 
• *Entry Time:* {entry_time} 
• *Duration:* {duration}{pnl_text} 

Use /exit to close this position manually.
"""
            await update.message.reply_text(position_message,parse_mode='Markdown')
            logger.info(f"Position command executed by user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error in position command: {e}")
            await update.message.reply_text("❌ Error getting position details. Please try again.")

    async def exit_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /exit command with enhanced confirmation"""
        try:
            if not self.trading_bot:
                await update.message.reply_text("❌ Bot is not connected to trading engine.")
                return
            current_position=getattr(self.trading_bot,'current_position',None)
            if not current_position:
                await update.message.reply_text("📭 *No Open Positions*\n\nThere are no positions to exit.",parse_mode='Markdown')
                return
            if not is_market_open():
                ms=get_market_status()
                no=time_until_market_open()
                await update.message.reply_text(
                    f"❌ *Cannot Exit Position*\n\nMarket is currently {ms}\nNext open: {no}", parse_mode='Markdown'
                )
                return
            proc=await update.message.reply_text("⏳ *Processing exit order...*",parse_mode='Markdown')
            success=await self._force_exit_position()
            if success:
                await proc.edit_text("✅ *Position Closed Successfully!*\n\nThe position has been exited at market price.\nCheck /status for updated P&L.",parse_mode='Markdown')
                logger.info(f"Position manually exited by user {update.effective_user.id}")
            else:
                await proc.edit_text("❌ *Failed to Close Position*\n\nThere was an error closing the position.\nPlease check manually or try again.\n\nUse /position to check current status.",parse_mode='Markdown')
                logger.error(f"Manual exit failed for user {update.effective_user.id}")
        except Exception as e:
            logger.error(f"Error in exit command: {e}")
            await update.message.reply_text("❌ Error processing exit command. Please try again.")

    async def _force_exit_position(self) -> bool:
        """Enhanced force exit with proper error handling and logging"""
        if not self.trading_bot or not hasattr(self.trading_bot,'current_position'):
            return False
        cp=self.trading_bot.current_position
        if not cp:
            return False
        try:
            if not is_market_open():
                logger.warning("Cannot exit position - market is closed")
                return False
            direction=cp['direction']
            qty=cp['quantity']
            sym=cp.get('symbol',Config.UNDERLYING_SYMBOL)
            opp='SELL' if direction=='BUY' else 'BUY'
            logger.info(f"Attempting to force exit: {direction} {qty} {sym}")
            if hasattr(self.trading_bot,'kite_client') and self.trading_bot.kite_client:
                orr=self.trading_bot.kite_client.kite.place_order(
                    variety=self.trading_bot.kite_client.kite.VARIETY_REGULAR,
                    exchange=self.trading_bot.kite_client.kite.EXCHANGE_NFO,
                    tradingsymbol=sym,
                    transaction_type=opp,
                    quantity=qty,
                    product=self.trading_bot.kite_client.kite.PRODUCT_MIS,
                    order_type=self.trading_bot.kite_client.kite.ORDER_TYPE_MARKET
                )
                if orr and 'order_id' in orr:
                    logger.info(f"Exit order placed: {orr['order_id']}")
                    await asyncio.sleep(3)
                    hist=self.trading_bot.kite_client.kite.order_history(orr['order_id'])
                    if hist and hist[-1]['status']=='COMPLETE':
                        ep=float(hist[-1].get('average_price') or hist[-1]['price'])
                        pnl=(ep-cp['entry_price'])*qty if direction=='BUY' else (cp['entry_price']-ep)*qty
                        try:
                            et=datetime.strptime(cp['entry_time'],'%Y-%m-%d %H:%M:%S')
                            dur=format_trade_duration(et)
                        except:
                            dur="Unknown"
                        tr={**cp,'exit_price':ep,'exit_time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'pnl':pnl,'exit_reason':'Manual Exit','duration':dur}
                        if hasattr(self.trading_bot,'trade_history'):
                            self.trading_bot.trade_history.append(tr)
                        if hasattr(self.trading_bot,'risk_manager'):
                            self.trading_bot.risk_manager.update_balance(pnl)
                        self.notify_trade_exit(tr)
                        self.trading_bot.current_position=None
                        logger.info(f"Position closed successfully. P&L: ₹{pnl:.2f}")
                        return True
                    else:
                        st=hist[-1]['status'] if hist else 'Unknown'
                        logger.error(f"Order not executed. Status: {st}")
                        return False
                else:
                    logger.error("Failed to place exit order - no order_id returned")
                    return False
            else:
                logger.error("Kite client not available")
                return False
        except Exception as e:
            logger.error(f"Error force exiting position: {e}")
            return False

    def notify_trade_entry(self, trade_data: Dict[str, Any]):
        """Send enhanced trade entry notification"""
        try:
            direction=trade_data.get('direction','N/A')
            entry_price=trade_data.get('entry_price',0)
            quantity=trade_data.get('quantity',0)
            stop_loss=trade_data.get('stop_loss',0)
            target=trade_data.get('target',0)
            timestamp=trade_data.get('timestamp','Now')
            risk=abs(entry_price-stop_loss)*quantity if stop_loss else 0
            reward=abs(target-entry_price)*quantity if target else 0
            rr=reward/risk if risk>0 else 0
            message=f"""
🚀 *Trade Entry Alert* 

📊 *Direction:* {direction} 
💲 *Entry Price:* ₹{entry_price:.2f} 
📦 *Quantity:* {quantity} 
🛑 *Stop Loss:* ₹{stop_loss:.2f} 
🎯 *Target:* ₹{target:.2f} 
💹 *Risk:Reward:* 1:{rr:.2f} 
⏰ *Time:* {timestamp}
"""
            asyncio.create_task(self.send_notification(message))
        except Exception as e:
            logger.error(f"Error in notify_trade_entry: {e}")

    def notify_trade_exit(self, trade_data: Dict[str, Any]):
        """Send trade exit notification"""
        pnl=trade_data.get('pnl',0)
        emoji="✅ Profit" if pnl>0 else "❌ Loss"
        message=f"""
{emoji} *Trade Exit Alert* 

📊 *Direction:* {trade_data.get('direction','N/A')} 
💲 *Exit Price:* ₹{trade_data.get('exit_price',0):.2f} 
💰 *P&L:* ₹{pnl:.2f} 
⏰ *Duration:* {trade_data.get('duration','N/A')}
"""
        asyncio.create_task(self.send_notification(message))

    def notify_circuit_breaker(self, loss_streak: int, pause_time: int):
        """Send circuit breaker notification"""
        message=f"""
🚨 *Circuit Breaker Activated* 

📉 *Consecutive Losses:* {loss_streak} 
⏱️ *Trading Paused For:* {pause_time} minutes 
🛑 *Auto-trading will resume automatically*

Please review your strategy!"""
        asyncio.create_task(self.send_notification(message))
