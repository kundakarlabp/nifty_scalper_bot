#!/usr/bin/env python3
"""
Nifty Scalper Bot - Telegram Trading Bot for Railway Deployment
Uses sync code for compatibility with Zerodha/Shoonya API
"""

import os
import logging
import pyotp
from datetime import datetime
from dotenv import load_dotenv
from telegram import ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from NorenRestApiPy.NorenApi import NorenApi

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==================== GLOBAL VARIABLES ====================

# Initialize Zerodha/Shoonya API
api = NorenApi(
    host='https://api.shoonya.com/NorenWClientTP/',
    websocket=''  # Empty - WebSocket disabled for Railway
)

# Bot state
BOT_ACTIVE = False

# ==================== BROKER API FUNCTIONS ====================

def login_to_broker():
    """Login to Zerodha/Shoonya broker"""
    global BOT_ACTIVE
    
    try:
        # Generate TOTP
        totp_secret = os.getenv('TOTP_SECRET')
        if not totp_secret:
            logger.error("TOTP_SECRET not found in environment variables")
            return None
        
        totp = pyotp.TOTP(totp_secret).now()
        logger.info(f"Generated TOTP: {totp}")
        
        # Login
        ret = api.login(
            userid=os.getenv('USER_ID'),
            password=os.getenv('PASSWORD'),
            twoFA=totp,
            vendor_code=os.getenv('VENDOR_CODE'),
            api_secret=os.getenv('API_SECRET'),
            imei='abc1234'
        )
        
        if ret and ret.get('stat') == 'Ok':
            logger.info(f"✅ Login successful for user: {ret.get('uname', 'N/A')}")
            BOT_ACTIVE = True
            return ret
        else:
            logger.error(f"❌ Login failed: {ret}")
            return None
            
    except Exception as e:
        logger.error(f"Login exception: {e}", exc_info=True)
        return None

def get_positions():
    """Get current positions"""
    try:
        positions = api.get_positions()
        return positions
    except Exception as e:
        logger.error(f"Get positions error: {e}")
        return None

def get_orderbook():
    """Get order book"""
    try:
        orders = api.get_order_book()
        return orders
    except Exception as e:
        logger.error(f"Get orderbook error: {e}")
        return None

def get_limits():
    """Get account limits"""
    try:
        limits = api.get_limits()
        return limits
    except Exception as e:
        logger.error(f"Get limits error: {e}")
        return None

def place_order(buy_or_sell, tradingsymbol, quantity, price_type='MKT', price=0.0, product_type='I'):
    """
    Place order
    
    Args:
        buy_or_sell: 'B' for Buy, 'S' for Sell
        tradingsymbol: Trading symbol (e.g., 'NIFTY24DECFUT')
        quantity: Quantity to trade
        price_type: 'MKT' or 'LMT'
        price: Price (for limit orders)
        product_type: 'I' for Intraday, 'C' for CNC
    """
    try:
        ret = api.place_order(
            buy_or_sell=buy_or_sell,
            product_type=product_type,
            exchange='NFO',  # NSE Futures & Options
            tradingsymbol=tradingsymbol,
            quantity=quantity,
            discloseqty=0,
            price_type=price_type,
            price=price,
            trigger_price=None,
            retention='DAY',
            remarks=f'bot_order_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        return ret
    except Exception as e:
        logger.error(f"Place order error: {e}")
        return {'stat': 'Not_Ok', 'emsg': str(e)}

def cancel_order(order_no):
    """Cancel order by order number"""
    try:
        ret = api.cancel_order(orderno=order_no)
        return ret
    except Exception as e:
        logger.error(f"Cancel order error: {e}")
        return None

def get_quotes(exchange, token):
    """Get live quotes for a symbol"""
    try:
        quotes = api.get_quotes(exchange=exchange, token=token)
        return quotes
    except Exception as e:
        logger.error(f"Get quotes error: {e}")
        return None

# ==================== TELEGRAM BOT HANDLERS ====================

def start(update, context):
    """Handle /start command"""
    user = update.effective_user
    logger.info(f"User {user.id} started bot")
    
    welcome_msg = (
        f"🤖 *Nifty Scalper Bot*

"
        f"Welcome {user.first_name}!

"
        f"*Available Commands:*
"
        f"/status - Check bot & account status
"
        f"/positions - View current positions
"
        f"/orders - View order book
"
        f"/limits - Check account limits
"
        f"/buy SYMBOL QTY - Place buy order
"
        f"/sell SYMBOL QTY - Place sell order
"
        f"/cancel ORDERID - Cancel order
"
        f"/help - Show this help message

"
        f"*Example:*
"
        f"`/buy NIFTY24DECFUT 50`
"
        f"`/sell BANKNIFTY24DECFUT 25`"
    )
    
    update.message.reply_text(welcome_msg, parse_mode=ParseMode.MARKDOWN)

def help_command(update, context):
    """Handle /help command"""
    start(update, context)

def status(update, context):
    """Handle /status command"""
    try:
        if not BOT_ACTIVE:
            update.message.reply_text("❌ Bot not logged in. Check logs.")
            return
        
        # Get account limits
        limits = get_limits()
        
        if limits and limits.get('stat') == 'Ok':
            cash = limits.get('cash', 'N/A')
            margin_used = limits.get('marginused', 'N/A')
            collateral = limits.get('collateral', '0')
            
            status_msg = (
                f"✅ *Bot Status: ACTIVE*

"
                f"*Account Info:*
"
                f"💰 Cash: ₹{cash}
"
                f"📊 Margin Used: ₹{margin_used}
"
                f"🏦 Collateral: ₹{collateral}
"
                f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            update.message.reply_text(status_msg, parse_mode=ParseMode.MARKDOWN)
        else:
            update.message.reply_text("❌ Failed to fetch account status. API may be down.")
            
    except Exception as e:
        logger.error(f"Status command error: {e}")
        update.message.reply_text(f"❌ Error: {str(e)}")

def positions_command(update, context):
    """Handle /positions command"""
    try:
        positions = get_positions()
        
        if not positions or len(positions) == 0:
            update.message.reply_text("📊 No open positions")
            return
        
        msg = "📊 *Current Positions:*

"
        
        for pos in positions:
            symbol = pos.get('tsym', 'N/A')
            netqty = pos.get('netqty', '0')
            avgprice = pos.get('netavgprc', '0')
            ltp = pos.get('lp', '0')
            pnl = pos.get('rpnl', '0')
            day_pnl = pos.get('daybuyavgprc', '0')
            
            msg += (
                f"*{symbol}*
"
                f"  Qty: {netqty}
"
                f"  Avg: ₹{avgprice} | LTP: ₹{ltp}
"
                f"  P&L: ₹{pnl}

"
            )
        
        update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error(f"Positions command error: {e}")
        update.message.reply_text(f"❌ Error: {str(e)}")

def orders_command(update, context):
    """Handle /orders command"""
    try:
        orders = get_orderbook()
        
        if not orders or len(orders) == 0:
            update.message.reply_text("📋 No orders found")
            return
        
        msg = "📋 *Order Book (Last 10):*

"
        
        for order in orders[:10]:
            symbol = order.get('tsym', 'N/A')
            status = order.get('status', 'N/A')
            qty = order.get('qty', '0')
            filled_qty = order.get('fillshares', '0')
            order_no = order.get('norenordno', 'N/A')
            order_type = order.get('trantype', 'N/A')
            
            emoji = '✅' if status == 'COMPLETE' else '⏳' if status == 'OPEN' else '❌'
            
            msg += (
                f"{emoji} *{symbol}*
"
                f"  {order_type} {qty} (Filled: {filled_qty})
"
                f"  Status: {status}
"
                f"  Order ID: `{order_no}`

"
            )
        
        update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error(f"Orders command error: {e}")
        update.message.reply_text(f"❌ Error: {str(e)}")

def limits_command(update, context):
    """Handle /limits command"""
    try:
        limits = get_limits()
        
        if limits and limits.get('stat') == 'Ok':
            msg = (
                f"💰 *Account Limits:*

"
                f"Cash: ₹{limits.get('cash', 'N/A')}
"
                f"Payin: ₹{limits.get('payin', 'N/A')}
"
                f"Payout: ₹{limits.get('payout', 'N/A')}
"
                f"Margin Used: ₹{limits.get('marginused', 'N/A')}
"
                f"Collateral: ₹{limits.get('collateral', 'N/A')}
"
                f"Unrealized M2M: ₹{limits.get('unrealizedMTM', 'N/A')}
"
                f"Realized M2M: ₹{limits.get('realizedMTM', 'N/A')}"
            )
            update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            update.message.reply_text("❌ Failed to fetch limits")
            
    except Exception as e:
        logger.error(f"Limits command error: {e}")
        update.message.reply_text(f"❌ Error: {str(e)}")

def buy_command(update, context):
    """Handle /buy SYMBOL QTY command"""
    try:
        if len(context.args) < 2:
            update.message.reply_text(
                "❌ *Usage:* `/buy SYMBOL QUANTITY`

"
                "*Example:*
"
                "`/buy NIFTY24DECFUT 50`
"
                "`/buy BANKNIFTY24DECFUT 25`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        symbol = context.args[0].upper()
        
        try:
            qty = int(context.args[1])
        except ValueError:
            update.message.reply_text("❌ Invalid quantity. Must be a number.")
            return
        
        # Optional: price type and price
        price_type = 'MKT'
        price = 0.0
        
        if len(context.args) >= 3:
            price_type = context.args[2].upper()
        if len(context.args) >= 4:
            try:
                price = float(context.args[3])
            except ValueError:
                price = 0.0
        
        update.message.reply_text(f"⏳ Placing BUY order for *{symbol}*...", parse_mode=ParseMode.MARKDOWN)
        
        result = place_order('B', symbol, qty, price_type, price)
        
        if result and result.get('stat') == 'Ok':
            order_id = result.get('norenordno', 'N/A')
            msg = (
                f"✅ *BUY Order Placed Successfully!*

"
                f"Order ID: `{order_id}`
"
                f"Symbol: {symbol}
"
                f"Quantity: {qty}
"
                f"Type: {price_type}"
            )
            if price > 0:
                msg += f"
Price: ₹{price}"
                
            update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            error_msg = result.get('emsg', 'Unknown error') if result else 'API Error'
            update.message.reply_text(f"❌ *Order Failed*

Reason: {error_msg}", parse_mode=ParseMode.MARKDOWN)
            
    except Exception as e:
        logger.error(f"Buy command error: {e}")
        update.message.reply_text(f"❌ Error: {str(e)}")

def sell_command(update, context):
    """Handle /sell SYMBOL QTY command"""
    try:
        if len(context.args) < 2:
            update.message.reply_text(
                "❌ *Usage:* `/sell SYMBOL QUANTITY`

"
                "*Example:*
"
                "`/sell NIFTY24DECFUT 50`
"
                "`/sell BANKNIFTY24DECFUT 25`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        symbol = context.args[0].upper()
        
        try:
            qty = int(context.args[1])
        except ValueError:
            update.message.reply_text("❌ Invalid quantity. Must be a number.")
            return
        
        price_type = 'MKT'
        price = 0.0
        
        if len(context.args) >= 3:
            price_type = context.args[2].upper()
        if len(context.args) >= 4:
            try:
                price = float(context.args[3])
            except ValueError:
                price = 0.0
        
        update.message.reply_text(f"⏳ Placing SELL order for *{symbol}*...", parse_mode=ParseMode.MARKDOWN)
        
        result = place_order('S', symbol, qty, price_type, price)
        
        if result and result.get('stat') == 'Ok':
            order_id = result.get('norenordno', 'N/A')
            msg = (
                f"✅ *SELL Order Placed Successfully!*

"
                f"Order ID: `{order_id}`
"
                f"Symbol: {symbol}
"
                f"Quantity: {qty}
"
                f"Type: {price_type}"
            )
            if price > 0:
                msg += f"
Price: ₹{price}"
                
            update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        else:
            error_msg = result.get('emsg', 'Unknown error') if result else 'API Error'
            update.message.reply_text(f"❌ *Order Failed*

Reason: {error_msg}", parse_mode=ParseMode.MARKDOWN)
            
    except Exception as e:
        logger.error(f"Sell command error: {e}")
        update.message.reply_text(f"❌ Error: {str(e)}")

def cancel_command(update, context):
    """Handle /cancel ORDERID command"""
    try:
        if len(context.args) < 1:
            update.message.reply_text(
                "❌ *Usage:* `/cancel ORDER_ID`

"
                "*Example:*
"
                "`/cancel 24112600000123`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
        
        order_no = context.args[0]
        
        update.message.reply_text(f"⏳ Cancelling order `{order_no}`...", parse_mode=ParseMode.MARKDOWN)
        
        result = cancel_order(order_no)
        
        if result and result.get('stat') == 'Ok':
            update.message.reply_text(f"✅ Order `{order_no}` cancelled successfully!", parse_mode=ParseMode.MARKDOWN)
        else:
            error_msg = result.get('emsg', 'Unknown error') if result else 'API Error'
            update.message.reply_text(f"❌ Cancel failed: {error_msg}", parse_mode=ParseMode.MARKDOWN)
            
    except Exception as e:
        logger.error(f"Cancel command error: {e}")
        update.message.reply_text(f"❌ Error: {str(e)}")

def error_handler(update, context):
    """Handle errors"""
    logger.error(f"Update {update} caused error: {context.error}")
    
    if update and update.effective_message:
        update.effective_message.reply_text(
            "❌ An error occurred while processing your request. Please try again."
        )

# ==================== MAIN FUNCTION ====================

def main():
    """Start the bot"""
    logger.info("=" * 60)
    logger.info("Starting Nifty Scalper Bot")
    logger.info("=" * 60)
    
    # Step 1: Login to broker
    logger.info("Step 1: Logging into broker...")
    login_result = login_to_broker()
    
    if not login_result:
        logger.error("❌ Failed to login to broker. Check credentials and TOTP.")
        return
    
    logger.info("✅ Broker login successful")
    
    # Step 2: Create Telegram bot
    logger.info("Step 2: Initializing Telegram bot...")
    
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ TELEGRAM_BOT_TOKEN not found in environment variables")
        return
    
    updater = Updater(token, use_context=True)
    dp = updater.dispatcher
    
    # Step 3: Register handlers
    logger.info("Step 3: Registering command handlers...")
    
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('help', help_command))
    dp.add_handler(CommandHandler('status', status))
    dp.add_handler(CommandHandler('positions', positions_command))
    dp.add_handler(CommandHandler('orders', orders_command))
    dp.add_handler(CommandHandler('limits', limits_command))
    dp.add_handler(CommandHandler('buy', buy_command))
    dp.add_handler(CommandHandler('sell', sell_command))
    dp.add_handler(CommandHandler('cancel', cancel_command))
    
    # Error handler
    dp.add_error_handler(error_handler)
    
    logger.info("✅ Handlers registered")
    
    # Step 4: Start polling
    logger.info("=" * 60)
    logger.info("🚀 Bot is now running! Press Ctrl+C to stop.")
    logger.info("=" * 60)
    
    updater.start_polling(drop_pending_updates=True)
    updater.idle()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("
👋 Bot stopped by user")
    except Exception as e:
        logger.critical(f"💥 Bot crashed: {e}", exc_info=True)