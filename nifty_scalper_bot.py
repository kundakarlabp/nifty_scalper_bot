#!/usr/bin/env python3
"""
Nifty Scalper Bot v2.0 - Production Ready
Enhanced trading bot with Telegram integration and proper auto-trading
"""

import os
import sys
import logging
import asyncio
import threading
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from flask import Flask, jsonify

# Import custom modules
from config import Config
from kite_client import KiteClient
from signal_generator import SignalGenerator
from monitor import Monitor
from utils import is_market_open, get_market_status, time_until_market_open
from telegram_bot import TelegramBot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RiskManager:
    """Enhanced Risk Management System"""
    
    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.todays_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = Config.MAX_CONSECUTIVE_LOSSES
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        self.daily_loss_limit = initial_balance * Config.MAX_DAILY_LOSS_PCT
        
    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed"""
        # Check market hours
        if not is_market_open():
            return False, "Market is closed"
        
        # Check daily trade limit
        if self.daily_trades >= Config.MAX_DAILY_TRADES:
            return False, f"Daily trade limit reached ({Config.MAX_DAILY_TRADES})"
        
        # Check daily loss limit
        if self.todays_pnl <= -self.daily_loss_limit:
            return False, f"Daily loss limit reached (₹{self.daily_loss_limit:,.2f})"
        
        # Check circuit breaker
        if self.circuit_breaker_active:
            if datetime.now() < self.circuit_breaker_until:
                remaining = (self.circuit_breaker_until - datetime.now()).seconds // 60
                return False, f"Circuit breaker active for {remaining} more minutes"
            else:
                self.reset_circuit_breaker()
        
        return True, "Trading allowed"
    
    def update_balance(self, pnl: float):
        """Update balance and risk metrics"""
        self.current_balance += pnl
        self.todays_pnl += pnl
        self.daily_trades += 1
        
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.activate_circuit_breaker()
        else:
            self.consecutive_losses = 0
    
    def activate_circuit_breaker(self):
        """Activate circuit breaker after consecutive losses"""
        self.circuit_breaker_active = True
        pause_minutes = Config.CIRCUIT_BREAKER_PAUSE_MINUTES
        self.circuit_breaker_until = datetime.now() + timedelta(minutes=pause_minutes)
        logger.warning(f"Circuit breaker activated for {pause_minutes} minutes after {self.consecutive_losses} consecutive losses")
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker"""
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        self.consecutive_losses = 0
        logger.info("Circuit breaker reset")
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.todays_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.reset_circuit_breaker()
        logger.info("Daily statistics reset")

class NiftyScalperBot:
    """Main Trading Bot Class with Enhanced Features"""
    
    def __init__(self):
        self.kite_client = None
        self.signal_generator = None
        self.monitor = None
        self.telegram_bot = None
        self.risk_manager = RiskManager()
        
        # Trading state
        self.auto_trade = True
        self.is_running = False
        self.current_position = None
        self.trade_history = []
        self.last_signal_time = None
        
        # Flask app for web service
        self.app = Flask(__name__)
        self.setup_web_routes()
        
        # Initialize components
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all bot components"""
        try:
            # Initialize Kite client
            self.kite_client = KiteClient()
            if not self.kite_client.connect():
                logger.error("Failed to connect to Kite")
                return False
            
            # Initialize signal generator
            self.signal_generator = SignalGenerator()
            
            # Initialize monitor
            self.monitor = Monitor()
            
            # Initialize Telegram bot
            self.setup_telegram_bot()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    def setup_telegram_bot(self):
        """Setup Telegram bot integration"""
        if not Config.TELEGRAM_BOT_TOKEN:
            logger.warning("Telegram bot token not provided - skipping Telegram integration")
            return
        
        try:
            self.telegram_bot = TelegramBot(trading_bot_instance=self)
            
            # Start Telegram bot in separate thread
            def start_telegram():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.telegram_bot.start_bot())
                    loop.run_forever()
                except Exception as e:
                    logger.error(f"Error in Telegram thread: {e}")
            
            telegram_thread = threading.Thread(target=start_telegram, daemon=True)
            telegram_thread.start()
            logger.info("Telegram bot thread started")
            
        except Exception as e:
            logger.error(f"Failed to setup Telegram bot: {e}")
    
    def setup_web_routes(self):
        """Setup Flask web routes for Render deployment"""
        @self.app.route('/')
        def health_check():
            return jsonify({
                "status": "Nifty Scalper Bot is running",
                "timestamp": datetime.now().isoformat(),
                "market_status": get_market_status(),
                "auto_trade": self.auto_trade,
                "current_balance": self.risk_manager.current_balance,
                "todays_pnl": self.risk_manager.todays_pnl,
                "daily_trades": self.risk_manager.daily_trades
            })
        
        @self.app.route('/status')
        def bot_status():
            return jsonify({
                "bot_status": "active" if self.is_running else "inactive",
                "auto_trade": self.auto_trade,
                "market_open": is_market_open(),
                "current_position": self.current_position,
                "balance": self.risk_manager.current_balance,
                "todays_pnl": self.risk_manager.todays_pnl,
                "daily_trades": self.risk_manager.daily_trades,
                "circuit_breaker": self.risk_manager.circuit_breaker_active,
                "last_update": datetime.now().isoformat()
            })
        
        @self.app.route('/trades')
        def recent_trades():
            return jsonify({
                "recent_trades": self.trade_history[-10:],  # Last 10 trades
                "total_trades": len(self.trade_history)
            })
    
    def get_market_data(self) -> Optional[Dict[str, Any]]:
        """Get current market data"""
        try:
            if not self.kite_client:
                return None
            
            # Get NIFTY data
            instrument_token = self.kite_client.get_instrument_token(Config.UNDERLYING_SYMBOL)
            if not instrument_token:
                return None
            
            quote = self.kite_client.kite.quote([instrument_token])
            if not quote or str(instrument_token) not in quote:
                return None
            
            data = quote[str(instrument_token)]
            return {
                'ltp': data['last_price'],
                'volume': data.get('volume', 0),
                'timestamp': datetime.now(),
                'ohlc': data.get('ohlc', {})
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def analyze_signals(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market data for trading signals"""
        try:
            if not self.signal_generator or not market_data:
                return None
            
            # Generate signal
            signal = self.signal_generator.generate_signal(market_data)
            
            if signal and signal.get('strength', 0) >= Config.SIGNAL_THRESHOLD:
                # Avoid duplicate signals
                current_time = datetime.now()
                if (self.last_signal_time and 
                    (current_time - self.last_signal_time).seconds < Config.MIN_SIGNAL_INTERVAL):
                    return None
                
                self.last_signal_time = current_time
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing signals: {e}")
            return None
    
    def calculate_position_size(self, signal_data: Dict[str, Any]) -> int:
    """Calculate number of lots based on capital and risk"""
    try:
        # Risk per trade (1% of current balance)
        risk_amount = self.risk_manager.current_balance * Config.RISK_PER_TRADE_PCT
        
        # Calculate stop loss distance
        entry_price = signal_data.get('entry_price', 0)
        stop_loss = signal_data.get('stop_loss', 0)
        
        if entry_price <= 0 or stop_loss <= 0:
            return Config.DEFAULT_LOTS
        
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return Config.DEFAULT_LOTS
        
        # Calculate number of lots
        total_shares = int(risk_amount / risk_per_share)
        num_lots = max(1, total_shares // Config.LOT_SIZE)
        
        # Apply lot limits
        num_lots = max(Config.MIN_LOTS, min(num_lots, Config.MAX_LOTS))
        
        return num_lots
        
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return Config.DEFAULT_LOTS
    
    def execute_trade(self, signal_data: Dict[str, Any]) -> bool:
        """Execute trade based on signal"""
        try:
            # Check if auto-trading is enabled
            if not self.auto_trade:
                logger.info("Auto-trading disabled, skipping trade")
                return False
            
            # Check if we can trade
            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                logger.info(f"Cannot trade: {reason}")
                return False
            
            # Check if we already have a position
            if self.current_position:
                logger.info("Already have an open position, skipping new trade")
                return False
            
            # Calculate position size
            # Calculate position size (in lots)
                num_lots = self.calculate_position_size(signal_data)
                quantity = num_lots * Config.LOT_SIZE  # Convert lots to shares for Kite API
            
            # Place order
            order_response = self.place_order(
                direction=signal_data['direction'],
                quantity=quantity,
                price=signal_data.get('entry_price')
            )
            
            if order_response and 'order_id' in order_response:
                # Wait for order execution
                time.sleep(2)
                
                # Check order status
                order_status = self.kite_client.kite.order_history(order_response['order_id'])
                if order_status and order_status[-1]['status'] == 'COMPLETE':
                    executed_price = float(order_status[-1]['average_price'] or order_status[-1]['price'])
                    
                    # Create position record
                    self.current_position = {
                        'order_id': order_response['order_id'],
                        'direction': signal_data['direction'],
                        'quantity': quantity,
                        'entry_price': executed_price,
                        'stop_loss': signal_data.get('stop_loss'),
                        'target': signal_data.get('target'),
                        'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': Config.UNDERLYING_SYMBOL
                    }
                    
                    logger.info(f"Trade executed: {signal_data['direction']} {quantity} @ ₹{executed_price:.2f}")
                    
                    # Send Telegram notification
                    if self.telegram_bot:
                        self.telegram_bot.notify_trade_entry({
                            'direction': signal_data['direction'],
                            'entry_price': executed_price,
                            'quantity': quantity,
                            'stop_loss': signal_data.get('stop_loss', 0),
                            'target': signal_data.get('target', 0),
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        })
                    
                    return True
                else:
                    logger.error(f"Order not executed. Status: {order_status[-1]['status'] if order_status else 'Unknown'}")
                    return False
            else:
                logger.error("Failed to place order")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def place_order(self, direction: str, quantity: int, price: float = None) -> Optional[Dict[str, Any]]:
        """Place order through Kite"""
        try:
            if not self.kite_client:
                return None
            
            order_params = {
                'variety': self.kite_client.kite.VARIETY_REGULAR,
                'exchange': self.kite_client.kite.EXCHANGE_NFO,
                'tradingsymbol': Config.UNDERLYING_SYMBOL,
                'transaction_type': direction,
                'quantity': quantity,
                'product': self.kite_client.kite.PRODUCT_MIS,
                'order_type': self.kite_client.kite.ORDER_TYPE_MARKET if not price else self.kite_client.kite.ORDER_TYPE_LIMIT
            }
            
            if price:
                order_params['price'] = price
            
            return self.kite_client.kite.place_order(**order_params)
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def check_exit_conditions(self) -> Optional[str]:
        """Check if current position should be exited"""
        if not self.current_position:
            return None
        
        try:
            # Get current market price
            market_data = self.get_market_data()
            if not market_data:
                return None
            
            current_price = market_data['ltp']
            entry_price = self.current_position['entry_price']
            direction = self.current_position['direction']
            stop_loss = self.current_position.get('stop_loss')
            target = self.current_position.get('target')
            
            # Check stop loss
            if stop_loss:
                if ((direction == 'BUY' and current_price <= stop_loss) or 
                    (direction == 'SELL' and current_price >= stop_loss)):
                    return 'stop_loss'
            
            # Check target
            if target:
                if ((direction == 'BUY' and current_price >= target) or 
                    (direction == 'SELL' and current_price <= target)):
                    return 'target'
            
            # Check time-based exit (end of day)
            now = datetime.now()
            if now.hour >= 15 and now.minute >= 20:  # Exit 10 minutes before market close
                return 'eod'
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None
    
    def close_position(self, exit_reason: str = 'manual') -> bool:
        """Close current position"""
        if not self.current_position:
            return False
        
        try:
            # Get opposite direction
            direction = self.current_position['direction']
            opposite_direction = 'SELL' if direction == 'BUY' else 'BUY'
            quantity = self.current_position['quantity']
            
            # Place exit order
            order_response = self.place_order(opposite_direction, quantity)
            
            if order_response and 'order_id' in order_response:
                # Wait for execution
                time.sleep(2)
                
                # Check order status
                order_status = self.kite_client.kite.order_history(order_response['order_id'])
                if order_status and order_status[-1]['status'] == 'COMPLETE':
                    exit_price = float(order_status[-1]['average_price'] or order_status[-1]['price'])
                    
                    # Calculate P&L
                    entry_price = self.current_position['entry_price']
                    if direction == 'BUY':
                        pnl = (exit_price - entry_price) * quantity
                    else:
                        pnl = (entry_price - exit_price) * quantity
                    
                    # Calculate trade duration
                    entry_time = datetime.strptime(self.current_position['entry_time'], '%Y-%m-%d %H:%M:%S')
                    duration = str(datetime.now() - entry_time).split('.')[0]
                    
                    # Create trade record
                    trade_record = {
                        **self.current_position,
                        'exit_price': exit_price,
                        'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'duration': duration
                    }
                    
                    # Update trade history
                    self.trade_history.append(trade_record)
                    
                    # Update risk manager
                    self.risk_manager.update_balance(pnl)
                    
                    logger.info(f"Position closed: {direction} @ ₹{exit_price:.2f}, P&L: ₹{pnl:.2f}")
                    
                    # Send Telegram notification
                    if self.telegram_bot:
                        self.telegram_bot.notify_trade_exit({
                            'direction': direction,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'duration': duration
                        })
                    
                    # Clear current position
                    self.current_position = None
                    
                    # Check circuit breaker
                    if (self.risk_manager.circuit_breaker_active and 
                        self.telegram_bot and 
                        exit_reason == 'stop_loss'):
                        self.telegram_bot.notify_circuit_breaker(
                            self.risk_manager.consecutive_losses,
                            Config.CIRCUIT_BREAKER_PAUSE_MINUTES
                        )
                    
                    return True
                else:
                    logger.error(f"Exit order not executed. Status: {order_status[-1]['status'] if order_status else 'Unknown'}")
                    return False
            else:
                logger.error("Failed to place exit order")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def trading_loop(self):
        """Main trading loop"""
        logger.info("Starting trading loop")
        
        while self.is_running:
            try:
                # Check if market is open
                if not is_market_open():
                    time.sleep(60)  # Check every minute when market is closed
                    continue
                
                # Get market data
                market_data = self.get_market_data()
                if not market_data:
                    time.sleep(Config.LOOP_DELAY)
                    continue
                
                # Check exit conditions for existing position
                if self.current_position:
                    exit_reason = self.check_exit_conditions()
                    if exit_reason:
                        self.close_position(exit_reason)
                else:
                    # Look for new trading opportunities
                    signal = self.analyze_signals(market_data)
                    if signal:
                        self.execute_trade(signal)
                
                # Monitor system
                if self.monitor:
                    self.monitor.update_metrics({
                        'current_price': market_data['ltp'],
                        'balance': self.risk_manager.current_balance,
                        'pnl': self.risk_manager.todays_pnl,
                        'trades': self.risk_manager.daily_trades
                    })
                
                time.sleep(Config.LOOP_DELAY)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(10)  # Wait before retrying
    
    def schedule_daily_reset(self):
        """Schedule daily reset of statistics"""
        schedule.every().day.at("00:01").do(self.risk_manager.reset_daily_stats)
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)
    
    def start(self):
        """Start the trading bot"""
        logger.info("Starting Nifty Scalper Bot v2.0")
        
        # Check initialization
        if not self.kite_client:
            logger.error("Bot not properly initialized")
            return
        
        self.is_running = True
        
        # Start trading loop in separate thread
        trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        trading_thread.start()
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self.schedule_daily_reset, daemon=True)
        scheduler_thread.start()
        
        logger.info("Bot started successfully")
        
        # Send startup notification
        if self.telegram_bot:
            asyncio.create_task(self.telegram_bot.send_notification(
                f"🚀 *Nifty Scalper Bot v2.0 Started!*\\n\\n"
                f"• *Market Status:* {get_market_status()}\\n"
                f"• *Auto-trading:* {'✅ ON' if self.auto_trade else '❌ OFF'}\\n"
                f"• *Balance:* ₹{self.risk_manager.current_balance:,.2f}\\n"
                f"• *Mode:* 💰 LIVE TRADING"
            ))
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping Nifty Scalper Bot")
        
        self.is_running = False
        
        # Close any open positions
        if self.current_position:
            self.close_position('shutdown')
        
        # Stop Telegram bot
        if self.telegram_bot:
            asyncio.create_task(self.telegram_bot.stop_bot())
        
        logger.info("Bot stopped")

def main():
    """Main function"""
    try:
        # Create bot instance
        bot = NiftyScalperBot()
        
        # Start bot
        bot.start()
        
        # Start Flask web service for Render
        port = int(os.environ.get('PORT', 5000))
        
        # Run Flask app in main thread
        bot.app.run(host='0.0.0.0', port=port, debug=False)
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        if 'bot' in locals():
            bot.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if 'bot' in locals():
            bot.stop()

if __name__ == "__main__":
    main()
