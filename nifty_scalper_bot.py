#!/usr/bin/env python3
"""
Nifty Scalper Bot - Production Ready Trading Bot
Supports Zerodha Kite API with Telegram notifications
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import threading

# Import our modules
from config import Config
from utils import setup_logging, is_market_open, CircuitBreaker, validate_trade_data
from kite_client import KiteClient
from signal_generator import SignalGenerator
from telegram_bot import TelegramBot

# Setup logging
logger = setup_logging()

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self):
        self.initial_capital = Config.TRADING_CAPITAL
        self.current_balance = Config.TRADING_CAPITAL
        self.peak_balance = Config.TRADING_CAPITAL
        self.daily_pnl = 0.0
        self.circuit_breaker = CircuitBreaker(Config.MAX_CONSEC_LOSSES, Config.LOSS_PAUSE_TIME)
        self.max_daily_loss = Config.TRADING_CAPITAL * (Config.MAX_DAILY_LOSS_PCT / 100)
        
    def update_balance(self, pnl: float):
        """Update balance and check risk limits"""
        self.current_balance += pnl
        self.daily_pnl += pnl
        
        # Update peak balance
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Update circuit breaker
        self.circuit_breaker.record_trade(pnl)
        
        logger.info(f"Balance updated: ₹{self.current_balance:.2f} (P&L: ₹{pnl:.2f})")
    
    def can_trade(self) -> bool:
        """Check if trading is allowed based on risk limits"""
        # Circuit breaker check
        if not self.circuit_breaker.can_trade():
            return False
        
        # Daily loss limit check
        if abs(self.daily_pnl) >= self.max_daily_loss and self.daily_pnl < 0:
            logger.warning(f"Daily loss limit reached: ₹{self.daily_pnl:.2f}")
            return False
        
        return True
    
    def get_position_size(self) -> int:
        """Calculate position size based on current drawdown"""
        drawdown = 0.0
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        base_size = Config.TRADE_LOT_SIZE
        
        # Reduce position size if in significant drawdown
        if drawdown > 0.15:  # 15% drawdown
            return max(Config.TRADE_LOT_SIZE // 2, 25)
        elif drawdown > 0.10:  # 10% drawdown
            return max(int(Config.TRADE_LOT_SIZE * 0.75), 50)
        
        return base_size
    
    def reset_daily_pnl(self):
        """Reset daily P&L (call at start of each trading day)"""
        self.daily_pnl = 0.0
        logger.info("Daily P&L reset")

class NiftyScalperBot:
    """Main trading bot class"""
    
    def __init__(self):
        # Validate configuration
        Config.validate_config()
        
        # Initialize components
        self.kite_client = KiteClient()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        self.telegram_bot = TelegramBot(self)
        
        # Trading state
        self.current_position = None
        self.trade_history = []
        self.price_data = pd.DataFrame()
        self.auto_trade = Config.AUTO_TRADE
        self.is_running = False
        
        # Flask app for health checks and webhooks
        self.flask_app = Flask(__name__)
        self.setup_flask_routes()
        
    def setup_flask_routes(self):
        """Setup Flask routes for monitoring and webhooks"""
        
        @self.flask_app.route('/health')
        def health_check():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'market_open': is_market_open(),
                'auto_trade': self.auto_trade,
                'balance': self.risk_manager.current_balance,
                'position': self.current_position is not None
            })
        
        @self.flask_app.route('/status')
        def get_status():
            return jsonify({
                'balance': self.risk_manager.current_balance,
                'daily_pnl': self.risk_manager.daily_pnl,
                'peak_balance': self.risk_manager.peak_balance,
                'current_position': self.current_position,
                'trade_count': len(self.trade_history),
                'circuit_breaker': self.risk_manager.circuit_breaker.is_active,
                'auto_trade': self.auto_trade
            })
        
        @self.flask_app.route('/trades')
        def get_trades():
            return jsonify({
                'trades': self.trade_history[-10:],  # Last 10 trades
                'total_trades': len(self.trade_history)
            })
        
        @self.flask_app.route('/control', methods=['POST'])
        def control_bot():
            data = request.json
            action = data.get('action')
            
            if action == 'start':
                self.auto_trade = True
                return jsonify({'status': 'success', 'message': 'Auto-trade started'})
            elif action == 'stop':
                self.auto_trade = False
                return jsonify({'status': 'success', 'message': 'Auto-trade stopped'})
            elif action == 'exit_position':
                success = self.force_exit_position()
                return jsonify({
                    'status': 'success' if success else 'error',
                    'message': 'Position closed' if success else 'Failed to close position'
                })
            else:
                return jsonify({'status': 'error', 'message': 'Invalid action'})
    
    def update_price_data(self, current_price: float):
        """Update price data for indicator calculations"""
        now = pd.Timestamp.now().floor('T')  # Round to minute
        
        # Create or update current minute candle
        if self.price_data.empty or now > self.price_data.index[-1]:
            # New minute candle
            new_candle = pd.DataFrame({
                'open': [current_price],
                'high': [current_price],
                'low': [current_price],
                'close': [current_price],
                'volume': [1000]  # Simulated volume
            }, index=[now])
            
            self.price_data = pd.concat([self.price_data, new_candle])
        else:
            # Update current minute candle
            idx = self.price_data.index[-1]
            self.price_data.loc[idx, 'close'] = current_price
            self.price_data.loc[idx, 'high'] = max(self.price_data.loc[idx, 'high'], current_price)
            self.price_data.loc[idx, 'low'] = min(self.price_data.loc[idx, 'low'], current_price)
            self.price_data.loc[idx, 'volume'] += 100
        
        # Keep only last 500 candles to manage memory
        if len(self.price_data) > 500:
            self.price_data = self.price_data.tail(500)
    
    def enter_trade(self, direction: str, current_price: float, signal_strength: float) -> bool:
        """Enter a new trade"""
        try:
            # Get position size
            quantity = self.risk_manager.get_position_size()
            
            # Calculate stop loss and target
            indicators = self.signal_generator.calculate_all_indicators(self.price_data)
            atr = indicators.get('atr', current_price * 0.01)  # Default to 1% if ATR not available
            
            stop_loss, target = self.signal_generator.get_stop_loss_target(
                current_price, direction, atr
            )
            
            # Place entry order
            symbol = f"{Config.UNDERLYING_SYMBOL}24DECFUT"  # Adjust based on current contract
            order_id = self.kite_client.place_order(
                symbol=symbol,
                transaction_type=direction,
                quantity=quantity,
                order_type="MARKET",
                exchange=Config.TRADE_EXCHANGE
            )
            
            if not order_id:
                logger.error("Failed to place entry order")
                return False
            
            # Create position record
            self.current_position = {
                'direction': direction,
                'entry_price': current_price,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'target': target,
                'entry_time': datetime.now(),
                'order_id': order_id,
                'signal_strength': signal_strength
            }
            
            logger.info(f"Trade entered: {direction} {quantity} @ ₹{current_price:.2f}")
            
            # Place stop loss and target orders
            self.place_exit_orders(stop_loss, target, quantity, direction)
            
            # Send Telegram notification
            self.telegram_bot.notify_trade_entry({
                'direction': direction,
                'entry_price': current_price,
                'quantity': quantity,
                'stop_loss': stop_loss,
                'target': target,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error entering trade: {e}")
            return False
    
    def place_exit_orders(self, stop_loss: float, target: float, quantity: int, direction: str):
        """Place stop loss and target orders"""
        try:
            symbol = f"{Config.UNDERLYING_SYMBOL}24DECFUT"
            exit_direction = 'SELL' if direction == 'BUY' else 'BUY'
            
            # Place stop loss order
            sl_order_id = self.kite_client.place_order(
                symbol=symbol,
                transaction_type=exit_direction,
                quantity=quantity,
                order_type="SL",
                price=stop_loss,
                trigger_price=stop_loss,
                exchange=Config.TRADE_EXCHANGE
            )
            
            # Place target order
            target_order_id = self.kite_client.place_order(
                symbol=symbol,
                transaction_type=exit_direction,
                quantity=quantity,
                order_type="LIMIT",
                price=target,
                exchange=Config.TRADE_EXCHANGE
            )
            
            if self.current_position:
                self.current_position['sl_order_id'] = sl_order_id
                self.current_position['target_order_id'] = target_order_id
            
            logger.info(f"Exit orders placed - SL: ₹{stop_loss:.2f}, Target: ₹{target:.2f}")
            
        except Exception as e:
            logger.error(f"Error placing exit orders: {e}")
    
    def check_position_status(self):
        """Check if current position is still open"""
        if not self.current_position:
            return
        
        try:
            # Get current positions from broker
            positions = self.kite_client.get_positions()
            symbol = f"{Config.UNDERLYING_SYMBOL}24DECFUT"
            
            # Check if position exists
            current_qty = 0
            for pos in positions.get('net', []):
                if pos['tradingsymbol'] == symbol:
                    current_qty = int(pos['quantity'])
                    break
            
            # If position closed, record the trade
            if current_qty == 0 and self.current_position:
                self.handle_position_exit()
                
        except Exception as e:
            logger.error(f"Error checking position status: {e}")
    
    def handle_position_exit(self):
        """Handle position exit and record trade"""
        if not self.current_position:
            return
        
        try:
            # Get current price as exit price (approximation)
            current_price = self.kite_client.get_ltp(
                f"{Config.UNDERLYING_SYMBOL}24DECFUT", 
                Config.TRADE_EXCHANGE
            )
            
            if not current_price:
                current_price = self.current_position['entry_price']  # Fallback
            
            # Calculate P&L
            direction = self.current_position['direction']
            entry_price = self.current_position['entry_price']
            quantity = self.current_position['quantity']
            
            if direction == 'BUY':
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
            
            # Record trade
            trade_record = {
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': current_price,
                'quantity': quantity,
                'pnl': pnl,
                'entry_time': self.current_position['entry_time'],
                'exit_time': datetime.now(),
                'signal_strength': self.current_position.get('signal_strength', 0)
            }
            
            self.trade_history.append(trade_record)
            
            # Update risk manager
            self.risk_manager.update_balance(pnl)
            
            # Update signal generator threshold
            self.signal_generator.adapt_threshold(self.trade_history[-Config.PERFORMANCE_WINDOW:])
            
            # Send Telegram notification
            duration = (trade_record['exit_time'] - trade_record['entry_time']).total_seconds() / 60
            self.telegram_bot.notify_trade_exit({
                'direction': direction,
                'exit_price': current_price,
                'pnl': pnl,
                'duration': f"{duration:.1f} min"
            })
            
            logger.info(f"Trade closed: {direction} P&L: ₹{pnl:.2f}")
            
            # Clear current position
            self.current_position = None
            
        except Exception as e:
            logger.error(f"Error handling position exit: {e}")
    
    def force_exit_position(self) -> bool:
        """Force exit current position"""
        if not self.current_position:
            return False
        
        try:
            direction = self.current_position['direction']
            quantity = self.current_position['quantity']
            symbol = f"{Config.UNDERLYING_SYMBOL}24DECFUT"
            exit_direction = 'SELL' if direction == 'BUY' else 'BUY'
            
            # Cancel existing exit orders
            if 'sl_order_id' in self.current_position:
                self.kite_client.cancel_order(self.current_position['sl_order_id'])
            if 'target_order_id' in self.current_position:
                self.kite_client.cancel_order(self.current_position['target_order_id'])
            
            # Place market exit order
            order_id = self.kite_client.place_order(
                symbol=symbol,
                transaction_type=exit_direction,
                quantity=quantity,
                order_type="MARKET",
                exchange=Config.TRADE_EXCHANGE
            )
            
            if order_id:
                logger.info("Position force-exited")
                time.sleep(2)  # Wait for order execution
                self.handle_position_exit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error force-exiting position: {e}")
            return False
    
    async def main_trading_loop(self):
        """Main trading logic loop"""
        logger.info("Starting main trading loop")
        
        while self.is_running:
            try:
                # Check if market is open
                if not is_market_open():
                    await asyncio.sleep(60)  # Check every minute when market closed
                    continue
                
                # Check if trading is allowed
                if not self.auto_trade or not self.risk_manager.can_trade():
                    await asyncio.sleep(5)
                    continue
                
                # Get current price
                current_price = self.kite_client.get_ltp(
                    f"{Config.UNDERLYING_SYMBOL}24DECFUT",
                    Config.TRADE_EXCHANGE
                )
                
                if not current_price:
                    logger.warning("Failed to get current price")
                    await asyncio.sleep(5)
                    continue
                
                # Update price data
                self.update_price_data(current_price)
                
                # Check existing position status
                self.check_position_status()
                
                # Generate new signals only if no current position
                if not self.current_position and len(self.price_data) > 50:
                    signal_strength, signal_components = self.signal_generator.calculate_signal_strength(
                        self.price_data, current_price
                    )
                    
                    should_trade, reason = self.signal_generator.should_trade(
                        signal_strength, signal_components
                    )
                    
                    if should_trade:
                        direction = "BUY" if signal_strength > 0 else "SELL"
                        logger.info(f"Trading signal: {reason}")
                        
                        # Enter trade
                        success = self.enter_trade(direction, current_price, signal_strength)
                        if success:
                            logger.info(f"Successfully entered {direction} trade")
                        else:
                            logger.error(f"Failed to enter {direction} trade")
                
                await asyncio.sleep(1)  # Wait 1 second before next iteration
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(5)
    
    def start_flask_server(self):
        """Start Flask server in a separate thread"""
        def run_flask():
            self.flask_app.run(
                host=Config.FLASK_HOST,
                port=Config.FLASK_PORT,
                debug=False,
                use_reloader=False
            )
        
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        logger.info(f"Flask server started on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    
    async def start(self):
        """Start the trading bot"""
        logger.info("🚀 Starting Nifty Scalper Bot")
        
        # Validate Kite connection
        if not self.kite_client.is_connected:
            logger.error("Kite client not connected. Please check your credentials.")
            return
        
        # Start Flask server
        self.start_flask_server()
        
        # Start Telegram bot
        await self.telegram_bot.start_bot()
        
        # Set running flag
        self.is_running = True
        
        # Start main trading loop
        await self.main_trading_loop()
    
    async def stop(self):
        """Stop the trading bot"""
        logger.info("🛑 Stopping Nifty Scalper Bot")
        
        self.is_running = False
        
        # Force exit any open position
        if self.current_position:
            self.force_exit_position()
        
        # Stop Telegram bot
        await self.telegram_bot.stop_bot()
        
        logger.info("Bot stopped successfully")

async def main():
    """Main function"""
    # Setup signal handlers for graceful shutdown
    bot = NiftyScalperBot()
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(bot.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await bot.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await bot.stop()

if __name__ == "__main__":
    # Check if all required environment variables are set
    try:
        Config.validate_config()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Run the bot
    asyncio.run(main())