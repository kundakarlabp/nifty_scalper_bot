import os
import time
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import redis
from flask import Flask, jsonify, request
from waitress import serve
import json
from kiteconnect import KiteConnect
from telegram import Bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NiftyScalperBot:
    def __init__(self):
        # Load configuration from environment variables
        self.config = {
            'ZERODHA_API_KEY': os.getenv('ZERODHA_API_KEY'),
            'ZERODHA_API_SECRET': os.getenv('ZERODHA_API_SECRET'),
            'ZERODHA_CLIENT_ID': os.getenv('ZERODHA_CLIENT_ID'),
            'ZERODHA_ACCESS_TOKEN': os.getenv('ZERODHA_ACCESS_TOKEN'),
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
            'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
            'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379'),
            'MARKET_START_HOUR': int(os.getenv('MARKET_START_HOUR', 9)),
            'MARKET_END_HOUR': int(os.getenv('MARKET_END_HOUR', 15)),
            'MARKET_START_MINUTE': int(os.getenv('MARKET_START_MINUTE', 15)),
            'MARKET_END_MINUTE': int(os.getenv('MARKET_END_MINUTE', 30)),
            'DRY_RUN': os.getenv('DRY_RUN', 'True').lower() == 'true'
        }
        
        # Validate required configuration
        required_configs = ['ZERODHA_API_KEY', 'ZERODHA_API_SECRET', 'ZERODHA_CLIENT_ID', 
                          'ZERODHA_ACCESS_TOKEN', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        
        for config in required_configs:
            if not self.config[config]:
                raise ValueError(f"Missing required configuration: {config}")
        
        # Initialize components
        self.kite = None
        self.telegram_bot = None
        self.redis_client = None
        self.ml_model = None
        self.scaler = StandardScaler()
        
        # Trading parameters
        self.symbol = "NIFTY"
        self.instrument_token = None
        self.lot_size = 50
        self.max_positions = 3
        self.stop_loss_pct = 0.5
        self.target_pct = 1.0
        self.risk_per_trade = 1000
        
        # Data storage
        self.price_data = []
        self.positions = {}
        self.pnl_history = []
        
        # Status flags
        self.is_running = False
        self.market_hours = False
        
        # Initialize all components
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all required components"""
        try:
            # Initialize Kite Connect
            self.kite = KiteConnect(api_key=self.config['ZERODHA_API_KEY'])
            self.kite.set_access_token(self.config['ZERODHA_ACCESS_TOKEN'])
            
            # Test connection
            profile = self.kite.profile()
            logger.info(f"Connected to Zerodha as: {profile['user_name']}")
            
            # Initialize Telegram Bot
            self.telegram_bot = Bot(token=self.config['TELEGRAM_BOT_TOKEN'])
            
            # Test telegram connection
            bot_info = self.telegram_bot.get_me()
            logger.info(f"Connected to Telegram bot: {bot_info.username}")
            
            # Initialize Redis
            self.redis_client = redis.from_url(self.config['REDIS_URL'])
            self.redis_client.ping()
            logger.info("Connected to Redis")
            
            # Get instrument token for NIFTY
            instruments = self.kite.instruments("NSE")
            nifty_instrument = next((inst for inst in instruments if inst['name'] == 'NIFTY 50'), None)
            if nifty_instrument:
                self.instrument_token = nifty_instrument['instrument_token']
                logger.info(f"NIFTY instrument token: {self.instrument_token}")
            
            # Initialize ML model
            self.initialize_ml_model()
            
            # Send startup message
            self.send_telegram_message("🚀 Nifty Scalper Bot initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def initialize_ml_model(self):
        """Initialize machine learning model for signal generation"""
        try:
            # Create a simple Random Forest model
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Load historical data if available
            historical_data = self.load_historical_data()
            if len(historical_data) > 100:
                self.train_model(historical_data)
            
            logger.info("ML model initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML model: {e}")
    
    def load_historical_data(self):
        """Load historical data for training"""
        try:
            # Try to load from Redis first
            cached_data = self.redis_client.get('historical_data')
            if cached_data:
                return json.loads(cached_data)
            
            # If not in cache, fetch from Kite
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            historical_data = self.kite.historical_data(
                instrument_token=self.instrument_token,
                from_date=start_date,
                to_date=end_date,
                interval="minute"
            )
            
            # Cache the data
            self.redis_client.setex(
                'historical_data', 
                3600,  # 1 hour expiry
                json.dumps(historical_data, default=str)
            )
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return []
    
    def train_model(self, data):
        """Train the ML model with historical data"""
        try:
            df = pd.DataFrame(data)
            df['datetime'] = pd.to_datetime(df['date'])
            df.set_index('datetime', inplace=True)
            
            # Calculate technical indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.bbands(df['close'], length=20).iloc[:, 0], ta.bbands(df['close'], length=20).iloc[:, 1], ta.bbands(df['close'], length=20).iloc[:, 2]
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            
            # Create features
            df['price_change'] = df['close'].pct_change()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['ema_signal'] = np.where(df['ema_9'] > df['ema_21'], 1, 0)
            
            # Create target (next candle direction)
            df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
            
            # Prepare features
            feature_columns = ['rsi', 'macd', 'bb_position', 'price_change', 'volume_ratio', 'ema_signal']
            
            # Remove NaN values
            df_clean = df[feature_columns + ['target']].dropna()
            
            if len(df_clean) > 50:
                X = df_clean[feature_columns]
                y = df_clean['target']
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train model
                self.ml_model.fit(X_scaled, y)
                
                logger.info(f"Model trained with {len(df_clean)} samples")
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
    
    def get_market_data(self):
        """Get current market data"""
        try:
            # Get current quote
            quote = self.kite.quote(f"NSE:{self.symbol}")
            
            if self.symbol in quote:
                data = quote[self.symbol]
                return {
                    'ltp': data['last_price'],
                    'volume': data['volume'],
                    'change': data['net_change'],
                    'change_pct': data['net_change'] / data['last_price'] * 100,
                    'timestamp': datetime.now()
                }
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            
        return None
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for current data"""
        try:
            if len(data) < 50:
                return None
            
            df = pd.DataFrame(data)
            
            # Calculate indicators
            df['rsi'] = ta.rsi(df['ltp'], length=14)
            df['macd'] = ta.macd(df['ltp'])['MACD_12_26_9']
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.bbands(df['ltp'], length=20).iloc[:, 0], ta.bbands(df['ltp'], length=20).iloc[:, 1], ta.bbands(df['ltp'], length=20).iloc[:, 2]
            df['ema_9'] = ta.ema(df['ltp'], length=9)
            df['ema_21'] = ta.ema(df['ltp'], length=21)
            
            # Get latest values
            latest = df.iloc[-1]
            
            return {
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'bb_upper': latest['bb_upper'],
                'bb_middle': latest['bb_middle'],
                'bb_lower': latest['bb_lower'],
                'ema_9': latest['ema_9'],
                'ema_21': latest['ema_21'],
                'bb_position': (latest['ltp'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']),
                'ema_signal': 1 if latest['ema_9'] > latest['ema_21'] else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {e}")
            return None
    
    def generate_signal(self, current_data, indicators):
        """Generate trading signal using ML model and technical analysis"""
        try:
            if not self.ml_model or not indicators:
                return 'HOLD'
            
            # Prepare features for ML model
            features = [
                indicators['rsi'],
                indicators['macd'],
                indicators['bb_position'],
                current_data['change_pct'] / 100,
                current_data['volume'] / 1000000,  # Normalize volume
                indicators['ema_signal']
            ]
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get ML prediction
            ml_signal = self.ml_model.predict(features_scaled)[0]
            ml_probability = self.ml_model.predict_proba(features_scaled)[0]
            
            # Technical analysis signals
            rsi_oversold = indicators['rsi'] < 30
            rsi_overbought = indicators['rsi'] > 70
            macd_bullish = indicators['macd'] > 0
            bb_oversold = indicators['bb_position'] < 0.2
            bb_overbought = indicators['bb_position'] > 0.8
            ema_bullish = indicators['ema_signal'] == 1
            
            # Combine signals
            bullish_signals = sum([
                ml_signal == 1 and ml_probability[1] > 0.6,
                rsi_oversold,
                macd_bullish,
                bb_oversold,
                ema_bullish
            ])
            
            bearish_signals = sum([
                ml_signal == 0 and ml_probability[0] > 0.6,
                rsi_overbought,
                not macd_bullish,
                bb_overbought,
                not ema_bullish
            ])
            
            # Decision logic
            if bullish_signals >= 3:
                return 'BUY'
            elif bearish_signals >= 3:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Failed to generate signal: {e}")
            return 'HOLD'
    
    def place_order(self, signal, current_price):
        """Place order based on signal"""
        try:
            if signal == 'HOLD' or len(self.positions) >= self.max_positions:
                return
            
            # Calculate position size based on risk
            stop_loss_points = current_price * (self.stop_loss_pct / 100)
            position_size = int(self.risk_per_trade / stop_loss_points)
            position_size = min(position_size, self.lot_size * 2)  # Max 2 lots
            
            if self.config['DRY_RUN']:
                # Simulate order
                order_id = f"DRY_{int(time.time())}"
                
                self.positions[order_id] = {
                    'signal': signal,
                    'entry_price': current_price,
                    'quantity': position_size,
                    'timestamp': datetime.now(),
                    'stop_loss': current_price - stop_loss_points if signal == 'BUY' else current_price + stop_loss_points,
                    'target': current_price + (current_price * self.target_pct / 100) if signal == 'BUY' else current_price - (current_price * self.target_pct / 100),
                    'status': 'OPEN'
                }
                
                message = f"🔄 DRY RUN ORDER\n"
                message += f"Signal: {signal}\n"
                message += f"Price: ₹{current_price:.2f}\n"
                message += f"Quantity: {position_size}\n"
                message += f"Stop Loss: ₹{self.positions[order_id]['stop_loss']:.2f}\n"
                message += f"Target: ₹{self.positions[order_id]['target']:.2f}"
                
                self.send_telegram_message(message)
                
            else:
                # Place actual order
                order = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=self.kite.EXCHANGE_NSE,
                    tradingsymbol=self.symbol,
                    transaction_type=self.kite.TRANSACTION_TYPE_BUY if signal == 'BUY' else self.kite.TRANSACTION_TYPE_SELL,
                    quantity=position_size,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET
                )
                
                if order:
                    self.positions[order['order_id']] = {
                        'signal': signal,
                        'entry_price': current_price,
                        'quantity': position_size,
                        'timestamp': datetime.now(),
                        'stop_loss': current_price - stop_loss_points if signal == 'BUY' else current_price + stop_loss_points,
                        'target': current_price + (current_price * self.target_pct / 100) if signal == 'BUY' else current_price - (current_price * self.target_pct / 100),
                        'status': 'OPEN',
                        'order_id': order['order_id']
                    }
                    
                    message = f"✅ ORDER PLACED\n"
                    message += f"Signal: {signal}\n"
                    message += f"Price: ₹{current_price:.2f}\n"
                    message += f"Quantity: {position_size}\n"
                    message += f"Order ID: {order['order_id']}"
                    
                    self.send_telegram_message(message)
                    
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            self.send_telegram_message(f"❌ Order failed: {str(e)}")
    
    def manage_positions(self, current_price):
        """Manage existing positions"""
        try:
            positions_to_close = []
            
            for pos_id, position in self.positions.items():
                if position['status'] != 'OPEN':
                    continue
                
                signal = position['signal']
                entry_price = position['entry_price']
                stop_loss = position['stop_loss']
                target = position['target']
                
                # Check stop loss and target
                should_close = False
                close_reason = ""
                
                if signal == 'BUY':
                    if current_price <= stop_loss:
                        should_close = True
                        close_reason = "Stop Loss Hit"
                    elif current_price >= target:
                        should_close = True
                        close_reason = "Target Achieved"
                elif signal == 'SELL':
                    if current_price >= stop_loss:
                        should_close = True
                        close_reason = "Stop Loss Hit"
                    elif current_price <= target:
                        should_close = True
                        close_reason = "Target Achieved"
                
                if should_close:
                    positions_to_close.append((pos_id, close_reason))
            
            # Close positions
            for pos_id, reason in positions_to_close:
                self.close_position(pos_id, current_price, reason)
                
        except Exception as e:
            logger.error(f"Failed to manage positions: {e}")
    
    def close_position(self, position_id, current_price, reason):
        """Close a specific position"""
        try:
            position = self.positions[position_id]
            
            if self.config['DRY_RUN']:
                # Simulate position close
                pnl = self.calculate_pnl(position, current_price)
                position['status'] = 'CLOSED'
                position['exit_price'] = current_price
                position['pnl'] = pnl
                position['close_reason'] = reason
                
                self.pnl_history.append(pnl)
                
                message = f"🔄 DRY RUN CLOSE\n"
                message += f"Reason: {reason}\n"
                message += f"Entry: ₹{position['entry_price']:.2f}\n"
                message += f"Exit: ₹{current_price:.2f}\n"
                message += f"PnL: ₹{pnl:.2f}"
                
                self.send_telegram_message(message)
                
            else:
                # Close actual position
                signal = position['signal']
                quantity = position['quantity']
                
                order = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=self.kite.EXCHANGE_NSE,
                    tradingsymbol=self.symbol,
                    transaction_type=self.kite.TRANSACTION_TYPE_SELL if signal == 'BUY' else self.kite.TRANSACTION_TYPE_BUY,
                    quantity=quantity,
                    product=self.kite.PRODUCT_MIS,
                    order_type=self.kite.ORDER_TYPE_MARKET
                )
                
                if order:
                    pnl = self.calculate_pnl(position, current_price)
                    position['status'] = 'CLOSED'
                    position['exit_price'] = current_price
                    position['pnl'] = pnl
                    position['close_reason'] = reason
                    
                    self.pnl_history.append(pnl)
                    
                    message = f"✅ POSITION CLOSED\n"
                    message += f"Reason: {reason}\n"
                    message += f"Entry: ₹{position['entry_price']:.2f}\n"
                    message += f"Exit: ₹{current_price:.2f}\n"
                    message += f"PnL: ₹{pnl:.2f}\n"
                    message += f"Order ID: {order['order_id']}"
                    
                    self.send_telegram_message(message)
                    
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
    
    def calculate_pnl(self, position, current_price):
        """Calculate PnL for a position"""
        entry_price = position['entry_price']
        quantity = position['quantity']
        signal = position['signal']
        
        if signal == 'BUY':
            return (current_price - entry_price) * quantity
        else:
            return (entry_price - current_price) * quantity
    
    def is_market_open(self):
        """Check if market is open"""
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check market hours
        market_start = now.replace(
            hour=self.config['MARKET_START_HOUR'],
            minute=self.config['MARKET_START_MINUTE'],
            second=0,
            microsecond=0
        )
        
        market_end = now.replace(
            hour=self.config['MARKET_END_HOUR'],
            minute=self.config['MARKET_END_MINUTE'],
            second=0,
            microsecond=0
        )
        
        return market_start <= now <= market_end
    
    def send_telegram_message(self, message):
        """Send message to Telegram"""
        try:
            self.telegram_bot.send_message(
                chat_id=self.config['TELEGRAM_CHAT_ID'],
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def get_status_message(self):
        """Get current bot status"""
        try:
            current_data = self.get_market_data()
            
            if not current_data:
                return "❌ Unable to fetch market data"
            
            # Calculate total PnL
            total_pnl = sum(self.pnl_history)
            open_positions = len([p for p in self.positions.values() if p['status'] == 'OPEN'])
            
            status_msg = "📊 NIFTY SCALPER BOT STATUS\n\n"
            status_msg += f"• Current Price: ₹{current_data['ltp']:.2f}\n"
            status_msg += f"• Change: {current_data['change']:+.2f} ({current_data['change_pct']:+.2f}%)\n"
            status_msg += f"• Market: {'🟢 OPEN' if self.is_market_open() else '🔴 CLOSED'}\n"
            
            # Fix the emoji syntax error by using proper string formatting
            mode_text = "🧪 DRY RUN" if self.config["DRY_RUN"] else "💰 LIVE TRADING"
            status_msg += f"• Mode: {mode_text}\n"
            
            status_msg += f"• Bot Status: {'🟢 RUNNING' if self.is_running else '🔴 STOPPED'}\n"
            status_msg += f"• Open Positions: {open_positions}/{self.max_positions}\n"
            status_msg += f"• Total PnL: ₹{total_pnl:.2f}\n"
            status_msg += f"• Total Trades: {len(self.pnl_history)}\n"
            
            if len(self.pnl_history) > 0:
                win_rate = len([pnl for pnl in self.pnl_history if pnl > 0]) / len(self.pnl_history) * 100
                status_msg += f"• Win Rate: {win_rate:.1f}%\n"
            
            return status_msg
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return f"❌ Error getting status: {str(e)}"
    
    def main_trading_loop(self):
        """Main trading loop"""
        logger.info("Starting main trading loop")
        
        while self.is_running:
            try:
                # Check if market is open
                if not self.is_market_open():
                    time.sleep(60)  # Check every minute
                    continue
                
                # Get current market data
                current_data = self.get_market_data()
                if not current_data:
                    time.sleep(5)
                    continue
                
                # Store price data
                self.price_data.append(current_data)
                
                # Keep only last 1000 data points
                if len(self.price_data) > 1000:
                    self.price_data = self.price_data[-1000:]
                
                # Calculate technical indicators
                if len(self.price_data) >= 50:
                    indicators = self.calculate_technical_indicators(self.price_data)
                    
                    if indicators:
                        # Generate trading signal
                        signal = self.generate_signal(current_data, indicators)
                        
                        # Manage existing positions
                        self.manage_positions(current_data['ltp'])
                        
                        # Place new orders if signal is generated
                        if signal in ['BUY', 'SELL']:
                            self.place_order(signal, current_data['ltp'])
                
                # Store data in Redis for persistence
                self.redis_client.setex(
                    'bot_data',
                    300,  # 5 minutes expiry
                    json.dumps({
                        'price_data': self.price_data[-100:],  # Last 100 points
                        'positions': self.positions,
                        'pnl_history': self.pnl_history
                    }, default=str)
                )
                
                # Sleep for 5 seconds before next iteration
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                time.sleep(10)
    
    def start_bot(self):
        """Start the trading bot"""
        if self.is_running:
            return "Bot is already running"
        
        self.is_running = True
        
        # Start trading loop in a separate thread
        trading_thread = threading.Thread(target=self.main_trading_loop)
        trading_thread.daemon = True
        trading_thread.start()
        
        message = "🚀 Nifty Scalper Bot started successfully!"
        self.send_telegram_message(message)
        
        return message
    
    def stop_bot(self):
        """Stop the trading bot"""
        if not self.is_running:
            return "Bot is not running"
        
        self.is_running = False
        
        # Close all open positions
        if self.positions:
            current_data = self.get_market_data()
            if current_data:
                for pos_id, position in self.positions.items():
                    if position['status'] == 'OPEN':
                        self.close_position(pos_id, current_data['ltp'], "Bot Stopped")
        
        message = "🛑 Nifty Scalper Bot stopped successfully!"
        self.send_telegram_message(message)
        
        return message

# Initialize Flask app
app = Flask(__name__)
bot_instance = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'bot_running': bot_instance.is_running if bot_instance else False
    })

@app.route('/start', methods=['POST'])
def start_bot():
    """Start the bot"""
    global bot_instance
    try:
        if not bot_instance:
            bot_instance = NiftyScalperBot()
        
        result = bot_instance.start_bot()
        return jsonify({'status': 'success', 'message': result})
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop_bot():
    """Stop the bot"""
    global bot_instance
    try:
        if bot_instance:
            result = bot_instance.stop_bot()
            return jsonify({'status': 'success', 'message': result})
        else:
            return jsonify({'status': 'error', 'message': 'Bot not initialized'}), 400
    except Exception as e:
        logger.error(f"Failed to stop bot: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get bot status"""
    global bot_instance
    try:
        if bot_instance:
            status = bot_instance.get_status_message()
            return jsonify({'status': 'success', 'data': status})
        else:
            return jsonify({'status': 'error', 'message': 'Bot not initialized'}), 400
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/positions', methods=['GET'])
def get_positions():
    """Get current positions"""
    global bot_instance
    try:
        if bot_instance:
            positions = {k: v for k, v in bot_instance.positions.items() if v['status'] == 'OPEN'}
            return jsonify({'status': 'success', 'data': positions})
        else:
            return jsonify({'status': 'error', 'message': 'Bot not initialized'}), 400
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/pnl', methods=['GET'])
def get_pnl():
    """Get PnL history"""
    global bot_instance
    try:
        if bot_instance:
            total_pnl = sum(bot_instance.pnl_history)
            return jsonify({
                'status': 'success',
                'data': {
                    'total_pnl': total_pnl,
                    'trade_count': len(bot_instance.pnl_history),
                    'pnl_history': bot_instance.pnl_history[-50:]  # Last 50 trades
                }
            })
        else:
            return jsonify({'status': 'error', 'message': 'Bot not initialized'}), 400
    except Exception as e:
        logger.error(f"Failed to get PnL: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    try:
        # Initialize bot on startup
        bot_instance = NiftyScalperBot()
        
        # Start the Flask app
        port = int(os.getenv('PORT', 10000))
        logger.info(f"Starting Flask app on port {port}")
        
        serve(app, host='0.0.0.0', port=port)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

