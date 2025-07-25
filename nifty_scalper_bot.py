# nifty_scalper_bot.py

import asyncio
import logging
from datetime import datetime
import pytz

# Import the classes from your other files
from config import Config
from utils import is_market_open, format_currency
from signal_generator import SignalGenerator
from telegram_bot import TelegramBot

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- RiskManager Class ---
# Manages capital, position sizing, and enforces risk rules.
class RiskManager:
    def __init__(self, initial_balance):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.todays_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None

    def can_trade(self):
        """Check if a new trade is allowed based on risk rules."""
        if self.circuit_breaker_active and datetime.now(pytz.UTC) < self.circuit_breaker_until:
            logger.warning(f"Circuit breaker is active. No new trades until {self.circuit_breaker_until}.")
            return False
        if self.daily_trades >= Config.MAX_DAILY_TRADES:
            logger.warning("Max daily trades limit reached.")
            return False
        if self.todays_pnl < - (self.initial_balance * Config.MAX_DAILY_LOSS_PCT):
            logger.error("Max daily loss limit reached. Stopping for the day.")
            return False
        return True

    def calculate_position_size(self, stop_loss_points):
        """Calculates position size based on risk per trade."""
        if stop_loss_points <= 0:
            return 0
        risk_amount = self.current_balance * Config.RISK_PER_TRADE_PCT
        quantity = risk_amount / stop_loss_points
        # Assuming Nifty lot size from config
        num_lots = max(1, round(quantity / Config.NIFTY_LOT_SIZE))
        return int(num_lots * Config.NIFTY_LOT_SIZE)

    def record_trade(self, pnl):
        """Update risk metrics after a trade is closed."""
        self.daily_trades += 1
        self.todays_pnl += pnl
        self.current_balance += pnl
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
                self.activate_circuit_breaker()
        else:
            self.consecutive_losses = 0

    def activate_circuit_breaker(self):
        """Activates the circuit breaker, pausing trading."""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now(pytz.UTC) + asyncio.timedelta(minutes=Config.CIRCUIT_BREAKER_PAUSE_MINUTES)
        logger.critical(f"CIRCUIT BREAKER ACTIVATED for {Config.CIRCUIT_BREAKER_PAUSE_MINUTES} minutes due to {self.consecutive_losses} consecutive losses.")
        # Optionally notify via Telegram
        if hasattr(self, 'telegram_bot') and self.telegram_bot:
            self.telegram_bot.notify_circuit_breaker(self.consecutive_losses, Config.CIRCUIT_BREAKER_PAUSE_MINUTES)

# --- Main Bot Class ---
# Orchestrates all components of the trading bot.
class NiftyScalperBot:
    def __init__(self):
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(Config.INITIAL_CAPITAL)
        self.telegram_bot = TelegramBot(trading_bot_instance=self)
        self.risk_manager.telegram_bot = self.telegram_bot # Link for notifications
        self.current_position = None
        self.auto_trade = True # Auto-trading is ON by default

    async def run(self):
        """Main execution loop for the bot."""
        logger.info("Nifty Scalper Bot v2.0 is starting...")
        
        # Start the Telegram bot as a background task
        telegram_task = asyncio.create_task(self.telegram_bot.start_bot())

        while True:
            try:
                if is_market_open():
                    market_data = self.get_market_data() # Placeholder for your data feed
                    
                    # If we have an open position, check if we should exit
                    if self.current_position:
                        self.check_exit_conditions(market_data)
                    
                    # If no position and auto-trading is on, check for new entry signals
                    elif self.auto_trade and self.risk_manager.can_trade():
                        signal = self.signal_generator.generate_signal(market_data)
                        if signal:
                            self.execute_trade(signal)
                
                # Log current status periodically
                logger.info(f"P&L: {format_currency(self.risk_manager.todays_pnl)} | Balance: {format_currency(self.risk_manager.current_balance)}")

            except Exception as e:
                logger.error(f"An error occurred in the main loop: {e}", exc_info=True)

            await asyncio.sleep(Config.TICK_INTERVAL_SECONDS) # Wait for the next tick

    def get_market_data(self):
        """
        Placeholder for your live market data feed API call.
        This should return a dictionary with 'ltp', 'volume', etc.
        """
        # In a real scenario, this would be an API call.
        # For demonstration, we simulate a price tick.
        # You MUST replace this with your actual data provider (e.g., Zerodha Kite, Angel One).
        import random
        simulated_price = 24800 + random.uniform(-50, 50)
        return {
            'ltp': simulated_price,
            'volume': random.randint(10000, 50000),
            'timestamp': datetime.now(pytz.UTC)
        }

    def execute_trade(self, signal):
        """Executes a new trade based on a signal."""
        logger.info(f"Executing trade for signal: {signal}")
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        
        stop_loss_points = abs(entry_price - stop_loss)
        quantity = self.risk_manager.calculate_position_size(stop_loss_points)

        if quantity == 0:
            logger.warning("Trade skipped: Calculated position size is zero.")
            return

        # --- PLACE ORDER VIA BROKER API HERE ---
        # This is a critical placeholder. You need to integrate your broker's API.
        # order_response = broker.place_order(symbol="NIFTY", direction=signal['direction'], quantity=quantity)
        # if not order_response.is_success:
        #     logger.error("Failed to place order with broker.")
        #     return
        
        self.current_position = {
            'direction': signal['direction'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'target': signal['target'],
            'quantity': quantity,
            'entry_time': datetime.now(pytz.UTC)
        }
        logger.critical(f"NEW POSITION OPENED: {self.current_position}")
        self.telegram_bot.notify_trade_entry(self.current_position)

    def check_exit_conditions(self, market_data):
        """Checks if the current position should be closed."""
        ltp = market_data['ltp']
        pos = self.current_position
        
        exit_reason = None
        if pos['direction'] == 'BUY':
            if ltp >= pos['target']: exit_reason = "Target hit"
            elif ltp <= pos['stop_loss']: exit_reason = "Stop-loss hit"
        elif pos['direction'] == 'SELL':
            if ltp <= pos['target']: exit_reason = "Target hit"
            elif ltp >= pos['stop_loss']: exit_reason = "Stop-loss hit"
        
        if exit_reason:
            self.close_position(exit_reason, ltp)

    def close_position(self, reason, exit_price):
        """Closes the current open position."""
        if not self.current_position:
            return

        # --- PLACE EXIT ORDER VIA BROKER API HERE ---
        logger.info(f"Closing position due to: {reason}")

        pos = self.current_position
        pnl = 0
        if pos['direction'] == 'BUY':
            pnl = (exit_price - pos['entry_price']) * pos['quantity']
        else: # SELL
            pnl = (pos['entry_price'] - exit_price) * pos['quantity']
        
        self.risk_manager.record_trade(pnl)
        
        exit_data = {**pos, 'exit_price': exit_price, 'pnl': pnl, 'reason': reason}
        logger.critical(f"POSITION CLOSED: P&L {format_currency(pnl)}")
        self.telegram_bot.notify_trade_exit(exit_data)
        
        self.current_position = None
        return True # Indicate success

# --- Main Execution Block ---
async def main():
    bot = NiftyScalperBot()
    try:
        await bot.run()
    except asyncio.CancelledError:
        logger.info("Main task cancelled. Shutting down.")
    finally:
        logger.info("Initiating graceful shutdown of Telegram bot...")
        await bot.telegram_bot.stop_bot()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually by user (Ctrl+C).")

