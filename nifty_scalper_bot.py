# nifty_scalper_bot.py
# CORRECTED AND CONSOLIDATED VERSION

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
import pytz

# Import all custom components.
# Make sure config.py, utils.py, signal_generator.py, telegram_bot.py, and broker_manager.py are present.
from config import Config
from utils import is_market_open, format_currency, get_market_status
from signal_generator import SignalGenerator
from telegram_bot import TelegramBot
from broker_manager import BrokerManager

# --- 1. Setup Professional Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("nifty_scalper_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
IST = pytz.timezone('Asia/Kolkata')

# --- 2. RiskManager Class (No changes needed here) ---
class RiskManager:
    def __init__(self, telegram_bot=None):
        self.initial_balance = Config.INITIAL_CAPITAL
        self.current_balance = Config.INITIAL_CAPITAL
        self.todays_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        self.telegram_bot = telegram_bot

    def can_trade(self):
        if self.circuit_breaker_active:
            if datetime.now(IST) < self.circuit_breaker_until:
                return False
            else:
                logger.info("Circuit breaker period ended. Resuming trading.")
                self.circuit_breaker_active = False
        if self.daily_trades >= Config.MAX_DAILY_TRADES:
            logger.warning("Max daily trades limit reached.")
            return False
        max_loss = self.initial_balance * Config.MAX_DAILY_LOSS_PCT
        if self.todays_pnl < -max_loss:
            logger.critical(f"Max daily loss limit of {format_currency(max_loss)} reached. Stopping trading.")
            return False
        return True

    def calculate_position_size(self, stop_loss_points):
        if stop_loss_points <= 0: return 0
        risk_amount = self.current_balance * Config.RISK_PER_TRADE_PCT
        quantity = risk_amount / stop_loss_points
        num_lots = max(1, round(quantity / Config.NIFTY_LOT_SIZE))
        return int(num_lots * Config.NIFTY_LOT_SIZE)

    def record_trade(self, pnl):
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
        pause_duration = timedelta(minutes=Config.CIRCUIT_BREAKER_PAUSE_MINUTES)
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now(IST) + pause_duration
        logger.critical(f"CIRCUIT BREAKER ACTIVATED for {Config.CIRCUIT_BREAKER_PAUSE_MINUTES} minutes.")
        if self.telegram_bot:
            self.telegram_bot.notify_circuit_breaker(self.consecutive_losses, Config.CIRCUIT_BREAKER_PAUSE_MINUTES)

    def reset_daily_stats(self):
        logger.info("Resetting daily trading statistics for the new day.")
        self.todays_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0

# --- 3. Main Bot Class with State Management ---
class NiftyScalperBot:
    def __init__(self):
        Config.validate()
        self.signal_generator = SignalGenerator()
        self.broker = BrokerManager()
        self.telegram_bot = TelegramBot(trading_bot_instance=self)
        self.risk_manager = RiskManager(telegram_bot=self.telegram_bot)
        self.auto_trade = not Config.DRY_RUN
        self.last_reset_day = -1
        self.current_position = self._load_position_state()

    def _load_position_state(self):
        if os.path.exists(Config.STATE_FILE):
            try:
                with open(Config.STATE_FILE, 'r') as f:
                    position = json.load(f)
                    logger.critical(f"Loaded existing position from state file: {position}")
                    return position
            except Exception as e:
                logger.error(f"Error loading state file '{Config.STATE_FILE}': {e}. Removing corrupt file.")
                os.remove(Config.STATE_FILE)
        return None

    def _save_position_state(self):
        if self.current_position:
            with open(Config.STATE_FILE, 'w') as f:
                json.dump(self.current_position, f, indent=4)
                logger.info(f"Saved current position state to '{Config.STATE_FILE}'.")

    def _clear_position_state(self):
        if os.path.exists(Config.STATE_FILE):
            os.remove(Config.STATE_FILE)
            logger.info(f"Cleared position state file '{Config.STATE_FILE}'.")

    # *** CHANGE 1: Renamed 'run' to 'start_trading_loop' and removed Telegram task creation ***
    async def start_trading_loop(self):
        """This is the dedicated loop for the core trading logic."""
        logger.info(f"Trading logic loop starting... | DRY RUN: {Config.DRY_RUN}")
        
        # The line that started the telegram bot has been REMOVED from here.

        while True:
            try:
                now = datetime.now(IST)
                if now.day != self.last_reset_day:
                    self.risk_manager.reset_daily_stats()
                    self.last_reset_day = now.day

                if is_market_open():
                    market_data = self.get_market_data()
                    if self.current_position:
                        await self.manage_open_position(market_data)
                    elif self.auto_trade and self.risk_manager.can_trade():
                        signal = self.signal_generator.generate_signal(market_data)
                        if signal:
                            await self.execute_trade(signal)

                await asyncio.sleep(Config.TICK_INTERVAL_SECONDS)
            except Exception as e:
                logger.error(f"Critical error in main loop: {e}", exc_info=True)
                await asyncio.sleep(Config.TICK_INTERVAL_SECONDS * 5)

    def get_market_data(self):
        import random
        price = 25000 + random.uniform(-50, 50)
        return {'ltp': price, 'volume': random.randint(10000, 50000), 'open': price-10, 'high': price+10, 'low': price-10, 'timestamp': datetime.now(IST)}

    async def execute_trade(self, signal: dict):
        instrument = self.broker.get_instrument_for_option(signal['underlying_price'], signal['option_type'])
        if not instrument: return

        option_ltp = self.broker.get_ltp(instrument)
        if option_ltp <= 0: return

        delta_factor = 0.5
        underlying_sl_points = abs(signal['underlying_price'] - signal['underlying_stop_loss'])
        option_sl_points = underlying_sl_points * delta_factor
        option_tp_points = abs(signal['underlying_price'] - signal['underlying_target']) * delta_factor

        quantity = self.risk_manager.calculate_position_size(underlying_sl_points)
        if quantity <= 0: return

        order_id = self.broker.place_gtt_oco_order(instrument, "BUY", quantity, option_ltp, option_ltp + option_tp_points, option_ltp - option_sl_points)
        if not order_id: return

        self.current_position = {
            'order_id': order_id, 'instrument': instrument, 'quantity': quantity,
            'entry_price': option_ltp, 'stop_loss': option_ltp - option_sl_points,
            'target': option_ltp + option_tp_points, 'status': 'OPEN'
        }
        self._save_position_state()
        logger.critical(f"NEW GTT OCO POSITION PLACED: {self.current_position}")
        self.telegram_bot.notify_trade_entry(self.current_position)

    async def manage_open_position(self, market_data: dict):
        pos = self.current_position
        pass

    async def handle_position_closure(self, exit_price: float, reason: str):
        if not self.current_position: return

        pos = self.current_position
        pnl = (exit_price - pos['entry_price']) * pos['quantity']

        self.risk_manager.record_trade(pnl)

        exit_data = {**pos, 'exit_price': exit_price, 'pnl': pnl, 'reason': reason}
        logger.critical(f"POSITION CLOSED: P&L {format_currency(pnl)} | Reason: {reason}")
        self.telegram_bot.notify_trade_exit(exit_data)

        self.current_position = None
        self._clear_position_state()

    async def manual_exit_position(self) -> bool:
        if not self.current_position:
            logger.warning("Manual exit requested, but no active position found.")
            return False

        order_id = self.current_position['order_id']
        logger.info(f"Attempting to manually cancel GTT order: {order_id}")

        success = self.broker.cancel_order(order_id)
        if success:
            await self.handle_position_closure(self.current_position['entry_price'], "Manual Exit")
            return True
        else:
            logger.error(f"Failed to cancel GTT order {order_id} via broker API.")
            return False

# --- 4. Main Execution Block ---
# *** CHANGE 2: Updated 'main' to run both tasks concurrently with asyncio.gather ***
async def main():
    bot = NiftyScalperBot()
    try:
        logger.info("Starting all bot components concurrently...")
        
        # Create a task for the trading loop using the renamed method
        trading_loop_task = asyncio.create_task(bot.start_trading_loop())

        # Create a task for the Telegram bot
        telegram_bot_task = asyncio.create_task(bot.telegram_bot.start_bot())

        # Use asyncio.gather to run both tasks concurrently.
        # This is the correct way to manage multiple long-running async functions.
        await asyncio.gather(trading_loop_task, telegram_bot_task)

    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    finally:
        logger.info("Initiating graceful shutdown...")
        if bot.telegram_bot and bot.telegram_bot.is_running:
            await bot.telegram_bot.stop_bot()
        logger.info("Bot has been shut down.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually by user (Ctrl+C).")
    except ValueError as e:
        logger.fatal(f"Configuration validation failed: {e}")

