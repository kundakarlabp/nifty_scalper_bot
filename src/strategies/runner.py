# src/strategies/runner.py
"""
The StrategyRunner is the central orchestrator of the trading bot.
It coordinates the data source, strategy, execution, and session management
to run the main trading loop.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from datetime import timedelta

from src.config import settings
from src.risk.position_sizing import PositionSizer
from src.risk.session import Trade
from src.signals.signal import Signal
from src.signals.regime_detector import detect_market_regime
from src.utils.strike_selector import get_instrument_tokens, is_market_open, fetch_cached_instruments

if TYPE_CHECKING:
    from src.data.source import DataSource
    from src.strategies.scalping_strategy import EnhancedScalpingStrategy
    from src.execution.order_executor import OrderExecutor
    from src.risk.session import TradingSession
    from src.notifications.telegram_controller import TelegramController

logger = logging.getLogger(__name__)


class StrategyRunner:
    def __init__(
        self,
        data_source: DataSource,
        strategy: EnhancedScalpingStrategy,
        order_executor: OrderExecutor,
        trading_session: TradingSession,
        position_sizer: PositionSizer,
        telegram_controller: TelegramController | None,
    ):
        self.data_source = data_source
        self.strategy = strategy
        self.executor = order_executor
        self.session = trading_session
        self.sizer = position_sizer
        self.telegram = telegram_controller
        self._running = False
        self._last_run_time = 0
        self.poll_interval_sec = 15  # Can be made configurable
        self.instrument_cache = {}
        self.cache_ttl_sec = 300
        self._last_signal_hash: str | None = None

    def start(self):
        self._running = True
        logger.info("StrategyRunner started.")
        while self._running:
            now = time.time()
            if (now - self._last_run_time) < self.poll_interval_sec:
                time.sleep(0.5)
                continue
            self._last_run_time = now
            self.tick()

    def stop(self):
        self._running = False
        logger.info("StrategyRunner stopped.")

    def _refresh_instrument_cache(self):
        if not self.instrument_cache or (time.time() - self.instrument_cache.get("timestamp", 0)) > self.cache_ttl_sec:
            logger.info("Refreshing instrument cache...")
            kite = getattr(self.executor, "kite", None)
            if kite:
                self.instrument_cache = fetch_cached_instruments(kite)
                self.instrument_cache["timestamp"] = time.time()

    def tick(self):
        if not is_market_open(settings.strategy.time_filter_start, settings.strategy.time_filter_end):
            logger.debug("Market is closed. Skipping tick.")
            return
        if (reason := self.session.check_risk_limits()) is not None:
            logger.warning(f"Trading halted due to risk limit breach: {reason}")
            self.stop()
            return

        self._refresh_instrument_cache()
        kite = getattr(self.executor, "kite", None)
        if not kite:
            return

        instruments = get_instrument_tokens(
            kite_instance=kite,
            spot_symbol=settings.strategy.spot_symbol,
            cached_nfo_instruments=self.instrument_cache.get("NFO", []),
            cached_nse_instruments=self.instrument_cache.get("NSE", []),
            strike_range=settings.strategy.strike_selection_range,
        )
        if not instruments:
            logger.debug("No instruments found for the current spot price.")
            return

        logger.debug(f"Found instruments: {instruments}")

        # --- Trailing Stop-Loss Logic ---
        for trade in list(self.session.active_trades.values()):
            ltp = self.data_source.get_spot_price(f"NFO:{trade.symbol}")
            if ltp:
                self.executor.update_trailing_stop(
                    order_id=trade.order_id,
                    current_price=ltp,
                    atr=trade.atr_at_entry,
                    atr_multiplier=settings.strategy.atr_sl_multiplier,
                )

        spot_token = instruments.get("spot_token")
        spot_df = self.data_source.fetch_ohlc(spot_token, datetime.now() - timedelta(minutes=60), datetime.now(), "minute") if spot_token else None

        for option_type in ["ce", "pe"]:
            option_token = instruments.get(f"{option_type}_token")
            option_symbol = instruments.get(f"{option_type}_symbol")
            if not option_token or not option_symbol:
                continue

            option_df = self.data_source.fetch_ohlc(option_token, datetime.now() - timedelta(minutes=100), datetime.now(), "minute")
            if option_df is None or len(option_df) < settings.strategy.min_bars_for_signal:
                logger.debug(f"Insufficient data for {option_symbol}. Have {len(option_df) if option_df is not None else 0} bars, need {settings.strategy.min_bars_for_signal}.")
                continue

            ltp = self.data_source.get_spot_price(f"NFO:{option_symbol}")
            if ltp is None: continue

            if spot_df is None or spot_df.empty:
                logger.debug("Spot OHLC data is missing, cannot generate signal.")
                continue

            signal: Signal | None = self.strategy.generate_signal(df=option_df, current_price=ltp, spot_df=spot_df)

            if not signal:
                continue

            # --- Quality Filter ---
            # Example: In a ranging market, require a higher score.
            # This logic can be expanded.
            regime = detect_market_regime(spot_df, spot_df["adx"], spot_df["di_plus"], spot_df["di_minus"], self.strategy.adx_trend_strength)
            if regime == "range" and signal.score < (settings.strategy.min_signal_score + 1):
                logger.debug(f"Skipping signal for {option_symbol} due to low score in ranging market.")
                continue

            # --- De-duplication Check ---
            if signal.hash and signal.hash == self._last_signal_hash:
                logger.debug(f"Duplicate signal {signal.hash} for {option_symbol}. Skipping.")
                continue
            self._last_signal_hash = signal.hash

            logger.info(f"Signal received for {option_symbol}: {signal}")

            quantity = self.sizer.calculate_quantity(
                session=self.session,
                entry_price=signal.entry_price,
                stop_loss_price=signal.stop_loss,
                lot_size=settings.executor.nifty_lot_size,
            )
            if quantity > 0:
                order_id = self.executor.place_entry_order(
                    symbol=option_symbol,
                    exchange="NFO",
                    transaction_type=signal.signal,
                    quantity=quantity,
                )
                if order_id:
                    trade = Trade(
                        symbol=option_symbol,
                        direction=signal.signal,
                        entry_price=signal.entry_price,
                        quantity=quantity,
                        order_id=order_id,
                        atr=signal.market_volatility,
                    )
                    self.session.add_trade(trade)
                    self.executor.setup_gtt_orders(
                        entry_order_id=order_id,
                        entry_price=signal.entry_price,
                        stop_loss_price=signal.stop_loss,
                        target_price=signal.target,
                        symbol=option_symbol,
                        exchange="NFO",
                        quantity=quantity,
                        transaction_type=signal.signal,
                    )
                    if self.telegram:
                        self.telegram.send_message(f"Trade executed: {signal}")
