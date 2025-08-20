# src/strategies/runner.py
"""
The StrategyRunner is the central orchestrator of the trading bot.
It coordinates the data source, strategy, execution, and session management
to run the main trading loop.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd
from ta.trend import ADXIndicator

from src.config import settings
from src.risk.position_sizing import PositionSizer
from src.risk.session import Trade
from src.signals.signal import Signal
from src.signals.regime_detector import detect_market_regime
from src.utils.strike_selector import (
    get_instrument_tokens,
    is_market_open,
    fetch_cached_instruments,
)

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
        data_source: "DataSource",
        strategy: "EnhancedScalpingStrategy",
        order_executor: "OrderExecutor",
        trading_session: "TradingSession",
        position_sizer: PositionSizer,
        telegram_controller: Optional["TelegramController"],
    ):
        self.data_source = data_source
        self.strategy = strategy
        self.executor = order_executor
        self.session = trading_session
        self.sizer = position_sizer
        self.telegram = telegram_controller

        self._running = False
        self._last_run_time = 0.0

        # Poll every 15s by default (configurable if needed)
        self.poll_interval_sec = 15

        # Instrument cache (NSE/NFO) refreshed periodically
        self.instrument_cache: Dict[str, object] = {}
        self.cache_ttl_sec = 300

        # De-dup signals per symbol
        self._last_signal_hash_by_symbol: Dict[str, str] = {}

    # ----------------------- lifecycle -----------------------

    def start(self):
        self._running = True
        logger.info("StrategyRunner started.")
        while self._running:
            now = time.time()
            if (now - self._last_run_time) < self.poll_interval_sec:
                time.sleep(0.5)
                continue
            self._last_run_time = now
            try:
                self.tick()
            except Exception as e:
                logger.error("Runner tick error: %s", e, exc_info=True)
                time.sleep(1)

    def stop(self):
        self._running = False
        logger.info("StrategyRunner stopped.")

    # ----------------------- internals -----------------------

    def _refresh_instrument_cache(self):
        ts = self.instrument_cache.get("timestamp", 0)
        if not self.instrument_cache or (time.time() - float(ts)) > self.cache_ttl_sec:
            logger.info("Refreshing instrument cache...")
            kite = getattr(self.executor, "kite", None)
            if kite:
                cache = fetch_cached_instruments(kite)
                if isinstance(cache, dict):
                    cache["timestamp"] = time.time()
                    self.instrument_cache = cache

    def _ensure_adx_di(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Ensure ADX, DI+ and DI- columns exist on df. Returns a new df with columns:
        ['open','high','low','close','volume','adx','di_plus','di_minus'].
        """
        if df is None or df.empty:
            return df

        cols = set(df.columns.str.lower())
        need_calc = not {"adx", "di_plus", "di_minus"}.issubset(cols)
        if not need_calc:
            # Normalize column names to lower-case for consistency
            out = df.copy()
            if "ADX" in df.columns:
                out.rename(columns={"ADX": "adx"}, inplace=True)
            if "DI+" in df.columns or "di+" in df.columns:
                out.rename(columns={"DI+": "di_plus", "di+": "di_plus"}, inplace=True)
            if "DI-" in df.columns or "di-" in df.columns:
                out.rename(columns={"DI-": "di_minus", "di-": "di_minus"}, inplace=True)
            return out

        try:
            hi = df["high"].astype(float)
            lo = df["low"].astype(float)
            cl = df["close"].astype(float)
            adx_ind = ADXIndicator(high=hi, low=lo, close=cl, window=window, fillna=False)
            out = df.copy()
            out["adx"] = adx_ind.adx()
            out["di_plus"] = adx_ind.adx_pos()
            out["di_minus"] = adx_ind.adx_neg()
            return out
        except KeyError:
            # Missing required OHLC columns
            return df
        except Exception as e:
            logger.debug("Failed to compute ADX/DI: %s", e)
            return df

    def _finalize_fills_if_any(self):
        """
        Ask executor to sync OCO state and finalize session trades for any fills detected.
        """
        try:
            fills = self.executor.sync_and_enforce_oco()
        except Exception as e:
            logger.debug("sync_and_enforce_oco failed: %s", e)
            return

        if not fills:
            return

        for order_id, exit_px in fills:
            try:
                closed = self.session.finalize_trade(order_id, float(exit_px))
                if closed and self.telegram:
                    self.telegram.send_message(
                        f"🏁 Closed {closed.symbol} ({closed.direction}) @ {exit_px:.2f} | "
                        f"PnL: {closed.net_pnl:.2f}"
                    )
            except Exception:
                logger.exception("Failed to finalize trade %s", order_id)

    # ----------------------- main tick -----------------------

    def tick(self):
        # 1) Market hours & risk gates
        if not is_market_open(settings.strategy.time_filter_start, settings.strategy.time_filter_end):
            logger.debug("Market is closed. Skipping tick.")
            return

        reason = self.session.check_risk_limits()
        if reason is not None:
            logger.warning("Trading halted due to risk limit breach: %s", reason)
            self.stop()
            return

        # 2) Instruments (with periodic cache refresh)
        self._refresh_instrument_cache()
        kite = getattr(self.executor, "kite", None)
        if not kite:
            logger.debug("No broker client attached; skipping tick.")
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

        logger.debug("Instrument selection: %s", instruments)

        # 3) Trailing SL updates for open trades
        for trade in list(self.session.active_trades.values()):
            try:
                ltp = self.data_source.get_spot_price(f"NFO:{trade.symbol}")
                if ltp:
                    self.executor.update_trailing_stop(
                        order_id=trade.order_id,
                        current_price=float(ltp),
                        atr=float(trade.atr_at_entry or 0.0),
                        atr_multiplier=float(settings.strategy.atr_sl_multiplier),
                    )
            except Exception as e:
                logger.debug("Trailing update failed for %s: %s", trade.order_id, e)

        # 4) Build spot dataframe with indicators (lookback = configured minutes)
        lookback_min = int(getattr(settings.executor, "data_lookback_minutes", 15))
        now = datetime.now()
        frm = now - timedelta(minutes=lookback_min)

        spot_token = instruments.get("spot_token")
        spot_df = None
        if spot_token:
            spot_df = self.data_source.fetch_ohlc(spot_token, frm, now, "minute")
            if spot_df is not None and not spot_df.empty:
                spot_df = self._ensure_adx_di(spot_df, window=int(settings.strategy.atr_period))
        if spot_df is None or spot_df.empty:
            logger.debug("Spot OHLC data is missing; skipping signal generation this tick.")
            self._finalize_fills_if_any()
            return

        # 5) Try both CE and PE legs around ATM
        min_bars = int(settings.strategy.min_bars_for_signal)
        adx_trend_threshold = int(getattr(self.strategy, "adx_trend_strength", 20))

        for option_type in ("ce", "pe"):
            option_token = instruments.get(f"{option_type}_token")
            option_symbol = instruments.get(f"{option_type}_symbol")
            if not option_token or not option_symbol:
                continue

            # Fetch option OHLC (slightly longer buffer than lookback for signals)
            opt_df = self.data_source.fetch_ohlc(option_token, frm - timedelta(minutes=lookback_min), now, "minute")
            if opt_df is None or len(opt_df) < min_bars:
                logger.debug(
                    "Insufficient data for %s. Have %d bars, need %d.",
                    option_symbol, (0 if opt_df is None else len(opt_df)), min_bars
                )
                continue

            # Latest trade price for the option
            ltp = self.data_source.get_spot_price(f"NFO:{option_symbol}")
            if ltp is None:
                continue
            ltp = float(ltp)

            # Ask strategy for a signal
            signal: Optional[Signal] = None
            try:
                signal = self.strategy.generate_signal(df=opt_df, current_price=ltp, spot_df=spot_df)
            except Exception as e:
                logger.error("Strategy error for %s: %s", option_symbol, e, exc_info=True)
                continue

            if not signal:
                continue

            # Ensure hashed identity for dedupe
            if not getattr(signal, "hash", None):
                try:
                    signal.compute_hash()
                except Exception:
                    pass

            # Regime-aware filter
            try:
                regime = detect_market_regime(
                    spot_df,
                    adx=spot_df.get("adx", pd.Series(dtype=float)),
                    di_plus=spot_df.get("di_plus", pd.Series(dtype=float)),
                    di_minus=spot_df.get("di_minus", pd.Series(dtype=float)),
                    adx_trend_strength=adx_trend_threshold,
                )
            except Exception:
                regime = "unknown"

            if regime == "range" and int(signal.score) < (int(settings.strategy.min_signal_score) + 1):
                logger.debug("Skipping %s in range regime due to low score.", option_symbol)
                continue

            # Per-symbol dedupe
            last_h = self._last_signal_hash_by_symbol.get(option_symbol)
            if signal.hash and last_h and signal.hash == last_h:
                logger.debug("Duplicate signal %s for %s. Skipping.", signal.hash, option_symbol)
                continue
            if signal.hash:
                self._last_signal_hash_by_symbol[option_symbol] = signal.hash

            logger.info("Signal for %s: %s", option_symbol, signal.to_dict() if hasattr(signal, "to_dict") else signal)

            # Position sizing (contracts, multiple of lot size)
            quantity = self.sizer.calculate_quantity(
                session=self.session,
                entry_price=float(signal.entry_price),
                stop_loss_price=float(signal.stop_loss),
                lot_size=int(settings.executor.nifty_lot_size),
            )
            if quantity <= 0:
                logger.debug("Sizer returned 0 for %s; skipping.", option_symbol)
                continue

            # Place entry order
            order_id = self.executor.place_entry_order(
                symbol=str(option_symbol),
                exchange="NFO",
                transaction_type=str(signal.signal),
                quantity=int(quantity),
            )
            if not order_id:
                logger.warning("Failed to place entry order for %s.", option_symbol)
                continue

            # Register trade in session
            trade = Trade(
                symbol=str(option_symbol),
                direction=str(signal.signal),
                entry_price=float(signal.entry_price),
                quantity=int(quantity),
                order_id=str(order_id),
                atr=float(getattr(signal, "market_volatility", 0.0) or 0.0),
            )
            self.session.add_trade(trade)

            # Setup exits (GTT or regular)
            ok = self.executor.setup_gtt_orders(
                entry_order_id=str(order_id),
                entry_price=float(signal.entry_price),
                stop_loss_price=float(signal.stop_loss),
                target_price=float(signal.target),
                symbol=str(option_symbol),
                exchange="NFO",
                quantity=int(quantity),
                transaction_type=str(signal.signal),
            )
            if not ok:
                logger.warning("Failed to create exits for %s (oid=%s).", option_symbol, order_id)

            # Telegram alert (structured)
            if self.telegram:
                try:
                    self.telegram.send_signal_alert(
                        token=int(option_token),
                        signal=(signal.to_dict() if hasattr(signal, "to_dict") else {
                            "signal": signal.signal,
                            "entry_price": signal.entry_price,
                            "stop_loss": signal.stop_loss,
                            "target": signal.target,
                            "confidence": signal.confidence,
                        }),
                        position={"quantity": int(quantity)},
                    )
                except Exception:
                    # fallback to simple text on any failure
                    self.telegram.send_message(f"Trade executed: {option_symbol} {signal.signal} x{quantity} @ {signal.entry_price}")

        # 6) After placing / managing orders, sync for any fills and finalize
        self._finalize_fills_if_any()
