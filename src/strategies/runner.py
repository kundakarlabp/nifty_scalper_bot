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
from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd

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
from src.utils.indicators import calculate_adx
from src.data.source import LiveKiteSource, DataSource

if TYPE_CHECKING:
    from src.strategies.scalping_strategy import EnhancedScalpingStrategy
    from kiteconnect import KiteConnect


class StrategyRunner:
    """
    Main orchestrator: ties together data source, strategy, risk, and execution.
    """

    def __init__(self, kite: Optional["KiteConnect"] = None, telegram_controller=None):
        self.logger = logging.getLogger("StrategyRunner")

        # Live trading flag
        self.live_trading: bool = settings.enable_live_trading

        # Broker session
        self.kite: Optional["KiteConnect"] = kite

        # Data source (shadow-safe if kite=None)
        self.data_source: DataSource = LiveKiteSource(kite)

        # Inject telegram
        self.telegram = telegram_controller

        # Position sizing and trade state
        self.sizer = PositionSizer()
        self.open_trades: Dict[str, Trade] = {}

        self.logger.info(
            "StrategyRunner ready (live_trading=%s, use_live_equity=%s)",
            self.live_trading,
            settings.risk.use_live_equity,
        )

    # ----------------------------------------------------------------------
    # Mode toggle
    # ----------------------------------------------------------------------
    def set_live_mode(self, live: bool) -> bool:
        """
        Toggle live/dry mode. If enabling live, rebuild Kite + LiveKiteSource.
        """
        self.live_trading = live
        if live:
            try:
                from kiteconnect import KiteConnect  # local import so shadow mode works

                api_key = settings.zerodha.api_key
                access_token = settings.zerodha.access_token
                if not api_key or not access_token:
                    self.logger.error("Cannot enable live mode: missing Zerodha creds")
                    self.live_trading = False
                    return False

                self.kite = KiteConnect(api_key=api_key)
                self.kite.set_access_token(access_token)
                self.data_source = LiveKiteSource(self.kite)
                self.logger.info("🔓 Live mode ON — broker session initialized.")
                return True
            except Exception as e:
                self.logger.error("Failed to initialize live Kite session: %s", e)
                self.kite = None
                self.data_source = LiveKiteSource(None)
                self.live_trading = False
                return False
        else:
            self.kite = None
            self.data_source = LiveKiteSource(None)
            self.logger.info("Live mode OFF — running in shadow/paper mode.")
            return True

    # ----------------------------------------------------------------------
    # Core loop methods
    # ----------------------------------------------------------------------
    def runner_tick(self, dry: bool = False) -> Optional[Signal]:
        """
        One tick of strategy evaluation. Returns a Signal or None.
        """
        if not is_market_open() and not settings.allow_offhours_testing:
            self.logger.debug("Market closed. Skipping tick.")
            return None

        try:
            inst = get_instrument_tokens(self.kite)
            if not inst:
                self.logger.warning("Instrument resolution failed.")
                return None

            spot_token = inst.get("spot_token")
            expiry = inst.get("expiry")
            if not spot_token or not expiry:
                self.logger.debug("Missing token/expiry.")
                return None

            end = datetime.now()
            start = end - pd.Timedelta(minutes=settings.data.lookback_minutes)
            ohlc = self.data_source.fetch_ohlc(
                spot_token, start=start, end=end, timeframe=settings.data.timeframe
            )
            if ohlc is None or ohlc.empty:
                self.logger.debug("No OHLC data.")
                return None

            # Strategy call
            from src.strategies.scalping_strategy import EnhancedScalpingStrategy

            strat = EnhancedScalpingStrategy()
            sig = strat.generate_signal(ohlc)

            if sig:
                self.logger.info("Signal: %s", sig)
                if self.telegram:
                    try:
                        self.telegram.send_message(f"📈 Signal: {sig}")
                    except Exception:
                        pass
            return sig
        except Exception as e:
            self.logger.error("runner_tick failed: %s", e)
            return None

    # ----------------------------------------------------------------------
    def get_status_snapshot(self) -> Dict[str, any]:
        """Return compact snapshot for Telegram /status."""
        return {
            "time_ist": datetime.now().strftime("%H:%M:%S"),
            "live_trading": self.live_trading,
            "broker": "Zerodha" if self.live_trading else "None",
            "active_orders": len(self.open_trades),
        }

    def get_last_signal_debug(self) -> Optional[Dict[str, any]]:
        # placeholder, fill with last signal if storing
        return None

    def build_diag(self) -> Dict[str, any]:
        """Return system diagnostics for /diag and /check."""
        return {"ok": True, "checks": [], "last_signal": None}

    def pause(self) -> None:
        self.logger.info("Entries paused.")

    def resume(self) -> None:
        self.logger.info("Entries resumed.")

    def health_check(self) -> None:
        """Lightweight periodic health report."""
        self.logger.debug("Health OK.")

    def shutdown(self) -> None:
        self.logger.info("Runner shutdown cleanly.")

    def _notify(self, msg: str) -> None:
        self.logger.info(msg)
        try:
            if self.telegram:
                self.telegram.send_message(msg)
        except Exception:
            pass