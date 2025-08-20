# src/strategies/runner.py
from __future__ import annotations

import logging
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd

from src.config import settings
from src.risk.position_sizing import PositionSizing
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.utils.account_info import get_equity_estimate
from src.utils.strike_selector import (
    get_instrument_tokens,
    is_market_open,
)
from src.data.source import DataSource, LiveKiteSource

# Optional broker SDK (keep imports safe and allow for "shadow mode")
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import (
        NetworkException,
        TokenException,
        InputException,
    )  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # fallbacks

log = logging.getLogger(__name__)


# -------------------- Internal Helper Functions --------------------
def _now_ist_naive() -> datetime:
    """Returns the current naive datetime in IST."""
    ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    return ist.replace(tzinfo=None)


def _ensure_adx_di(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Computes ADX and DI+, DI- indicators on the provided DataFrame.

    Uses `ta` if available; otherwise falls back to a lightweight manual calc.
    """
    if df is None or df.empty or not {"high", "low", "close"}.issubset(df.columns):
        log.warning("ADX/DI: Missing 'high', 'low', or 'close' columns.")
        return df

    # Try ta.trend.ADXIndicator if available
    try:
        from ta.trend import ADXIndicator  # type: ignore
        adxi = ADXIndicator(df["high"], df["low"], df["close"], window=window)
        df[f"adx_{window}"] = adxi.adx()
        df[f"di_plus_{window}"] = adxi.adx_pos()
        df[f"di_minus_{window}"] = adxi.adx_neg()
        return df
    except Exception:
        # Fallback manual estimate
        pass

    # Manual approximation if ta is not installed
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = (df["high"] - df["low"]).abs()
    atr = tr.ewm(alpha=1 / window, adjust=False).mean().replace(0, 1e-9)

    plus_di = (plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr) * 100.0
    minus_di = (minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr) * 100.0
    dx = (plus_di.subtract(minus_di).abs() / (plus_di.add(minus_di).abs() + 1e-9)) * 100.0
    adx = dx.ewm(alpha=1 / window, adjust=False).mean()

    df[f"adx_{window}"] = adx
    df[f"di_plus_{window}"] = plus_di
    df[f"di_minus_{window}"] = minus_di
    return df


def _fetch_and_prepare_df(
    data_source: Optional[DataSource],
    token: Optional[int],
    lookback: timedelta,
    timeframe: str,
) -> pd.DataFrame:
    """
    Helper to fetch OHLC data and ensure it's in the correct format.
    """
    if data_source is None or token is None:
        return pd.DataFrame()

    end_date = _now_ist_naive()
    start_date = end_date - lookback
    df = data_source.fetch_ohlc(token, start_date, end_date, timeframe)

    if df.empty:
        log.warning("Data fetch for token %s returned empty DataFrame.", token)
        return pd.DataFrame()

    # Ensure required columns are present for the strategy
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        log.error("Fetched data for token %s is missing required columns: %s", token, required_cols)
        return pd.DataFrame()

    return df


# -------------------- Main Runner Class --------------------
class StrategyRunner:
    """
    Orchestrates the bot's core trading loop:
      1) Market hours check
      2) Resolve strike/instrument tokens
      3) Fetch spot + option OHLC
      4) Compute ADX/DI on spot
      5) Strategy.generate_signal(...)
      6) Position sizing
      7) Return actionable signal dict
    """

    def __init__(
        self,
        strategy: Optional[EnhancedScalpingStrategy] = None,
        data_source: Optional[DataSource] = None,
        spot_source: Optional[DataSource] = None,
        kite: Optional["KiteConnect"] = None,
    ) -> None:
        """
        Initializes the runner with optional strategy and data sources.
        If not provided, builds sensible defaults (shadow-mode safe).
        """
        self.strategy = strategy or EnhancedScalpingStrategy()
        self._kite = kite or self._build_kite()
        self.data_source = data_source or self._build_live_source(self._kite)
        self.spot_source = spot_source or self.data_source
        self._last_token_warn_ts = 0.0

    # ---- wiring helpers ----
    def _build_kite(self) -> Optional["KiteConnect"]:
        if KiteConnect is None:
            log.warning("KiteConnect not installed; StrategyRunner in shadow mode (no broker).")
            return None

        api_key = getattr(getattr(settings, "zerodha", object()), "api_key", None)
        access_token = getattr(getattr(settings, "zerodha", object()), "access_token", None)
        if not api_key:
            log.warning("ZERODHA_API_KEY missing; StrategyRunner will run without broker connectivity.")
            return None

        kc = KiteConnect(api_key=api_key)
        if access_token:
            try:
                kc.set_access_token(access_token)
                log.info("Kite session OK via ACCESS_TOKEN")
            except Exception as e:
                log.warning("Unable to set Kite access token: %s", e)
        return kc

    def _build_live_source(self, kite: Optional["KiteConnect"]) -> Optional[DataSource]:
        if kite is None:
            return None
        try:
            ds = LiveKiteSource(kite)
            ds.connect()
            log.info("LiveKiteSource connected.")
            return ds
        except Exception as e:
            log.warning("LiveKiteSource unavailable: %s", e)
            return None

    # ---- misc helpers ----
    def _throttled_token_warn(self, message: str) -> None:
        """Rate-limit token warnings to avoid log spam."""
        now = time.time()
        if now - self._last_token_warn_ts > 60:
            log.warning(message)
            self._last_token_warn_ts = now

    def to_status_dict(self) -> Dict[str, Any]:
        """Minimal status snapshot for health/Telegram."""
        return {
            "time_ist": _now_ist_naive().isoformat(sep=" ", timespec="seconds"),
            "broker": "Kite" if self._kite else "none",
            "data_source": type(self.data_source).__name__ if self.data_source else None,
        }

    # ---- main step ----
    def run_once(self, stop_event: threading.Event) -> Optional[Dict[str, Any]]:
        """
        Perform a single cycle of the trading loop.
        Returns a structured signal dict or None.
        """
        # 1) market hours gate
        if not is_market_open():
            log.info("Market is closed. Skipping run.")
            return None

        if stop_event.is_set():
            return None

        try:
            # 2) resolve tokens (pass broker if selector uses it)
            token_info = get_instrument_tokens(kite_instance=self._kite)
            if not token_info:
                self._throttled_token_warn("Could not resolve strike tokens.")
                return None

            spot_token = token_info.get("spot_token")
            ce_token = token_info.get("tokens", {}).get("ce")
            if not spot_token or not ce_token:
                self._throttled_token_warn("Spot or CE token missing. Skipping.")
                return None

            # 3) lookback + timeframe
            lookback_minutes = int(getattr(getattr(settings, "data", object()), "lookback_minutes", 60))
            lookback = timedelta(minutes=lookback_minutes)
            timeframe = str(getattr(getattr(settings, "data", object()), "timeframe", "minute"))

            # 4) fetch OHLC
            spot_df = _fetch_and_prepare_df(self.spot_source, spot_token, lookback, timeframe)
            if spot_df.empty:
                return None

            opt_df = _fetch_and_prepare_df(self.data_source, ce_token, lookback, timeframe)
            if opt_df.empty:
                return None

            # 5) indicators on spot
            adx_window = int(getattr(getattr(settings, "strategy", object()), "atr_period", 14))
            spot_df = _ensure_adx_di(spot_df, window=adx_window)

            current_price = float(opt_df["close"].iloc[-1])

            # 6) strategy
            signal = self.strategy.generate_signal(opt_df, current_price, spot_df)
            if not signal:
                log.info("No signal generated by strategy.")
                return None

            # 7) position sizing (use broker-based equity when possible)
            equity = float(get_equity_estimate(self._kite))
            if equity <= 0:
                log.warning("Equity estimate is zero or negative. Skipping trade.")
                return None

            sl_points = float(signal.get("sl_points", 0.0))
            if sl_points <= 0:
                log.warning("Calculated stop loss points are zero or negative. Skipping.")
                return None

            lots = int(PositionSizing.lots_from_equity(equity=equity, sl_points=sl_points))
            if lots <= 0:
                log.info("Calculated lots is zero. Skipping trade.")
                return None

            # 8) enrich and return
            enriched_signal: Dict[str, Any] = {
                **signal,
                "equity": equity,
                "lots": lots,
                "instrument": {
                    "atm_strike": token_info.get("atm_strike"),
                    "target_strike": token_info.get("target_strike"),
                    "expiry": token_info.get("expiry"),
                    "token_ce": ce_token,
                },
            }
            log.info("Signal generated: %s", enriched_signal)
            return enriched_signal

        except (NetworkException, TokenException, InputException) as e:
            log.error("Transient broker error: %s", e)
        except Exception as e:
            log.exception("Unexpected error in run_once: %s", e)

        return None
