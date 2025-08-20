"""
StrategyRunner orchestrates:
- Time-gate via is_market_open()
- Resolve strikes/tokens (only if Kite is available)
- Ensure ADX/DI columns on SPOT DF
- Generate a signal using EnhancedScalpingStrategy(df, current_price)  # 2-arg signature
- Compute position size (lots) using PositionSizing and equity estimate

Shadow-mode safe: if Kite or data source is missing, run_once() returns None without raising.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import pandas as pd

from src.config import settings
from src.utils.strike_selector import (
    fetch_cached_instruments,
    get_instrument_tokens,
    is_market_open,
)
from src.utils.account_info import get_equity_estimate
from src.risk.position_sizing import PositionSizing
from src.strategies.scalping_strategy import EnhancedScalpingStrategy

log = logging.getLogger(__name__)

try:
    from ta.trend import ADXIndicator  # type: ignore
except Exception:  # pragma: no cover
    ADXIndicator = None  # type: ignore


def _ensure_adx_di(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    If ADX/DI columns are missing, compute them on the given DF (spot DF).
    """
    if df is None or df.empty:
        return df
    cols = set(c.lower() for c in df.columns)
    need = {"adx", "di_plus", "di_minus"}
    if need.issubset(cols):
        return df

    try:
        if ADXIndicator is None:
            return df  # skip if ta not installed
        hi = df["high"].astype(float)
        lo = df["low"].astype(float)
        cl = df["close"].astype(float)
        adx_ind = ADXIndicator(high=hi, low=lo, close=cl, window=window, fillna=False)
        out = df.copy()
        out["adx"] = adx_ind.adx()
        out["di_plus"] = adx_ind.adx_pos()
        out["di_minus"] = adx_ind.adx_neg()
        return out
    except Exception:
        return df


def _fetch_df(
    source: Any,
    kind: str,
    *,
    symbol_or_token: Any,
    lookback_minutes: int,
    timeframe: str,
) -> Optional[pd.DataFrame]:
    """
    Duck-typed data fetcher:
      - tries source.get_spot_ohlc / get_option_ohlc
      - falls back to source.fetch_ohlc(symbol_or_token, ...)
    """
    if source is None:
        return None
    try:
        if kind == "spot":
            if hasattr(source, "get_spot_ohlc"):
                return source.get_spot_ohlc(symbol_or_token, lookback_minutes, timeframe)
        else:
            if hasattr(source, "get_option_ohlc"):
                return source.get_option_ohlc(symbol_or_token, lookback_minutes, timeframe)
        if hasattr(source, "fetch_ohlc"):
            return source.fetch_ohlc(symbol_or_token, lookback_minutes, timeframe)
    except Exception as e:
        log.warning("Data source fetch error (%s): %s", kind, e)
    return None


class StrategyRunner:
    """
    Minimal orchestrator; call .run_once(...) from your loop.
    """

    def __init__(self, *, data_source: Any, kite: Any, strategy: Optional[EnhancedScalpingStrategy] = None) -> None:
        self.data_source = data_source
        self.kite = kite
        self.strategy = strategy or EnhancedScalpingStrategy()
        self._last_token_warn_ts: float = 0.0  # throttle noisy warnings

    def _throttled_token_warn(self, msg: str, period_sec: int = 60) -> None:
        now = time.time()
        if now - self._last_token_warn_ts >= period_sec:
            log.warning(msg)
            self._last_token_warn_ts = now
        else:
            log.debug(msg)

    def run_once(self) -> Optional[Dict[str, Any]]:
        """
        Execute a single decision cycle:
          - skip if market closed (unless ALLOW_OFFHOURS_TESTING)
          - if Kite is unavailable (shadow mode), return None gracefully
          - else produce signal dict with side/confidence/sl/tp/score, plus size and tokens
        """
        if not is_market_open() and not getattr(settings, "allow_offhours_testing", False):
            log.debug("Market closed (IST gate).")
            return None

        # Shadow-mode guard: cannot resolve tokens without Kite
        if self.kite is None:
            log.debug("Runner: kite instance missing; skipping trading cycle (shadow mode).")
            return None

        # Resolve instruments
        nfo, _ = fetch_cached_instruments(self.kite)
        token_info = get_instrument_tokens(self.kite, nfo)
        if not token_info or not token_info.get("tokens"):
            self._throttled_token_warn("Could not resolve CE/PE tokens.")
            return None

        # ---- SPOT DF (context) ----
        spot_symbol = getattr(settings, "SPOT_SYMBOL", getattr(settings.instruments, "spot_symbol", "NSE:NIFTY 50"))
        spot_df = _fetch_df(
            self.data_source,
            "spot",
            symbol_or_token=spot_symbol,
            lookback_minutes=int(getattr(settings, "DATA_LOOKBACK_MINUTES", getattr(settings.data, "lookback_minutes", 60))),
            timeframe=str(getattr(settings, "HISTORICAL_TIMEFRAME", getattr(settings.data, "timeframe", "minute"))),
        )
        if spot_df is not None and not spot_df.empty:
            spot_df = _ensure_adx_di(spot_df, window=int(getattr(settings, "ATR_PERIOD", getattr(settings.strategy, "atr_period", 14))))

        # ---- OPTION DF (default CE) ----
        ce_token = token_info["tokens"].get("ce")
        if ce_token is None:
            self._throttled_token_warn("CE token missing; skipping.")
            return None

        opt_df = _fetch_df(
            self.data_source,
            "option",
            symbol_or_token=ce_token,
            lookback_minutes=int(getattr(settings, "DATA_LOOKBACK_MINUTES", getattr(settings.data, "lookback_minutes", 60))),
            timeframe=str(getattr(settings, "HISTORICAL_TIMEFRAME", getattr(settings.data, "timeframe", "minute"))),
        )
        if opt_df is None or opt_df.empty:
            log.debug("Option DF empty; skipping.")
            return None

        current_price = float(opt_df["close"].iloc[-1])

        # Strategy call (STRICT signature: df, current_price)
        signal = self.strategy.generate_signal(opt_df, current_price)
        if not signal:
            return None

        equity = get_equity_estimate()
        lots = PositionSizing.lots_from_equity(
            equity=equity,
            sl_points=float(signal["sl_points"]),
        )

        enriched = {
            **signal,
            "equity": float(equity),
            "lots": int(lots),
            "instrument": {
                "atm_strike": token_info["atm_strike"],
                "target_strike": token_info["target_strike"],
                "expiry": token_info["expiry"],
                "token_ce": token_info["tokens"]["ce"],
                "token_pe": token_info["tokens"]["pe"],
            },
        }
        return enriched