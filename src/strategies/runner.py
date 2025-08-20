"""
StrategyRunner orchestrates:
- Time-gate via is_market_open()
- Resolve strikes/tokens
- Ensure ADX/DI columns on SPOT DF
- Generate a signal using EnhancedScalpingStrategy(df, current_price)  # 2-arg signature
- Compute position size (lots) using PositionSizing and equity estimate

Note: This runner is intentionally conservative and duck-typed for the data source.
"""

from __future__ import annotations

import logging
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
    Minimal orchestrator; you can call .run_once(...) from your loop.
    """

    def __init__(self, *, data_source: Any, kite: Any, strategy: Optional[EnhancedScalpingStrategy] = None) -> None:
        self.data_source = data_source
        self.kite = kite
        self.strategy = strategy or EnhancedScalpingStrategy()

    def run_once(self) -> Optional[Dict[str, Any]]:
        """
        Execute a single decision cycle:
          - skip if market closed (unless ALLOW_OFFHOURS_TESTING)
          - produce signal dict with side/confidence/sl/tp/score
          - attach lots, equity, and token info
        """
        if not is_market_open() and not getattr(settings.toggles, "allow_offhours_testing", getattr(settings, "ALLOW_OFFHOURS_TESTING", False)):
            log.info("Market closed (IST gate).")
            return None

        # Resolve instruments
        nfo, _ = fetch_cached_instruments(self.kite)
        token_info = get_instrument_tokens(self.kite, nfo)
        if not token_info or not token_info.get("tokens"):
            log.warning("Could not resolve CE/PE tokens.")
            return None

        # ---- SPOT DF (for ADX/DI context if you need it later) ----
        spot_symbol = getattr(settings.instruments, "spot_symbol", getattr(settings, "SPOT_SYMBOL", "NSE:NIFTY 50"))
        spot_df = _fetch_df(
            self.data_source,
            "spot",
            symbol_or_token=spot_symbol,
            lookback_minutes=int(getattr(settings.data, "lookback_minutes", getattr(settings, "DATA_LOOKBACK_MINUTES", 60))),
            timeframe=str(getattr(settings.data, "timeframe", getattr(settings, "HISTORICAL_TIMEFRAME", "minute"))),
        )
        if spot_df is not None and not spot_df.empty:
            spot_df = _ensure_adx_di(spot_df, window=int(getattr(settings.strategy, "atr_period", getattr(settings, "ATR_PERIOD", 14))))

        # ---- OPTION DF (we’ll default to CE; you can choose based on bias externally) ----
        ce_token = token_info["tokens"].get("ce")
        if ce_token is None:
            log.warning("CE token missing; cannot build option DF.")
            return None

        opt_df = _fetch_df(
            self.data_source,
            "option",
            symbol_or_token=ce_token,
            lookback_minutes=int(getattr(settings.data, "lookback_minutes", 60)),
            timeframe=str(getattr(settings.data, "timeframe", "minute")),
        )
        if opt_df is None or opt_df.empty:
            log.warning("Option DF empty.")
            return None

        # LTP/current price for the option
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