# src/data_streaming/kite_data_provider.py
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
from src.config import Config

logger = logging.getLogger(__name__)

def _ist_now() -> datetime:
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

class KiteDataProvider:
    """
    Zerodha Kite wrapper:
      - get_ohlc(symbol, minutes, timeframe) -> DataFrame[open,high,low,close,volume?]
      - get_last_price(symbol) -> float | None

    `symbol` must be a full tradingsymbol like "NSE:NIFTY 50", "NFO:NIFTY24AUG22600CE".
    """

    _TF_MAP = {
        "minute": "minute", "1m": "minute",
        "3minute": "3minute", "3m": "3minute",
        "5minute": "5minute", "5m": "5minute",
        "10minute": "10minute",
        "15minute": "15minute", "15m": "15minute",
        "30minute": "30minute", "30m": "30minute",
        "60minute": "60minute", "1h": "60minute",
        "day": "day", "d": "day",
    }

    def __init__(self, kite, default_token: Optional[int] = None) -> None:
        self.kite = kite
        self.default_token = int(default_token or getattr(Config, "INSTRUMENT_TOKEN", 256265))
        self._ltp_cache: Dict[str, Dict[str, Any]] = {}
        self._ltp_ttl = 1.8

    @staticmethod
    def build_from_env() -> "KiteDataProvider":
        from kiteconnect import KiteConnect
        api_key = getattr(Config, "ZERODHA_API_KEY", None)
        access_token = getattr(Config, "KITE_ACCESS_TOKEN", None) or getattr(Config, "ZERODHA_ACCESS_TOKEN", None)
        if not api_key or not access_token:
            raise RuntimeError("ZERODHA_API_KEY or KITE_ACCESS_TOKEN missing for KiteDataProvider")
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        return KiteDataProvider(kite=kite, default_token=getattr(Config, "INSTRUMENT_TOKEN", 256265))

    def _with_retries(self, fn, *args, tries=2, backoff=0.6, **kwargs):
        last = None
        for i in range(tries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last = e
                if i == tries - 1:
                    break
                time.sleep(backoff * (2 ** i))
        raise last

    def _resolve_token(self, symbol: Optional[str]) -> int:
        if not symbol:
            return int(self.default_token)
        if symbol == getattr(Config, "SPOT_SYMBOL", "NSE:NIFTY 50"):
            return int(self.default_token)
        try:
            q = self._with_retries(self.kite.quote, [symbol])
            tok = (q.get(symbol, {}) or {}).get("instrument_token")
            if tok:
                return int(tok)
        except Exception:
            pass
        raise ValueError(f"Cannot resolve instrument token for {symbol}")

    def get_ohlc(self, symbol: str, minutes: int, timeframe: str = "minute") -> pd.DataFrame:
        interval = self._TF_MAP.get(str(timeframe).lower(), "minute")
        token = self._resolve_token(symbol)
        end_ts = _ist_now()
        start_ts = end_ts - timedelta(minutes=max(2 * minutes, minutes + 60))

        def _hist():
            return self.kite.historical_data(
                instrument_token=token,
                from_date=start_ts,
                to_date=end_ts,
                interval=interval,
                continuous=False,
                oi=False,
            )

        candles: List[Dict[str, Any]] = self._with_retries(_hist)
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        if "date" in df.columns:
            df = df.set_index("date", drop=True)
        df = df.sort_index()
        cutoff = end_ts - timedelta(minutes=minutes + 1)
        df = df[df.index >= cutoff]
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep].copy()

    def get_last_price(self, symbol: str) -> Optional[float]:
        now = time.time()
        c = self._ltp_cache.get(symbol)
        if c and (now - c["ts"] <= self._ltp_ttl):
            return float(c["ltp"])
        try:
            data = self._with_retries(self.kite.ltp, [symbol])
            px = (data.get(symbol, {}) or {}).get("last_price")
            if px is None:
                data = self._with_retries(self.kite.quote, [symbol])
                px = (data.get(symbol, {}) or {}).get("last_price")
            if px is None:
                return None
            self._ltp_cache[symbol] = {"ts": now, "ltp": float(px)}
            return float(px)
        except Exception:
            return None