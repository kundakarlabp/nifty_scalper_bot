# src/data_providers/kite_data_provider.py
from __future__ import annotations

"""
KiteDataProvider: thin, resilient adapter around KiteConnect

Exposed API (used by RealTimeTrader):
- build_from_env() -> KiteDataProvider
- get_ohlc(symbol, minutes, timeframe="minute") -> pd.DataFrame[open,high,low,close,volume]
- get_last_price(symbol) -> float | None

Extras:
- get_quote(symbol) -> dict | None        # raw quote()
- get_depth(symbol) -> {"bid":float,"ask":float} | None
- get_mid_price(symbol) -> float | None
- get_token_for_symbol(symbol) -> int
- fetch_cached_instruments() -> dict[str, list[dict]]  # {"NFO":[...], "NSE":[...]}

Environment / Config:
- Uses Config.ZERODHA_API_KEY and (Config.KITE_ACCESS_TOKEN or Config.ZERODHA_ACCESS_TOKEN)
- Falls back to env vars with the same names if Config is missing those
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class KiteDataProvider:
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

        # tiny in-process caches
        self._ltp_cache: Dict[str, Dict[str, Any]] = {}
        self._ltp_ttl = float(getattr(Config, "LTP_CACHE_TTL_SEC", 1.8))

        self._inst_cache: Dict[str, list] = {}
        self._inst_ts: float = 0.0
        self._inst_ttl = float(getattr(Config, "INSTRUMENTS_CACHE_TTL_SEC", 900.0))  # 15 min

    # -------- factory -------- #

    @staticmethod
    def build_from_env() -> "KiteDataProvider":
        from kiteconnect import KiteConnect

        api_key = getattr(Config, "ZERODHA_API_KEY", None)
        access_token = (getattr(Config, "KITE_ACCESS_TOKEN", None)
                        or getattr(Config, "ZERODHA_ACCESS_TOKEN", None))
        if not api_key or not access_token:
            raise RuntimeError("ZERODHA_API_KEY or KITE/ ZERODHA_ACCESS_TOKEN missing for KiteDataProvider")

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        return KiteDataProvider(kite=kite, default_token=getattr(Config, "INSTRUMENT_TOKEN", 256265))

    # -------- utils -------- #

    @staticmethod
    def _ist_now() -> datetime:
        # UTC+5:30 without external tz deps
        return datetime.utcnow() + timedelta(hours=5, minutes=30)

    def _with_retries(self, fn, *args, tries=2, backoff=0.6, **kwargs):
        last = None
        for i in range(tries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last = e
                if i == tries - 1:
                    break
                sleep = backoff * (2 ** i)
                logger.debug("retrying %s in %.2fs (err: %s)", getattr(fn, "__name__", "call"), sleep, e)
                time.sleep(sleep)
        raise last

    def _format_quote_token(self, symbol: str, exchange: Optional[str] = None) -> str:
        if ":" in symbol:
            return symbol
        ex = exchange or getattr(Config, "TRADE_EXCHANGE", "NFO")
        return f"{ex}:{symbol}"

    # -------- instruments cache -------- #

    def fetch_cached_instruments(self) -> Dict[str, list]:
        now = time.time()
        if self._inst_cache and (now - self._inst_ts) <= self._inst_ttl:
            return self._inst_cache
        try:
            nfo = self._with_retries(self.kite.instruments, "NFO") or []
        except Exception as e:
            logger.warning("NFO instruments fetch failed: %s", e)
            nfo = []
        try:
            nse = self._with_retries(self.kite.instruments, "NSE") or []
        except Exception as e:
            logger.warning("NSE instruments fetch failed: %s", e)
            nse = []
        self._inst_cache = {"NFO": nfo, "NSE": nse}
        self._inst_ts = now
        return self._inst_cache

    def get_token_for_symbol(self, symbol: Optional[str]) -> int:
        if not symbol:
            return int(self.default_token)
        # Fast path: quote can expose instrument_token
        try:
            q = self._with_retries(self.kite.quote, [symbol])
            tok = (q.get(symbol, {}) or {}).get("instrument_token")
            if tok:
                return int(tok)
        except Exception:
            pass
        # Fallback: search instruments (NFO/NSE)
        inst = self.fetch_cached_instruments()
        sym = symbol.split(":", 1)[-1].strip().upper()
        for seg in ("NFO", "NSE"):
            for row in inst.get(seg, []):
                tsym = (row.get("tradingsymbol") or "").strip().upper()
                if tsym == sym:
                    t = row.get("instrument_token")
                    if t:
                        return int(t)
        # Last resort
        return int(self.default_token)

    # -------- public API -------- #

    def get_ohlc(self, symbol: str, minutes: int, timeframe: str = "minute") -> pd.DataFrame:
        interval = self._TF_MAP.get(str(timeframe).lower(), "minute")
        token = self.get_token_for_symbol(symbol)
        end_ts = self._ist_now()
        # generous lookback to smooth gaps (min extra one hour)
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

        try:
            candles = self._with_retries(_hist)
        except Exception as e:
            logger.error("historical_data failed for %s (%s): %s", symbol, interval, e)
            return pd.DataFrame()

        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        if "date" in df.columns:
            df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        # keep last N minutes
        cutoff = end_ts - timedelta(minutes=minutes + 1)
        df = df[df.index >= cutoff]
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        return df[keep].copy()

    def get_last_price(self, symbol: str) -> Optional[float]:
        now = time.time()
        c = self._ltp_cache.get(symbol)
        if c and (now - c["ts"]) <= self._ltp_ttl:
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

    # -------- handy extras (used by spread guard / status) -------- #

    def get_quote(self, symbol: str) -> Optional[dict]:
        try:
            return (self._with_retries(self.kite.quote, [symbol]) or {}).get(symbol)
        except Exception:
            return None

    def get_depth(self, symbol: str) -> Optional[Dict[str, float]]:
        try:
            q = self.get_quote(symbol) or {}
            dd = q.get("depth") or {}
            bids = (dd.get("buy") or [])
            asks = (dd.get("sell") or [])
            bid = float(bids[0]["price"]) if bids else None
            ask = float(asks[0]["price"]) if asks else None
            if bid and ask and ask >= bid:
                return {"bid": bid, "ask": ask}
        except Exception:
            pass
        return None

    def get_mid_price(self, symbol: str) -> Optional[float]:
        bb = self.get_depth(symbol)
        if bb:
            return 0.5 * (bb["bid"] + bb["ask"])
        return self.get_last_price(symbol)