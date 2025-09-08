# src/data/source.py
from __future__ import annotations

import logging
import os
import time
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
import datetime as dt
from typing import Any, Dict, Optional, Tuple, Callable, List


import numpy as np
import pandas as pd
from src.utils.atr_helper import compute_atr
from src.utils.indicators import calculate_vwap
from src.utils.circuit_breaker import CircuitBreaker
from src.data.base_source import BaseDataSource
from src.boot.validate_env import (
    data_warmup_disable,
)

log = logging.getLogger(__name__)

# Optional broker SDK (keep imports tolerant so paper mode works)
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import (  # type: ignore
        NetworkException,
        TokenException,
        InputException,
        DataException,
        GeneralException,
    )
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    # Collapse to base Exception so retry wrapper still works in paper mode
    NetworkException = TokenException = InputException = DataException = GeneralException = Exception  # type: ignore


# --------------------------------------------------------------------------------------
# Base Interface
# --------------------------------------------------------------------------------------
class DataSource:
    """Minimal interface used by StrategyRunner/OrderExecutor."""

    def connect(self) -> None:
        """Connect or noop."""
        return

    def fetch_ohlc(
        self, token: int, start: datetime, end: datetime, timeframe: str
    ) -> Optional[pd.DataFrame]:
        """
        Return an OHLC DataFrame with columns: open, high, low, close, volume.
        Index should be pandas Timestamps (timezone‑naive IST is fine).
        A subclass may return ``None`` if no data could be retrieved.
        """
        raise NotImplementedError

    def get_last_price(self, symbol_or_token: Any) -> Optional[float]:
        """Return LTP for a trading symbol (e.g., 'NSE:NIFTY 50') or instrument token."""
        raise NotImplementedError

    def api_health(self) -> Dict[str, Dict[str, object]]:
        """Return circuit breaker health metrics if available."""
        return {}

    def get_last_bars(self, n: int):
        """Return the last ``n`` 1m bars with ATR% and VWAP if available."""
        try:
            token = int(getattr(settings.instruments, "instrument_token", 0) or 0)
        except Exception:
            return None
        from src.utils.time_windows import now_ist, floor_to_minute
        from src.data.ohlc_builder import prepare_ohlc

        now = now_ist()
        cutoff = now - timedelta(seconds=5)
        end = floor_to_minute(cutoff, None)
        lookback = max(60, n + 50)
        start = end - timedelta(minutes=lookback)
        df = self.fetch_ohlc(token=token, start=start, end=end, timeframe="minute")

        # If no bars were fetched, attempt a backfill and retry once.  This helps
        # recover when the bot starts with an empty cache or the broker API drops
        # a request, which previously resulted in a permanent warmup failure.
        if df is None or df.empty:
            try:
                ensure = getattr(self, "ensure_backfill", None)
                if callable(ensure):
                    ensure(required_bars=n, token=token, timeframe="minute")
                    df = self.fetch_ohlc(
                        token=token, start=start, end=end, timeframe="minute"
                    )
            except Exception:
                log.warning("ensure_backfill failed", exc_info=True)
                df = None
        if df is None or df.empty:
            return None
        df = prepare_ohlc(df, now)
        if "vwap" not in df.columns:
            try:
                df["vwap"] = calculate_vwap(df)
            except Exception:
                log.debug("calculate_vwap failed", exc_info=True)
        if "atr_pct" not in df.columns:
            atr = compute_atr(df, period=14)
            df["atr_pct"] = atr / df["close"] * 100.0
        return df.tail(n)


# Accept a few common aliases; default to 'minute'
_INTERVAL_MAP: Dict[str, str] = {
    "minute": "minute",
    "1minute": "minute",
    "1m": "minute",
    "3minute": "3minute",
    "3m": "3minute",
    "5minute": "5minute",
    "5m": "5minute",
    "10minute": "10minute",
    "10m": "10minute",
    "15minute": "15minute",
    "15m": "15minute",
}

# Map intervals to their minute counts and pandas frequency strings
_INTERVAL_TO_MINUTES: Dict[str, int] = {
    "minute": 1,
    "3minute": 3,
    "5minute": 5,
    "10minute": 10,
    "15minute": 15,
    "day": 1440,
}

_INTERVAL_TO_FREQ: Dict[str, str] = {
    "minute": "1min",
    "3minute": "3min",
    "5minute": "5min",
    "10minute": "10min",
    "15minute": "15min",
    "day": "1D",
}

from typing import Any, cast

try:  # pragma: no cover - imported lazily to avoid circular dependency during settings init
    import src.config as _cfg  # type: ignore[import]

    settings = cast(Any, getattr(_cfg, "settings"))
    WARMUP_BARS = int(
        max(
            getattr(settings, "warmup_bars", 0),
            settings.data.lookback_minutes,
            settings.strategy.min_bars_for_signal,
        )
    )
except Exception:  # pragma: no cover
    # Fallback for early imports or missing settings
    WARMUP_BARS = 20


@dataclass
class _CacheEntry:
    df: pd.DataFrame
    ts: float  # insertion timestamp (epoch seconds)
    fetched_window: Tuple[datetime, datetime]
    requested_window: Tuple[datetime, datetime]


class _TTLCache:
    """
    Extremely simple per‑(token,interval) cache to soften historical_data pressure
    during frequent ticks or Telegram diagnostics. Keeps a tiny TTL.
    """
    def __init__(self, ttl_sec: float = 4.0) -> None:
        self._ttl = float(ttl_sec)
        self._data: Dict[Tuple[int, str], _CacheEntry] = {}

    def get(self, token: int, interval: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        key = (int(token), interval)
        ent = self._data.get(key)
        if not ent:
            return None
        if time.time() - ent.ts > self._ttl:
            self._data.pop(key, None)
            return None
        s0, e0 = ent.fetched_window
        if s0 <= start and e0 >= end:
            try:
                return ent.df.loc[(ent.df.index >= start) & (ent.df.index <= end)].copy()
            except Exception:
                return None
        return None

    def set(
        self,
        token: int,
        interval: str,
        df: pd.DataFrame,
        fetched_window: Tuple[datetime, datetime],
        requested_window: Tuple[datetime, datetime],
    ) -> None:
        key = (int(token), interval)
        self._data[key] = _CacheEntry(
            df=df.copy(),
            ts=time.time(),
            fetched_window=fetched_window,
            requested_window=requested_window,
        )


def render_last_bars(ds: DataSource, n: int = 5) -> str:
    """Return formatted last ``n`` 1m bars with key indicators."""
    try:
        token = int(getattr(settings.instruments, "instrument_token", 0) or 0)
        if token <= 0:
            return "instrument_token missing"
        from src.utils.time_windows import now_ist, floor_to_minute
        from src.data.ohlc_builder import prepare_ohlc

        now = now_ist()
        cutoff = now - timedelta(seconds=5)
        end = floor_to_minute(cutoff, None)
        lookback = max(60, n + 50)
        start = end - timedelta(minutes=lookback)
        df = ds.fetch_ohlc(token=token, start=start, end=end, timeframe="minute")
        if df is None or df.empty:
            return "no data"
        df = prepare_ohlc(df, now)
        vwap = calculate_vwap(df)
        ema21 = df["close"].ewm(span=21, adjust=False).mean()
        ema50 = df["close"].ewm(span=50, adjust=False).mean()
        atr = compute_atr(df, period=14)
        lines: List[str] = []
        for ts, row in df.tail(n).iterrows():
            atr_val = float(atr.loc[ts]) if ts in atr.index else float("nan")
            ema21_val = float(ema21.loc[ts]) if ts in ema21.index else float("nan")
            ema50_val = float(ema50.loc[ts]) if ts in ema50.index else float("nan")
            vwap_val = float(vwap.loc[ts]) if ts in vwap.index else float("nan")
            lines.append(
                f"{ts:%H:%M} O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f} VWAP={vwap_val:.2f} ATR={atr_val:.2f} EMA21={ema21_val:.2f} EMA50={ema50_val:.2f}"
            )
        last_ts = df.index[-1]
        atr_pct = (float(atr.iloc[-1]) / float(df["close"].iloc[-1]) * 100.0) if len(atr) else 0.0
        lines.append(f"last_bar_ts={last_ts.to_pydatetime().isoformat()} ATR%={atr_pct:.2f}")
        return "\n".join(lines)
    except Exception as e:  # pragma: no cover - diagnostic helper
        return f"bars error: {e}"


def _safe_dataframe(rows: Any) -> pd.DataFrame:
    """Return a sanitized OHLC frame.

    The input is copied, column names are lowercased and required OHLC columns
    are verified. Numeric columns are coerced, infinite values are replaced with
    ``pd.NA`` and rows with ``NaN`` values are dropped. Only rows with positive
    prices and logical relationships (``low`` ≤ ``open``/``close`` ≤ ``high``)
    are kept. The index is sorted and duplicate timestamps are removed keeping
    the latest occurrence.
    """
    try:
        df = pd.DataFrame(rows or []).copy()
        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns=lambda c: str(c).lower())
        df = df.T.groupby(level=0).sum(min_count=1).T

        if "date" in df.columns:
            from src.utils.time_windows import TZ

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["date"] = (
                df["date"].dt.tz_convert(TZ)
                if df["date"].dt.tz is not None
                else df["date"].dt.tz_localize(TZ)
            )
            df["date"] = df["date"].dt.tz_localize(None)
            df = df.set_index("date")

        need = ["open", "high", "low", "close"]
        if not set(need).issubset(df.columns):
            return pd.DataFrame()

        if "volume" not in df.columns:
            df["volume"] = 0

        cols = ["open", "high", "low", "close", "volume"]
        df = df[cols].copy()
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace([np.inf, -np.inf], pd.NA)
        df = df.dropna()

        mask = (
            df[need].gt(0).all(axis=1)
            & df["low"].le(df[["open", "close", "high"]].min(axis=1))
            & df[["open", "close"]].max(axis=1).le(df["high"])
        )
        df = df[mask]

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        return df

    except Exception as e:
        log.warning("Failed to normalize OHLC frame: %s", e)
        return pd.DataFrame()


def _retry(fn: Callable, *args, tries: int = 3, base_delay: float = 0.25, **kwargs):
    """
    Simple retry with exponential backoff. Retries on common Kite exceptions and
    common request errors.  Falls back to a generic Exception catch to avoid
    breaking the calling code, but keeps the retry window tight so failures are
    surfaced quickly.
    """
    delay = float(base_delay)
    last: Optional[BaseException] = None

    # Import lazily to avoid hard dependency on requests/urllib3 during tests
    try:  # pragma: no cover - simple import guard
        from requests import exceptions as req_exc  # type: ignore
        import urllib3  # type: ignore
        http_exc: Tuple[type[BaseException], ...] = (
            req_exc.RequestException,
            urllib3.exceptions.HTTPError,
        )
    except Exception:  # pragma: no cover
        http_exc = ()

    for i in range(max(1, int(tries))):
        try:
            return fn(*args, **kwargs)
        except (NetworkException, TokenException, InputException, DataException, GeneralException) as e:  # type: ignore
            last = e
        except http_exc as e:  # type: ignore[misc]
            last = e
        except Exception as e:
            last = e
        if i == tries - 1:
            break
        time.sleep(delay + random.uniform(0, 0.05))
        delay = min(8.0, delay * 2.0)
    if last:
        raise last


def _coerce_interval(s: str) -> str:
    s = (s or "").strip().lower()
    interval = _INTERVAL_MAP.get(s)
    if interval:
        return interval
    log.warning("Unsupported timeframe '%s' — falling back to 'minute'.", s)
    return "minute"


def _naive_ist(dt: datetime) -> datetime:
    """Return ``dt`` converted to timezone-naive IST."""
    ts = pd.Timestamp(dt)
    try:
        from src.utils.time_windows import TZ

        if ts.tzinfo is None:
            ts = ts.tz_localize(TZ)
        else:
            ts = ts.tz_convert(TZ)
        return ts.tz_localize(None).to_pydatetime()
    except Exception:
        return ts.tz_localize(None).to_pydatetime()


def _clip_window(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    try:
        s = pd.Timestamp(start).tz_localize(None)
        e = pd.Timestamp(end).tz_localize(None)
        idx = pd.DatetimeIndex(df.index).tz_localize(None)
        mask = (idx >= s) & (idx <= e)
        return df.loc[mask].copy()
    except Exception:
        return df.copy()


def _kite_symbol(symbol_or_token: Any) -> Any:
    """Best effort normalization for symbols passed to Kite LTP.

    ``LiveKiteSource.get_last_price`` accepts either instrument tokens or
    trading symbols.  Users often supply shorthand strings like ``"NIFTY50"``
    which Kite expects as ``"NSE:NIFTY 50"``. This helper preserves numeric
    tokens and returns a best‑guess exchange‑qualified symbol for strings.
    """

    if not isinstance(symbol_or_token, str):
        return symbol_or_token

    sym = symbol_or_token.strip()
    if ":" in sym:
        return sym

    cleaned = sym.replace(" ", "").upper()
    if cleaned in {"NIFTY50", "NIFTY"}:
        return "NSE:NIFTY 50"

    return f"NSE:{sym}"


def _synthetic_ohlc(
    price: float, end: datetime, interval: str, bars: int = WARMUP_BARS
) -> pd.DataFrame:
    """Generate a synthetic OHLC frame repeating ``price`` for ``bars`` rows."""
    try:
        bars = int(bars)
    except Exception:
        bars = WARMUP_BARS
    if bars <= 0:
        bars = 1

    minutes = _INTERVAL_TO_MINUTES.get(_coerce_interval(interval), 1)
    freq = _INTERVAL_TO_FREQ.get(_coerce_interval(interval), "1min")
    start = end - timedelta(minutes=minutes * bars)
    idx = pd.date_range(start=start, periods=bars, freq=freq)
    data = {
        "open": [price] * bars,
        "high": [price] * bars,
        "low": [price] * bars,
        "close": [price] * bars,
        "volume": [0] * bars,
    }
    return pd.DataFrame(data, index=idx)


def get_historical_data(
    source: DataSource,
    token: int,
    end: datetime,
    timeframe: str,
    warmup_bars: int = WARMUP_BARS,
) -> Optional[pd.DataFrame]:
    """Fetch at least ``warmup_bars`` rows of OHLC data.

    The helper expands the lookback window progressively until the desired
    number of bars is retrieved or returns whatever data could be obtained
    after a few attempts. Returned data is capped to the most recent
    ``warmup_bars`` rows.
    """
    if data_warmup_disable():
        log.info("Warmup disabled via DATA__WARMUP_DISABLE=true")
        return pd.DataFrame()

    try:
        warmup = int(warmup_bars)
    except Exception:
        warmup = 0

    if warmup <= 0:
        warmup = 1

    interval = _coerce_interval(timeframe)
    step = timedelta(minutes=_INTERVAL_TO_MINUTES.get(interval, 1) * warmup)
    start = end - step

    attempts = 0
    df: Optional[pd.DataFrame] = None

    while attempts < 4:
        df = source.fetch_ohlc(token=token, start=start, end=end, timeframe=timeframe)
        if isinstance(df, pd.DataFrame) and len(df) >= warmup:
            return df.tail(warmup).copy()
        if attempts == 2:
            start -= step + timedelta(days=2)
        else:
            start -= step
        attempts += 1

    # Return whatever data was fetched (may be ``None`` or short).
    if isinstance(df, pd.DataFrame):
        if len(df) > warmup:
            return df.tail(warmup).copy()
        return df.copy()
    return df


# --------------------------------------------------------------------------------------
# LiveKiteSource
# --------------------------------------------------------------------------------------
class LiveKiteSource(DataSource, BaseDataSource):
    """
    Reads candles via Kite's historical API + LTP for quick checks.
    Adds a tiny TTL cache to stay under rate limits during frequent ticks/diags.
    Provides a safe fallback (synthetic bar at LTP) so downstream diags don't break.
    """

    def __init__(self, kite: Optional["KiteConnect"]) -> None:
        self.kite = kite
        self._cache = _TTLCache(ttl_sec=4.0)
        self.cb_hist = CircuitBreaker("historical")
        self.cb_quote = CircuitBreaker("quote")
        self._last_tick_ts: Optional[datetime] = None
        self._last_bar_open_ts = None
        self._tf_seconds = 60
        self._last_backfill: Optional[dt.datetime] = None
        self._backfill_cooldown_s = 60

    # ---- lifecycle ----
    def connect(self) -> None:
        """Connect to broker, subscribe to SPOT token and warm up cache."""
        token = 0
        try:
            token = int(
                getattr(settings.instruments, "spot_token", 0)
                or getattr(settings.instruments, "instrument_token", 0)
            )
        except Exception:
            token = 0

        if not self.kite:
            log.info("LiveKiteSource: kite is None (shadow mode).")
        else:
            log.info("LiveKiteSource: connected to Kite.")
            if token > 0:
                try:
                    sub_fn = getattr(self.kite, "subscribe", None)
                    mode_fn = getattr(self.kite, "set_mode", None)
                    full_mode = getattr(self.kite, "MODE_FULL", "full")
                    if callable(sub_fn) and callable(mode_fn):
                        sub_fn([token])
                        mode_fn(full_mode, [token])
                except Exception as e:
                    log.warning("LiveKiteSource connect: subscribe failed: %s", e)

        if token > 0:
            try:
                self.ensure_backfill(
                    required_bars=WARMUP_BARS, token=token, timeframe="minute"
                )
            except Exception as e:
                log.warning("LiveKiteSource connect: backfill failed: %s", e)

    def disconnect(self) -> None:
        """Close the underlying Kite session if available."""
        if not self.kite:
            return
        close_fn = getattr(self.kite, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                log.debug("LiveKiteSource: kite close failed", exc_info=True)
        log.info("LiveKiteSource: disconnected from Kite.")

    def api_health(self) -> Dict[str, Dict[str, object]]:
        """Return circuit breaker health for broker APIs."""
        return {"hist": self.cb_hist.health(), "quote": self.cb_quote.health()}

    # ---- quick LTP ----
    def get_last_price(self, symbol_or_token: Any) -> Optional[float]:
        if not self.kite or not self.cb_quote.allow():
            return None
        t0 = time.monotonic()
        try:
            sym_or_token = _kite_symbol(symbol_or_token)
            data = _retry(self.kite.ltp, [sym_or_token], tries=2)
            lat = int((time.monotonic() - t0) * 1000)
            self.cb_quote.record_success(lat)
            key = str(sym_or_token)
            v = (data or {}).get(key)
            if not isinstance(v, dict):
                log.warning("get_last_price: %s not found in LTP response", sym_or_token)
                return None
            val = v.get("last_price")
            price = float(val) if isinstance(val, (int, float)) else None
            if price is not None:
                self._last_tick_ts = datetime.utcnow()
            return price
        except Exception as e:
            lat = int((time.monotonic() - t0) * 1000)
            self.cb_quote.record_failure(lat, reason=str(e))
            log.debug("get_last_price failed for %s: %s", symbol_or_token, e)
            return None

    def ensure_backfill(
        self, *, required_bars: int, token: int = 256265, timeframe: str = "minute"
    ) -> None:
        """Best-effort backfill to reach ``required_bars`` bars."""
        if data_warmup_disable():
            log.info("ensure_backfill skipped via DATA__WARMUP_DISABLE=true")
            return

        now = dt.datetime.now()
        if (
            self._last_backfill
            and (now - self._last_backfill).total_seconds() < self._backfill_cooldown_s
        ):
            return
        self._last_backfill = now
        try:
            if self.kite:
                to_dt = now
                from_dt = to_dt - dt.timedelta(minutes=max(required_bars + 5, 60))
                hist = self.kite.historical_data(token, from_dt, to_dt, timeframe)
                if hist:
                    df = pd.DataFrame(hist)
                    if not df.empty and "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df.set_index("date", inplace=True)
                        df.sort_index(inplace=True)
                        if hasattr(self, "seed_ohlc"):
                            try:
                                self.seed_ohlc(df[["open", "high", "low", "close", "volume"]])
                            except Exception:
                                pass
                    if len(df) >= required_bars:
                        return
        except Exception as e:
            log.warning("kite historical backfill failed: %s", e)

    # ---- main candle fetch ----
    def fetch_ohlc(
        self, token: int, start: datetime, end: datetime, timeframe: str
    ) -> Optional[pd.DataFrame]:
        """
        Primary path: Kite ``historical_data``.
        Returns ``None`` if no candles were retrieved even after all retries.
        """
        # Guard inputs
        try:
            token = int(token)
        except Exception:
            log.error("fetch_ohlc: invalid token %r", token)
            return None

        if not isinstance(start, datetime) or not isinstance(end, datetime):
            log.error(
                "fetch_ohlc: start/end must be datetime, got %r %r", type(start), type(end)
            )
            return None

        start = _naive_ist(start)
        end = _naive_ist(end)
        if start >= end:
            # Soft auto-correct: if equal or reversed, nudge start back 10 minutes
            start = end - timedelta(minutes=10)

        interval = _coerce_interval(str(timeframe))
        self._tf_seconds = _INTERVAL_TO_MINUTES.get(interval, 1) * 60

        # Ensure warmup window
        needed = timedelta(
            minutes=_INTERVAL_TO_MINUTES.get(interval, 1) * WARMUP_BARS
        )
        if end - start < needed:
            start = end - needed

        # Try cache first
        cached = self._cache.get(token, interval, start, end)
        if cached is not None and not cached.empty:
            bars = _clip_window(cached, start, end)
            if not bars.empty:
                self._last_bar_open_ts = pd.to_datetime(bars.index[-1]).to_pydatetime()
            return bars

        if not self.kite or not self.cb_hist.allow():
            log.warning("LiveKiteSource.fetch_ohlc: broker unavailable.")
            ltp = self.get_last_price(token)
            if isinstance(ltp, (int, float)):
                syn = _synthetic_ohlc(float(ltp), end, interval, WARMUP_BARS)
                fetched_window = (
                    pd.to_datetime(syn.index.min()).to_pydatetime(),
                    pd.to_datetime(syn.index.max()).to_pydatetime(),
                )
                self._cache.set(
                    int(token),
                    interval,
                    syn,
                    fetched_window,
                    (start, end),
                )
                if not syn.empty:
                    self._last_bar_open_ts = pd.to_datetime(syn.index[-1]).to_pydatetime()
                return syn
            return None

        # Kite expects naive or tz-aware UTC; we'll pass naive (already)
        frm = pd.to_datetime(start).to_pydatetime()
        to = pd.to_datetime(end).to_pydatetime()

        # Kite limits historical_data to ~2000 candles per call.  Chunk long
        # requests (e.g., multi‑day backfills) into smaller ranges and stitch
        # the results together.  This keeps the external behaviour the same
        # while avoiding silent truncation.
        step: Optional[timedelta] = None
        if interval in _INTERVAL_TO_MINUTES:
            step = timedelta(minutes=_INTERVAL_TO_MINUTES[interval] * 2000)
        elif interval == "day":
            step = timedelta(days=2000)

        frames: List[pd.DataFrame] = []
        cur = frm
        try:
            while cur < to:
                cur_end = to if step is None else min(to, cur + step)
                t0 = time.monotonic()
                try:
                    rows = _retry(
                        self.kite.historical_data,
                        token,
                        cur,
                        cur_end,
                        interval,
                        continuous=False,
                        oi=False,
                        tries=3,
                    )
                except Exception as e:
                    lat = int((time.monotonic() - t0) * 1000)
                    msg = str(e)
                    if (
                        "Historical fetch API call should not be made before the market opens"
                        in msg
                    ):
                        self.cb_hist.record_failure(lat, reason="pre_open")
                        if not getattr(self, "_preopen_logged", False):
                            log.info(
                                "historical_data before market open; using previous session"
                            )
                            self._preopen_logged = True
                        from src.utils.market_time import prev_session_bounds
                        from src.utils.time_windows import TZ

                        prev_start, prev_end = prev_session_bounds(dt.datetime.now(TZ))
                        prev_start = _naive_ist(prev_start)
                        prev_end = _naive_ist(prev_end)
                        try:
                            rows = _retry(
                                self.kite.historical_data,
                                token,
                                prev_start,
                                prev_end,
                                interval,
                                continuous=False,
                                oi=False,
                                tries=3,
                            )
                        except Exception:
                            return None
                        part = _safe_dataframe(rows)
                        if not part.empty:
                            frames.append(part)
                            start, end = prev_start, prev_end
                            break
                        return None
                    else:
                        self.cb_hist.record_failure(lat, reason=msg)
                        cur = cur_end
                        continue
                else:
                    lat = int((time.monotonic() - t0) * 1000)
                    self.cb_hist.record_success(lat)
                    part = _safe_dataframe(rows)
                    if not part.empty:
                        frames.append(part)
                cur = cur_end

            df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
            df = df[~df.index.duplicated(keep="last")]

            if df.empty:
                log.error(
                    "historical_data empty for token=%s interval=%s window=%s→%s",
                    token,
                    interval,
                    start,
                    end,
                )
                ltp = self.get_last_price(token)
                if isinstance(ltp, (int, float)):
                    syn = _synthetic_ohlc(float(ltp), end, interval, WARMUP_BARS)
                    fetched_window = (
                        pd.to_datetime(syn.index.min()).to_pydatetime(),
                        pd.to_datetime(syn.index.max()).to_pydatetime(),
                    )
                    self._cache.set(
                        token,
                        interval,
                        syn,
                        fetched_window,
                        (start, end),
                    )
                    return syn
                return None

            clipped = _clip_window(df, start, end)
            fetched_window = (
                pd.to_datetime(df.index.min()).to_pydatetime(),
                pd.to_datetime(df.index.max()).to_pydatetime(),
            )
            self._cache.set(
                token,
                interval,
                df,
                fetched_window,
                (start, end),
            )
            need = {"open", "high", "low", "close"}
            if not clipped.empty and need.issubset(clipped.columns):
                self._last_bar_open_ts = pd.to_datetime(clipped.index[-1]).to_pydatetime()
                return clipped

            return None

        except Exception as e:
            log.warning("fetch_ohlc failed token=%s interval=%s: %s", token, interval, e)
            ltp = self.get_last_price(token)
            if isinstance(ltp, (int, float)):
                syn = _synthetic_ohlc(float(ltp), end, interval, WARMUP_BARS)
                fetched_window = (
                    pd.to_datetime(syn.index.min()).to_pydatetime(),
                    pd.to_datetime(syn.index.max()).to_pydatetime(),
                )
                self._cache.set(
                    token,
                    interval,
                    syn,
                    fetched_window,
                    (start, end),
                )
                if not syn.empty:
                    self._last_bar_open_ts = pd.to_datetime(syn.index[-1]).to_pydatetime()
                return syn
            return None


def _livekite_health() -> float:
    """Simple health score for LiveKiteSource."""
    return 100.0


_auto_atm_last_check_ts = 0.0


def auto_resubscribe_atm(self: Any) -> None:
    """
    Periodically ensure the current ATM option tokens are subscribed for L1.
    Works with common method/attr names; no-ops if nothing is detectable.
    Controlled by: AUTO_ATM_RESUB_INTERVAL_S (default 30s)
    """
    global _auto_atm_last_check_ts
    try:
        interval = int(os.getenv("AUTO_ATM_RESUB_INTERVAL_S", "30"))
    except Exception:
        interval = 30
    now = time.time()
    if now - _auto_atm_last_check_ts < interval:
        return
    _auto_atm_last_check_ts = now

    lg = logging.getLogger(__name__)

    # 1) Discover ATM tokens
    tokens = None
    for name in ("atm_tokens", "current_atm_tokens", "_atm_tokens"):
        if hasattr(self, name):
            v = getattr(self, name)
            if isinstance(v, (list, tuple, set)) and v:
                tokens = list(v)
                break
    if tokens is None:
        for meth in ("get_atm_tokens", "compute_atm_tokens", "atm_tokens"):
            fn = getattr(self, meth, None)
            if callable(fn):
                try:
                    v = fn()
                    if isinstance(v, (list, tuple, set)) and v:
                        tokens = list(v)
                        break
                except Exception:
                    pass
    if not tokens:
        return  # can't infer ATM tokens; nothing to do

    # 2) Check if we have fresh L1 for each token
    stale: list[int] = []
    for t in tokens:
        ok = False
        # try self-level getters
        for get in ("get_l1", "ltp", "get_quote", "quote"):
            fn = getattr(self, get, None)
            if callable(fn):
                try:
                    q = fn([t]) if get in ("get_quote", "quote") else fn(t)
                    ok = bool(q)
                    break
                except Exception:
                    pass
        # try broker-level as fallback
        if not ok and hasattr(self, "broker"):
            for bget in ("ltp", "quote", "get_ltp"):
                fn = getattr(self.broker, bget, None)
                if callable(fn):
                    try:
                        q = fn([t]) if bget in ("quote",) else fn(t)
                        ok = bool(q)
                        break
                    except Exception:
                        pass
        if not ok:
            stale.append(t)

    if not stale:
        return

    # 3) Resubscribe missing tokens
    resubbed = False
    for sub in ("subscribe_tokens", "subscribe", "subscribe_l1"):
        fn = getattr(self, sub, None)
        if callable(fn):
            try:
                fn(stale)
                resubbed = True
                break
            except Exception:
                pass
    if not resubbed and hasattr(self, "broker"):
        for bsub in ("subscribe_tokens", "subscribe"):
            fn = getattr(self.broker, bsub, None)
            if callable(fn):
                try:
                    fn(stale)
                    resubbed = True
                    break
                except Exception:
                    pass

    if resubbed:
        lg.info("auto_resubscribe_atm: resubscribed tokens=%s", stale)
    else:
        lg.warning("auto_resubscribe_atm: could not resubscribe tokens=%s", stale)


# Bind helper to DataSource for easy access
DataSource.auto_resubscribe_atm = auto_resubscribe_atm  # type: ignore[attr-defined]


