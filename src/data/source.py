# src/data/source.py
from __future__ import annotations

import datetime as dt
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import pandas as pd

from src.server.logging_setup import LogSuppressor

from src.boot.validate_env import (
    data_clamp_to_market_open,
    data_warmup_backfill_min,
    data_warmup_disable,
)
from src.config import settings
from src.data.base_source import BaseDataSource
from src.data.types import HistResult, HistStatus
from src.diagnostics import healthkit
from src.diagnostics.checks import emit_quote_diag
from src.logs import structured_log
from src.utils.atr_helper import compute_atr
from src.utils.circuit_breaker import CircuitBreaker
from src.utils.indicators import calculate_vwap
from src.utils.market_time import prev_session_bounds
from src.utils.strike_selector import _nearest_strike
from src.utils.time_windows import TZ

if TYPE_CHECKING:
    from src.utils.log_gate import LogGate

log = logging.getLogger(__name__)

# Warn only once when historical data access is denied
_warn_perm_once = False

_hist_log_suppressor = LogSuppressor()
_tick_log_suppressor = LogSuppressor(window_sec=180.0)


def _as_float(val: Any) -> float:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(num):
        return 0.0
    return num


def _as_int(val: Any) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _to_epoch_ms(value: Any) -> int | None:
    """Best-effort conversion of ``value`` to epoch milliseconds."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        if val >= 1e12:
            return int(val)
        if val >= 1e9:
            return int(val * 1000.0)
        return int(val)
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000.0)
    if isinstance(value, str):
        try:
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
        except Exception:
            try:
                parsed = pd.to_datetime(value).to_pydatetime()
            except Exception:
                return None
        return int(parsed.timestamp() * 1000.0)
    return None


def get_option_quote_safe(
    *,
    option: Mapping[str, Any] | None,
    quote: Mapping[str, Any] | None,
    fetch_ltp: Callable[[Any], Optional[float]] | None = None,
) -> tuple[Optional[Dict[str, Any]], str]:
    """Return a normalized option quote with sensible fallbacks.

    Parameters
    ----------
    option:
        Metadata for the option. Only ``token``/``instrument_token`` and
        ``tradingsymbol`` keys are consulted when a REST LTP fallback is needed.
    quote:
        Raw quote payload from the broker. The helper tolerates partially
        populated dictionaries (missing depth/bid/ask fields).
    fetch_ltp:
        Callable used to fetch a fresh LTP if depth/bid/ask data is unavailable.
        Invoked at most once.

    Returns
    -------
    tuple
        ``(quote_dict, mode)`` where ``quote_dict`` contains ``bid``, ``ask``,
        ``mid``, ``ltp`` and depth metrics. ``mode`` indicates which source was
        used for pricing (``"depth"``, ``"ltp"``, ``"bid"``, ``"ask"``,
        ``"rest_ltp"``). When a usable price cannot be determined the function
        returns ``(None, "no_quote")``.
    """

    if not option:
        return None, "no_quote"

    data: Mapping[str, Any] = quote or {}
    depth = data.get("depth")
    if not isinstance(depth, Mapping):
        depth = {}

    buy_levels = depth.get("buy") if isinstance(depth, Mapping) else None
    sell_levels = depth.get("sell") if isinstance(depth, Mapping) else None
    buy_levels = buy_levels if isinstance(buy_levels, list) else []
    sell_levels = sell_levels if isinstance(sell_levels, list) else []

    bid = _as_float(data.get("bid"))
    ask = _as_float(data.get("ask"))

    if (bid <= 0.0 or ask <= 0.0) and buy_levels:
        lvl = buy_levels[0] if isinstance(buy_levels[0], Mapping) else {}
        bid = _as_float(lvl.get("price")) if bid <= 0.0 else bid
    if (bid <= 0.0 or ask <= 0.0) and sell_levels:
        lvl = sell_levels[0] if isinstance(sell_levels[0], Mapping) else {}
        ask = _as_float(lvl.get("price")) if ask <= 0.0 else ask

    bid_qty = _as_int(data.get("bid_qty"))
    ask_qty = _as_int(data.get("ask_qty"))
    if bid_qty <= 0 and buy_levels:
        lvl = buy_levels[0] if isinstance(buy_levels[0], Mapping) else {}
        bid_qty = _as_int(lvl.get("quantity"))
    if ask_qty <= 0 and sell_levels:
        lvl = sell_levels[0] if isinstance(sell_levels[0], Mapping) else {}
        ask_qty = _as_int(lvl.get("quantity"))

    bid5 = _as_int(data.get("bid5_qty"))
    ask5 = _as_int(data.get("ask5_qty"))
    if bid5 <= 0 and buy_levels:
        total = 0
        for lvl in buy_levels[:5]:
            total += _as_int(lvl.get("quantity") if isinstance(lvl, Mapping) else 0)
        bid5 = total
    if ask5 <= 0 and sell_levels:
        total = 0
        for lvl in sell_levels[:5]:
            total += _as_int(lvl.get("quantity") if isinstance(lvl, Mapping) else 0)
        ask5 = total

    ltp = _as_float(data.get("last_price")) or _as_float(data.get("ltp"))

    mode = "depth"
    mid = 0.0
    if bid > 0.0 and ask > 0.0:
        mid = (bid + ask) / 2.0
    else:
        mode = "ltp"
        if ltp > 0.0:
            mid = ltp
        elif bid > 0.0:
            mode = "bid"
            mid = bid
            if ltp <= 0.0:
                ltp = bid
        elif ask > 0.0:
            mode = "ask"
            mid = ask
            if ltp <= 0.0:
                ltp = ask
        else:
            mode = "rest_ltp"
            identifier: Any | None = option.get("token") or option.get(
                "instrument_token"
            )
            if identifier is None:
                ident_sym = option.get("tradingsymbol") or option.get("symbol")
                if isinstance(ident_sym, str) and ident_sym.strip():
                    sym = ident_sym.strip()
                    identifier = sym if ":" in sym else f"NFO:{sym}"
            fetched = None
            if fetch_ltp is not None and identifier is not None:
                try:
                    fetched = fetch_ltp(identifier)
                except Exception:
                    fetched = None
            if fetched and fetched > 0.0:
                mid = float(fetched)
                ltp = mid
            else:
                return None, "no_quote"

    if mid <= 0.0:
        return None, "no_quote"

    if ltp <= 0.0:
        ltp = mid

    ts_val = data.get("timestamp") or data.get("last_trade_time")
    if isinstance(ts_val, datetime):
        timestamp: Optional[str] = ts_val.isoformat()
    elif isinstance(ts_val, str):
        timestamp = ts_val
    else:
        timestamp = None

    result: Dict[str, Any] = {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "ltp": ltp,
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "bid5_qty": bid5,
        "ask5_qty": ask5,
        "source": data.get("source", "kite"),
    }

    if timestamp is not None:
        result["timestamp"] = timestamp

    if "oi" in data:
        result["oi"] = data.get("oi")

    if buy_levels or sell_levels:
        result["depth"] = {"buy": buy_levels, "sell": sell_levels}

    return result, mode

# Optional broker SDK (keep imports tolerant so paper mode works)
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import (  # type: ignore
        DataException,
        GeneralException,
        InputException,
        NetworkException,
        TokenException,
        PermissionException,
    )
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore

    class PermissionException(Exception):  # type: ignore[no-redef]
        """Fallback used when kiteconnect is unavailable."""

    # Collapse to base Exception so retry wrapper still works in paper mode
    NetworkException = TokenException = InputException = DataException = GeneralException = Exception  # type: ignore


class MinuteBarBuilder:
    """Aggregate ticks into 1-minute OHLCV bars.

    The builder maintains a deque of the most recent ``max_bars`` fully-formed
    bars and a partial bar for the current minute. Call :meth:`on_tick` for each
    tick and retrieve bars via :meth:`get_recent_bars`.
    """

    def __init__(self, max_bars: int = 120) -> None:
        self.max_bars = int(max_bars)
        self.bars: Deque[dict[str, Any]] = deque(maxlen=self.max_bars)
        self._current: dict[str, Any] | None = None
        self._minute: datetime | None = None

    def on_tick(self, tick: Dict[str, Any]) -> None:
        """Incorporate ``tick`` into the current minute bar."""
        ts = tick.get("timestamp") or tick.get("exchange_timestamp")
        if ts is None:
            return
        if not isinstance(ts, datetime):
            ts = datetime.fromtimestamp(float(ts))
        minute = ts.replace(second=0, microsecond=0)
        price = tick.get("last_price")
        if price is None:
            return

        vol = cast(float, tick.get("volume") or 0)
        if minute != self._minute:
            if self._current:
                self.bars.append(self._current)
            self._current = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": vol,
                "vwap": price,
                "count": 1,
                "timestamp": minute,
            }
            self._minute = minute
            return

        bar = self._current
        if bar is None:
            return
        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        prev_vol = cast(float, bar["volume"])
        bar["volume"] = prev_vol + vol
        if bar["volume"]:
            bar["vwap"] = (bar["vwap"] * prev_vol + price * vol) / bar["volume"]
        bar["count"] += 1

    def get_recent_bars(self, n: int) -> List[dict[str, Any]]:
        """Return up to ``n`` most recent bars including the current partial bar."""
        out: List[dict[str, Any]] = list(self.bars)
        if self._current:
            out.append(self._current)
        return out[-n:]

    def have_min_bars(self, n: int) -> bool:
        """Return ``True`` if at least ``n`` bars (including partial) are available."""
        count = len(self.bars) + (1 if self._current else 0)
        return count >= n


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
    ) -> HistResult:
        """Return historical bars for ``token``.

        Implementations must always return a :class:`HistResult` where ``df`` is
        a DataFrame (possibly empty) and ``status`` reflects whether data was
        retrieved.
        """
        raise NotImplementedError

    def fetch_ohlc_df(
        self, token: int, start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        """Compatibility wrapper returning only the DataFrame from
        :meth:`fetch_ohlc`.

        Always returns a DataFrame, which may be empty.
        """
        res = self.fetch_ohlc(token=token, start=start, end=end, timeframe=timeframe)
        return _normalize_ohlc_df(res.df)

    def get_last_price(self, symbol_or_token: Any) -> Optional[float]:
        """Return LTP for a trading symbol (e.g., 'NSE:NIFTY 50') or instrument token."""
        raise NotImplementedError

    def prime_option_quote(
        self, token: int | str
    ) -> tuple[float | None, str | None, int | None]:
        """Return the best-effort price snapshot for ``token``.

        The default implementation indicates absence of a quote. Live data
        sources should override this to surface the latest option book state.
        """

        return None, None, None

    def current_tokens(self) -> tuple[int | None, int | None]:
        """Return the currently tracked CE/PE instrument tokens if available."""

        def _coerce(value: Any) -> int | None:
            if value in (None, "", 0):
                return None
            try:
                return int(value)
            except Exception:
                return None

        ce_pref = getattr(self, "_current_ce_token", None)
        pe_pref = getattr(self, "_current_pe_token", None)

        tokens = getattr(self, "atm_tokens", None)
        if isinstance(tokens, (list, tuple)) and tokens:
            ce_fallback = tokens[0] if len(tokens) > 0 else None
            pe_fallback = tokens[1] if len(tokens) > 1 else None
        else:
            ce_fallback = None
            pe_fallback = None

        ce = _coerce(ce_pref if ce_pref not in (None, "", 0) else ce_fallback)
        pe = _coerce(pe_pref if pe_pref not in (None, "", 0) else pe_fallback)

        return ce, pe

    def api_health(self) -> Dict[str, Dict[str, object]]:
        """Return circuit breaker health metrics if available."""
        return {}

    def get_recent_bars(self, n: int) -> pd.DataFrame:
        """Return the last ``n`` 1m bars with ATR% and VWAP if available."""
        try:
            token = int(getattr(settings.instruments, "instrument_token", 0) or 0)
        except Exception:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "ts"]
            )

        from src.data.ohlc_builder import prepare_ohlc
        from src.utils.time_windows import floor_to_minute, now_ist

        now = now_ist()
        cutoff = now - timedelta(seconds=5)
        end = floor_to_minute(cutoff, None)
        lookback = max(60, n + 50)
        start = end - timedelta(minutes=lookback)
        df = self.fetch_ohlc_df(token=token, start=start, end=end, timeframe="minute")

        if df.empty:
            try:
                ensure = getattr(self, "ensure_backfill", None)
                if callable(ensure):
                    ensure(required_bars=n, token=token, timeframe="minute")
                    df = self.fetch_ohlc_df(
                        token=token, start=start, end=end, timeframe="minute"
                    )
            except Exception:
                log.warning("ensure_backfill failed", exc_info=True)
                df = pd.DataFrame()

        df = prepare_ohlc(df, now)
        if "vwap" not in df.columns:
            try:
                df["vwap"] = calculate_vwap(df)
            except Exception:
                log.debug("calculate_vwap failed", exc_info=True)
        if "atr_pct" not in df.columns and not df.empty:
            atr = compute_atr(df, period=14)
            df["atr_pct"] = atr / df["close"] * 100.0
        return df.tail(n)

    def get_last_bars(self, n: int) -> pd.DataFrame:  # pragma: no cover - legacy alias
        return self.get_recent_bars(n)

    def have_min_bars(self, n: int) -> bool:
        """Return ``True`` if at least ``n`` recent bars are available."""

        try:
            token = int(getattr(settings.instruments, "instrument_token", 0) or 0)
        except Exception:
            return False

        from src.utils.time_windows import floor_to_minute, now_ist

        end = floor_to_minute(now_ist(), None)
        lookback = max(60, n + 50)
        start = end - timedelta(minutes=lookback)
        res = self.fetch_ohlc(token=token, start=start, end=end, timeframe="minute")
        df = _normalize_ohlc_df(res.df)
        return len(df) >= n

    def ensure_warmup(self, n: int) -> bool:
        """Best-effort warmup to have at least ``n`` recent bars.

        Default implementation simply checks :meth:`have_min_bars`.
        Subclasses may override to provide broker backfill or live tick
        aggregation. Returns ``True`` once the requirement is met.
        """
        return self.have_min_bars(n)


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
    "day": "day",
    "1day": "day",
    "1d": "day",
    "daily": "day",
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


def _floor_to_interval_end(ts: datetime, interval: str) -> datetime:
    """Floor ``ts`` to the last fully-closed bar for ``interval``."""
    if interval == "day":
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    mins = _INTERVAL_TO_MINUTES.get(interval, 1)
    return ts - timedelta(
        minutes=ts.minute % mins,
        seconds=ts.second,
        microseconds=ts.microsecond,
    )


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
    WARMUP_BARS = 15


@dataclass
class _CacheEntry:
    df: pd.DataFrame
    ts: float  # insertion timestamp (epoch seconds)
    fetched_window: Tuple[datetime, datetime]
    requested_window: Tuple[datetime, datetime]


@dataclass(slots=True)
class QuoteState:
    token: int
    ts: float
    bid: float | None
    ask: float | None
    bid_qty: int | None
    ask_qty: int | None
    spread_pct: float | None
    has_depth: bool


@dataclass(slots=True)
class QuoteReadyStatus:
    ok: bool
    reason: str
    retries: int = 0
    bid: float | None = None
    ask: float | None = None
    bid_qty: int | None = None
    ask_qty: int | None = None
    last_tick_age_ms: int | None = None
    source: str | None = None


class _TTLCache:
    """
    Extremely simple per‑(token,interval) cache to soften historical_data pressure
    during frequent ticks or Telegram diagnostics. Keeps a tiny TTL.
    """

    def __init__(self, ttl_sec: float = 4.0) -> None:
        self._ttl = float(ttl_sec)
        self._data: Dict[Tuple[int, str], _CacheEntry] = {}

    def get(
        self, token: int, interval: str, start: datetime, end: datetime
    ) -> Optional[pd.DataFrame]:
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
                return ent.df.loc[
                    (ent.df.index >= start) & (ent.df.index <= end)
                ].copy()
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
        from src.data.ohlc_builder import prepare_ohlc
        from src.utils.time_windows import floor_to_minute, now_ist

        now = now_ist()
        cutoff = now - timedelta(seconds=5)
        end = floor_to_minute(cutoff, None)
        lookback = max(60, n + 50)
        start = end - timedelta(minutes=lookback)
        res = ds.fetch_ohlc(token=token, start=start, end=end, timeframe="minute")
        df = _normalize_ohlc_df(res.df)
        if df.empty:
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
        atr_pct = (
            (float(atr.iloc[-1]) / float(df["close"].iloc[-1]) * 100.0)
            if len(atr)
            else 0.0
        )
        lines.append(
            f"last_bar_ts={last_ts.to_pydatetime().isoformat()} ATR%={atr_pct:.2f}"
        )
        return "\n".join(lines)
    except Exception as e:  # pragma: no cover - diagnostic helper
        return f"bars error: {e}"


def _normalize_ohlc_df(rows: Any) -> pd.DataFrame:
    """Return a sanitized OHLC frame with a ``ts`` column.

    The input is copied, column names are normalised and required OHLC columns
    are verified. Numeric columns are coerced and invalid rows dropped. The
    index is sorted, duplicate timestamps removed, and a ``ts`` column is
    always present.  If ``rows`` is falsy an empty DataFrame is returned.
    """
    try:
        df = pd.DataFrame(rows if rows is not None else []).copy()
        if df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "ts"]
            )

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
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "ts"]
            )

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

        df["ts"] = pd.to_datetime(df.index)
        return df[["open", "high", "low", "close", "volume", "ts"]]

    except Exception as e:
        log.warning("Failed to normalize OHLC frame: %s", e)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "ts"])


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
        import urllib3  # type: ignore
        from requests import exceptions as req_exc  # type: ignore

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


_HIST_WARN_TS: Dict[int, float] = {}
_GLOBAL_AUTH_WARN = 0.0


def _hist_warn(
    token: int, interval: str, start: datetime, end: datetime, reason: str
) -> None:
    global _GLOBAL_AUTH_WARN
    now = time.monotonic()
    last = _HIST_WARN_TS.get(token, 0.0)
    data_cfg = getattr(settings, "data", None)
    ratelimit = getattr(data_cfg, "hist_warn_ratelimit_seconds", 300)
    try:
        ratelimit_sec = max(0, int(ratelimit))
    except (TypeError, ValueError):
        ratelimit_sec = 300

    if now - last < ratelimit_sec or now - _GLOBAL_AUTH_WARN < ratelimit_sec:
        return
    _HIST_WARN_TS[token] = now
    sym = getattr(settings.instruments, "trade_symbol", token)
    log.warning(
        "Historical data unavailable from broker token=%s symbol=%s interval=%s window=%s→%s reason=%s",
        token,
        sym,
        interval,
        start,
        end,
        reason,
    )
    rlow = reason.lower()
    if any(x in rlow for x in ("credential", "subscription", "api_key", "access_token", "auth")):
        _GLOBAL_AUTH_WARN = now


def get_historical_data(
    source: DataSource,
    token: int,
    end: datetime,
    timeframe: str,
    warmup_bars: int = WARMUP_BARS,
) -> pd.DataFrame:
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
    end = _floor_to_interval_end(end, interval)
    step = timedelta(minutes=_INTERVAL_TO_MINUTES.get(interval, 1) * warmup)
    start = end - step

    attempts = 0
    res: HistResult | None = None
    df: pd.DataFrame = pd.DataFrame()

    while attempts < 4:
        res = source.fetch_ohlc(token=token, start=start, end=end, timeframe=timeframe)
        df = _normalize_ohlc_df(res.df)
        if len(df) >= warmup:
            return df.tail(warmup).copy()
        if attempts == 2:
            start -= step + timedelta(days=2)
        else:
            start -= step
        attempts += 1

    if len(df) > warmup:
        return df.tail(warmup).copy()
    return df.copy()


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
        self.log = logging.getLogger(self.__class__.__name__)
        self.telegram = getattr(self, "telegram", None)
        gate_factory = getattr(settings, "build_log_gate", None)
        gate_obj: Any | None = None
        if callable(gate_factory):
            try:
                gate_obj = gate_factory()
            except Exception:
                gate_obj = None
                self.log.debug("log_gate.build_error", exc_info=True)
        if gate_obj is None:
            try:
                from src.utils.log_gate import LogGate as _RuntimeLogGate

                gate_obj = _RuntimeLogGate()
            except Exception:
                class _NoopGate:
                    def should_emit(self, *args: Any, **kwargs: Any) -> bool:
                        return True

                gate_obj = _NoopGate()
        self._gate = cast("LogGate", gate_obj)
        self.kite_ticker = getattr(kite, "ticker", None)
        self._quotes: dict[int, QuoteState] = {}
        self._full_depth_tokens: set[int] = set()
        self._stale_tokens: set[int] = set()
        self.settings = self._build_micro_settings()
        self._cache = _TTLCache(ttl_sec=4.0)
        self.cb_hist = CircuitBreaker("historical")
        self.cb_quote = CircuitBreaker("quote")
        self._last_tick_ts: Optional[datetime] = None
        # Epoch timestamp (seconds) of the most recent tick heartbeat.
        self._last_tick_epoch: float | None = None
        self._last_bar_open_ts = None
        self._tf_seconds = 60
        self._last_backfill: Optional[dt.datetime] = None
        self._backfill_cooldown_s = 60
        self.atm_tokens: tuple[int | None, int | None] = (None, None)
        self.current_atm_strike: int | None = None
        self.current_atm_expiry: dt.date | None = None
        self._atm_next_resolve_ts: float = 0.0
        self._atm_resolve_date: dt.date | None = None
        # Historical mode defaults to using broker backfill.  When backfill is
        # skipped (e.g., warmup disabled), switch to live warmup using in-memory
        # tick aggregation.
        self.hist_mode = "backfill"
        self.bar_builder: MinuteBarBuilder | None = None
        if data_warmup_disable():
            self.hist_mode = "live_warmup"
            self.bar_builder = MinuteBarBuilder(max_bars=120)

        self._reconnect_backoff = 1.0
        self._stale_tick_checks = 0
        self._stale_tick_thresh = 3
        self._last_refresh_min: tuple[int, int] | None = None
        self._last_auth_warn = 0.0
        self._option_quote_cache: dict[int, dict[str, Any]] = {}
        self._atm_reconnect_hook_set = False
        self._last_quote_ready_attempt: dict[int, float] = {}

    def _build_micro_settings(self) -> SimpleNamespace:
        try:
            tick_stale = float(
                getattr(settings, "TICK_STALE_SECONDS", getattr(settings, "TICK_MAX_LAG_S", 5.0))
            )
        except Exception:
            tick_stale = 5.0
        micro_cfg = getattr(settings, "micro", None)
        try:
            spread_max = float(
                getattr(micro_cfg, "max_spread_pct", getattr(micro_cfg, "spread_cap_pct", 1.0))
            ) if micro_cfg is not None else 1.0
        except Exception:
            spread_max = 1.0
        try:
            lot_size = int(getattr(getattr(settings, "instruments", None), "nifty_lot_size", 0) or 0)
        except Exception:
            lot_size = 0
        try:
            depth_min_lots = float(getattr(micro_cfg, "depth_min_lots", 0) or 0) if micro_cfg is not None else 0.0
        except Exception:
            depth_min_lots = 0.0
        depth_min_qty = int(max(depth_min_lots, 0.0) * max(lot_size, 0))
        return SimpleNamespace(
            TICK_STALE_SECONDS=max(tick_stale, 0.0),
            DEPTH_MIN_QTY=max(depth_min_qty, 0),
            SPREAD_MAX_PCT=max(spread_max, 0.0),
        )

    @property
    def last_tick_ts(self) -> float | None:
        """Epoch seconds of the most recent tick heartbeat."""

        return self._last_tick_epoch

    @last_tick_ts.setter
    def last_tick_ts(self, value: float | None) -> None:
        self._last_tick_epoch = value

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
                    if time.time() - self._last_auth_warn > 60:
                        log.warning("LiveKiteSource connect: subscribe failed: %s", e)
                        self._last_auth_warn = time.time()

            try:
                self.ensure_atm_tokens()  # type: ignore[attr-defined]
            except Exception as e:
                if time.time() - self._last_auth_warn > 60:
                    log.warning(
                        "LiveKiteSource connect: ensure_atm_tokens failed: %s", e
                    )
                    self._last_auth_warn = time.time()
            tokens_now = [int(t) for t in getattr(self, "atm_tokens", ()) if t]
            if tokens_now:
                self._subscribe_tokens_full(tokens_now)
                if not self._atm_reconnect_hook_set:
                    hook = getattr(self.kite, "on_reconnect", None)
                    if callable(hook):
                        try:
                            hook(lambda: self._subscribe_tokens_full(tokens_now))
                        except Exception:
                            log.debug(
                                "LiveKiteSource connect: on_reconnect hook failed", exc_info=True
                            )
                        else:
                            self._atm_reconnect_hook_set = True

        warmup_disabled = data_warmup_disable()
        warmup_failed = False
        if token > 0 and not warmup_disabled:
            try:
                self.ensure_backfill(
                    required_bars=WARMUP_BARS, token=token, timeframe="minute"
                )
            except Exception as e:
                if time.time() - self._last_auth_warn > 60:
                    log.warning("LiveKiteSource connect: backfill failed: %s", e)
                    self._last_auth_warn = time.time()
                warmup_failed = True
        else:
            warmup_failed = warmup_disabled

        if warmup_failed:  # pragma: no cover - rare path
            self.hist_mode = "live_warmup"
            if self.bar_builder is None:
                self.bar_builder = MinuteBarBuilder(max_bars=120)

    def _trace_force(self) -> bool:
        telegram = getattr(self, "telegram", None)
        if telegram is None:
            return False
        checker = getattr(telegram, "_window_active", None)
        if not callable(checker):
            return False
        try:
            return bool(checker("trace"))
        except Exception:  # pragma: no cover - defensive
            return False

    def _emit_tick_log(self, event: str, payload: dict[str, Any]) -> None:
        if healthkit.trace_active():
            self.log.info(event, extra=payload)
        else:
            self.log.debug(event)

    def _subscribe_tokens_full(self, tokens: list[int]) -> None:
        if not tokens:
            return
        payload: list[int] = []
        for token in tokens:
            if token is None:
                continue
            try:
                payload.append(int(token))
            except Exception:
                continue
        if not payload:
            return
        ordered_payload: list[int] = []
        seen: set[int] = set()
        for token in payload:
            if token in seen:
                continue
            seen.add(token)
            ordered_payload.append(token)
        subscribed = False
        mode_applied = False
        candidates = [
            getattr(self, "kite_ticker", None),
            getattr(self, "kws", None),
            getattr(self, "ticker", None),
            getattr(self.kite, "ticker", None) if self.kite else None,
            self.kite,
        ]
        for target in candidates:
            if target is None:
                continue
            subscribe = getattr(target, "subscribe", None)
            if callable(subscribe):
                try:
                    subscribe(ordered_payload)
                    subscribed = True
                except Exception:
                    continue
            set_mode = getattr(target, "set_mode", None)
            if callable(set_mode):
                mode = getattr(target, "MODE_FULL", getattr(self.kite, "MODE_FULL", "full"))
                try:
                    set_mode(mode, ordered_payload)
                    mode_applied = True
                except Exception:
                    continue
            if subscribed and mode_applied:
                break
        if not (subscribed and mode_applied):
            _subscribe_tokens(self, ordered_payload)
        self.log.info("data.subscribe", extra={"mode": "FULL", "tokens": ordered_payload})

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

    def reconnect_with_backoff(self) -> None:
        delay = self._reconnect_backoff
        while True:
            try:
                self.disconnect()
                self.connect()
                self._reconnect_backoff = 1.0
                break
            except Exception:
                time.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 2.0, 60.0)
                self._reconnect_backoff = delay

    def reconnect_ws(self) -> None:
        """Reconnect the streaming channel after a detected tick stall."""

        self.reconnect_with_backoff()

    def maybe_refresh_session(self, now: datetime | None = None) -> None:
        now = now or datetime.now(TZ)
        key = (now.hour, now.minute)
        if key in {(12, 30), (14, 30)} and key != self._last_refresh_min:
            self.reconnect_with_backoff()
            self._last_refresh_min = key

    def tick_watchdog(self, max_age_s: float = 3.0) -> bool:
        """Return ``True`` when consecutive tick gaps exceed the watchdog threshold."""

        heartbeat_ts = self.last_tick_ts
        now = time.time()
        tick_max_lag_s = float(
            getattr(
                settings,
                "TICK_MAX_LAG_S",
                getattr(getattr(settings, "strategy", None), "max_tick_lag_s", max_age_s),
            )
        )
        if heartbeat_ts:
            lag = now - heartbeat_ts
            if lag > tick_max_lag_s:
                bucket = round(lag)
                if _tick_log_suppressor.should_log("warn", "tick_stale", bucket):
                    log.warning("tick_stale", {"tick_lag": lag})
                try:
                    self.reconnect_ws()
                    log.info("tick_reconnect_attempt", {"tick_lag": lag})
                except Exception as exc:  # pragma: no cover - defensive reconnect guard
                    log.error("tick_reconnect_error", {"err": str(exc), "tick_lag": lag})

        ts = self._last_tick_ts
        if ts is None:
            return False
        age = (datetime.utcnow() - ts).total_seconds()
        if age > max_age_s:
            self._stale_tick_checks += 1
        else:
            self._stale_tick_checks = 0
        if self._stale_tick_checks >= self._stale_tick_thresh:
            structured_log.event(
                "stale_block",
                age_s=round(age, 3) if age is not None else None,
                checks=int(self._stale_tick_checks),
                threshold_checks=int(self._stale_tick_thresh),
                max_age_s=float(max_age_s),
            )
            return True
        return False

    def tick_watchdog_details(self) -> dict[str, float | int | None]:
        ts = self._last_tick_ts
        age = (datetime.utcnow() - ts).total_seconds() if ts else None
        return {"age": age, "checks": self._stale_tick_checks}

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
                log.warning(
                    "get_last_price: %s not found in LTP response", sym_or_token
                )
                return None
            val = v.get("last_price")
            price = float(val) if isinstance(val, (int, float)) else None
            if price is not None:
                self._last_tick_ts = datetime.utcnow()
                self.last_tick_ts = time.time()
            return price
        except Exception as e:
            lat = int((time.monotonic() - t0) * 1000)
            self.cb_quote.record_failure(lat, reason=str(e))
            log.debug("get_last_price failed for %s: %s", symbol_or_token, e)
            return None

    def _ingest_option_tick(self, tick: Mapping[str, Any]) -> None:
        token_raw = tick.get("instrument_token") or tick.get("token")
        if token_raw is None:
            return
        try:
            token = int(token_raw)
        except Exception:
            return
        tokens = getattr(self, "atm_tokens", ())
        if tokens and token not in tokens:
            return

        depth_raw = tick.get("depth") if isinstance(tick, Mapping) else None
        depth = depth_raw if isinstance(depth_raw, Mapping) else {}
        buy_levels_raw = depth.get("buy") if isinstance(depth, Mapping) else None
        sell_levels_raw = depth.get("sell") if isinstance(depth, Mapping) else None
        buy_levels = buy_levels_raw if isinstance(buy_levels_raw, Sequence) else []
        sell_levels = sell_levels_raw if isinstance(sell_levels_raw, Sequence) else []

        def _sum_levels(levels: Sequence[Any]) -> int:
            total = 0
            for lvl in levels[:5]:
                if isinstance(lvl, Mapping):
                    total += _as_int(lvl.get("quantity"))
                else:
                    total += _as_int(getattr(lvl, "quantity", 0))
            return total

        bid = _as_float(
            buy_levels[0].get("price") if buy_levels and isinstance(buy_levels[0], Mapping) else 0.0
        )
        ask = _as_float(
            sell_levels[0].get("price") if sell_levels and isinstance(sell_levels[0], Mapping) else 0.0
        )

        bid_qty = _as_int(
            buy_levels[0].get("quantity")
            if buy_levels and isinstance(buy_levels[0], Mapping)
            else tick.get("total_buy_quantity")
        )
        ask_qty = _as_int(
            sell_levels[0].get("quantity")
            if sell_levels and isinstance(sell_levels[0], Mapping)
            else tick.get("total_sell_quantity")
        )

        bid5 = _sum_levels(buy_levels) if buy_levels else _as_int(tick.get("total_buy_quantity"))
        ask5 = _sum_levels(sell_levels) if sell_levels else _as_int(tick.get("total_sell_quantity"))

        ltp = _as_float(
            tick.get("last_price")
            or tick.get("ltp")
            or tick.get("close")
            or tick.get("close_price")
        )
        mid = (bid + ask) / 2.0 if bid > 0.0 and ask > 0.0 else 0.0
        ts_val = (
            tick.get("timestamp")
            or tick.get("last_trade_time")
            or tick.get("exchange_timestamp")
        )
        ts_ms = _to_epoch_ms(ts_val) or int(time.time() * 1000)

        payload: dict[str, Any] = {
            "bid": bid,
            "ask": ask,
            "ltp": ltp,
            "mid": mid if mid > 0.0 else 0.0,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
            "bid5_qty": bid5,
            "ask5_qty": ask5,
            "timestamp": ts_val,
            "ts_ms": ts_ms,
            "source": "ws",
            "mode": "mid" if mid > 0.0 else ("ltp" if ltp > 0.0 else "bid" if bid > 0.0 else "ask"),
        }
        if isinstance(depth_raw, Mapping):
            payload["depth"] = depth_raw

        self._option_quote_cache[token] = payload

    def _on_tick(self, tick: Mapping[str, Any]) -> None:
        token_raw = tick.get("instrument_token") or tick.get("token")
        if token_raw is None:
            return
        try:
            token = int(token_raw)
        except Exception:
            return
        runner = getattr(self, "_runner", None) or getattr(self, "owner", None)
        if runner and hasattr(runner, "on_market_tick"):
            try:
                runner.on_market_tick()
            except Exception:
                pass
        now = time.time()
        depth = tick.get("depth") if isinstance(tick, Mapping) else None
        bid = ask = None
        bid_qty = ask_qty = None
        has_depth = False
        if isinstance(depth, Mapping):
            buy = depth.get("buy")
            sell = depth.get("sell")
            if isinstance(buy, Sequence) and isinstance(sell, Sequence) and buy and sell:
                try:
                    best_buy = buy[0]
                    best_sell = sell[0]
                    bid_val = (
                        best_buy.get("price")
                        if isinstance(best_buy, Mapping)
                        else getattr(best_buy, "price", None)
                    )
                    ask_val = (
                        best_sell.get("price")
                        if isinstance(best_sell, Mapping)
                        else getattr(best_sell, "price", None)
                    )
                    bid_qty_val = (
                        best_buy.get("quantity")
                        if isinstance(best_buy, Mapping)
                        else getattr(best_buy, "quantity", None)
                    )
                    ask_qty_val = (
                        best_sell.get("quantity")
                        if isinstance(best_sell, Mapping)
                        else getattr(best_sell, "quantity", None)
                    )
                    bid = float(bid_val) if bid_val is not None else None
                    ask = float(ask_val) if ask_val is not None else None
                    bid_qty = int(bid_qty_val) if bid_qty_val is not None else None
                    ask_qty = int(ask_qty_val) if ask_qty_val is not None else None
                    has_depth = bool(
                        bid is not None and ask is not None and bid > 0.0 and ask > 0.0
                    )
                except Exception:
                    has_depth = False
        ltp_tick = _as_float(
            tick.get("last_price")
            or tick.get("ltp")
            or tick.get("close")
            or tick.get("close_price")
        )
        spread_pct = None
        if bid is not None and ask is not None:
            try:
                if bid <= 0.0 or ask <= 0.0:
                    spread_pct = 999.0
                else:
                    ref_price = ltp_tick if ltp_tick > 0 else (bid + ask) / 2.0
                    spread_pct = abs(ask - bid) / max(ref_price, 1e-6) * 100.0
            except Exception:
                spread_pct = None
        state = QuoteState(
            token=token,
            ts=now,
            bid=bid,
            ask=ask,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            spread_pct=spread_pct,
            has_depth=bool(has_depth),
        )
        self._quotes[token] = state
        self._stale_tokens.discard(token)
        if has_depth:
            force_trace = self._trace_force()
            if token not in self._full_depth_tokens:
                self._full_depth_tokens.add(token)
                key_ready = f"tick_full_ready:{token}"
                if self._gate.should_emit(key_ready, force=force_trace):
                    self.log.debug(
                        "data.tick_full_ready",
                        extra={
                            "token": token,
                            "bid": bid,
                            "ask": ask,
                            "bid_qty": bid_qty,
                            "ask_qty": ask_qty,
                            "spread_pct": spread_pct,
                        },
                    )
            key = f"tick_full:{token}"
            if self._gate.should_emit(key, force=force_trace):
                self._emit_tick_log(
                    "data.tick_full",
                    {
                        "token": token,
                        "bid": bid,
                        "ask": ask,
                        "bid_qty": bid_qty,
                        "ask_qty": ask_qty,
                        "spread_pct": spread_pct,
                    },
                )
        else:
            self._emit_tick_log("data.tick_ltp_only", {"token": token})

    def get_cached_full_quote(
        self, token: int | str
    ) -> dict[str, Any] | None:
        """Return a copy of the cached FULL quote for ``token`` if available."""

        try:
            token_i = int(token)
        except Exception:
            return None

        cached = self._option_quote_cache.get(token_i)
        if not isinstance(cached, Mapping):
            return None
        return dict(cached)

    def on_tick(self, tick: Dict[str, Any]) -> None:
        """Handle incoming tick by updating warmup bar builder."""
        try:
            self._on_tick(tick)
        except Exception:
            self.log.debug("data.tick_state_error", exc_info=True)
        try:
            self._ingest_option_tick(tick)
        except Exception:  # pragma: no cover - defensive
            log.debug("option_tick_ingest_failed", exc_info=True)
        runner = getattr(self, "_runner", None) or getattr(self, "owner", None)
        if runner and hasattr(runner, "on_market_tick"):
            try:
                runner.on_market_tick()
            except Exception:
                pass
        self._last_tick_ts = datetime.utcnow()
        self.last_tick_ts = time.time()
        if self.hist_mode == "live_warmup" and self.bar_builder:
            try:
                self.bar_builder.on_tick(tick)
            except Exception:  # pragma: no cover - defensive
                log.debug("bar_builder.on_tick failed", exc_info=True)

    def get_micro_state(self, token: int) -> dict[str, Any]:
        try:
            token_i = int(token)
        except Exception:
            token_i = token if isinstance(token, int) else 0
        q = self._quotes.get(token_i)
        if not q:
            return {
                "stale": True,
                "age": None,
                "age_sec": None,
                "has_depth": False,
                "depth_ok": False,
                "spread_pct": None,
                "spread_ok": False,
                "bid": None,
                "ask": None,
                "bid_qty": None,
                "ask_qty": None,
                "last_tick_ts": None,
                "reason": "no_quote",
            }
        now = time.time()
        age = max(now - q.ts, 0.0)
        tick_stale = float(getattr(self.settings, "TICK_STALE_SECONDS", 5.0))
        stale = age > tick_stale
        depth_min_qty = int(getattr(self.settings, "DEPTH_MIN_QTY", 0))
        depth_ok = bool(
            q.has_depth
            and q.bid_qty is not None
            and q.ask_qty is not None
            and q.bid_qty >= depth_min_qty
            and q.ask_qty >= depth_min_qty
        )
        spread_cap = float(getattr(self.settings, "SPREAD_MAX_PCT", 1.0))
        spread_ok = bool(q.spread_pct is not None and q.spread_pct <= spread_cap)
        if stale:
            if token_i not in self._stale_tokens:
                self._stale_tokens.add(token_i)
                key = f"tick_stale:{token_i}"
                if self._gate.should_emit(key, force=self._trace_force()):
                    self.log.debug(
                        "data.tick_stale",
                        extra={
                            "token": token_i,
                            "age": round(age, 3),
                            "threshold": tick_stale,
                        },
                    )
        else:
            self._stale_tokens.discard(token_i)
        return {
            "stale": stale,
            "age": round(age, 3),
            "age_sec": round(age, 3),
            "has_depth": q.has_depth,
            "depth_ok": depth_ok,
            "spread_pct": q.spread_pct,
            "spread_ok": spread_ok,
            "bid": q.bid,
            "ask": q.ask,
            "bid_qty": q.bid_qty,
            "ask_qty": q.ask_qty,
            "last_tick_ts": q.ts,
        }

    def quote_snapshot(self, token: int | str) -> dict[str, Any] | None:
        """Return a lightweight quote summary for Telegram diagnostics."""

        try:
            token_i = int(token)
        except Exception:
            return None
        q = self._quotes.get(token_i)
        if not q:
            return None
        age = max(time.time() - q.ts, 0.0)
        return {
            "token": token_i,
            "bid": q.bid,
            "ask": q.ask,
            "bid_qty": q.bid_qty,
            "ask_qty": q.ask_qty,
            "spread_pct": q.spread_pct,
            "has_depth": q.has_depth,
            "age_sec": round(age, 3),
        }

    def get_last_bbo(
        self, token: int | str
    ) -> tuple[float | None, float | None, int | None, int | None, int | None]:
        """Return the last seen best-bid/ask tuple for ``token``."""

        try:
            token_i = int(token)
        except Exception:
            return None, None, None, None, None
        state = self._quotes.get(int(token_i))
        if not state:
            return None, None, None, None, None
        ts_ms = int(max(state.ts, 0.0) * 1000.0)
        return state.bid, state.ask, state.bid_qty, state.ask_qty, ts_ms

    def _update_quote_state_from_payload(
        self,
        token: int | str,
        payload: Mapping[str, Any] | None,
        *,
        ts_ms: int | None = None,
    ) -> None:
        if payload is None:
            return
        try:
            token_i = int(token)
        except Exception:
            return
        if token_i <= 0:
            return
        bid = _as_float(payload.get("bid"))
        ask = _as_float(payload.get("ask"))
        bid_qty = _as_int(payload.get("bid_qty"))
        ask_qty = _as_int(payload.get("ask_qty"))
        if bid_qty <= 0:
            depth = payload.get("depth") if isinstance(payload, Mapping) else None
            if isinstance(depth, Mapping):
                try:
                    bid_lvl = depth.get("buy", [])[0]
                except Exception:
                    bid_lvl = None
                if isinstance(bid_lvl, Mapping):
                    bid_qty = _as_int(bid_lvl.get("quantity"))
                    if bid <= 0:
                        bid = _as_float(bid_lvl.get("price"))
        if ask_qty <= 0:
            depth = payload.get("depth") if isinstance(payload, Mapping) else None
            if isinstance(depth, Mapping):
                try:
                    ask_lvl = depth.get("sell", [])[0]
                except Exception:
                    ask_lvl = None
                if isinstance(ask_lvl, Mapping):
                    ask_qty = _as_int(ask_lvl.get("quantity"))
                    if ask <= 0:
                        ask = _as_float(ask_lvl.get("price"))
        ts_val = ts_ms if isinstance(ts_ms, int) else _to_epoch_ms(payload.get("ts"))
        if ts_val is None:
            ts_val = _to_epoch_ms(payload.get("timestamp"))
        if ts_val is None:
            ts_val = int(time.time() * 1000)
        spread_pct = None
        if bid > 0 and ask > 0:
            ref = (bid + ask) / 2.0
            spread_pct = abs(ask - bid) / max(ref, 1e-6) * 100.0
        has_depth = bid > 0 and ask > 0 and bid_qty > 0 and ask_qty > 0
        self._quotes[token_i] = QuoteState(
            token=token_i,
            ts=float(ts_val) / 1000.0,
            bid=bid if bid > 0 else None,
            ask=ask if ask > 0 else None,
            bid_qty=bid_qty if bid_qty > 0 else None,
            ask_qty=ask_qty if ask_qty > 0 else None,
            spread_pct=spread_pct,
            has_depth=has_depth,
        )

    def ensure_quote_ready(
        self,
        token: int | str,
        mode: str | None = None,
        *,
        symbol: str | None = None,
    ) -> QuoteReadyStatus:
        """Ensure ``token`` has a live best-bid/offer before strategy evaluation."""

        try:
            token_i = int(token)
        except Exception:
            return QuoteReadyStatus(ok=False, reason="invalid_token")
        if token_i <= 0:
            return QuoteReadyStatus(ok=False, reason="invalid_token")

        mode_pref = mode or getattr(settings, "QUOTES__MODE", getattr(_cfg, "QUOTES__MODE", "FULL"))
        mode_norm = str(mode_pref).upper() or "FULL"
        attempts = max(
            1,
            int(
                getattr(
                    settings,
                    "QUOTES__RETRY_ATTEMPTS",
                    getattr(_cfg, "QUOTES__RETRY_ATTEMPTS", 3),
                )
            ),
        )
        timeout_ms = max(
            0,
            int(
                getattr(
                    settings,
                    "QUOTES__PRIME_TIMEOUT_MS",
                    getattr(_cfg, "QUOTES__PRIME_TIMEOUT_MS", 1500),
                )
            ),
        )
        jitter_ms = max(
            0,
            int(
                getattr(
                    settings,
                    "QUOTES__RETRY_JITTER_MS",
                    getattr(_cfg, "QUOTES__RETRY_JITTER_MS", 150),
                )
            ),
        )
        stale_ms = max(
            0,
            int(
                getattr(
                    settings,
                    "MICRO__STALE_MS",
                    getattr(_cfg, "MICRO__STALE_MS", 1500),
                )
            ),
        )
        snapshot_delay_ms = max(
            100,
            int(
                getattr(
                    settings,
                    "QUOTES__SNAPSHOT_DELAY_MS",
                    getattr(_cfg, "QUOTES__SNAPSHOT_DELAY_MS", 1000),
                )
            ),
        )

        def _state_status(state: QuoteState | None) -> QuoteReadyStatus | None:
            if not state or state.bid is None or state.ask is None:
                return None
            age_ms = int(max(0.0, (time.time() - float(state.ts)) * 1000.0))
            ok = age_ms <= stale_ms and state.bid > 0 and state.ask > 0
            reason = "ready" if ok else "stale_quote"
            return QuoteReadyStatus(
                ok=ok,
                reason=reason,
                bid=state.bid,
                ask=state.ask,
                bid_qty=state.bid_qty,
                ask_qty=state.ask_qty,
                last_tick_age_ms=age_ms,
                source=mode_norm,
            )

        ensure_subscribe = getattr(self, "ensure_token_subscribed", None)
        status = QuoteReadyStatus(ok=False, reason="no_quote")

        for attempt in range(attempts):
            status.retries = attempt
            resub_logged = False
            try:
                if callable(ensure_subscribe):
                    ensure_subscribe(token_i, mode="FULL")
                else:
                    _subscribe_tokens(self, [token_i], mode="FULL")
            except Exception:
                self.log.debug("ensure_quote_ready.subscribe_failed", exc_info=True)

            state = self._quotes.get(token_i)
            state_status = _state_status(state)
            if state_status is not None:
                status = state_status
                if (
                    state_status.last_tick_age_ms is not None
                    and state_status.last_tick_age_ms > stale_ms
                    and not resub_logged
                ):
                    try:
                        self.resubscribe_if_stale(token_i)
                    except Exception:
                        self.log.debug(
                            "ensure_quote_ready.resubscribe_failed", exc_info=True
                        )
                    resub_logged = True
            if state_status and state_status.ok:
                state_status.retries = attempt
                self._last_quote_ready_attempt[token_i] = time.time()
                emit_quote_diag(
                    token=token_i,
                    symbol=symbol,
                    sub_mode=mode_norm,
                    bid=state_status.bid,
                    ask=state_status.ask,
                    bid_qty=state_status.bid_qty,
                    ask_qty=state_status.ask_qty,
                    last_tick_age_ms=state_status.last_tick_age_ms,
                    retries=attempt,
                    reason="ready",
                    source=state_status.source,
                )
                return state_status

            deadline = time.time() + (timeout_ms / 1000.0 if timeout_ms else 0.0)
            snapshot_deadline = time.time() + (snapshot_delay_ms / 1000.0)
            snapshot_taken = False
            while timeout_ms == 0 or time.time() < deadline:
                state = self._quotes.get(token_i)
                state_status = _state_status(state)
                if state_status and state_status.ok:
                    state_status.retries = attempt
                    self._last_quote_ready_attempt[token_i] = time.time()
                    emit_quote_diag(
                        token=token_i,
                        symbol=symbol,
                        sub_mode=mode_norm,
                        bid=state_status.bid,
                        ask=state_status.ask,
                        bid_qty=state_status.bid_qty,
                        ask_qty=state_status.ask_qty,
                        last_tick_age_ms=state_status.last_tick_age_ms,
                        retries=attempt,
                        reason="ready",
                        source=state_status.source,
                    )
                    return state_status
                if state_status is not None:
                    status = state_status
                now_ts = time.time()
                if (
                    not snapshot_taken
                    and now_ts >= snapshot_deadline
                    and (status.bid in {None, 0.0} or status.ask in {None, 0.0})
                ):
                    try:
                        self.prime_option_quote(token_i)
                    except Exception:
                        self.log.debug(
                            "ensure_quote_ready.snapshot_failed", exc_info=True
                        )
                    snapshot_taken = True
                    continue
                if timeout_ms == 0:
                    break
                time.sleep(0.05)

            if attempt < attempts - 1 and jitter_ms:
                time.sleep((jitter_ms + random.uniform(0, jitter_ms)) / 1000.0)

        status.source = mode_norm
        status.reason = status.reason or "no_quote"
        self._last_quote_ready_attempt[token_i] = time.time()
        emit_quote_diag(
            token=token_i,
            symbol=symbol,
            sub_mode=mode_norm,
            bid=status.bid,
            ask=status.ask,
            bid_qty=status.bid_qty,
            ask_qty=status.ask_qty,
            last_tick_age_ms=status.last_tick_age_ms,
            retries=status.retries,
            reason=status.reason,
            source=status.source,
        )
        return status

    def resubscribe_if_stale(self, token: int | str) -> None:
        """Resubscribe to ``token`` if the cached quote appears stale."""

        try:
            token_i = int(token)
        except Exception:
            return
        if token_i <= 0:
            return
        state = self._quotes.get(token_i)
        stale_ms = int(
            getattr(settings, "MICRO__STALE_MS", getattr(_cfg, "MICRO__STALE_MS", 1500))
        )
        now = time.time()
        age_ms: float | None = None
        if state is not None:
            age_ms = (now - float(state.ts)) * 1000.0
            if age_ms <= stale_ms:
                return
        last = self._last_quote_ready_attempt.get(token_i, 0.0)
        if now - last < 0.5:
            return
        ensure_subscribe = getattr(self, "ensure_token_subscribed", None)
        try:
            if callable(ensure_subscribe):
                ensure_subscribe(token_i, mode="FULL")
            else:
                _subscribe_tokens(self, [token_i], mode="FULL")
            try:
                structured_log.event(
                    "ws_resubscribe",
                    token=int(token_i),
                    last_tick_age_ms=int(age_ms) if age_ms is not None else None,
                    reason="stale_quote",
                )
            except Exception:
                self.log.debug("resubscribe_if_stale.log_failed", exc_info=True)
            self._last_quote_ready_attempt[token_i] = now
        except Exception:
            self.log.debug("resubscribe_if_stale.ensure_failed", exc_info=True)

    def prime_option_quote(
        self, token: int | str
    ) -> tuple[float | None, str | None, int | None]:
        try:
            token_i = int(token)
        except Exception:
            return None, None, None

        def _emit_snapshot(
            *,
            price: float,
            source: str,
            ts_ms: int | None,
            mode: str | None,
        ) -> None:
            age_ms: int | None = None
            if ts_ms is not None:
                now_ms = int(time.time() * 1000)
                age_ms = int(max(0, now_ms - int(ts_ms)))
            structured_log.event(
                "market_data_snapshot",
                token=int(token_i),
                price=float(price),
                source=str(source),
                ts_ms=ts_ms,
                age_ms=age_ms,
                mode=mode,
            )

        try:
            cached = self._option_quote_cache.get(token_i)
        except Exception:
            cached = None
        if cached:
            ts_ms_cached = cached.get("ts_ms")
            ts_ms = ts_ms_cached if isinstance(ts_ms_cached, int) else _to_epoch_ms(cached.get("timestamp"))
            for key, label in (("mid", "mid"), ("ltp", "ltp"), ("bid", "bid"), ("ask", "ask")):
                val = cached.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    if ts_ms is None:
                        ts_ms = int(time.time() * 1000)
                    source_label = f"ws_{label}"
                    cached_mode = cached.get("mode")
                    mode_str = (
                        str(cached_mode)
                        if cached_mode is not None and cached_mode != ""
                        else None
                    )
                    _emit_snapshot(
                        price=float(val),
                        source=source_label,
                        ts_ms=ts_ms,
                        mode=mode_str,
                    )
                    return float(val), source_label, ts_ms

        quote_payload: dict[str, Any] | None = None
        if self.kite:
            try:
                data = self.kite.quote([token_i])
            except Exception as exc:  # pragma: no cover - broker errors
                log.debug("prime_option_quote: quote failed token=%s err=%s", token_i, exc)
                data = None
            if isinstance(data, Mapping):
                data_map = cast(Mapping[Any, Any], data)
                entry: Mapping[str, Any] | None = None
                for candidate_key in (token_i, str(token_i), f"NFO:{token_i}"):
                    candidate = data_map.get(candidate_key)
                    if isinstance(candidate, Mapping):
                        entry = cast(Mapping[str, Any], candidate)
                        break
                if entry is not None:
                    quote_payload = dict(entry)
        if not quote_payload:
            ltp_price = self.get_last_price(token_i)
            if isinstance(ltp_price, (int, float)) and ltp_price > 0:
                ts_ms = int(time.time() * 1000)
                self._option_quote_cache[token_i] = {
                    "ltp": float(ltp_price),
                    "ts_ms": ts_ms,
                    "source": "rest",
                    "mode": "rest_ltp",
                }
                _emit_snapshot(
                    price=float(ltp_price),
                    source="rest_ltp",
                    ts_ms=ts_ms,
                    mode="rest_ltp",
                )
                return float(ltp_price), "rest_ltp", ts_ms
            return None, None, None

        quote_payload.setdefault("source", "kite")
        fetch_ltp = getattr(self, "get_last_price", None)
        quote_dict, mode = get_option_quote_safe(
            option={"token": token_i},
            quote=quote_payload,
            fetch_ltp=fetch_ltp,
        )
        if not quote_dict:
            return None, None, None

        ts_ms = _to_epoch_ms(quote_dict.get("timestamp")) or int(time.time() * 1000)
        quote_dict["mode"] = mode
        quote_dict["ts_ms"] = ts_ms
        quote_dict.setdefault("source", quote_payload.get("source", "kite"))
        self._option_quote_cache[token_i] = quote_dict
        self._update_quote_state_from_payload(token_i, quote_dict, ts_ms=ts_ms)

        for key, label in (("mid", "mid"), ("ltp", "ltp"), ("bid", "bid"), ("ask", "ask")):
            val = quote_dict.get(key)
            if isinstance(val, (int, float)) and val > 0:
                if label == "ltp" and mode == "rest_ltp":
                    source_label = "rest_ltp"
                else:
                    source_label = label
                mode_val = mode if isinstance(mode, str) else str(mode) if mode else None
                _emit_snapshot(
                    price=float(val),
                    source=source_label,
                    ts_ms=ts_ms,
                    mode=mode_val,
                )
                return float(val), source_label, ts_ms

        rest_ltp: float | None = None
        if mode != "rest_ltp":
            try:
                fetched = self.get_last_price(token_i)
            except Exception:  # pragma: no cover - defensive fetch guard
                fetched = None
            if isinstance(fetched, (int, float)) and fetched > 0:
                rest_ltp = float(fetched)

        if rest_ltp is not None:
            ts_ms = int(time.time() * 1000)
            merged = dict(quote_dict)
            merged["ltp"] = rest_ltp
            merged.setdefault("mid", rest_ltp)
            merged["mode"] = "rest_ltp"
            merged["ts_ms"] = ts_ms
            merged.setdefault("source", "rest")
            self._option_quote_cache[token_i] = merged
            self._update_quote_state_from_payload(token_i, merged, ts_ms=ts_ms)
            _emit_snapshot(
                price=float(rest_ltp),
                source="rest_ltp",
                ts_ms=ts_ms,
                mode="rest_ltp",
            )
            return rest_ltp, "rest_ltp", ts_ms

        ltp_val = quote_dict.get("ltp")
        if isinstance(ltp_val, (int, float)) and ltp_val > 0:
            mode_val = mode if isinstance(mode, str) else str(mode) if mode else "ltp"
            source_label = mode_val or "ltp"
            _emit_snapshot(
                price=float(ltp_val),
                source=source_label,
                ts_ms=ts_ms,
                mode=mode_val,
            )
            return float(ltp_val), source_label, ts_ms
        return None, mode, ts_ms

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

        bars_needed = max(int(required_bars), data_warmup_backfill_min())
        lookback = bars_needed + 5

        to_dt = now
        from_dt = to_dt - dt.timedelta(minutes=lookback)
        if timeframe == "minute" and data_clamp_to_market_open():
            now_ist = dt.datetime.now(TZ)
            if now_ist.time() < dt.time(9, 16):
                prev_start, prev_end = prev_session_bounds(now_ist)
                to_dt = prev_end.replace(tzinfo=None)
                from_dt = max(
                    prev_start.replace(tzinfo=None),
                    to_dt - dt.timedelta(minutes=lookback),
                )
            else:
                mkt_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                if from_dt < mkt_open:
                    from_dt = mkt_open

        hist: list[dict[str, Any]] | None = None
        try:
            if self.kite:
                hist = self.kite.historical_data(token, from_dt, to_dt, timeframe)
        except Exception as e:
            log.warning("kite historical backfill failed: %s", e)

        df = pd.DataFrame(hist or [])
        if df.empty:
            if self.hist_mode != "broker":
                log.info(
                    "Backfill skipped (no OHLC). Using live_warmup bars."
                )
                return
            raise RuntimeError("historical_data empty ...")

        if "date" in df.columns:
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
    
    def get_recent_bars(self, n: int) -> pd.DataFrame:
        """Return last ``n`` bars using live ticks when warmup is active."""
        if self.hist_mode == "live_warmup" and self.bar_builder:
            bars = self.bar_builder.get_recent_bars(n)
            if not bars:
                return pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume", "ts"]
                )
            df = pd.DataFrame(bars).drop(columns=["count"], errors="ignore")
            df = df.rename(columns={"timestamp": "ts"}).set_index("ts")
            if "vwap" not in df.columns:  # pragma: no cover - vwap always present
                try:
                    df["vwap"] = calculate_vwap(df)
                except Exception:  # pragma: no cover - defensive
                    log.debug("calculate_vwap failed", exc_info=True)
            if "atr_pct" not in df.columns and not df.empty:
                atr = compute_atr(df, period=14)
                df["atr_pct"] = atr / df["close"] * 100.0
            return df.tail(n)
        return super().get_recent_bars(n)

    def have_min_bars(self, n: int) -> bool:
        """Return ``True`` if at least ``n`` bars are available."""
        if self.hist_mode == "live_warmup" and self.bar_builder:
            return self.bar_builder.have_min_bars(n)

        try:
            token = int(getattr(settings.instruments, "instrument_token", 0) or 0)
        except Exception:
            return False

        from src.utils.time_windows import floor_to_minute, now_ist

        end = floor_to_minute(now_ist(), None)
        lookback = max(60, n + 50)
        start = end - timedelta(minutes=lookback)

        cached = self._cache.get(token, "minute", start, end)
        if cached is not None and len(cached) >= n:
            return True

        try:
            synth_n = int(getattr(self, "_synth_bars_n", 0))
            if synth_n >= n:
                return True
        except Exception:
            pass

        res = self.fetch_ohlc(token=token, start=start, end=end, timeframe="minute")
        df = _normalize_ohlc_df(res.df)
        return len(df) >= n

    def ensure_warmup(self, n: int) -> bool:
        """Try broker backfill, else fall back to live tick aggregation."""
        if self.have_min_bars(n):
            return True
        token = 0
        try:
            token = int(getattr(settings.instruments, "instrument_token", 0) or 0)
        except Exception:
            token = 0

        if self.hist_mode != "live_warmup" and token > 0:
            try:
                self.ensure_backfill(
                    required_bars=n, token=token, timeframe="minute"
                )
            except Exception:
                pass
            if self.have_min_bars(n):
                return True

        # Broker backfill unavailable; switch to live warmup using bar builder
        self.hist_mode = "live_warmup"
        if self.bar_builder is None:
            self.bar_builder = MinuteBarBuilder(max_bars=120)
        return self.have_min_bars(n)

    # ---- main candle fetch ----
    def _fetch_ohlc_df(
        self, token: int, start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        self._last_hist_reason = ""
        # Guard inputs
        try:
            token = int(token)
        except Exception:
            log.error("fetch_ohlc: invalid token %r", token)
            self._last_hist_reason = "invalid_token"
            return pd.DataFrame()

        if not isinstance(start, datetime) or not isinstance(end, datetime):
            log.error(
                "fetch_ohlc: start/end must be datetime, got %r %r",
                type(start),
                type(end),
            )
            self._last_hist_reason = "invalid_time"
            return pd.DataFrame()

        start = _naive_ist(start)
        end = _naive_ist(end)
        if start >= end:
            # Soft auto-correct: if equal or reversed, nudge start back 10 minutes
            start = end - timedelta(minutes=10)

        interval = _coerce_interval(str(timeframe))
        self._tf_seconds = _INTERVAL_TO_MINUTES.get(interval, 1) * 60

        # Ensure warmup window
        needed = timedelta(minutes=_INTERVAL_TO_MINUTES.get(interval, 1) * WARMUP_BARS)
        if end - start < needed:
            start = end - needed

        def _use_prev_close() -> pd.DataFrame:
            """Return synthetic bars seeded from previous session's close."""
            if not self.kite:
                return pd.DataFrame()
            try:
                from src.utils.market_time import prev_session_bounds
                from src.utils.time_windows import TZ

                prev_start, prev_end = prev_session_bounds(dt.datetime.now(TZ))
                prev_start = _naive_ist(prev_start)
                prev_end = _naive_ist(prev_end)
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
                part = _normalize_ohlc_df(rows)
                if part.empty:
                    return pd.DataFrame()
                close = float(part["close"].iloc[-1])
                syn = _synthetic_ohlc(close, end, interval, WARMUP_BARS)
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
                    self._last_bar_open_ts = pd.to_datetime(
                        syn.index[-1]
                    ).to_pydatetime()
                self._last_hist_reason = "synthetic_prev_close"
                return syn
            except Exception:
                return pd.DataFrame()

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
                    self._last_bar_open_ts = pd.to_datetime(
                        syn.index[-1]
                    ).to_pydatetime()
                self._last_hist_reason = "synthetic_ltp"
                return syn
            prev_syn = _use_prev_close()
            if not prev_syn.empty:
                return prev_syn
            self._last_hist_reason = "broker_unavailable"
            return pd.DataFrame()

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
                            self._last_hist_reason = "preopen_fallback_failed"
                            return pd.DataFrame()
                        part = _normalize_ohlc_df(rows)
                        if not part.empty:
                            frames.append(part)
                            start, end = prev_start, prev_end
                            break
                        self._last_hist_reason = "preopen_empty"
                        return pd.DataFrame()
                    else:
                        self.cb_hist.record_failure(lat, reason=msg)
                        cur = cur_end
                        continue
                else:
                    lat = int((time.monotonic() - t0) * 1000)
                    self.cb_hist.record_success(lat)
                    part = _normalize_ohlc_df(rows)
                    if not part.empty:
                        frames.append(part)
                cur = cur_end

            df = pd.concat(frames).sort_index() if frames else pd.DataFrame()
            df = df[~df.index.duplicated(keep="last")]

            if df.empty:
                if _hist_log_suppressor.should_log(
                    "hist.empty",
                    f"{token}:{interval}",
                    f"{start}->{end}",
                ):
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
                    self._last_hist_reason = "synthetic_ltp"
                    return syn
                prev_syn = _use_prev_close()
                if not prev_syn.empty:
                    return prev_syn
                self._last_hist_reason = "empty"
                return pd.DataFrame()

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
                self._last_bar_open_ts = pd.to_datetime(
                    clipped.index[-1]
                ).to_pydatetime()
                return clipped

            self._last_hist_reason = "missing_cols"
            return pd.DataFrame()

        except PermissionException as e:
            self._last_hist_reason = "permission_denied"
            global _warn_perm_once
            if not _warn_perm_once:
                log.warning(
                    "fetch_ohlc permission denied token=%s interval=%s: %s; falling back to live warmup",
                    token,
                    interval,
                    e,
                )
                _warn_perm_once = True
            self.hist_mode = "live_warmup"
            return pd.DataFrame()

        except Exception as e:
            log.warning(
                "fetch_ohlc failed token=%s interval=%s: %s", token, interval, e
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
                if not syn.empty:
                    self._last_bar_open_ts = pd.to_datetime(
                        syn.index[-1]
                    ).to_pydatetime()
                self._last_hist_reason = "synthetic_ltp"
                return syn
            prev_syn = _use_prev_close()
            if not prev_syn.empty:
                return prev_syn
            self._last_hist_reason = str(e)
            return pd.DataFrame()

    def fetch_ohlc(
        self, token: int, start: datetime, end: datetime, timeframe: str
    ) -> HistResult:
        df = self._fetch_ohlc_df(token=token, start=start, end=end, timeframe=timeframe)
        df = _normalize_ohlc_df(df)
        status = HistStatus.OK if not df.empty else HistStatus.NO_DATA
        reason = (
            "" if status is HistStatus.OK else getattr(self, "_last_hist_reason", "")
        )
        if status is HistStatus.NO_DATA:
            interval = _coerce_interval(str(timeframe))
            _hist_warn(int(token), interval, start, end, reason)
        return HistResult(status, df, reason)


def _livekite_health() -> float:
    """Simple health score for LiveKiteSource."""
    return 100.0


_instruments_cache: Dict[str, Any] = {"ts": 0.0, "items": []}


def _refresh_instruments_nfo(broker: Any) -> list[dict]:
    """Return cached NFO instruments, refreshing periodically."""
    global _instruments_cache
    now = time.time()
    opt_cfg = getattr(settings, "option_selector", None)
    mins_val = getattr(opt_cfg, "instruments_refresh_minutes", 15)
    try:
        mins = max(0, int(mins_val))
    except (TypeError, ValueError):
        mins = 15
    if (
        _instruments_cache["items"] and now - _instruments_cache["ts"] < mins * 60
    ) or not broker:
        return list(_instruments_cache["items"])
    items: list[dict] = []
    fn = getattr(broker, "instruments", None)
    if callable(fn):
        try:
            items = fn("NFO") or []
        except Exception:
            items = []
    _instruments_cache = {"ts": now, "items": items}
    return list(items)


def _pick_expiry(
    items: list[dict], underlying: str, today: dt.date
) -> Optional[dt.date]:
    """Pick weekly expiry for ``underlying`` on/after ``today``."""
    opt_cfg = getattr(settings, "option_selector", None)
    weekday_val = getattr(opt_cfg, "weekly_expiry_weekday", 2)
    prefer_monthly = bool(getattr(opt_cfg, "prefer_monthly_expiry", False))
    try:
        week_wd = int(weekday_val) - 1
    except (TypeError, ValueError):
        week_wd = 1
    week_wd = max(0, min(6, week_wd))

    dates: list[dt.date] = []
    for it in items:
        if it.get("name") != underlying:
            continue
        exp = it.get("expiry")
        if isinstance(exp, dt.datetime):
            d = exp.date()
        elif isinstance(exp, dt.date):
            d = exp
        else:
            try:
                d = dt.datetime.strptime(str(exp), "%Y-%m-%d").date()
            except Exception:
                continue
        dates.append(d)
    dates = sorted(set(d for d in dates if d >= today))
    if not dates:
        return None

    def _last_weekday(d: dt.date, wd: int) -> dt.date:
        n = d.replace(day=28) + dt.timedelta(days=4)
        last = n - dt.timedelta(days=n.day)
        return last - dt.timedelta(days=(last.weekday() - wd) % 7)

    if prefer_monthly:
        for d in dates:
            if d.weekday() == week_wd and d == _last_weekday(d, week_wd):
                return d
    for d in dates:
        if d.weekday() == week_wd:
            return d
    return dates[0]


def _match_token(
    items: list[dict],
    underlying: str,
    expiry: dt.date,
    strike: int,
    opt: str,
) -> Optional[int]:
    for it in items:
        try:
            if (
                it.get("name") == underlying
                and _to_date(it.get("expiry")) == expiry
                and int(float(it.get("strike", 0))) == int(strike)
                and it.get("instrument_type") == opt
            ):
                return int(it.get("instrument_token", 0) or 0)
        except Exception:
            continue
    return None


def _to_date(val: Any) -> dt.date:
    if isinstance(val, dt.datetime):
        return val.date()
    if isinstance(val, dt.date):
        return val
    return dt.datetime.strptime(str(val), "%Y-%m-%d").date()


def _subscribe_tokens(
    obj: Any, tokens: list[int], *, mode: str | None = None
) -> bool:
    if not tokens:
        return False
    seen: set[int] = {int(t) for t in tokens if t is not None}
    if not seen:
        return False
    payload = list(seen)

    def _iter_targets(source: Any) -> list[Any]:
        targets: list[Any] = []
        seen_ids: set[int] = set()

        def _append(candidate: Any) -> None:
            if candidate is None:
                return
            ident = id(candidate)
            if ident in seen_ids:
                return
            seen_ids.add(ident)
            targets.append(candidate)

        _append(source)
        for attr in ("kws", "ticker", "ws", "stream", "client"):
            _append(getattr(source, attr, None))
        return targets

    subscribed = False
    mode_applied = False

    for target in _iter_targets(obj):
        for name in ("subscribe_tokens", "subscribe", "subscribe_l1"):
            fn = getattr(target, name, None)
            if callable(fn):
                try:
                    fn(payload)
                    subscribed = True
                except Exception:
                    continue
                else:
                    break
        for attr in ("set_mode", "set_mode_full"):
            mode_fn = getattr(target, attr, None)
            if callable(mode_fn):
                mode_value_raw = mode or getattr(
                    target, "MODE_FULL", getattr(obj, "MODE_FULL", "full")
                )
                mode_value: Any = mode_value_raw
                if attr == "set_mode" and isinstance(mode_value_raw, str):
                    mode_value = mode_value_raw.lower()
                try:
                    if attr == "set_mode":
                        mode_fn(mode_value, payload)
                    else:
                        mode_fn(payload)
                    mode_applied = True
                except Exception:
                    continue
                else:
                    break
        hook = getattr(target, "on_reconnect", None)
        if callable(hook):
            try:
                hook(lambda t=target, p=payload: _subscribe_tokens(t, p, mode=mode))
            except Exception:
                log.debug("subscribe_tokens: reconnect hook failed", exc_info=True)

    if subscribed or mode_applied:
        return True

    broker = getattr(obj, "broker", None)
    if broker and _subscribe_tokens(broker, tokens, mode=mode):
        return True

    kite = getattr(obj, "kite", None)
    if kite and _subscribe_tokens(kite, tokens, mode=mode):
        return True

    return False


def ensure_token_subscribed(
    self: Any, token: int | str, *, mode: str | None = None
) -> bool:
    """Ensure ``token`` is subscribed via the underlying broker."""

    try:
        token_i = int(token)
    except Exception:
        return False

    if token_i <= 0:
        return False

    if _subscribe_tokens(self, [token_i], mode=mode):
        return True

    broker = getattr(self, "kite", None) or getattr(self, "broker", None)
    if broker and _subscribe_tokens(broker, [token_i], mode=mode):
        return True
    return False


def _have_quote(obj: Any, token: int) -> bool:
    for name in ("get_l1", "ltp", "get_quote", "quote"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                q = fn([token]) if name in ("get_quote", "quote") else fn(token)
                if q:
                    return True
            except Exception:
                continue
    broker = getattr(obj, "broker", None)
    if broker:
        return _have_quote(broker, token)
    return False


def ensure_atm_tokens(self: Any, underlying: str | None = None) -> None:
    """Resolve and subscribe current ATM option tokens.

    Re-resolves when drift from current strike exceeds ``ATM_ROLL_DRIFT_PTS``
    or at scheduled rolls (daily 00:05, Tuesday 09:16 and 13:45). Calls are
    throttled per instance via ``ATM_RESOLVE_COOLDOWN_S`` (min 45s).
    """

    try:
        cooldown = max(45, int(os.getenv("ATM_RESOLVE_COOLDOWN_S", "45")))
    except Exception:
        cooldown = 45
    try:
        drift_pts = float(os.getenv("ATM_ROLL_DRIFT_PTS", "50"))
    except Exception:
        drift_pts = 50.0

    now = time.time()
    under = str(underlying or getattr(settings.instruments, "trade_symbol", "NIFTY"))
    spot = self.get_last_price(getattr(settings.instruments, "spot_symbol", under))
    current_strike = float(getattr(self, "current_atm_strike", 0) or 0)
    drift = abs(float(spot or 0) - current_strike)

    roll = False
    try:
        now_dt = dt.datetime.now(TZ)
        marks: set[str] = getattr(self, "_atm_roll_marks", set())
        checks: list[tuple[str, bool]] = []
        checks.append((f"day-{now_dt.date()}", now_dt.time() >= dt.time(0, 5)))
        if now_dt.weekday() == 1:
            checks.append((f"tue-am-{now_dt.date()}", now_dt.time() >= dt.time(9, 16)))
            checks.append((f"tue-pm-{now_dt.date()}", now_dt.time() >= dt.time(13, 45)))
        for key, cond in checks:
            if cond and key not in marks:
                roll = True
                marks.add(key)
                break
        self._atm_roll_marks = marks
    except Exception:
        roll = False

    next_ts = float(getattr(self, "_atm_next_resolve_ts", 0.0))
    existing = getattr(self, "atm_tokens", None)
    have_tokens = isinstance(existing, (list, tuple)) and all(t is not None for t in existing)
    if now < next_ts and have_tokens:
        return
    self._atm_next_resolve_ts = now + cooldown
    if have_tokens and drift < drift_pts and not roll:
        return

    if drift >= drift_pts or roll:
        log.info("atm_roll: reason=%s", "drift" if drift >= drift_pts else "expiry")

    broker = getattr(self, "kite", None) or getattr(self, "broker", None)
    items = _refresh_instruments_nfo(broker)
    if not items:
        return
    today = dt.date.today()
    expiry = _pick_expiry(items, under, today)
    if not expiry:
        return
    if spot is None:
        spot = self.get_last_price(getattr(settings.instruments, "spot_symbol", under))
    if spot is None:
        return
    try:
        step = int(getattr(settings.instruments, "strike_step", 50))
    except Exception:
        step = 50
    base = _nearest_strike(float(spot), step)
    ce = pe = None
    strike = base
    for widen in range(0, 7):
        for sign in (0, 1, -1) if widen else (0,):
            s = base + sign * widen * step
            ce = _match_token(items, under, expiry, s, "CE")
            pe = _match_token(items, under, expiry, s, "PE")
            if ce and pe:
                strike = s
                break
        if ce and pe:
            break
    if not (ce and pe):
        return
    ce_token, pe_token = int(ce), int(pe)
    tokens = [ce_token, pe_token]
    self.atm_tokens = tuple(tokens)
    self.current_atm_strike = strike
    self.current_atm_expiry = expiry
    self._atm_resolve_date = today
    log.info(
        "ATM tokens resolved: expiry=%s, strike=%d, ce=%d, pe=%d",
        expiry.isoformat(),
        int(strike),
        ce_token,
        pe_token,
    )
    subscribe_full = getattr(self, "_subscribe_tokens_full", None)
    logger = getattr(self, "log", log)
    if callable(subscribe_full):
        subscribe_full(tokens)
    else:
        _subscribe_tokens(self, tokens)
    if hasattr(logger, "info"):
        try:
            logger.info(
                "data.tokens_resolved",
                extra={"ce": ce_token, "pe": pe_token, "mode": "FULL"},
            )
        except Exception:
            logger.info(
                "data.tokens_resolved ce=%s pe=%s mode=FULL", ce_token, pe_token
            )
    for _ in range(2):
        missing = [t for t in tokens if not _have_quote(self, t)]
        if not missing:
            break
        if callable(subscribe_full):
            subscribe_full(missing)
        else:
            _subscribe_tokens(self, missing)
        time.sleep(0.1)


def auto_resubscribe_atm(self: Any) -> None:
    """Ensure current ATM tokens remain subscribed and have quotes.

    Respects ``AUTO_ATM_RESUB_INTERVAL_S`` (default ``30``) to avoid
    spamming the broker with subscription requests. Throttled per
    instance via ``self._atm_next_check_ts``.
    """

    try:
        interval = int(os.getenv("AUTO_ATM_RESUB_INTERVAL_S", "30"))
    except Exception:
        interval = 30
    now = time.time()
    next_check = float(getattr(self, "_atm_next_check_ts", 0.0))
    if now < next_check:
        return
    self._atm_next_check_ts = now + interval
    tokens = list(getattr(self, "atm_tokens", []) or [])
    if not tokens:
        return
    missing = [t for t in tokens if not _have_quote(self, t)]
    if not missing:
        return
    subscribed = False
    subscribe_full = getattr(self, "_subscribe_tokens_full", None)
    if callable(subscribe_full):
        try:
            subscribe_full(missing)
            subscribed = True
        except Exception:
            subscribed = False
    if not subscribed:
        subscribed = _subscribe_tokens(self, missing)
    if subscribed:
        log.info("auto_resubscribe_atm: resubscribed tokens=%s", missing)
    else:
        log.warning("auto_resubscribe_atm: could not resubscribe tokens=%s", missing)


# Bind helpers to DataSource for easy access
DataSource.auto_resubscribe_atm = auto_resubscribe_atm  # type: ignore[attr-defined]
DataSource.ensure_atm_tokens = ensure_atm_tokens  # type: ignore[attr-defined]
DataSource.ensure_token_subscribed = ensure_token_subscribed  # type: ignore[attr-defined]
