"""Data-source integrity helpers for strategy execution.

Runtime role:
- Low-level broker data adapter for quote/LTP/historical fetches.
- Consumed by MarketDataManager hydration.
- Must not select contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import time

import pandas as pd


class DataIntegrityError(ValueError):
    """Raised when required market data is missing or inconsistent."""


@dataclass(slots=True)
class CandleFrame:
    """OHLCV wrapper with integrity checks. Args: df. Returns: CandleFrame. Raises: DataIntegrityError."""

    dataframe: pd.DataFrame

    def validate(self) -> pd.DataFrame:
        """Validate the underlying dataframe. Args: none. Returns: DataFrame. Raises: DataIntegrityError."""
        required = {"timestamp", "open", "high", "low", "close"}
        missing = [name for name in required if name not in self.dataframe.columns]
        if missing:
            raise DataIntegrityError(f"Missing OHLC fields: {missing}")
        if self.dataframe.empty:
            raise DataIntegrityError("Empty OHLC dataframe")
        if self.dataframe["close"].isna().any():
            raise DataIntegrityError("Close column contains nulls")
        ts = pd.to_datetime(self.dataframe["timestamp"], utc=True, errors="coerce")
        if ts.isna().any():
            raise DataIntegrityError("Invalid timestamps in OHLC dataframe")
        if not ts.is_monotonic_increasing:
            raise DataIntegrityError("OHLC timestamps are not aligned/monotonic")
        return self.dataframe


def ensure_ltp(ltp: float | None) -> float:
    """Validate LTP value. Args: ltp. Returns: float. Raises: DataIntegrityError."""
    if ltp is None:
        raise DataIntegrityError("Missing LTP")
    return float(ltp)


def ensure_indicator_values(indicators: dict[str, float | int | None]) -> None:
    """Validate indicator map has no missing/NaN values. Args: indicators. Returns: None. Raises: DataIntegrityError."""
    if not indicators:
        raise DataIntegrityError("Missing indicator values")
    for name, raw_value in indicators.items():
        if raw_value is None:
            raise DataIntegrityError(f"Indicator {name} is missing")
        value = float(raw_value)
        if pd.isna(value):
            raise DataIntegrityError(f"Indicator {name} is NaN")


def is_symbol_valid(dataframe: pd.DataFrame, *, min_required_bars: int) -> bool:
    """Validate candle-frame readiness. Args: dataframe/min_required_bars. Returns: bool. Raises: None."""
    if dataframe is None or dataframe.empty:
        return False
    if len(dataframe) < int(min_required_bars):
        return False
    if dataframe.index.has_duplicates:
        return False
    if not dataframe.index.is_monotonic_increasing:
        return False
    if dataframe.isnull().values.any():
        return False
    return True


@dataclass(slots=True)
class HistoricalLiveOHLCProvider:
    """Compose historical + live OHLC safely. Args: callbacks. Returns: provider. Raises: DataIntegrityError."""

    fetch_historical: Callable[[str, str], pd.DataFrame]
    get_current_live_candle: Callable[[str], Mapping[str, Any] | pd.Series | None]

    def get_clean_ohlc(self, symbol: str, timeframe: str = "minute") -> pd.DataFrame:
        """Return cleaned OHLC using historical source-of-truth. Args: symbol/timeframe. Returns: DataFrame. Raises: DataIntegrityError."""
        df = self.fetch_historical(symbol, timeframe)
        if df is None or len(df) == 0:
            raise DataIntegrityError(f"No historical bars for {symbol}")
        cleaned = df.copy()
        live_candle = self.get_current_live_candle(symbol)
        if live_candle is None:
            return cleaned

        live_row: pd.Series
        if isinstance(live_candle, pd.Series):
            live_row = live_candle.copy()
        elif isinstance(live_candle, dict):
            live_row = pd.Series(live_candle)
        else:
            raise DataIntegrityError(f"Invalid live candle payload for {symbol}")

        for col in cleaned.columns:
            if col in live_row:
                cleaned.at[cleaned.index[-1], col] = live_row[col]

        # Keep historical index as the source of truth for alignment.
        if "timestamp" in cleaned.columns:
            cleaned["timestamp"] = pd.to_datetime(
                cleaned["timestamp"], utc=True, errors="coerce"
            )
            if cleaned["timestamp"].isna().any():
                raise DataIntegrityError("Invalid timestamps after live candle merge")

        return cleaned


class MarketDataSource:
    """Read LTP/OHLC with WS-first, polling fallback hierarchy."""

    def __init__(self, kite: Any, market_state: Any) -> None:
        """Initialize source with kite + market state. Args: kite/market_state. Returns: none. Raises: none."""

        self._kite = kite
        self._state = market_state

    def get_ltp_poll(self, tokens: Sequence[int]) -> dict[int, float]:
        """Fetch LTP by polling API. Args: tokens. Returns: token->price. Raises: DataIntegrityError."""

        normalized_tokens = [int(token) for token in tokens]
        if not normalized_tokens:
            return {}
        try:
            if hasattr(self._kite, "get_ltp_bulk"):
                payload = self._kite.get_ltp_bulk(normalized_tokens)
                return self._extract_token_ltps(payload, normalized_tokens)
            if hasattr(self._kite, "get_quote_bulk"):
                payload = self._kite.get_quote_bulk(normalized_tokens)
                return self._extract_token_ltps(payload, normalized_tokens)
            if hasattr(self._kite, "quote_any"):
                payload = self._kite.quote_any(normalized_tokens)
                return self._extract_token_ltps(payload, normalized_tokens)

            token_to_symbol = self._token_symbol_lookup(normalized_tokens)
            if token_to_symbol and hasattr(self._kite, "ltp"):
                query = [token_to_symbol[token] for token in normalized_tokens if token in token_to_symbol]
                if not query:
                    return {}
                payload = self._kite.ltp(query)
                return self._extract_token_ltps(payload, normalized_tokens, token_to_symbol=token_to_symbol)
        except Exception as e:
            raise DataIntegrityError(f"Polling LTP failed: {e}") from e
        return {}

    def _extract_token_ltps(
        self,
        payload: Any,
        tokens: Sequence[int],
        *,
        token_to_symbol: Mapping[int, str] | None = None,
    ) -> dict[int, float]:
        """Extract token-keyed LTPs from broker bulk payloads."""

        if not isinstance(payload, Mapping):
            return {}
        parsed: dict[int, float] = {}
        symbol_to_token = {str(symbol): int(token) for token, symbol in (token_to_symbol or {}).items()}
        wanted = {int(token) for token in tokens}
        for raw_key, raw_entry in payload.items():
            token: int | None = None
            if isinstance(raw_key, int) or (isinstance(raw_key, str) and raw_key.isdigit()):
                token = int(raw_key)
            elif str(raw_key) in symbol_to_token:
                token = symbol_to_token[str(raw_key)]
            if isinstance(raw_entry, (int, float)):
                if token is not None and token in wanted and float(raw_entry) > 0:
                    parsed[int(token)] = float(raw_entry)
                continue
            entry = raw_entry if isinstance(raw_entry, Mapping) else {}
            if token is None:
                entry_token = entry.get("instrument_token") or entry.get("token")
                try:
                    token = int(entry_token) if entry_token is not None else None
                except (TypeError, ValueError):
                    token = None
            if token is None or token not in wanted:
                continue
            ltp = entry.get("last_price") or entry.get("ltp")
            if ltp is None:
                continue
            parsed[int(token)] = float(ltp)
        return parsed

    def _token_symbol_lookup(self, tokens: Sequence[int]) -> dict[int, str]:
        """Return broker tradingsymbol lookup for raw kite.ltp fallback only."""

        wanted = {int(token) for token in tokens}
        lookup: dict[int, str] = {}
        for attr in ("token_to_symbol", "symbol_by_token", "_token_to_symbol", "_symbol_by_token"):
            mapping = getattr(self._kite, attr, None) or getattr(self._state, attr, None)
            if not isinstance(mapping, Mapping):
                continue
            for raw_token, raw_symbol in mapping.items():
                try:
                    token = int(raw_token)
                except (TypeError, ValueError):
                    continue
                if token in wanted and raw_symbol:
                    lookup[token] = str(raw_symbol)
        return lookup

    def get_ltp(self, tokens: Sequence[int], *, stale_after_seconds: int = 5) -> dict[int, float]:
        """Get token LTPs using WS first and polling fallback. Args: tokens/stale_after_seconds. Returns: token->price. Raises: none."""

        snapshot = self._state.get_snapshot()
        now = time.time()
        ws_fresh = snapshot.last_ws_update is not None and (now - snapshot.last_ws_update) <= float(stale_after_seconds)
        resolved = {int(token): float(snapshot.option_ltps[int(token)]) for token in tokens if int(token) in snapshot.option_ltps}
        if ws_fresh and len(resolved) == len(tokens):
            return resolved
        polled = self.get_ltp_poll(tokens)
        for token, price in polled.items():
            self._state.update_tick(token, price, source='poll')
        merged = dict(resolved)
        merged.update(polled)
        return merged

    def get_ohlc(
        self,
        token: int,
        interval: str,
        ttl: int = 60,
        *,
        min_required_bars: int = 30,
    ) -> list[dict[str, Any]]:
        """Get token OHLC with TTL cache. Args: token/interval/ttl/min_required_bars. Returns: rows. Raises: DataIntegrityError."""

        cached = self._state.get_ohlc_cache(token, interval, ttl=ttl)
        if cached is not None:
            return cached
        try:
            rows = self._kite.historical_data(
                int(token),
                datetime.now(timezone.utc).replace(hour=3, minute=45, second=0, microsecond=0),
                datetime.now(timezone.utc),
                interval,
            )
        except Exception as e:
            raise DataIntegrityError(f'OHLC fetch failed for token {token}: {e}') from e
        required = max(1, int(min_required_bars))
        if len(rows) < required:
            raise DataIntegrityError(
                f"Insufficient OHLC candles for token {token}: {len(rows)}/{required}"
            )
        normalized = [dict(row) for row in rows]
        self._state.set_ohlc_cache(int(token), interval, normalized)
        return normalized
