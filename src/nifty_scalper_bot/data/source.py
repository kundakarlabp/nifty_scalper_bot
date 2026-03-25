"""Data-source integrity helpers for strategy execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pandas as pd


class DataIntegrityError(ValueError):
    """Raised when required market data is missing or inconsistent."""


@dataclass(slots=True)
class MarketDataValidator:
    """Central OHLC validator. Args: min_candles. Returns: MarketDataValidator. Raises: None."""

    min_candles: int = 1

    def validate_ohlc(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate OHLCV frame. Args: df/symbol. Returns: validated frame. Raises: DataIntegrityError."""
        if df is None or df.empty:
            raise DataIntegrityError(f"{symbol}: empty dataframe")

        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise DataIntegrityError(f"{symbol}: missing columns {missing}")

        if len(df) < int(self.min_candles):
            raise DataIntegrityError(
                f"{symbol}: insufficient candles {len(df)}<{int(self.min_candles)}"
            )

        validated = df
        if validated[required].isnull().any().any():
            validated = validated.ffill()
            validated = validated.dropna(subset=required)

        if validated.empty:
            raise DataIntegrityError(f"{symbol}: no rows after null cleanup")

        if "timestamp" in validated.columns:
            ts = pd.to_datetime(validated["timestamp"], utc=True, errors="coerce")
            if ts.isna().any():
                raise DataIntegrityError(f"{symbol}: invalid timestamps")
            if not ts.is_monotonic_increasing:
                raise DataIntegrityError(f"{symbol}: non-monotonic timestamps")
        elif isinstance(validated.index, pd.DatetimeIndex):
            if not validated.index.is_monotonic_increasing:
                raise DataIntegrityError(f"{symbol}: non-monotonic datetime index")

        highs_ok = validated["high"] >= validated[["open", "close"]].max(axis=1)
        if not bool(highs_ok.all()):
            raise DataIntegrityError(f"{symbol}: invalid high values")

        lows_ok = validated["low"] <= validated[["open", "close"]].min(axis=1)
        if not bool(lows_ok.all()):
            raise DataIntegrityError(f"{symbol}: invalid low values")

        return validated


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
