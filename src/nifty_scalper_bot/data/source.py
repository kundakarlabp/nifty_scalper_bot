"""Data-source integrity helpers for strategy execution."""

from __future__ import annotations

from dataclasses import dataclass

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
