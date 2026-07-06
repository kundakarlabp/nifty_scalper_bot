from __future__ import annotations

from typing import Any

import pandas as pd

from nifty_scalper_bot.utils.time_series_ist import as_ist


def apply(obj: Any) -> None:
    if getattr(obj, '_IST_PATCHED', False):
        return

    def validate(self: Any):
        required = {'timestamp', 'open', 'high', 'low', 'close'}
        missing = [name for name in required if name not in self.dataframe.columns]
        if missing:
            raise obj.DataIntegrityError(f'Missing OHLC fields: {missing}')
        if self.dataframe.empty:
            raise obj.DataIntegrityError('Empty OHLC dataframe')
        if self.dataframe['close'].isna().any():
            raise obj.DataIntegrityError('Close column contains nulls')
        ts = as_ist(self.dataframe['timestamp'])
        if ts.isna().any():
            raise obj.DataIntegrityError('Invalid timestamps in OHLC dataframe')
        if not ts.is_monotonic_increasing:
            raise obj.DataIntegrityError('OHLC timestamps are not aligned/monotonic')
        return self.dataframe

    def get_clean_ohlc(self: Any, symbol: str, timeframe: str = 'minute'):
        df = self.fetch_historical(symbol, timeframe)
        if df is None or len(df) == 0:
            raise obj.DataIntegrityError(f'No historical bars for {symbol}')
        cleaned = df.copy()
        live_candle = self.get_current_live_candle(symbol)
        if live_candle is None:
            return cleaned
        live_row = live_candle.copy() if isinstance(live_candle, pd.Series) else pd.Series(live_candle) if isinstance(live_candle, dict) else None
        if live_row is None:
            raise obj.DataIntegrityError(f'Invalid live candle payload for {symbol}')
        for col in cleaned.columns:
            if col in live_row:
                cleaned.at[cleaned.index[-1], col] = live_row[col]
        if 'timestamp' in cleaned.columns:
            cleaned['timestamp'] = as_ist(cleaned['timestamp'])
            if cleaned['timestamp'].isna().any():
                raise obj.DataIntegrityError('Invalid timestamps after live candle merge')
        return cleaned

    obj.CandleFrame.validate = validate
    obj.HistoricalLiveOHLCProvider.get_clean_ohlc = get_clean_ohlc
    obj._IST_PATCHED = True


__all__ = ['apply']
