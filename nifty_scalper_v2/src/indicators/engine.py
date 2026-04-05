"""IndicatorEngine: subscribes to candle events, maintains rolling indicator snapshots."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import TYPE_CHECKING

import numpy as np

from .technical import (
    calc_adx, calc_atr, calc_bollinger, calc_ema, calc_macd, calc_rsi, calc_vwap, calc_cpr
)
from .greeks import calc_delta, calc_gamma, calc_iv, calc_theta, calc_vega
from ..config.constants import (
    WARMUP_CANDLES, TRADING_DAYS_PER_YEAR, RISK_FREE_RATE,
    EMA_FAST, EMA_MID, EMA_SLOW, RSI_PERIOD, ATR_PERIOD,
    BB_PERIOD, BB_STD, ADX_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, MAX_INDICATOR_HISTORY
)

if TYPE_CHECKING:
    from ..data.candle_engine import OHLCV
    from ..data.tick_hub import EventBus


@dataclass(slots=True)
class IndicatorSnapshot:
    symbol: str
    timestamp: datetime
    ltp: float
    vwap: float
    ema_9: float | None
    ema_21: float | None
    ema_50: float | None
    adx: float | None
    adx_plus_di: float | None
    adx_minus_di: float | None
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    atr_14: float | None
    bb_upper: float | None
    bb_lower: float | None
    bb_mid: float | None
    bb_width: float | None
    bb_squeeze: bool | None
    volume_sma_20: float | None
    volume_ratio: float | None
    iv: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    day_high: float | None
    day_low: float | None
    opening_range_high: float | None
    opening_range_low: float | None
    cpr_top: float | None
    cpr_pivot: float | None
    cpr_bottom: float | None
    vix: float | None
    max_pain_strike: float | None
    is_valid: bool = True


class _SymbolBuffer:
    """Per-symbol rolling OHLCV buffer."""

    def __init__(self, maxlen: int) -> None:
        self._candles: deque["OHLCV"] = deque(maxlen=maxlen)
        self._bb_widths: deque[float] = deque(maxlen=100)
        self.day_high: float | None = None
        self.day_low: float | None = None
        self.opening_range_high: float | None = None
        self.opening_range_low: float | None = None
        self.cpr_top: float | None = None
        self.cpr_pivot: float | None = None
        self.cpr_bottom: float | None = None
        self.vix: float | None = None
        self.max_pain_strike: float | None = None

    def add(self, candle: "OHLCV") -> None:
        self._candles.append(candle)
        # Day H/L
        if self.day_high is None or candle.high > self.day_high:
            self.day_high = candle.high
        if self.day_low is None or candle.low < self.day_low:
            self.day_low = candle.low

    @property
    def count(self) -> int:
        return len(self._candles)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        candles = list(self._candles)
        opens = np.array([c.open for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles], dtype=float)
        return opens, highs, lows, closes, volumes


class IndicatorEngine:
    """Computes and caches IndicatorSnapshot per symbol on each completed candle."""

    def __init__(self, event_bus: "EventBus", warmup_candles: int = WARMUP_CANDLES) -> None:
        self._warmup = warmup_candles
        self._buffers: dict[str, _SymbolBuffer] = {}
        self._snapshots: dict[str, IndicatorSnapshot] = {}
        self._spot_price: dict[str, float] = {}  # underlying spot for Greeks
        event_bus.subscribe("candle_complete", self._on_candle)

    def _on_candle(self, candle: "OHLCV") -> None:
        sym = candle.symbol
        if sym not in self._buffers:
            self._buffers[sym] = _SymbolBuffer(MAX_INDICATOR_HISTORY)
        buf = self._buffers[sym]
        buf.add(candle)
        self._compute(sym, candle.close)

    def set_spot_price(self, underlying: str, price: float) -> None:
        self._spot_price[underlying] = price

    def set_vix(self, symbol: str, vix: float) -> None:
        if symbol in self._buffers:
            self._buffers[symbol].vix = vix

    def set_max_pain(self, symbol: str, strike: float) -> None:
        if symbol in self._buffers:
            self._buffers[symbol].max_pain_strike = strike

    def set_opening_range(self, symbol: str, high: float, low: float) -> None:
        if symbol in self._buffers:
            self._buffers[symbol].opening_range_high = high
            self._buffers[symbol].opening_range_low = low

    def set_cpr(self, symbol: str, top: float, pivot: float, bottom: float) -> None:
        if symbol in self._buffers:
            buf = self._buffers[symbol]
            buf.cpr_top, buf.cpr_pivot, buf.cpr_bottom = top, pivot, bottom

    def _compute(self, sym: str, ltp: float) -> None:
        buf = self._buffers[sym]
        valid = buf.count >= self._warmup

        if buf.count < 2:
            self._snapshots[sym] = IndicatorSnapshot(
                symbol=sym, timestamp=datetime.now(timezone.utc),
                ltp=ltp, vwap=ltp,
                ema_9=None, ema_21=None, ema_50=None,
                adx=None, adx_plus_di=None, adx_minus_di=None,
                rsi_14=None, macd=None, macd_signal=None, macd_hist=None,
                atr_14=None, bb_upper=None, bb_lower=None, bb_mid=None,
                bb_width=None, bb_squeeze=None,
                volume_sma_20=None, volume_ratio=None,
                iv=None, delta=None, gamma=None, theta=None, vega=None,
                day_high=buf.day_high, day_low=buf.day_low,
                opening_range_high=buf.opening_range_high,
                opening_range_low=buf.opening_range_low,
                cpr_top=buf.cpr_top, cpr_pivot=buf.cpr_pivot, cpr_bottom=buf.cpr_bottom,
                vix=buf.vix, max_pain_strike=buf.max_pain_strike,
                is_valid=False,
            )
            return

        _, highs, lows, closes, volumes = buf.arrays()

        vwap = calc_vwap(highs, lows, closes, volumes) or ltp
        ema9 = calc_ema(closes, EMA_FAST)
        ema21 = calc_ema(closes, EMA_MID)
        ema50 = calc_ema(closes, EMA_SLOW)
        rsi = calc_rsi(closes, RSI_PERIOD)
        atr = calc_atr(highs, lows, closes, ATR_PERIOD)
        bb = calc_bollinger(closes, BB_PERIOD, BB_STD)
        adx_res = calc_adx(highs, lows, closes, ADX_PERIOD)
        macd_res = calc_macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

        bb_upper = bb_lower = bb_mid = bb_width = None
        bb_squeeze = None
        if bb:
            bb_upper, bb_mid, bb_lower = bb
            if bb_mid and bb_mid > 0:
                bb_width = (bb_upper - bb_lower) / bb_mid
                buf._bb_widths.append(bb_width)
                if len(buf._bb_widths) >= 20:
                    pct20 = float(np.percentile(list(buf._bb_widths), 20))
                    bb_squeeze = bb_width < pct20

        adx = adx_pdi = adx_ndi = None
        if adx_res:
            adx, adx_pdi, adx_ndi = adx_res

        macd_line = macd_sig = macd_hist_val = None
        if macd_res:
            macd_line, macd_sig, macd_hist_val = macd_res

        vol_sma = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else None
        vol_ratio = (volumes[-1] / vol_sma) if vol_sma and vol_sma > 0 else None

        self._snapshots[sym] = IndicatorSnapshot(
            symbol=sym, timestamp=datetime.now(timezone.utc),
            ltp=ltp, vwap=vwap,
            ema_9=ema9, ema_21=ema21, ema_50=ema50,
            adx=adx, adx_plus_di=adx_pdi, adx_minus_di=adx_ndi,
            rsi_14=rsi, macd=macd_line, macd_signal=macd_sig, macd_hist=macd_hist_val,
            atr_14=atr, bb_upper=bb_upper, bb_lower=bb_lower, bb_mid=bb_mid,
            bb_width=bb_width, bb_squeeze=bb_squeeze,
            volume_sma_20=vol_sma, volume_ratio=vol_ratio,
            iv=None, delta=None, gamma=None, theta=None, vega=None,
            day_high=buf.day_high, day_low=buf.day_low,
            opening_range_high=buf.opening_range_high,
            opening_range_low=buf.opening_range_low,
            cpr_top=buf.cpr_top, cpr_pivot=buf.cpr_pivot, cpr_bottom=buf.cpr_bottom,
            vix=buf.vix, max_pain_strike=buf.max_pain_strike,
            is_valid=valid,
        )

    def get_snapshot(self, symbol: str) -> IndicatorSnapshot | None:
        return self._snapshots.get(symbol)

    def get_all_snapshots(self) -> dict[str, IndicatorSnapshot]:
        return dict(self._snapshots)
