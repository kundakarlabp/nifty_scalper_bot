"""Market regime classifier based on ADX/ATR/VWAP composites."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Deque, Mapping, Sequence

import pandas as pd


class RegimeKind(Enum):
    """Enumerated market regimes recognised by :class:`RegimeGate`."""

    TREND = "trend"
    RANGE = "range"
    VOLATILE = "volatile"
    QUIET = "quiet"


@dataclass(slots=True)
class RegimeMetrics:
    """Indicator snapshot backing a :class:`RegimeSnapshot`."""

    adx: float
    atr: float
    atr_pct: float
    atr_smooth_pct: float
    vwap: float
    vwap_distance_pct: float


@dataclass(slots=True)
class RegimeSnapshot:
    """Classification state produced by :class:`RegimeGate`."""

    timestamp: datetime
    regime: RegimeKind
    metrics: RegimeMetrics
    bias: str


@dataclass(slots=True)
class RegimeGateConfig:
    """Runtime tuning parameters for :class:`RegimeGate`."""

    adx_period: int = 14
    adx_min_trend: float = 18.0
    atr_period: int = 14
    atr_smooth_period: int = 30
    atr_range_multiple: float = 1.0
    vwap_lookback: int = 30
    vwap_max_trend_distance_pct: float = 0.0025
    quiet_threshold_pct: float = 0.0007
    volatile_threshold_pct: float = 0.0025
    warmup_bars: int = 30
    session_open: time | None = time(9, 15)
    session_close: time | None = time(15, 30)
    lunch_start: time | None = time(12, 15)
    lunch_end: time | None = time(13, 15)
    open_buffer_minutes: int = 5
    preclose_buffer_minutes: int = 10

    def validate(self) -> None:
        if self.adx_period <= 0:
            msg = "adx_period must be positive"
            raise ValueError(msg)
        if self.atr_period <= 0:
            msg = "atr_period must be positive"
            raise ValueError(msg)
        if self.atr_smooth_period < self.atr_period:
            msg = "atr_smooth_period must be >= atr_period"
            raise ValueError(msg)
        if self.vwap_lookback <= 0:
            msg = "vwap_lookback must be positive"
            raise ValueError(msg)
        if self.warmup_bars <= 0:
            msg = "warmup_bars must be positive"
            raise ValueError(msg)
        if self.atr_range_multiple <= 0:
            msg = "atr_range_multiple must be positive"
            raise ValueError(msg)
        if self.quiet_threshold_pct < 0 or self.volatile_threshold_pct < 0:
            msg = "volatility thresholds must be non-negative"
            raise ValueError(msg)
        if (
            self.lunch_start is not None
            and self.lunch_end is not None
            and self.lunch_end <= self.lunch_start
        ):
            msg = "lunch_end must be after lunch_start"
            raise ValueError(msg)


class RegimeGate:
    """Classify the active market regime and gate signals accordingly."""

    def __init__(self, config: RegimeGateConfig | None = None) -> None:
        self._config: RegimeGateConfig = config or RegimeGateConfig()
        self._config.validate()
        maxlen = int(
            max(self._config.atr_smooth_period, self._config.vwap_lookback) * 4
        )
        self._bars: Deque[dict[str, float | datetime]] = deque(maxlen=maxlen)
        self._last_snapshot: RegimeSnapshot | None = None

    @property
    def snapshot(self) -> RegimeSnapshot | None:
        """Return the latest computed :class:`RegimeSnapshot`."""

        return self._last_snapshot

    def observe(
        self, timestamp: datetime, bar: Mapping[str, float]
    ) -> RegimeSnapshot | None:
        """Ingest a completed minute bar and return the updated regime."""

        open_price = float(bar.get("open", 0.0) or bar.get("o", 0.0) or 0.0)
        high_price = float(bar.get("high", 0.0) or bar.get("h", 0.0) or 0.0)
        low_price = float(bar.get("low", 0.0) or bar.get("l", 0.0) or 0.0)
        close_price = float(bar.get("close", 0.0) or bar.get("c", 0.0) or 0.0)
        volume_value = float(bar.get("volume", 0.0) or bar.get("v", 0.0) or 0.0)
        record: dict[str, float | datetime] = {
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume_value,
        }
        if close_price <= 0:
            self._last_snapshot = None
            return None
        self._bars.append(record)
        if len(self._bars) < self._config.warmup_bars:
            self._last_snapshot = None
            return None
        frame = pd.DataFrame(self._bars)
        metrics = self._compute_metrics(frame)
        regime = self._classify(metrics)
        bias = self._derive_bias(regime)
        snapshot = RegimeSnapshot(
            timestamp=timestamp,
            regime=regime,
            metrics=metrics,
            bias=bias,
        )
        self._last_snapshot = snapshot
        return snapshot

    def allow_signal(self, timestamp: datetime, *, bias: str | None = None) -> bool:
        """Return ``True`` if a signal is allowed to trigger at ``timestamp``."""

        if self._is_in_no_trade_window(timestamp):
            return False
        snapshot = self._last_snapshot
        if snapshot is None:
            return False
        if bias is None:
            return snapshot.regime is not RegimeKind.QUIET
        bias_lower = bias.lower()
        if bias_lower == "trend":
            return snapshot.regime is RegimeKind.TREND
        if bias_lower in {"mean_revert", "range"}:
            return snapshot.regime in {RegimeKind.RANGE, RegimeKind.QUIET}
        return snapshot.regime is not RegimeKind.QUIET

    # ------------------------------------------------------------------
    # Internal helpers
    def _compute_metrics(self, frame: pd.DataFrame) -> RegimeMetrics:
        highs = frame["high"]
        lows = frame["low"]
        closes = frame["close"]
        adx = float(_compute_adx(highs, lows, closes, self._config.adx_period))
        atr_series = _compute_atr(highs, lows, closes, self._config.atr_period)
        atr = float(atr_series.iloc[-1])
        close = float(closes.iloc[-1])
        atr_pct = atr / close if close else 0.0
        smooth_period = min(self._config.atr_smooth_period, len(atr_series))
        atr_smooth_pct = 0.0
        if smooth_period > 0:
            atr_smooth_pct = float(
                (atr_series.tail(smooth_period) / closes.tail(smooth_period)).mean()
            )
        vwap = float(_compute_vwap(frame.tail(self._config.vwap_lookback)))
        vwap_distance_pct = (close - vwap) / vwap if vwap else 0.0
        return RegimeMetrics(
            adx=adx,
            atr=atr,
            atr_pct=atr_pct,
            atr_smooth_pct=atr_smooth_pct,
            vwap=vwap,
            vwap_distance_pct=vwap_distance_pct,
        )

    def _classify(self, metrics: RegimeMetrics) -> RegimeKind:
        cfg = self._config
        if (
            metrics.adx >= cfg.adx_min_trend
            and abs(metrics.vwap_distance_pct) <= cfg.vwap_max_trend_distance_pct
        ):
            return RegimeKind.TREND
        baseline = metrics.atr_smooth_pct or metrics.atr_pct
        if not baseline:
            baseline = metrics.atr_pct
        upper = baseline * (1.0 + cfg.atr_range_multiple)
        lower = baseline / (1.0 + cfg.atr_range_multiple)
        if metrics.atr_pct >= max(cfg.volatile_threshold_pct, upper):
            return RegimeKind.VOLATILE
        quiet_threshold = (
            cfg.quiet_threshold_pct if cfg.quiet_threshold_pct > 0 else lower
        )
        threshold = min(quiet_threshold, lower)
        if metrics.atr_pct <= threshold:
            return RegimeKind.QUIET
        return RegimeKind.RANGE

    def _derive_bias(self, regime: RegimeKind) -> str:
        if regime is RegimeKind.TREND:
            return "trend"
        if regime is RegimeKind.RANGE:
            return "mean_revert"
        if regime is RegimeKind.VOLATILE:
            return "trend"
        return "flat"

    def _is_in_no_trade_window(self, timestamp: datetime) -> bool:
        cfg = self._config
        if cfg.session_open is None or cfg.session_close is None:
            return False
        session_start = datetime.combine(timestamp.date(), cfg.session_open)
        session_end = datetime.combine(timestamp.date(), cfg.session_close)
        if (
            cfg.open_buffer_minutes > 0
            and session_start
            <= timestamp
            < session_start + timedelta(minutes=cfg.open_buffer_minutes)
        ):
            return True
        if (
            cfg.preclose_buffer_minutes > 0
            and session_end - timedelta(minutes=cfg.preclose_buffer_minutes)
            <= timestamp
            < session_end
        ):
            return True
        if cfg.lunch_start is not None and cfg.lunch_end is not None:
            lunch_start = datetime.combine(timestamp.date(), cfg.lunch_start)
            lunch_end = datetime.combine(timestamp.date(), cfg.lunch_end)
            if lunch_start <= timestamp < lunch_end:
                return True
        return False


def _compute_atr(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int
) -> pd.Series:
    series = pd.Series(closes)
    high = pd.Series(highs)
    low = pd.Series(lows)
    prev_close = series.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    atr = atr.bfill()
    return atr


def _compute_adx(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int
) -> float:
    high = pd.Series(highs)
    low = pd.Series(lows)
    close = pd.Series(closes)
    plus_dm = (high.diff() > low.shift(1) - low).astype(float) * (high.diff().abs())
    minus_dm = (low.shift(1) - low).where(
        low.shift(1) - low > high - high.shift(1), 0.0
    )
    plus_dm = plus_dm.fillna(0.0)
    minus_dm = minus_dm.fillna(0.0)
    atr = _compute_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, math.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    if adx.empty:
        return 0.0
    return float(adx.iloc[-1]) if math.isfinite(float(adx.iloc[-1])) else 0.0


def _compute_vwap(frame: pd.DataFrame) -> float:
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = frame["volume"].replace(0, 1.0)
    cumulative_volume = volume.cumsum()
    cumulative_value = (typical_price * volume).cumsum()
    if cumulative_volume.iloc[-1] <= 0:
        return float(frame["close"].iloc[-1])
    return float(cumulative_value.iloc[-1] / cumulative_volume.iloc[-1])


__all__ = [
    "RegimeGate",
    "RegimeGateConfig",
    "RegimeKind",
    "RegimeMetrics",
    "RegimeSnapshot",
]
