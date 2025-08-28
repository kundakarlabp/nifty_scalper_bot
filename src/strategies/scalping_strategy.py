# Path: src/strategies/scalping_strategy.py
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple, Literal

import pandas as pd

from src.config import settings
from src.utils.atr_helper import compute_atr, latest_atr_value
from src.utils.indicators import (
    calculate_vwap,
    calculate_macd,
)

logger = logging.getLogger(__name__)

# --- 60s log throttle (avoid spam in deploy logs) ---
_LOG_EVERY = 60.0
_last_log_ts = {
    "drop_strict": 0.0,
    "drop_relaxed": 0.0,
    "auto_relax": 0.0,
    "generated": 0.0,
}


def _log_throttled(key: str, level: int, msg: str, *args) -> None:
    now = time.time()
    if now - _last_log_ts.get(key, 0.0) >= _LOG_EVERY:
        _last_log_ts[key] = now
        logger.log(level, msg, *args)


Side = Literal["BUY", "SELL"]
SignalOutput = Optional[Dict[str, Any]]


class EnhancedScalpingStrategy:
    """
    Regime-aware scalping strategy (minimal, runner-aligned).

    Inputs
    ------
    - df:           OHLCV dataframe (ascending index). In current wiring this is SPOT.
    - current_tick: optional dict from broker stream; may contain 'ltp', 'spot_ltp', 'option_ltp'
    - current_price: explicit option LTP (overrides tick if provided)
    - spot_df:      optional SPOT dataframe (if 'df' were option candles in the future)

    Outputs (runner/executor expect these keys)
    ------------------------------------------
    - action: "BUY" | "SELL"
    - option_type: "CE" | "PE"
    - strike: int (nearest 50-point ATM from spot)
    - entry_price: float
    - stop_loss: float
    - take_profit: float
    - rr: float
    - score, confidence, regime, reasons, diagnostics...
    """

    def __init__(
        self,
        *,
        ema_fast: int = settings.strategy.ema_fast,
        ema_slow: int = settings.strategy.ema_slow,
        rsi_period: int = settings.strategy.rsi_period,
        adx_period: int = 14,
        adx_trend_strength: int = 20,
        atr_period: int = settings.strategy.atr_period,
        min_bars_for_signal: int = settings.strategy.min_bars_for_signal,
        confidence_threshold: float = settings.strategy.confidence_threshold,
        min_signal_score: int = settings.strategy.min_signal_score,
        atr_sl_multiplier: float = settings.strategy.atr_sl_multiplier,
        atr_tp_multiplier: float = settings.strategy.atr_tp_multiplier,
    ) -> None:
        # Core lookbacks
        self.ema_fast = int(ema_fast)
        self.ema_slow = int(ema_slow)
        self.rsi_period = int(rsi_period)
        self.adx_period = int(adx_period)
        self.adx_trend_strength = int(adx_trend_strength)
        self.atr_period = int(atr_period)

        # Regime shaping (add to multipliers; can be negative)
        self.trend_tp_boost = float(getattr(settings.strategy, "trend_tp_boost", 0.6))
        self.trend_sl_relax = float(getattr(settings.strategy, "trend_sl_relax", 0.2))
        self.range_tp_tighten = float(getattr(settings.strategy, "range_tp_tighten", -0.4))
        self.range_sl_tighten = float(getattr(settings.strategy, "range_sl_tighten", -0.2))

        # Bars threshold for validity
        self.min_bars_for_signal = int(min_bars_for_signal)

        # Thresholds (normalize confidence scale)
        # Config is typically 0..100; internal scoring below returns ~0..8
        raw_conf = float(confidence_threshold)  # e.g., 55 (%)
        raw_conf_rel = float(
            getattr(settings.strategy, "confidence_threshold_relaxed", max(0.0, raw_conf - 20))
        )
        self.min_conf_strict = raw_conf / 10.0   # 55 -> 5.5 on a 0..10-ish scale
        self.min_conf_relaxed = raw_conf_rel / 10.0

        self.min_score_strict = int(min_signal_score)
        self.auto_relax_enabled = bool(getattr(settings.strategy, "auto_relax_enabled", True))
        self.min_score_relaxed = int(
            getattr(settings.strategy, "min_signal_score_relaxed", max(2, self.min_score_strict - 1))
        )

        # ATR & confidence shaping
        self.base_sl_mult = float(atr_sl_multiplier)
        self.base_tp_mult = float(atr_tp_multiplier)
        self.sl_conf_adj = float(getattr(settings.strategy, "sl_confidence_adj", 0.2))
        self.tp_conf_adj = float(getattr(settings.strategy, "tp_confidence_adj", 0.3))

        # Exportable debug snapshot
        self._last_debug: Dict[str, Any] = {"note": "no_evaluation_yet"}

    # ---------- tech utils ----------
    @staticmethod
    def _ema(s: pd.Series, period: int) -> pd.Series:
        return s.ewm(span=max(1, int(period)), adjust=False).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
        rs = (avg_gain / (avg_loss.replace(0.0, 1e-9))).fillna(0.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @staticmethod
    def _extract_adx_columns(spot_df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """Return (adx, di_plus, di_minus) from spot_df; tolerant to suffix (_{n}) naming."""
        if spot_df is None or spot_df.empty:
            return None, None, None
        adx_cols = sorted([c for c in spot_df.columns if c.startswith("adx_")])
        dip_cols = sorted([c for c in spot_df.columns if c.startswith("di_plus_")])
        dim_cols = sorted([c for c in spot_df.columns if c.startswith("di_minus_")])
        adx = spot_df[adx_cols[-1]] if adx_cols else spot_df.get("adx")
        di_plus = spot_df[dip_cols[-1]] if dip_cols else spot_df.get("di_plus")
        di_minus = spot_df[dim_cols[-1]] if dim_cols else spot_df.get("di_minus")
        return adx, di_plus, di_minus

    # ---------- thresholds ----------
    def _score_confidence(self, score: int) -> float:
        # compact 0..8-ish scale mapped to our thresholds above
        if score >= 8:
            return 8.0
        if score >= 6:
            return 6.0
        if score >= 4:
            return 2.5
        return 0.0

    def _passes(self, score: int, conf: float, *, strict: bool) -> bool:
        if strict:
            return (score >= self.min_score_strict) and (conf >= self.min_conf_strict)
        return (score >= self.min_score_relaxed) and (conf >= self.min_conf_relaxed)

    # ---------- debug export ----------
    def get_debug(self) -> Dict[str, Any]:
        return dict(self._last_debug)

    # ---------- main ----------
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_tick: Optional[Dict[str, Any]] = None,
        current_price: Optional[float] = None,
        spot_df: Optional[pd.DataFrame] = None,
    ) -> SignalOutput:

        dbg: Dict[str, Any] = {"reason_block": None}
        try:
            if df is None or df.empty or len(df) < self.min_bars_for_signal:
                dbg["reason_block"] = "insufficient_bars"
                return None
            if spot_df is None:
                spot_df = df

            spot_last = float(spot_df["close"].iloc[-1])
            if current_price is None:
                current_price = float(current_tick.get("ltp", spot_last)) if current_tick else spot_last
            if current_price is None or current_price <= 0:
                dbg["reason_block"] = "invalid_price"
                return None

            ema21 = self._ema(df["close"], 21)
            ema50 = self._ema(df["close"], 50)
            vwap = calculate_vwap(spot_df)
            macd_line, macd_signal, macd_hist = calculate_macd(df["close"])
            rsi = self._rsi(df["close"], 14)
            atr_series = compute_atr(df, period=14)
            atr_val = latest_atr_value(atr_series, default=0.0)

            if vwap is None or len(vwap) == 0 or atr_val <= 0:
                dbg["reason_block"] = "indicators_missing"
                return None

            price = float(spot_last)
            ema21_val, ema50_val = float(ema21.iloc[-1]), float(ema50.iloc[-1])
            ema21_slope = float(ema21.iloc[-1] - ema21.iloc[-2])
            macd_val = float(macd_line.iloc[-1])
            rsi_val = float(rsi.iloc[-1])
            rsi_rising = rsi_val > float(rsi.iloc[-2])

            swing_high = df["high"].rolling(window=20, min_periods=2).max().iloc[-2]
            breakout_dist = abs(price - swing_high) / price * 100.0

            long_ok = (
                price > float(vwap.iloc[-1])
                and ema21_val > ema50_val
                and ema21_slope > 0
                and macd_val > 0
                and rsi_val >= 48 and rsi_rising
                and breakout_dist >= 0.15
            )
            short_ok = (
                price < float(vwap.iloc[-1])
                and ema21_val < ema50_val
                and ema21_slope < 0
                and macd_val < 0
                and rsi_val <= 52 and not rsi_rising
                and breakout_dist >= 0.15
            )

            regime = None
            side: Optional[Side] = None
            if long_ok or short_ok:
                regime = "trend"
                side = "BUY" if long_ok else "SELL"
            else:
                dbg["reason_block"] = "trend_gates_failed"
                self._last_debug = dbg
                return None

            atr_pct = atr_val / price
            if not (0.0030 <= atr_pct <= 0.0150):
                dbg["reason_block"] = "atr_pct_out_of_range"
                self._last_debug = dbg
                return None

            score = 10
            reasons = ["trend_playbook"]
            entry_price = float(current_price)
            sl_dist = max(0.8 * atr_val, 0.8 * atr_val)
            if side == "BUY":
                stop_loss = entry_price - sl_dist
                tp1 = entry_price + 1.1 * sl_dist
                tp2 = entry_price + 1.8 * sl_dist
                option_type = "CE"
            else:
                stop_loss = entry_price + sl_dist
                tp1 = entry_price - 1.1 * sl_dist
                tp2 = entry_price - 1.8 * sl_dist
                option_type = "PE"

            risk = abs(entry_price - stop_loss)
            rr = (abs(tp2 - entry_price) / risk) if risk > 0 else 0.0

            from src.utils.strike_selector import select_strike, StrikeInfo

            strike_info: Optional[StrikeInfo] = select_strike(price, score)
            strike = int(strike_info.strike) if strike_info else int(round(price / 50.0) * 50)

            signal: Dict[str, Any] = {
                "action": side,
                "option_type": option_type,
                "strike": strike,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": tp2,
                "tp1": tp1,
                "tp2": tp2,
                "trail_atr_mult": 0.8,
                "time_stop_min": 12,
                "rr": round(rr, 2),
                "regime": regime,
                "score": score,
                "reasons": reasons,
                "side": side,
                "confidence": 1.0,
                "target": tp2,
            }

            self._last_debug = {
                "score": score,
                "regime": regime,
                "rr": rr,
                "reason_block": None,
                "atr_pct": atr_pct,
            }
            return signal

        except Exception as e:
            dbg["reason_block"] = f"exception:{e.__class__.__name__}"
            self._last_debug = dbg
            logger.debug("generate_signal exception: %s", e, exc_info=True)
            return None
