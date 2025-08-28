# Path: src/strategies/scalping_strategy.py
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple, Literal

import pandas as pd

from src.config import settings
from src.utils.atr_helper import compute_atr, atr_sl_tp_points, latest_atr_value
from src.utils.indicators import (
    calculate_vwap,
    calculate_macd,
    calculate_bollinger_bands,
)
from src.signals.regime_detector import detect_market_regime

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

            # Default spot_df to df (current wiring provides SPOT df through runner)
            if spot_df is None:
                spot_df = df

            # Derive spot last and option current price
            spot_last = None
            if current_tick and isinstance(current_tick, dict):
                spot_last = current_tick.get("spot_ltp", None)
            if spot_last is None and spot_df is not None and not spot_df.empty:
                spot_last = float(spot_df["close"].iloc[-1])

            # Priority for option price: explicit -> tick.option_ltp -> tick.ltp -> fall back to spot last
            if current_price is None and current_tick:
                current_price = current_tick.get("option_ltp", None)
            if current_price is None and current_tick:
                current_price = current_tick.get("ltp", None)
            if current_price is None:
                current_price = spot_last  # last resort to keep flow running

            if current_price is None or float(current_price) <= 0:
                dbg["reason_block"] = "invalid_current_price"
                return None

            # --- Trend/momentum (on the provided df; currently SPOT in runner) ---
            ema_fast = self._ema(df["close"], self.ema_fast)
            ema_slow = self._ema(df["close"], self.ema_slow)
            ema_bias_up = bool(ema_fast.iloc[-1] > ema_slow.iloc[-1])
            ema_cross_up = bool((ema_fast.iloc[-2] <= ema_slow.iloc[-2]) and ema_bias_up)
            ema_cross_down = bool((ema_fast.iloc[-2] >= ema_slow.iloc[-2]) and not ema_bias_up)

            score = 0
            reasons: list[str] = []

            if ema_cross_up:
                score += 2
                reasons.append(f"EMA fast({self.ema_fast}) crossed above slow({self.ema_slow}).")
            elif ema_cross_down:
                score += 2
                reasons.append(f"EMA fast({self.ema_fast}) crossed below slow({self.ema_slow}).")

            rsi_series = self._rsi(df["close"], self.rsi_period)
            rsi_val = float(rsi_series.iloc[-1])
            macd_line, _, _ = calculate_macd(df["close"])
            bb_upper, bb_lower = calculate_bollinger_bands(df["close"])
            if ema_bias_up and rsi_val > 50:
                score += 1
                reasons.append(f"RSI({self.rsi_period}) > 50 (up momentum).")
            elif not ema_bias_up and rsi_val < 50:
                score += 1
                reasons.append(f"RSI({self.rsi_period}) < 50 (down momentum).")

            # --- SPOT regime / VWAP filters ---
            regime = None
            if spot_df is not None and len(spot_df) >= max(10, self.adx_period):
                adx_series, di_plus_series, di_minus_series = self._extract_adx_columns(spot_df)
                regime = detect_market_regime(
                    df=spot_df,
                    adx=adx_series,
                    di_plus=di_plus_series,
                    di_minus=di_minus_series,
                )
                if regime == "trend":
                    regime = "trend_up" if ema_bias_up else "trend_down"
                elif regime == "no_trade":
                    dbg["reason_block"] = "no_trade_regime"
                    self._last_debug = dbg
                    return None
                if regime == "trend_up" and ema_bias_up:
                    score += 2
                    reasons.append("Spot ADX trend_up (aligned).")
                elif regime == "trend_down" and not ema_bias_up:
                    score += 2
                    reasons.append("Spot ADX trend_down (aligned).")
                elif regime == "range":
                    reasons.append("Spot ADX range.")

            vwap_val = None
            current_spot_price = None
            if spot_df is not None and len(spot_df) > 0:
                vwap_series = calculate_vwap(spot_df)
                if vwap_series is not None and len(vwap_series) > 0:
                    vwap_val = float(vwap_series.iloc[-1])
                    current_spot_price = float(spot_df["close"].iloc[-1])
                    if ema_bias_up and current_spot_price > vwap_val:
                        score += 1
                        reasons.append("Spot > VWAP (risk-on).")
                    elif not ema_bias_up and current_spot_price < vwap_val:
                        score += 1
                        reasons.append("Spot < VWAP (risk-off).")

            adx_val = float(adx_series.iloc[-1]) if 'adx_series' in locals() and adx_series is not None and len(adx_series) else 0.0
            bb_width = float((bb_upper.iloc[-1] - bb_lower.iloc[-1])) if len(bb_upper) else 0.0
            bb_width_pct = bb_width / float(current_spot_price) if current_spot_price else 0.0
            if adx_val < (self.adx_trend_strength / 2) and bb_width_pct < 0.01:
                dbg["reason_block"] = "indecisive_market"
                self._last_debug = dbg
                return None

            macd_val = float(macd_line.iloc[-1]) if len(macd_line) else 0.0
            ema_fast_slope = float(ema_fast.iloc[-1] - ema_fast.iloc[-2])
            ema_slow_slope = float(ema_slow.iloc[-1] - ema_slow.iloc[-2])

            if regime in ("trend_up", "trend_down") and vwap_val is not None and current_spot_price is not None:
                vwap_ok = (current_spot_price > vwap_val) if regime == "trend_up" else (current_spot_price < vwap_val)
                ema_slope_ok = (
                    (ema_fast_slope > 0 and ema_slow_slope > 0)
                    if regime == "trend_up"
                    else (ema_fast_slope < 0 and ema_slow_slope < 0)
                )
                macd_ok = macd_val > 0 if regime == "trend_up" else macd_val < 0
                if not (vwap_ok and ema_slope_ok and macd_ok):
                    dbg["reason_block"] = "trend_gates_failed"
                    self._last_debug = dbg
                    return None
            elif regime == "range" and vwap_val is not None and current_spot_price is not None:
                price = current_spot_price
                std_from_vwap = df["close"].rolling(window=20, min_periods=20).std().iloc[-1]
                if abs(price - vwap_val) < std_from_vwap:
                    dbg["reason_block"] = "not_far_from_vwap"
                    self._last_debug = dbg
                    return None
                prev_open, prev_close = float(df["open"].iloc[-2]), float(df["close"].iloc[-2])
                curr_open, curr_close = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
                rsi_prev = float(rsi_series.iloc[-2])
                if price >= float(bb_upper.iloc[-1]):
                    reversal = prev_close > prev_open and curr_close < curr_open
                    rsi_roll = rsi_prev > rsi_val
                    if not (reversal and rsi_roll):
                        dbg["reason_block"] = "range_gates_failed"
                        self._last_debug = dbg
                        return None
                    ema_bias_up = False
                elif price <= float(bb_lower.iloc[-1]):
                    reversal = prev_close < prev_open and curr_close > curr_open
                    rsi_roll = rsi_prev < rsi_val
                    if not (reversal and rsi_roll):
                        dbg["reason_block"] = "range_gates_failed"
                        self._last_debug = dbg
                        return None
                    ema_bias_up = True
                else:
                    dbg["reason_block"] = "no_range_setup"
                    self._last_debug = dbg
                    return None

            # --- Score -> confidence; apply thresholds (strict, then relaxed) ---
            confidence = self._score_confidence(score)
            strict_ok = self._passes(score, confidence, strict=True)
            relaxed_ok = self._passes(score, confidence, strict=False)

            if not strict_ok:
                _log_throttled(
                    "drop_strict",
                    logging.INFO,
                    "Signal drop (strict): confidence(%.2f) < threshold(%.2f) | score=%d",
                    confidence, self.min_conf_strict, score,
                )
                if not self.auto_relax_enabled or not relaxed_ok:
                    if not relaxed_ok:
                        _log_throttled(
                            "drop_relaxed",
                            logging.INFO,
                            "Signal drop (relaxed): confidence(%.2f) < threshold(%.2f) | score=%d",
                            confidence, self.min_conf_relaxed, score,
                        )
                        dbg["reason_block"] = "confidence_below_threshold"
                    else:
                        dbg["reason_block"] = "auto_relax_disabled"
                    self._last_debug = {**dbg, "score": score, "confidence": confidence, "reasons": reasons}
                    return None
                _log_throttled(
                    "auto_relax",
                    logging.INFO,
                    "Auto-relax applied: min_score->%d, confidence_threshold->%.2f",
                    self.min_score_relaxed, self.min_conf_relaxed,
                )

            # --- ATR-based SL/TP (on df; currently SPOT in runner) ---
            atr_series = compute_atr(df, period=self.atr_period)
            atr_val = latest_atr_value(atr_series, default=0.0)
            if atr_val <= 0:
                dbg["reason_block"] = "atr_unavailable"
                self._last_debug = {**dbg, "score": score, "confidence": confidence, "reasons": reasons}
                return None

            sl_mult = self.base_sl_mult
            tp_mult = self.base_tp_mult
            if regime in ("trend_up", "trend_down"):
                tp_mult += self.trend_tp_boost
                sl_mult += self.trend_sl_relax
                reasons.append(f"Regime boost: trend (tp+{self.trend_tp_boost}, sl+{self.trend_sl_relax}).")
            elif regime == "range":
                tp_mult += self.range_tp_tighten
                sl_mult += self.range_sl_tighten
                reasons.append(f"Regime tighten: range (tp{self.range_tp_tighten:+}, sl{self.range_sl_tighten:+}).")

            sl_mult = max(0.2, sl_mult)
            tp_mult = max(0.4, tp_mult)

            base_sl = atr_val * sl_mult
            base_tp = atr_val * tp_mult

            sl_points, tp_points = atr_sl_tp_points(
                base_sl_points=base_sl,
                base_tp_points=base_tp,
                atr_value=atr_val,
                sl_mult=sl_mult,
                tp_mult=tp_mult,
                confidence=confidence,
                sl_conf_adj=self.sl_conf_adj,
                tp_conf_adj=self.tp_conf_adj,
            )

            side: Side = "BUY" if ema_bias_up else "SELL"
            entry_price = float(current_price)
            eps = 1e-4
            if side == "BUY":
                stop_loss = max(eps, entry_price - sl_points)
                target = entry_price + tp_points
            else:
                stop_loss = max(eps, entry_price + sl_points)
                target = entry_price - tp_points

            # Option selection fields (nearest 50‑point ATM for NIFTY)
            try:
                s_last = float(spot_last) if spot_last is not None else float(df["close"].iloc[-1])
            except Exception:
                s_last = float(df["close"].iloc[-1])
            strike = int(round(s_last / 50.0) * 50)
            option_type = "CE" if side == "BUY" else "PE"
            action = side

            # RR
            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            rr = round((reward / risk) if risk > 0 else 0.0, 2)

            signal: Dict[str, Any] = {
                "action": action,
                "option_type": option_type,
                "strike": strike,
                "entry_price": float(entry_price),
                "stop_loss": float(stop_loss),
                "take_profit": float(target),
                "rr": rr,
                "side": side,
                "confidence": float(confidence),
                "sl_points": float(sl_points),
                "tp_points": float(tp_points),
                "score": int(score),
                "target": float(target),
                "reasons": reasons,
                "regime": regime or "unknown",
                "emas": {"fast": float(ema_fast.iloc[-1]), "slow": float(ema_slow.iloc[-1])},
                "rsi": float(rsi_val),
                "atr": float(atr_val),
            }

            _log_throttled(
                "generated",
                logging.INFO,
                "Generated signal (%s): %s",
                "relaxed" if (relaxed_ok and not strict_ok) else "strict",
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in signal.items()
                    if k not in ("reasons", "emas")
                },
            )

            self._last_debug = {
                "score": score,
                "confidence": confidence,
                "rr": rr,
                "regime": regime or "unknown",
                "ema_bias_up": ema_bias_up,
                "reason_block": None,
                "reasons": reasons[-6:],
                "thresholds": {
                    "min_score_strict": self.min_score_strict,
                    "min_conf_strict": self.min_conf_strict,
                    "min_score_relaxed": self.min_score_relaxed,
                    "min_conf_relaxed": self.min_conf_relaxed,
                },
                "atr_val": atr_val,
                "sl_mult": sl_mult,
                "tp_mult": tp_mult,
            }

            return signal

        except Exception as e:
            dbg["reason_block"] = f"exception:{e.__class__.__name__}"
            self._last_debug = dbg
            logger.debug("generate_signal exception: %s", e, exc_info=True)
            return None
