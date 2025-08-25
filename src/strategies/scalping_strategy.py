# Path: src/strategies/scalping_strategy.py
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple, Literal

import pandas as pd

from src.config import settings
from src.utils.atr_helper import compute_atr, atr_sl_tp_points, latest_atr_value
from src.utils.indicators import calculate_vwap
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
    Regime‑aware scalping strategy (runner‑aligned).

    Inputs
    ------
    - df: OHLCV dataframe (ascending index). In current wiring this is SPOT.
    - current_tick: optional dict; may contain 'ltp', 'spot_ltp', 'option_ltp'
    - current_price: explicit option LTP (overrides tick if provided)
    - spot_df: optional SPOT dataframe (if 'df' were option candles in the future)

    Output (executor expects)
    -------------------------
    dict with keys:
      action, option_type, strike, entry_price, stop_loss, take_profit, rr
    plus diagnostics (score, confidence, regime, reasons, etc.)
    """

    def __init__(
        self,
        *,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        adx_period: int = 14,
        adx_trend_strength: int = 20,
    ) -> None:
        strat = getattr(settings, "strategy", object())

        # Core lookbacks
        self.ema_fast = int(getattr(strat, "ema_fast", ema_fast))
        self.ema_slow = int(getattr(strat, "ema_slow", ema_slow))
        self.rsi_period = int(getattr(strat, "rsi_period", rsi_period))
        self.adx_period = int(getattr(strat, "adx_period", adx_period))
        self.adx_trend_strength = int(getattr(strat, "adx_trend_strength", adx_trend_strength))
        self.atr_period = int(getattr(strat, "atr_period", 14))

        # Regime shaping (add to multipliers; can be negative)
        self.trend_tp_boost = float(getattr(strat, "trend_tp_boost", 0.6))
        self.trend_sl_relax = float(getattr(strat, "trend_sl_relax", 0.2))
        self.range_tp_tighten = float(getattr(strat, "range_tp_tighten", -0.4))
        self.range_sl_tighten = float(getattr(strat, "range_sl_tighten", -0.2))

        # Bars threshold for validity
        self.min_bars_for_signal = int(getattr(strat, "min_bars_for_signal", max(self.ema_slow, 10)))

        # Confidence thresholds (config often 0..100; internal 0..~8)
        raw_conf = float(getattr(strat, "confidence_threshold", 55))
        raw_conf_rel = float(getattr(strat, "confidence_threshold_relaxed", max(0.0, raw_conf - 20)))
        self.min_conf_strict = raw_conf / 10.0           # 55 -> 5.5
        self.min_conf_relaxed = raw_conf_rel / 10.0

        self.min_score_strict = int(getattr(strat, "min_signal_score", 3))
        self.auto_relax_enabled = bool(getattr(strat, "auto_relax_enabled", True))
        self.min_score_relaxed = int(getattr(strat, "min_signal_score_relaxed", max(2, self.min_score_strict - 1)))

        # ATR & confidence shaping
        self.base_sl_mult = float(getattr(strat, "atr_sl_multiplier", 1.3))
        self.base_tp_mult = float(getattr(strat, "atr_tp_multiplier", 2.2))
        self.sl_conf_adj = float(getattr(strat, "sl_confidence_adj", 0.2))
        self.tp_conf_adj = float(getattr(strat, "tp_confidence_adj", 0.3))

        # Exportable debug snapshot
        self._last_debug: Dict[str, Any] = {"note": "no_evaluation_yet"}

    # ---------- tech utils ----------
    @staticmethod
    def _ema(s: pd.Series, period: int) -> pd.Series:
        period = max(1, int(period))
        return pd.Series(pd.to_numeric(s, errors="coerce")).ewm(span=period, adjust=False).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        close = pd.Series(pd.to_numeric(close, errors="coerce"))
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
        rs = (avg_gain / (avg_loss.replace(0.0, 1e-9))).fillna(0.0)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _extract_adx_columns(
        spot_df: pd.DataFrame,
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
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
        # compact 0..8-ish scale mapped to thresholds above
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
            # ---- basic guards
            if df is None or len(df) < self.min_bars_for_signal:
                dbg["reason_block"] = "insufficient_bars"
                self._last_debug = dbg
                return None

            # ensure required columns exist and numeric
            need = {"open", "high", "low", "close"}
            if not need.issubset(df.columns):
                dbg["reason_block"] = "missing_ohlc_columns"
                self._last_debug = dbg
                return None
            df = df.copy()
            for c in ("open", "high", "low", "close", "volume"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["close"])
            if df.empty:
                dbg["reason_block"] = "no_valid_prices"
                self._last_debug = dbg
                return None
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

            # Default spot_df to df (current wiring provides SPOT df)
            if spot_df is None:
                spot_df = df

            # ---- derive spot last and option current price
            spot_last = None
            if current_tick and isinstance(current_tick, dict):
                spot_last = current_tick.get("spot_ltp", None)
            if spot_last is None and spot_df is not None and len(spot_df) > 0:
                spot_last = float(pd.to_numeric(spot_df["close"].iloc[-1], errors="coerce"))

            if current_price is None and current_tick:
                current_price = current_tick.get("option_ltp", None)
            if current_price is None and current_tick:
                current_price = current_tick.get("ltp", None)
            if current_price is None:
                current_price = spot_last  # last resort to keep flow running

            if current_price is None or float(current_price) <= 0:
                dbg["reason_block"] = "invalid_current_price"
                self._last_debug = dbg
                return None

            # ---- trend/momentum on df (SPOT)
            ema_fast = self._ema(df["close"], self.ema_fast)
            ema_slow = self._ema(df["close"], self.ema_slow)
            if len(ema_fast) < 2 or len(ema_slow) < 2:
                dbg["reason_block"] = "ema_insufficient"
                self._last_debug = dbg
                return None

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
            if ema_bias_up and rsi_val > 50:
                score += 1
                reasons.append(f"RSI({self.rsi_period}) > 50 (up momentum).")
            elif not ema_bias_up and rsi_val < 50:
                score += 1
                reasons.append(f"RSI({self.rsi_period}) < 50 (down momentum).")

            # ---- SPOT regime / VWAP filters
            regime = None
            if spot_df is not None and len(spot_df) >= max(10, self.adx_period):
                adx_series, di_plus_series, di_minus_series = self._extract_adx_columns(spot_df)
                try:
                    regime = detect_market_regime(
                        df=spot_df,
                        adx=adx_series,
                        di_plus=di_plus_series,
                        di_minus=di_minus_series,
                        adx_trend_strength=self.adx_trend_strength,
                    )
                except Exception:
                    regime = None

                if regime == "trend_up" and ema_bias_up:
                    score += 2
                    reasons.append("Spot ADX trend_up (aligned).")
                elif regime == "trend_down" and not ema_bias_up:
                    score += 2
                    reasons.append("Spot ADX trend_down (aligned).")
                elif regime == "range":
                    reasons.append("Spot ADX range.")

            if spot_df is not None and len(spot_df) > 0:
                try:
                    vwap_series = calculate_vwap(spot_df)
                    if vwap_series is not None and len(vwap_series) > 0:
                        vwap_val = float(vwap_series.iloc[-1])
                        current_spot_price = float(pd.to_numeric(spot_df["close"].iloc[-1], errors="coerce"))
                        if ema_bias_up and current_spot_price > vwap_val:
                            score += 1
                            reasons.append("Spot > VWAP (risk-on).")
                        elif not ema_bias_up and current_spot_price < vwap_val:
                            score += 1
                            reasons.append("Spot < VWAP (risk-off).")
                except Exception:
                    # VWAP is advisory; ignore failures
                    pass

            # ---- Score -> confidence; apply thresholds
            confidence = self._score_confidence(score)
            strict_ok = self._passes(score, confidence, strict=True)
            relaxed_ok = self._passes(score, confidence, strict=False)

            if not strict_ok:
                _log_throttled(
                    "drop_strict",
                    logging.INFO,
                    "Signal drop (strict): confidence(%.2f) < threshold(%.2f) | score=%d",
                    confidence,
                    self.min_conf_strict,
                    score,
                )
                if not self.auto_relax_enabled or not relaxed_ok:
                    if not relaxed_ok:
                        _log_throttled(
                            "drop_relaxed",
                            logging.INFO,
                            "Signal drop (relaxed): confidence(%.2f) < threshold(%.2f) | score=%d",
                            confidence,
                            self.min_conf_relaxed,
                            score,
                        )
                        dbg["reason_block"] = "confidence_below_threshold"
                    else:
                        dbg["reason_block"] = "auto_relax_disabled"
                    self._last_debug = {**dbg, "score": score, "confidence": confidence, "reasons": reasons}
                    return None
                _log_throttled(
                    "auto_relax",
                    logging.INFO,
                    "Auto‑relax applied: min_score->%d, confidence_threshold->%.2f",
                    self.min_score_relaxed,
                    self.min_conf_relaxed,
                )

            # ---- ATR‑based SL/TP (on df; currently SPOT)
            atr_series = compute_atr(df, period=self.atr_period)
            atr_val = latest_atr_value(atr_series, default=0.0)
            if atr_val is None or atr_val <= 0:
                dbg["reason_block"] = "atr_unavailable"
                self._last_debug = {**dbg, "score": score, "confidence": confidence, "reasons": reasons}
                return None

            sl_mult = max(0.2, float(self.base_sl_mult))
            tp_mult = max(0.4, float(self.base_tp_mult))
            if regime in ("trend_up", "trend_down"):
                tp_mult += float(self.trend_tp_boost)
                sl_mult += float(self.trend_sl_relax)
                reasons.append(f"Regime boost: trend (tp+{self.trend_tp_boost}, sl+{self.trend_sl_relax}).")
            elif regime == "range":
                tp_mult += float(self.range_tp_tighten)
                sl_mult += float(self.range_sl_tighten)
                reasons.append(f"Regime tighten: range (tp{self.range_tp_tighten:+}, sl{self.range_sl_tighten:+}).")

            base_sl = float(atr_val) * sl_mult
            base_tp = float(atr_val) * tp_mult

            sl_points, tp_points = atr_sl_tp_points(
                base_sl_points=base_sl,
                base_tp_points=base_tp,
                atr_value=float(atr_val),
                sl_mult=sl_mult,
                tp_mult=tp_mult,
                confidence=float(confidence),
                sl_conf_adj=float(self.sl_conf_adj),
                tp_conf_adj=float(self.tp_conf_adj),
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

            # Option selection: nearest 50‑point ATM from spot
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
                # required by runner/executor
                "action": action,
                "option_type": option_type,
                "strike": strike,
                "entry_price": float(entry_price),
                "stop_loss": float(stop_loss),
                "take_profit": float(target),
                "rr": rr,
                # diagnostics
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
                "reasons": reasons[-6:],  # compact tail
                "thresholds": {
                    "min_score_strict": self.min_score_strict,
                    "min_conf_strict": self.min_conf_strict,
                    "min_score_relaxed": self.min_score_relaxed,
                    "min_conf_relaxed": self.min_conf_relaxed,
                },
                "atr_val": float(atr_val),
                "sl_mult": sl_mult,
                "tp_mult": tp_mult,
            }
            return signal

        except Exception as e:
            dbg["reason_block"] = f"exception:{e.__class__.__name__}"
            self._last_debug = dbg
            logger.debug("generate_signal exception: %s", e, exc_info=True)
            return None