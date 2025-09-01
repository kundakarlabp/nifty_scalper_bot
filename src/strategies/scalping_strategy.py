# Path: src/strategies/scalping_strategy.py
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, time as dt_time
from typing import Any, Dict, Optional, Tuple, Literal, Deque
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import settings
from src.utils.atr_helper import compute_atr, latest_atr_value
from src.utils.indicators import (
    calculate_vwap,
    calculate_macd,
)
from src.signals.regime_detector import detect_market_regime
from src.execution.order_executor import fetch_quote_with_depth, micro_ok
from src.utils.strike_selector import (
    get_instrument_tokens,
    select_strike,
    resolve_weekly_atm,
)
from random import uniform as rand_uniform

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


def is_thu_after(threshold: dt_time, *, now: Optional[datetime] = None) -> bool:
    """Return True if current time is Thursday after ``threshold``."""
    now = now or datetime.now(ZoneInfo("Asia/Kolkata"))
    return now.weekday() == 3 and now.time() >= threshold


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
        self._iv_window: Deque[float] = getattr(self, "_iv_window", deque(maxlen=20))
        self.last_atr_pct: float = 0.0

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

    def _est_iv_pct(self, S: float, K: float, T: float) -> Optional[int]:
        """Estimate rolling IV percentile."""
        try:
            from src.risk.greeks import implied_vol_newton

            tv = max(0.5, S * (self.last_atr_pct / 100.0) * 0.25)
            iv = implied_vol_newton(tv, S, K, T, 0.06, 0.0, "CE", guess=0.20) or 0.20
        except Exception:
            iv = 0.20
        self._iv_window.append(iv)
        if len(self._iv_window) < 5:
            return None
        arr = sorted(self._iv_window)
        rk = arr.index(iv)
        return int(round(100 * rk / max(1, len(arr) - 1)))

    def _iv_adx_reject_reason(
        self, plan: Dict[str, Any], close: float
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        cfg = getattr(getattr(self, "runner", None), "strategy_cfg", None)
        if not cfg:
            return None
        adx = plan.get("adx")
        if plan.get("regime") == "TREND" and adx is not None and adx < int(cfg.adx_min_trend):
            return "weak_trend", {"adx": adx}
        ivpct = self._est_iv_pct(S=close, K=plan.get("atm_strike", int(close)), T=plan.get("T", 3 / 365))
        plan["iv_pct"] = ivpct
        limit = int(cfg.iv_percentile_limit)
        need = int(cfg.score_trend_min) + 1
        if ivpct is not None and ivpct > limit and plan.get("score", 0) < need:
            return "iv_extreme", {"iv_pct": ivpct, "need_score": need}
        return None

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
        cfg = getattr(getattr(self, "runner", None), "strategy_cfg", None)
        if cfg is None:
            from src.strategies.strategy_config import StrategyConfig, resolve_config_path
            cfg = StrategyConfig.load(resolve_config_path())

        plan = {
            "has_signal": False,
            "action": "NONE",
            "option_type": None,
            "strike": None,
            "qty_lots": None,
            "regime": "NO_TRADE",
            "score": 0,
            "atr_pct": 0.0,
            "micro": {"spread_pct": 0.0, "depth_ok": False},
            "rr": 0.0,
            "entry": None,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "trail_atr_mult": None,
            "time_stop_min": None,
            "reasons": [],
            "reason_block": None,
            "ts": datetime.utcnow().isoformat(),
            "bar_count": int(len(df) if isinstance(df, pd.DataFrame) else 0),
            "last_bar_ts": (
                pd.to_datetime(df.index[-1]).to_pydatetime().isoformat()
                if isinstance(df, pd.DataFrame) and len(df)
                else None
            ),
            # legacy extras for compatibility
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "target": None,
            "side": None,
            "confidence": 0.0,
            "breakeven_ticks": None,
            "tp1_qty_ratio": None,
        }

        plan["_event_post_widen"] = float(getattr(self, "_event_post_widen", 0.0))

        dbg: Dict[str, Any] = {"reason_block": None}

        def plan_block(reason: str, **extra: Any) -> Dict[str, Any]:
            plan.update(extra)
            plan["reason_block"] = reason
            dbg["reason_block"] = reason
            self._last_debug = dbg
            return plan

        try:
            if df is None or df.empty:
                return plan_block("indicator_unready")
            if spot_df is None:
                spot_df = df
            last_ts = pd.to_datetime(df.index[-1])
            plan["last_bar_ts"] = last_ts.to_pydatetime().isoformat()
            spot_last = float(spot_df["close"].iloc[-1])
            if current_price is None:
                current_price = float(current_tick.get("ltp", spot_last)) if current_tick else spot_last
            if len(df) < cfg.min_bars_required:
                return plan_block("insufficient_bars", bar_count=len(df))
            if current_price is None or current_price <= 0 or len(df) < cfg.indicator_min_bars:
                return plan_block("indicator_unready", bar_count=len(df))

            ema21 = self._ema(df["close"], 21)
            ema50 = self._ema(df["close"], 50)
            vwap = calculate_vwap(df)
            macd_line, macd_signal, macd_hist = calculate_macd(df["close"])
            rsi = self._rsi(df["close"], 14)
            atr_series = compute_atr(df, period=14)
            atr_val = latest_atr_value(atr_series, default=0.0)

            if (
                len(df) < 14
                or pd.isna(ema21.iloc[-1])
                or pd.isna(ema50.iloc[-1])
                or vwap is None
                or len(vwap) == 0
                or pd.isna(vwap.iloc[-1])
                or atr_val <= 0
            ):
                return plan_block("indicator_unready")

            price = float(spot_last)
            ema21_val, ema50_val = float(ema21.iloc[-1]), float(ema50.iloc[-1])
            ema21_slope = float(ema21.iloc[-1] - ema21.iloc[-2])
            macd_val = float(macd_line.iloc[-1])
            rsi_val = float(rsi.iloc[-1])
            rsi_prev = float(rsi.iloc[-2])
            rsi_rising = rsi_val > rsi_prev

            # --- regime detection ---
            adx_series = spot_df.get("adx")
            plan["adx"] = float(adx_series.iloc[-1]) if adx_series is not None else None
            reg = detect_market_regime(
                df=df,
                adx=adx_series,
                di_plus=spot_df.get("di_plus"),
                di_minus=spot_df.get("di_minus"),
            )
            if reg.regime == "RANGE" and adx_series is None:
                reg.regime = "TREND"  # fallback when ADX not available
            plan["regime"] = reg.regime
            if reg.regime == "NO_TRADE":
                return plan_block("regime_no_trade")

            # swing levels for breakout guard
            swing_high = df["high"].rolling(window=20, min_periods=2).max().iloc[-2]
            swing_low = df["low"].rolling(window=20, min_periods=2).min().iloc[-2]
            breakout_dist_long = abs(price - swing_high) / price * 100.0
            breakout_dist_short = abs(price - swing_low) / price * 100.0

            side: Optional[Side] = None
            option_type: Optional[str] = None
            reasons: list[str] = []

            if reg.regime == "TREND":
                long_ok = (
                    price > float(vwap.iloc[-1])
                    and ema21_val > ema50_val
                    and ema21_slope > 0
                    and macd_val > 0
                    and rsi_val >= 48
                    and rsi_rising
                    and breakout_dist_long >= 0.15
                )
                short_ok = (
                    price < float(vwap.iloc[-1])
                    and ema21_val < ema50_val
                    and ema21_slope < 0
                    and macd_val < 0
                    and rsi_val <= 52
                    and not rsi_rising
                    and breakout_dist_short >= 0.15
                )
                if long_ok or short_ok:
                    side = "BUY" if long_ok else "SELL"
                    option_type = "CE" if long_ok else "PE"
                    reasons.append("trend_playbook")
                else:
                    return plan_block("score_low")
            elif reg.regime == "RANGE":
                close = float(df["close"].iloc[-1])
                bb_mid = df["close"].rolling(20).mean()
                bb_std = df["close"].rolling(20).std()
                upper = float(bb_mid.iloc[-1] + 2 * bb_std.iloc[-1])
                lower = float(bb_mid.iloc[-1] - 2 * bb_std.iloc[-1])
                vwap_val = float(vwap.iloc[-1])
                std_val = float(bb_std.iloc[-1])

                upper_fade = (
                    (close >= upper or close >= vwap_val + 1.9 * std_val)
                    and rsi_val > 65
                    and rsi_val < rsi_prev
                    and float(df["close"].iloc[-1]) < float(df["open"].iloc[-1])
                )
                lower_fade = (
                    (close <= lower or close <= vwap_val - 1.9 * std_val)
                    and rsi_val < 35
                    and rsi_val > rsi_prev
                    and float(df["close"].iloc[-1]) > float(df["open"].iloc[-1])
                )

                if upper_fade:
                    side = "SELL"
                    option_type = "PE"
                    reasons.append("range_playbook_upper")
                elif lower_fade:
                    side = "BUY"
                    option_type = "CE"
                    reasons.append("range_playbook_lower")
                else:
                    return plan_block("score_low")
            else:
                return plan_block("regime_no_trade")

            atr_pct = (atr_val / price) * 100.0
            plan["atr_pct"] = round(atr_pct, 2)
            self.last_atr_pct = plan["atr_pct"]
            if not (cfg.atr_min <= plan["atr_pct"] <= cfg.atr_max):
                return plan_block("atr_out_of_band", atr_pct=plan["atr_pct"])

            # ----- scoring -----
            regime_score = 2
            momentum_score = 0
            macd_rising = macd_line.iloc[-1] > macd_line.iloc[-2]
            if (side == "BUY" and macd_val > 0) or (side == "SELL" and macd_val < 0):
                momentum_score += 1
            if (side == "BUY" and macd_rising) or (side == "SELL" and not macd_rising):
                momentum_score += 1
            if (side == "BUY" and rsi_rising) or (side == "SELL" and not rsi_rising):
                momentum_score += 1
            obv = spot_df.get("obv") or df.get("obv")
            if (
                isinstance(obv, pd.Series)
                and pd.api.types.is_numeric_dtype(obv)
                and len(obv) >= 2
            ):
                obv_rising = float(obv.iloc[-1]) > float(obv.iloc[-2])
                if (side == "BUY" and obv_rising) or (side == "SELL" and not obv_rising):
                    momentum_score += 1
            else:
                logger.debug("Invalid OBV series for momentum scoring: %r", obv)
            momentum_score = min(3, momentum_score)

            structure_score = 0
            candle_bull = float(df["close"].iloc[-1]) > float(df["open"].iloc[-1])
            candle_bear = float(df["close"].iloc[-1]) < float(df["open"].iloc[-1])
            if side == "BUY" and candle_bull:
                structure_score += 1
            if side == "SELL" and candle_bear:
                structure_score += 1
            if (side == "BUY" and breakout_dist_long >= 0.15) or (
                side == "SELL" and breakout_dist_short >= 0.15
            ):
                structure_score += 1

            vol_score = 1 if cfg.atr_min <= atr_pct <= cfg.atr_max else 0

            micro_score = 2
            atm = resolve_weekly_atm(price)
            info_atm = atm.get(option_type.lower()) if atm else None
            if not info_atm:
                return plan_block("no_option_token", micro={"spread_pct": None, "depth_ok": None})
            tsym, lot_sz = info_atm
            q = fetch_quote_with_depth(getattr(settings, "kite", None), tsym)
            runner_obj = getattr(self, "runner", None)
            nowt = (runner_obj.now_ist if runner_obj else datetime.now(ZoneInfo(cfg.tz))).time()
            max_spread = cfg.max_spread_pct_regular
            if nowt <= dt_time(10, 0):
                max_spread = cfg.max_spread_pct_open
            elif nowt >= dt_time(15, 10):
                max_spread = cfg.max_spread_pct_last20m
            event_widen = float(plan.get("_event_post_widen", 0.0))
            max_spread = max_spread + event_widen
            ok_micro, micro = micro_ok(
                q,
                qty_lots=1,
                lot_size=lot_sz,
                max_spread_pct=max_spread,
                depth_mult=cfg.depth_multiplier,
            )
            plan["micro"] = micro
            if not ok_micro:
                micro_score = 0

            score = regime_score + momentum_score + structure_score + vol_score + micro_score
            plan["score"] = score
            plan["reasons"] = reasons

            need = cfg.score_trend_min if reg.regime == "TREND" else cfg.score_range_min
            if cfg.lower_score_temp:
                need = min(need, 6)
            if score < need:
                return plan_block("score_low", score=score, need=need)

            rej = self._iv_adx_reject_reason(plan, price)
            if rej:
                reason, extra = rej
                return plan_block(reason, **extra)

            entry_price = float(current_price)
            tick_size = float(getattr(getattr(settings, "executor", object()), "tick_size", 0.05))

            if reg.regime == "RANGE" and side == "SELL":
                struct_sl_price = float(df["high"].iloc[-1]) + 0.25 * atr_val
                struct_dist = struct_sl_price - entry_price
            elif reg.regime == "RANGE" and side == "BUY":
                struct_sl_price = float(df["low"].iloc[-1]) - 0.25 * atr_val
                struct_dist = entry_price - struct_sl_price
            else:
                struct_dist = 0.8 * atr_val

            sl_dist = max(0.8 * atr_val, struct_dist)
            if side == "BUY":
                stop_loss = entry_price - sl_dist
            else:
                stop_loss = entry_price + sl_dist

            R = abs(entry_price - stop_loss)
            tp1_mult = rand_uniform(cfg.tp1_R_min, cfg.tp1_R_max)
            tp2_mult = cfg.tp2_R_trend if reg.regime == "TREND" else cfg.tp2_R_range
            trail_mult = cfg.trail_atr_mult
            time_stop = cfg.time_stop_min
            runner_obj = getattr(self, "runner", None)
            now_dt = runner_obj.now_ist if runner_obj else datetime.now(ZoneInfo(cfg.tz))
            if cfg.gamma_enabled and is_thu_after(cfg.gamma_after, now=now_dt):
                tp2_mult = min(tp2_mult, cfg.gamma_tp2_cap)
                trail_mult = min(trail_mult, cfg.gamma_trail_mult)
                time_stop = min(time_stop, cfg.gamma_time_stop_min)
                reasons.append("gamma_mode")
            if side == "BUY":
                tp1 = entry_price + tp1_mult * R
                tp2 = entry_price + tp2_mult * R
            else:
                tp1 = entry_price - tp1_mult * R
                tp2 = entry_price - tp2_mult * R

            breakeven_ticks = int(max(1, round(0.1 * R / tick_size)))
            rr = (abs(tp2 - entry_price) / R) if R > 0 else 0.0

            # strike selection & liquidity
            try:
                _ = get_instrument_tokens(spot_price=price)
            except Exception as e:
                logger.debug("instrument token lookup failed: %s", e)
            strike_info = select_strike(price, score)
            if not strike_info:
                liquidity_info: Optional[Dict[str, Any]] = None
                try:
                    from src.utils import strike_selector as ss
                    liquidity_info = ss._option_info_fetcher(int(round(price / 50.0) * 50))
                except Exception:
                    liquidity_info = None
                if liquidity_info is not None:
                    return plan_block("liquidity_fail")
                strike = int(round(price / 50.0) * 50)
            else:
                strike = int(strike_info.strike)

            plan.update({
                "has_signal": True,
                "action": side,
                "option_type": option_type or None,
                "strike": str(strike),
                "entry": entry_price,
                "sl": stop_loss,
                "tp1": tp1,
                "tp2": tp2,
                "trail_atr_mult": trail_mult,
                "time_stop_min": time_stop,
                "rr": round(rr, 2),
                "regime": reg.regime,
                "score": score,
                "reasons": reasons,
            })
            # backward compatibility extras
            plan["entry_price"] = entry_price
            plan["stop_loss"] = stop_loss
            plan["take_profit"] = tp2
            plan["target"] = tp2
            plan["side"] = side
            plan["confidence"] = min(1.0, max(0.0, score / 10.0))
            plan["breakeven_ticks"] = breakeven_ticks
            plan["tp1_qty_ratio"] = cfg.tp1_partial

            self._last_debug = {
                "score": score,
                "regime": reg.regime,
                "rr": rr,
                "reason_block": None,
                "atr_pct": plan["atr_pct"],
            }
            return plan

        except Exception as e:
            plan["reason_block"] = f"exception:{e.__class__.__name__}"
            dbg["reason_block"] = plan["reason_block"]
            self._last_debug = dbg
            logger.debug("generate_signal exception: %s", e, exc_info=True)
            return plan


class ScalpingStrategy(EnhancedScalpingStrategy):
    """Alias with helpers used for tests and backtesting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Allow an optional ``settings`` kwarg for tests.

        The upstream ``EnhancedScalpingStrategy`` pulls configuration from the
        global ``settings`` object. The lightweight test harness in this kata
        instantiates ``ScalpingStrategy(settings=None)`` to avoid importing the
        real config. We accept and discard this keyword to keep the public API
        stable.
        """

        kwargs.pop("settings", None)
        super().__init__(*args, **kwargs)

    def evaluate_from_backtest(
        self, ts: datetime, o: float, h: float, l: float, c: float, v: float
    ) -> Dict[str, Any] | None:
        """Update internal buffer with a bar and reuse ``generate_signal``."""

        if not hasattr(self, "_bt_df"):
            self._bt_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self._bt_df.loc[ts] = {"open": o, "high": h, "low": l, "close": c, "volume": v}
        return self.generate_signal(self._bt_df)

