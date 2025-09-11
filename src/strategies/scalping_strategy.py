# Path: src/strategies/scalping_strategy.py
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from random import uniform as rand_uniform
from typing import Any, Deque, Dict, Literal, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import settings
from src.execution.micro_filters import cap_for_mid, evaluate_micro
from src.execution.order_executor import fetch_quote_with_depth
from src.signals.regime_detector import detect_market_regime
from src.strategies.strategy_config import StrategyConfig
from src.strategies.warmup import warmup_status
from src.strategies.patches import _resolve_min_atr_pct
from src.utils.atr_helper import compute_atr, latest_atr_value
from src.utils.indicators import (
    calculate_adx,
    calculate_bb_width,
    calculate_macd,
    calculate_vwap,
)
from src.utils import strike_selector
from src.utils.strike_selector import (
    get_instrument_tokens,
    resolve_weekly_atm,
    select_strike,
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


def is_tue_after(threshold: dt_time, *, now: Optional[datetime] = None) -> bool:
    """Return True if current time is Tuesday after ``threshold``."""
    now = now or datetime.now(ZoneInfo("Asia/Kolkata"))
    return now.weekday() == 1 and now.time() >= threshold


def _token_to_symbol_and_lot(
    kite: Any, token: int
) -> Optional[Tuple[str, int]]:
    """Return trading symbol and lot size for ``token``.

    Parameters
    ----------
    kite:
        Optional broker client used to fetch instrument dump.
    token:
        Instrument token to resolve.

    Returns
    -------
    Optional[Tuple[str, int]]
        Tuple of trading symbol and lot size if resolution succeeds.
    """

    try:
        inst_dump = strike_selector._fetch_instruments_nfo(kite) or []
        token_int = int(token)
        for row in inst_dump:
            if int(row.get("instrument_token", 0)) == token_int:
                tsym = str(row.get("tradingsymbol"))
                lot_sz = int(row.get("lot_size") or 0)
                return tsym, lot_sz
    except Exception:  # pragma: no cover
        return None
    return None


# ---------- scoring helpers ----------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp ``x`` between ``lo`` and ``hi``."""

    return max(lo, min(hi, x))


@dataclass
class ScoreDetails:
    """Detailed score breakdown for diagnostics."""

    regime: str
    parts: Dict[str, float]
    total: float


def _trend_score(df: pd.DataFrame, cfg: Any) -> Tuple[float, ScoreDetails]:
    """Compute trend-following score using EMA slope."""

    ema_fast = df["close"].ewm(span=getattr(cfg, "ema_fast", 9), adjust=False).mean()
    ema_slow = df["close"].ewm(span=getattr(cfg, "ema_slow", 21), adjust=False).mean()
    slope = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / (df["atr"].iloc[-1] or 1e-9)
    s = _clamp(abs(slope))
    det = ScoreDetails("TREND", {"ema_slope": s}, s)
    return s, det


def _range_score(df: pd.DataFrame, cfg: Any) -> Tuple[float, ScoreDetails]:
    """Compute mean‑reversion score for range‑bound markets."""

    period = getattr(cfg, "bb_period", 20)
    mid = df["close"].rolling(period).mean()
    dev = df["close"].rolling(period).std(ddof=0)
    bandw = (dev.iloc[-1] * 2.0) or 1e-9
    dist = abs(df["close"].iloc[-1] - mid.iloc[-1]) / (df["atr"].iloc[-1] or 1e-9)
    raw = dist / (bandw / (df["atr"].iloc[-1] or 1e-9))
    s = _clamp(raw)
    wide_pen = _clamp((dev.iloc[-1] / (mid.iloc[-1] or 1e-9)) * 4.0)
    s = _clamp(s * (1.0 - 0.5 * wide_pen))
    det = ScoreDetails("RANGE", {"mr_dist": s, "wide_pen": wide_pen}, s)
    return s, det


def compute_score(
    df: Optional[pd.DataFrame], regime: str, cfg: Any
) -> Tuple[float, Optional[ScoreDetails]]:
    """Return total score and breakdown for ``regime``.

    Parameters
    ----------
    df:
        Input dataframe containing ``close`` and ``atr`` columns.
    regime:
        Detected market regime, e.g. ``TREND`` or ``RANGE``.
    cfg:
        Strategy configuration providing lookback parameters.
    """

    if df is None or len(df) < getattr(cfg, "warmup_bars_min", 15):
        return 0.0, None
    if "atr" not in df.columns:
        period = getattr(cfg, "atr_period", 14)
        df = df.copy()
        df.loc[:, "atr"] = compute_atr(df, period=period)
    if regime == "TREND":
        return _trend_score(df, cfg)
    if regime == "RANGE" and getattr(cfg, "enable_range_scoring", True):
        return _range_score(df, cfg)
    return 0.0, None


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
        self.range_tp_tighten = float(
            getattr(settings.strategy, "range_tp_tighten", -0.4)
        )
        self.range_sl_tighten = float(
            getattr(settings.strategy, "range_sl_tighten", -0.2)
        )

        # Bars threshold for validity
        self.min_bars_for_signal = int(min_bars_for_signal)

        # Thresholds (normalize confidence scale)
        # Config is typically 0..100; internal scoring below returns ~0..8
        raw_conf = float(confidence_threshold)  # e.g., 55 (%)
        raw_conf_rel = float(
            getattr(
                settings.strategy,
                "confidence_threshold_relaxed",
                max(0.0, raw_conf - 20),
            )
        )
        self.min_conf_strict = raw_conf / 10.0  # 55 -> 5.5 on a 0..10-ish scale
        self.min_conf_relaxed = raw_conf_rel / 10.0

        self.min_score_strict = int(min_signal_score)
        self.auto_relax_enabled = bool(
            getattr(settings.strategy, "auto_relax_enabled", True)
        )
        self.auto_relax_after_min = int(
            getattr(settings.strategy, "auto_relax_after_min", 30)
        )
        self.min_score_relaxed = int(
            getattr(
                settings.strategy,
                "min_signal_score_relaxed",
                max(2, self.min_score_strict - 1),
            )
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

    @staticmethod
    def _normalize_min_score(val: float) -> float:
        """Return ``val`` on a 0..1 scale, accepting percentages."""
        return val / 100.0 if val > 1 else val

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
        cfg: StrategyConfig | None = getattr(
            getattr(self, "runner", None), "strategy_cfg", None
        )
        if not isinstance(cfg, StrategyConfig):
            return None
        adx = plan.get("adx")
        if (
            plan.get("regime") == "TREND"
            and adx is not None
            and adx < int(cfg.adx_min_trend)
        ):
            return "weak_trend", {"adx": adx}
        ivpct = self._est_iv_pct(
            S=close, K=plan.get("atm_strike", int(close)), T=plan.get("T", 3 / 365)
        )
        plan["iv_pct"] = ivpct
        limit = int(cfg.iv_percentile_limit)
        min_score = self._normalize_min_score(
            float(cfg.raw.get("strategy", {}).get("min_score", 0.35))
        )
        need = min_score + 0.1
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
        cfg: StrategyConfig | None = getattr(
            getattr(self, "runner", None), "strategy_cfg", None
        )
        if not isinstance(cfg, StrategyConfig):
            from src.strategies.strategy_config import resolve_config_path

            cfg = StrategyConfig.load(resolve_config_path())

        plan: Dict[str, Any] = {
            "has_signal": False,
            "action": "NONE",
            "option_type": None,
            "strike": None,
            "qty_lots": None,
            "regime": "NO_TRADE",
            # Score and microstructure metrics are unknown until later stages.
            # Use ``None`` so diagnostics can report "N/A" rather than misleading
            # zero values when signal generation exits early.
            "score": None,
            "atr_pct": 0.0,
            "micro": {"spread_pct": None, "depth_ok": None},
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
            "atm_strike": None,
            "option_token": None,
        }

        plan["score_dbg"] = {
            "components": {},
            "weights": {},
            "penalties": {},
            "raw": 0.0,
            "final": 0.0,
            "threshold": 0.0,
        }

        plan["_event_post_widen"] = float(getattr(self, "_event_post_widen", 0.0))

        dbg: Dict[str, Any] = {"reason_block": None}

        def plan_block(reason: str, **extra: Any) -> Dict[str, Any]:
            sd = plan.setdefault(
                "score_dbg",
                {
                    "components": {},
                    "weights": {},
                    "penalties": {},
                    "raw": 0.0,
                    "final": 0.0,
                    "threshold": 0.0,
                },
            )
            sd.setdefault("components", {})
            sd.setdefault("weights", {})
            sd.setdefault("penalties", {})
            sd.setdefault("raw", 0.0)
            sd.setdefault("final", 0.0)
            sd.setdefault("threshold", 0.0)
            plan.update(extra)
            plan["reason_block"] = reason
            dbg["reason_block"] = reason
            if reason == "score_low" and plan.get("score") is None:
                plan["score"] = float(extra.get("score", 0.0))
            if "score_dbg" not in plan:
                min_score_cfg = 0.0
                if isinstance(cfg, StrategyConfig):
                    strat_cfg = getattr(cfg, "raw", {}).get("strategy", {})  # type: ignore[arg-type]
                    min_score_cfg = self._normalize_min_score(
                        float(strat_cfg.get("min_score", 0.35))
                    )
                final = float(plan.get("score") or 0.0)
                plan["score_dbg"] = {
                    "components": {},
                    "weights": {},
                    "penalties": {},
                    "raw": 0.0,
                    "final": round(final, 4),
                    "threshold": min_score_cfg,
                }
            dbg["score"] = plan.get("score")
            dbg["score_dbg"] = plan.get("score_dbg")
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
                current_price = (
                    float(current_tick.get("ltp", spot_last))
                    if current_tick
                    else spot_last
                )
            atr_p = (
                int(getattr(cfg, "atr_period", 14))
                if hasattr(cfg, "atr_period")
                else 14
            )
            config_min = int(
                getattr(cfg, "min_bars_required", self.min_bars_for_signal)
            )
            warmup_bars = (
                int(getattr(cfg, "warmup_bars", 0))
                if hasattr(cfg, "warmup_bars")
                else 0
            )
            required_bars = max(
                self.min_bars_for_signal,
                config_min,
                atr_p + 5,
                warmup_bars,
            )
            have_bars = len(df)
            w = warmup_status(have_bars, required_bars)
            try:
                plan["features"] = {
                    "reasons": w.reasons,
                    "have_bars": have_bars,
                    "need_bars": required_bars,
                }
            except Exception:
                pass
            if not w.ok:
                try:
                    if hasattr(self, "data_source") and hasattr(
                        self.data_source, "ensure_backfill"
                    ):
                        self.data_source.ensure_backfill(required_bars=required_bars)
                        # refresh bars after backfill attempt
                        if hasattr(self.data_source, "get_last_bars"):
                            try:
                                df = self.data_source.get_last_bars(required_bars)
                            except Exception:
                                df = None
                        have_bars = len(df) if isinstance(df, pd.DataFrame) else 0
                        if have_bars >= required_bars:
                            w = warmup_status(have_bars, required_bars)
                            plan["features"] = {
                                "reasons": w.reasons,
                                "have_bars": have_bars,
                                "need_bars": required_bars,
                            }
                        else:
                            return plan_block(
                                "insufficient_bars",
                                bar_count=have_bars,
                                need_bars=required_bars,
                            )
                    else:
                        return plan_block(
                            "insufficient_bars",
                            bar_count=have_bars,
                            need_bars=required_bars,
                        )
                except Exception:
                    return plan_block(
                        "insufficient_bars",
                        bar_count=have_bars,
                        need_bars=required_bars,
                    )
            if current_price is None or current_price <= 0:
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
            adx_series, di_plus, di_minus = self._extract_adx_columns(spot_df)
            if adx_series is None or di_plus is None or di_minus is None:
                try:
                    adx_series, di_plus, di_minus = calculate_adx(spot_df)
                except Exception:
                    adx_series = di_plus = di_minus = None
            bb_width = spot_df.get("bb_width")
            if bb_width is None:
                try:
                    bb_width = calculate_bb_width(spot_df["close"], use_percentage=True)
                except Exception:
                    bb_width = None
            plan["adx"] = float(adx_series.iloc[-1]) if adx_series is not None else None
            reg = detect_market_regime(
                df=df,
                adx=adx_series,
                di_plus=di_plus,
                di_minus=di_minus,
                bb_width=bb_width,
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
                    and rsi_val >= 45
                    and rsi_rising
                    and breakout_dist_long >= 0.10
                )
                short_ok = (
                    price < float(vwap.iloc[-1])
                    and ema21_val < ema50_val
                    and ema21_slope < 0
                    and macd_val < 0
                    and rsi_val <= 55
                    and not rsi_rising
                    and breakout_dist_short >= 0.10
                )
                if long_ok or short_ok:
                    side = "BUY" if long_ok else "SELL"
                    option_type = "CE" if long_ok else "PE"
                    reasons.append("trend_playbook")
                else:
                    side = "BUY" if price >= float(vwap.iloc[-1]) else "SELL"
                    option_type = "CE" if side == "BUY" else "PE"
                    reasons.append("trend_fallback")
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
                    and rsi_val > 60
                    and rsi_val < rsi_prev
                    and float(df["close"].iloc[-1]) < float(df["open"].iloc[-1])
                )
                lower_fade = (
                    (close <= lower or close <= vwap_val - 1.9 * std_val)
                    and rsi_val < 40
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
                    side = "SELL" if close >= vwap_val else "BUY"
                    option_type = "PE" if side == "SELL" else "CE"
                    reasons.append("range_fallback")
            else:
                return plan_block("regime_no_trade")

            atr_pct = float((atr_val / price) * 100.0)
            plan["atr_pct"] = round(atr_pct, 2)
            self.last_atr_pct = float(plan["atr_pct"])
            atr_min = min(float(cfg.atr_min), _resolve_min_atr_pct())
            atr_max = float(cfg.atr_max)
            atr_pct_val = float(plan["atr_pct"])
            if not (atr_min <= atr_pct_val <= atr_max):
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
                if (side == "BUY" and obv_rising) or (
                    side == "SELL" and not obv_rising
                ):
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

            vol_score = 1 if atr_min <= atr_pct_val <= atr_max else 0

            ds = getattr(getattr(self, "runner", None), "data_source", None) or getattr(
                self, "data_source", None
            )
            quote_id: Any | None = None
            option_token = None
            lot_sz = int(
                getattr(getattr(settings, "instruments", object()), "nifty_lot_size", 75)
            )
            if ds is not None:
                idx = 0 if str(option_type).upper() == "CE" else 1
                strike_ds = getattr(ds, "current_atm_strike", None)
                need_resolve = strike_ds is None or abs(price - float(strike_ds)) >= 75
                now_dt = datetime.now(ZoneInfo("Asia/Kolkata"))
                expiry_ds = getattr(ds, "current_atm_expiry", None)
                last_resolved = getattr(ds, "_atm_resolve_date", None)
                if getattr(expiry_ds, "month", now_dt.month) != now_dt.month:
                    need_resolve = True
                if now_dt.weekday() == 1 and last_resolved != now_dt.date():
                    need_resolve = True
                if need_resolve:
                    try:
                        ds.ensure_atm_tokens()
                    except Exception:
                        logger.debug("ensure_atm_tokens failed", exc_info=True)
                    strike_ds = getattr(ds, "current_atm_strike", strike_ds)
                tokens_ds = getattr(ds, "atm_tokens", None)
                if (
                    isinstance(tokens_ds, (list, tuple))
                    and len(tokens_ds) == 2
                    and tokens_ds[idx] is not None
                ):
                    option_token = tokens_ds[idx]
                    quote_id = option_token
                    res = _token_to_symbol_and_lot(
                        getattr(settings, "kite", None), option_token
                    )
                    if res:
                        _, lot_sz = res
                else:
                    symbols_ds = getattr(ds, "atm_tradingsymbols", None)
                    if (
                        isinstance(symbols_ds, (list, tuple))
                        and len(symbols_ds) == 2
                        and symbols_ds[idx]
                    ):
                        quote_id = symbols_ds[idx]
                if strike_ds is not None:
                    plan["atm_strike"] = int(strike_ds)
            plan["option_token"] = option_token

            if quote_id is None:
                atm = resolve_weekly_atm(price)
                info_atm = atm.get(option_type.lower()) if atm else None
                if not info_atm:
                    return plan_block(
                        "no_option_token", micro={"spread_pct": None, "depth_ok": None}
                    )
                quote_id, lot_sz = info_atm

            # Fetch quote/depth using whichever identifier we have
            q = fetch_quote_with_depth(getattr(settings, "kite", None), quote_id)
            plan["_last_quote"] = q
            mid = (q.get("bid", 0.0) + q.get("ask", 0.0)) / 2.0
            cap_pct = cap_for_mid(mid, cfg)
            micro = evaluate_micro(q, lot_size=lot_sz, atr_pct=atr_pct_val, cfg=cfg)
            if not isinstance(micro, dict):
                micro = {}
            micro["cap_pct"] = cap_pct
            plan["micro"] = micro
            sp = micro.get("spread_pct") if isinstance(micro, dict) else None
            cap = micro.get("cap_pct") if isinstance(micro, dict) else None
            depth_ok = bool(micro.get("depth_ok")) if isinstance(micro, dict) else False
            over_spread = bool(sp is not None and cap is not None and sp > cap)
            ok_micro = not (over_spread or not depth_ok)
            if micro.get("mode") == "HARD" and not ok_micro:
                return plan_block("microstructure", micro=micro)

            comps = {
                "trend": float(regime_score),
                "momentum": float(momentum_score),
                "pullback": float(structure_score),
                "breakout": float(vol_score),
            }
            strat_cfg = getattr(cfg, "raw", {}).get("strategy", {})  # type: ignore[arg-type]
            weights = strat_cfg.get(
                "weights",
                {"trend": 0.4, "momentum": 0.3, "pullback": 0.2, "breakout": 0.1},
            )
            raw_score = sum(comps[k] * float(weights.get(k, 0.0)) for k in comps)
            max_score = sum(float(weights.get(k, 0.0)) for k in comps)
            penalties: Dict[str, float] = {}
            m_val = plan.get("micro")
            m = m_val if isinstance(m_val, dict) else {}
            if m.get("mode") == "SOFT":
                sp2 = m.get("spread_pct")
                cap2 = m.get("cap_pct")
                if sp2 is not None and cap2 is not None and sp2 > cap2:
                    penalties["micro_spread"] = 0.1
                if m.get("depth_ok") is False:
                    penalties["micro_depth"] = 0.1
            final_score = 0.0
            if max_score > 0:
                final_score = max(0.0, raw_score - sum(penalties.values())) / max_score
            plan["score"] = final_score
            score = final_score
            plan["score_dbg"] = {
                "components": comps,
                "weights": weights,
                "raw": round(raw_score, 4),
                "penalties": penalties,
                "final": round(final_score, 4),
                "threshold": self._normalize_min_score(
                    float(strat_cfg.get("min_score", 0.35))
                ),
            }
            plan["reasons"] = reasons

            min_score_cfg = self._normalize_min_score(
                float(strat_cfg.get("min_score", 0.35))
            )
            if final_score < min_score_cfg:
                return plan_block("score_low", score=final_score, need=min_score_cfg)

            rej = self._iv_adx_reject_reason(plan, price)
            if rej:
                reason, extra = rej
                return plan_block(reason, **extra)

            entry_price = float(current_price)
            tick_size = float(
                getattr(getattr(settings, "executor", object()), "tick_size", 0.05)
            )

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
            now_dt = (
                runner_obj.now_ist if runner_obj else datetime.now(ZoneInfo(cfg.tz))
            )
            if cfg.gamma_enabled and is_tue_after(cfg.gamma_after, now=now_dt):
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
            strike_info = select_strike(price, int(score))
            if not strike_info:
                liquidity_info: Optional[Dict[str, Any]] = None
                try:
                    from src.utils import strike_selector as ss

                    liquidity_info = ss._option_info_fetcher(
                        int(round(price / 50.0) * 50)
                    )
                except Exception:
                    liquidity_info = None
                if liquidity_info is not None:
                    return plan_block("liquidity_fail")
                strike = int(round(price / 50.0) * 50)
            else:
                strike = int(strike_info.strike)

            plan.update(
                {
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
                }
            )
            # --- premium-basis equivalents ---
            _ = int(getattr(settings.instruments, "nifty_lot_size", 75))
            q = plan.get("_last_quote") or {}
            mid = q.get("mid") or (
                (
                    (q.get("bid", 0) + q.get("ask", 0)) / 2
                    if q.get("bid") and q.get("ask")
                    else q.get("ltp")
                )
            )
            if mid:
                opt_entry = float(mid)
                delta = float(plan.get("delta") or 0.5)
                delta = max(0.25, min(delta, 0.75))
                spot_entry = float(plan["entry"])
                elasticity = _clamp(
                    delta * (spot_entry / opt_entry), 0.3, 1.2
                )

                def _opt_target(spot_target: float) -> float:
                    spot_offset = spot_target - spot_entry
                    parity = 1.0 if plan["option_type"] == "CE" else -1.0
                    offset = elasticity * spot_offset * parity
                    if plan["action"] == "BUY":
                        prem = opt_entry + offset
                    else:
                        prem = opt_entry - offset
                    return round(round(prem / tick_size) * tick_size, 2)

                plan["opt_entry"] = opt_entry
                plan["opt_sl"] = (
                    _opt_target(plan["sl"]) if plan["sl"] is not None else None
                )
                plan["opt_tp1"] = (
                    _opt_target(plan["tp1"]) if plan["tp1"] is not None else None
                )
                plan["opt_tp2"] = (
                    _opt_target(plan["tp2"]) if plan["tp2"] is not None else None
                )
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
                "score_dbg": plan.get("score_dbg"),
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
            self._bt_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )
        self._bt_df.loc[ts] = {"open": o, "high": h, "low": l, "close": c, "volume": v}
        return self.generate_signal(self._bt_df)
