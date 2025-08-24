# Path: src/strategies/scalping_strategy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

from src.config import settings


@dataclass
class TradingSignal:
    action: str               # "BUY" or "SELL"
    option_type: str          # "CE" or "PE"
    strike: float
    entry_price: float
    stop_loss: float
    take_profit: float
    score: float
    confidence: float         # 0..100
    rr: float
    regime: str               # "TREND" | "RANGE"
    reasons: List[str]


class EnhancedScalpingStrategy:
    def __init__(self) -> None:
        self.min_bars = int(settings.strategy_min_bars_for_signal)
        self.ema_fast = int(settings.strategy_ema_fast)
        self.ema_slow = int(settings.strategy_ema_slow)
        self.rsi_period = int(settings.strategy_rsi_period)
        self.bb_period = int(settings.strategy_bb_period)
        self.bb_std = float(settings.strategy_bb_std)
        self.atr_period = int(settings.strategy_atr_period)
        self.min_score = float(settings.strategy_min_signal_score)
        self.conf_thresh = float(settings.strategy_confidence_threshold)
        self.sl_mult = float(settings.strategy_atr_sl_multiplier)
        self.tp_mult = float(settings.strategy_atr_tp_multiplier)

        # last debug snapshot for /check
        self._last_debug: Dict = {"note": "no_evaluation_yet"}

    def get_debug(self) -> Dict:
        return dict(self._last_debug)

    def generate_signal(self, df: pd.DataFrame, current_tick: Optional[Dict] = None) -> Optional[TradingSignal]:
        dbg = {
            "ok": False,
            "reason_block": None,
            "bars": int(len(df) if isinstance(df, pd.DataFrame) else 0),
            "features": {},
            "scores": {},
            "selected_side": None,
            "score": None,
            "confidence": None,
            "rr": None,
            "regime": None,
            "reasons": [],
            "gates": {"min_bars": False, "nan_free": False, "score_gate": False, "confidence_gate": False, "rr_gate": False},
        }
        try:
            if df is None or not isinstance(df, pd.DataFrame) or len(df) < self.min_bars:
                dbg["reason_block"] = "min_bars"
                self._last_debug = dbg
                return None
            dbg["gates"]["min_bars"] = True

            df = df.copy().dropna(subset=["open", "high", "low", "close"]).sort_index()

            self._ema(df, self.ema_fast, "ema_fast")
            self._ema(df, self.ema_slow, "ema_slow")
            self._rsi(df, self.rsi_period, "rsi")
            self._bb(df, self.bb_period, self.bb_std, "bb_mid", "bb_up", "bb_dn")
            self._atr(df, self.atr_period, "atr")
            self._vwap(df, "vwap")

            last = df.iloc[-1]
            if any(pd.isna([last.close, last.ema_fast, last.ema_slow, last.rsi, last.atr, last.vwap])):
                dbg["reason_block"] = "nan_features"
                self._last_debug = dbg
                return None
            dbg["gates"]["nan_free"] = True

            dbg["features"] = {
                "close": float(last.close), "ema_fast": float(last.ema_fast), "ema_slow": float(last.ema_slow),
                "rsi": float(last.rsi), "atr": float(last.atr), "vwap": float(last.vwap), "bb_mid": float(last.bb_mid),
            }

            regime = self._regime(df); dbg["regime"] = regime

            slong, rlong = self._score_long(df)
            sshort, rshort = self._score_short(df)
            dbg["scores"] = {"long": slong, "short": sshort}

            if slong >= sshort:
                action, opt, score, reasons = "BUY", "CE", slong, rlong
                sl = last.close - self.sl_mult * last.atr
                tp = last.close + self.tp_mult * last.atr
                dbg["selected_side"] = "long"
            else:
                action, opt, score, reasons = "SELL", "PE", sshort, rshort
                sl = last.close + self.sl_mult * last.atr
                tp = last.close - self.tp_mult * last.atr
                dbg["selected_side"] = "short"

            sl_dist = abs(last.close - sl); tp_dist = abs(tp - last.close)
            if sl_dist <= 0 or tp_dist <= 0:
                dbg["reason_block"] = "rr_degenerate"; self._last_debug = dbg; return None
            rr = round(tp_dist / sl_dist, 2); dbg["rr"] = rr; dbg["gates"]["rr_gate"] = True

            confidence = max(0.0, min(100.0, score * 20.0))
            dbg["score"] = float(round(score, 2)); dbg["confidence"] = float(round(confidence, 1)); dbg["reasons"] = reasons[:6]

            if score < self.min_score:
                dbg["reason_block"] = "score_gate"; self._last_debug = dbg; return None
            dbg["gates"]["score_gate"] = True

            if confidence < self.conf_thresh:
                dbg["reason_block"] = "confidence_gate"; self._last_debug = dbg; return None
            dbg["gates"]["confidence_gate"] = True

            strike = self._round(last.close, 50)
            entry = float(last.close)

            sig = TradingSignal(
                action=action, option_type=opt, strike=float(strike),
                entry_price=float(entry), stop_loss=float(sl), take_profit=float(tp),
                score=float(round(score, 2)), confidence=float(round(confidence, 1)), rr=float(rr),
                regime=regime, reasons=reasons[:6],
            )
            dbg["ok"] = True; dbg["reason_block"] = None; self._last_debug = dbg
            return sig

        except Exception as e:
            dbg["reason_block"] = f"exception:{e.__class__.__name__}"
            self._last_debug = dbg
            return None

    @staticmethod
    def _ema(df: pd.DataFrame, n: int, c: str) -> None:
        df[c] = df["close"].ewm(span=n, adjust=False).mean()

    @staticmethod
    def _rsi(df: pd.DataFrame, n: int, c: str) -> None:
        delta = df["close"].diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = (-delta.clip(upper=0)).rolling(n).mean()
        rs = up / (down.replace(0, np.nan))
        df[c] = 100 - (100 / (1 + rs))
        df[c] = df[c].fillna(50.0)

    @staticmethod
    def _bb(df: pd.DataFrame, n: int, std: float, mid: str, up: str, dn: str) -> None:
        ma = df["close"].rolling(n).mean()
        sd = df["close"].rolling(n).std(ddof=0)
        df[mid] = ma; df[up] = ma + std * sd; df[dn] = ma - std * sd

    @staticmethod
    def _atr(df: pd.DataFrame, n: int, c: str) -> None:
        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift(1)).abs()
        lc = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df[c] = tr.rolling(n).mean().fillna(method="bfill")

    @staticmethod
    def _vwap(df: pd.DataFrame, c: str) -> None:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].replace(0, np.nan).fillna(method="ffill")
        df[c] = (tp * vol).cumsum() / vol.cumsum()

    def _regime(self, df: pd.DataFrame) -> str:
        slow = df["ema_slow"]; fast = df["ema_fast"]
        if len(df) < max(self.ema_slow, self.ema_fast) + 2:
            return "RANGE"
        slope = (slow.iloc[-1] - slow.iloc[-3])
        spread = abs(fast.iloc[-1] - slow.iloc[-1])
        atr = df["atr"].iloc[-1]
        return "TREND" if (slope * 10 > 0.1 * atr and spread > 0.1 * atr) else "RANGE"

    def _score_long(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        last = df.iloc[-1]; score = 0.0; reasons: List[str] = []
        if last.ema_fast > last.ema_slow: score += 1.0; reasons.append("EMA fast>slow")
        if last.close > last.vwap: score += 0.5; reasons.append("Close>VWAP")
        if last.close > last.bb_mid: score += 0.5; reasons.append("Close>BB mid")
        if last.rsi >= 55: score += 0.75; reasons.append("RSI>=55")
        if last.rsi >= 60: score += 0.25; reasons.append("RSI>=60")
        if last.atr > 0 and (last.atr / max(1e-6, last.close)) > 0.002: score += 0.25; reasons.append("ATR ok")
        if self._regime(df) == "TREND" and last.ema_fast > last.ema_slow: score += 0.5; reasons.append("Trend-aligned")
        return round(score, 2), reasons

    def _score_short(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        last = df.iloc[-1]; score = 0.0; reasons: List[str] = []
        if last.ema_fast < last.ema_slow: score += 1.0; reasons.append("EMA fast<slow")
        if last.close < last.vwap: score += 0.5; reasons.append("Close<VWAP")
        if last.close < last.bb_mid: score += 0.5; reasons.append("Close<BB mid")
        if last.rsi <= 45: score += 0.75; reasons.append("RSI<=45")
        if last.rsi <= 40: score += 0.25; reasons.append("RSI<=40")
        if last.atr > 0 and (last.atr / max(1e-6, last.close)) > 0.002: score += 0.25; reasons.append("ATR ok")
        if self._regime(df) == "TREND" and last.ema_fast < last.ema_slow: score += 0.5; reasons.append("Trend-aligned")
        return round(score, 2), reasons

    @staticmethod
    def _round(x: float, step: int) -> float:
        return float(int(round(x / step) * step)) if step > 0 else float(x)
