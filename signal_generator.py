#!/usr/bin/env python3
"""
SignalGenerator module for Nifty Scalper Bot

Calculates technical indicators, builds composite signals, adapts thresholds,
and exposes a `generate_signal` method for use by the trading loop.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

import pandas as pd

from config import Config
from utils import TechnicalIndicators, safe_float

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Generate trading signals based on multiple technical criteria."""

    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.signal_history = []  # list of dicts with past trade results
        self.adaptive_threshold = Config.SIGNAL_THRESHOLD

    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute and return all required technical indicators."""
        if df.empty or len(df) < max(
            Config.RSI_PERIOD,
            Config.EMA_SLOW,
            Config.ATR_PERIOD,
            Config.BB_PERIOD
        ):
            return {}

        out: Dict[str, float] = {}
        close = df['close']

        try:
            # RSI
            out['rsi'] = self.indicators.calculate_rsi(close, Config.RSI_PERIOD)
            # EMAs
            out['ema_fast'] = self.indicators.calculate_ema(close, Config.EMA_FAST)
            out['ema_slow'] = self.indicators.calculate_ema(close, Config.EMA_SLOW)
            # MACD
            macd = self.indicators.calculate_macd(
                close, Config.MACD_FAST, Config.MACD_SLOW, Config.MACD_SIGNAL
            )
            out.update(macd)
            # Bollinger Bands
            bb = self.indicators.calculate_bollinger_bands(
                close, Config.BB_PERIOD, Config.BB_STDDEV
            )
            out.update(bb)
            # ATR
            out['atr'] = self.indicators.calculate_atr(df, Config.ATR_PERIOD)
            # Volume SMA & ratio
            if 'volume' in df:
                vsma = df['volume'].rolling(window=Config.VOL_SMA_PERIOD).mean()
                out['volume_sma'] = float(vsma.iloc[-1] if not vsma.empty else 0.0)
                out['volume_ratio'] = (
                    df['volume'].iloc[-1] / out['volume_sma']
                    if out['volume_sma'] > 0 else 1.0
                )
            # VWAP
            if 'volume' in df:
                window = df.tail(Config.VWAP_WINDOW) if Config.VWAP_WINDOW > 0 else df
                tv = window['volume'].sum()
                out['vwap'] = (
                    (window['close'] * window['volume']).sum() / tv
                    if tv > 0 else float(close.iloc[-1])
                )

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

        return out

    def generate_trend_signal(self, ind: Dict[str, float]) -> float:
        s = 0.0
        if ind['ema_fast'] > ind['ema_slow']:
            s += 2.0
        else:
            s -= 2.0
        if ind['macd'] > ind['signal']:
            s += 1.5
        else:
            s -= 1.5
        return s

    def generate_momentum_signal(self, ind: Dict[str, float]) -> float:
        s = 0.0
        rsi = ind.get('rsi', 50.0)
        if rsi > 70:
            s -= 1.0
        elif rsi < 30:
            s += 1.0
        elif rsi > 60:
            s += 0.5
        elif rsi < 40:
            s -= 0.5
        s += 0.5 if ind.get('histogram', 0) > 0 else -0.5
        return s

    def generate_volume_signal(self, ind: Dict[str, float], price: float) -> float:
        s = 0.0
        vr = ind.get('volume_ratio', 1.0)
        if vr > 1.5:
            if price > ind.get('vwap', price):
                s += 0.5
            else:
                s -= 0.5
        return s

    def generate_mean_reversion_signal(self, ind: Dict[str, float], price: float) -> float:
        s = 0.0
        upper, mid, lower = ind.get('upper'), ind.get('middle'), ind.get('lower')
        if lower is not None and price <= lower:
            s += 1.0
        elif upper is not None and price >= upper:
            s -= 1.0
        elif price > mid:
            s += 0.3
        else:
            s -= 0.3
        return s

    def calculate_signal_strength(
        self, df: pd.DataFrame, price: float
    ) -> Tuple[float, Dict[str, float]]:
        ind = self.calculate_all_indicators(df)
        if not ind:
            return 0.0, {}
        trend = self.generate_trend_signal(ind)
        mom = self.generate_momentum_signal(ind)
        vol = self.generate_volume_signal(ind, price)
        mr = self.generate_mean_reversion_signal(ind, price)
        total = trend*0.4 + mom*0.3 + vol*0.2 + mr*0.1
        comps = {'trend': trend, 'momentum': mom, 'volume': vol, 'mean_reversion': mr, 'atr': ind.get('atr', 0)}
        return total, comps

    def adapt_threshold(self) -> None:
        if not Config.ADAPT_THRESHOLD or len(self.signal_history) < Config.PERFORMANCE_WINDOW:
            return
        wins = sum(1 for t in self.signal_history if t.get('pnl', 0) > 0)
        wr = wins / len(self.signal_history)
        if wr < 0.3:
            self.adaptive_threshold = min(Config.MAX_THRESHOLD, self.adaptive_threshold + 0.5)
        elif wr > 0.7:
            self.adaptive_threshold = max(Config.MIN_THRESHOLD, self.adaptive_threshold - 0.3)
        else:
            if self.adaptive_threshold > Config.SIGNAL_THRESHOLD:
                self.adaptive_threshold -= 0.1
            elif self.adaptive_threshold < Config.SIGNAL_THRESHOLD:
                self.adaptive_threshold += 0.1
        logger.info(f"Adaptive threshold: {self.adaptive_threshold:.2f} (WR {wr:.0%})")

    def should_trade(self, strength: float, comps: Dict[str, float]) -> Tuple[bool, str]:
        abs_s = abs(strength)
        if abs_s < self.adaptive_threshold:
            return False, f"Weak signal {abs_s:.2f} < {self.adaptive_threshold:.2f}"
        if abs_s > Config.MAX_THRESHOLD:
            t, m = comps['trend'], comps['momentum']
            if (t>0 and m< -1) or (t<0 and m>1):
                return False, "Trend/momentum divergence"
        return True, "OK"

    def get_stop_loss_target(
        self, entry: float, direction: str, atr: float
    ) -> Tuple[float, float]:
        if Config.USE_ATR_SL and Config.USE_ATR_TP and atr > 0:
            sl = entry - atr*Config.ATR_SL_MULT if direction=="BUY" else entry + atr*Config.ATR_SL_MULT
            tp = entry + atr*Config.ATR_TP_MULT if direction=="BUY" else entry - atr*Config.ATR_TP_MULT
        else:
            sl = entry*(1 - Config.SL_PERCENT/100) if direction=="BUY" else entry*(1 + Config.SL_PERCENT/100)
            tp = entry*(1 + Config.TP_PERCENT/100) if direction=="BUY" else entry*(1 - Config.TP_PERCENT/100)
        return round(sl,2), round(tp,2)

    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Public method: returns a trade dict if a valid signal, else None.
        """
        ohlc = market_data.get('ohlc')
        price = safe_float(market_data.get('ltp'))
        if not ohlc or price <= 0:
            return None

        df = pd.DataFrame(ohlc)
        strength, comps = self.calculate_signal_strength(df, price)
        can_trade, reason = self.should_trade(strength, comps)
        if not can_trade:
            logger.debug(f"Skip: {reason}")
            return None

        direction = "BUY" if strength > 0 else "SELL"
        sl, tp = self.get_stop_loss_target(price, direction, comps.get('atr', 0))
        return {
            'direction': direction,
            'strength': strength,
            'entry_price': price,
            'stop_loss': sl,
            'target': tp
        }
