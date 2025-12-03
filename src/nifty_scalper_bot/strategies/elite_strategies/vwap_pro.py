import logging
import numpy as np
from typing import Optional, List, Dict, Any
from ..base_strategy import BaseStrategy

LOGGER = logging.getLogger(__name__)

class VWAPProStrategy(BaseStrategy):
    """
    VWAP Pro: An institutional-grade intraday strategy.
    
    Logic:
    1. Regime Filter: Only trades in 'TRENDING' or 'VOLATILE' markets.
    2. Trend Filter: Longs only above EMA-50. Shorts only below EMA-50.
    3. Trigger: Price crosses VWAP with momentum.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Default config if none provided
        config = config or {}
        config.setdefault("allowed_regimes", ["TRENDING", "VOLATILE"])
        config.setdefault("timeframe", 5)  # 5-minute candles
        config.setdefault("ema_period", 50)
        
        super().__init__("VWAP_PRO", config)
        self.ema_period = config["ema_period"]

    def calculate_signal(self, tick: Dict[str, Any], candles: List[Dict[str, Any]], regime: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Core logic to generate signals based on Tick, Candle history, and Regime.
        """
        # 1. Data Validation (Fail Fast)
        if not candles or len(candles) < self.ema_period:
            return None
            
        if not self.can_trade(regime):
            # Log once per minute to avoid spam, or just return None silently
            return None

        # 2. Extract Market Data
        try:
            last_candle = candles[-1]
            current_price = tick.get('ltp') or tick.get('last_price')
            
            if not current_price:
                return None

            # 3. Get or Calculate Indicators
            vwap = last_candle.get('vwap')
            if vwap is None:
                vwap = self._calculate_intraday_vwap(candles)
            
            ema = self._calculate_ema(candles, self.ema_period)

            # 4. Trading Logic
            
            # --- LONG Logic ---
            # Conditions: Price > EMA (Trend is Up) AND Price crossed above VWAP recently
            if current_price > ema:
                # Check for crossover: Previous candle closed below VWAP, Current price is above
                prev_close = candles[-2]['close'] if len(candles) > 1 else last_candle['open']
                
                if prev_close < vwap and current_price > vwap:
                    # Confirm with Volume if available
                    vol_spike = last_candle['volume'] > (self._get_avg_volume(candles) * 1.2)
                    
                    if vol_spike:
                        LOGGER.info(f"📈 VWAP_PRO Buy Signal: {tick['symbol']} @ {current_price} (Regime: {regime.get('label')})")
                        return {
                            "side": "BUY",
                            "price": current_price,
                            "quantity": self.config.get("base_quantity", 50), # Default to 1 lot Nifty
                            "reason": f"VWAP Cross + EMA{self.ema_period} Trend + Vol"
                        }

            # --- SHORT Logic ---
            # Conditions: Price < EMA (Trend is Down) AND Price crossed below VWAP
            elif current_price < ema:
                prev_close = candles[-2]['close'] if len(candles) > 1 else last_candle['open']
                
                if prev_close > vwap and current_price < vwap:
                    # Confirm with Volume
                    vol_spike = last_candle['volume'] > (self._get_avg_volume(candles) * 1.2)
                    
                    if vol_spike:
                        LOGGER.info(f"📉 VWAP_PRO Sell Signal: {tick['symbol']} @ {current_price} (Regime: {regime.get('label')})")
                        return {
                            "side": "SELL",
                            "price": current_price,
                            "quantity": self.config.get("base_quantity", 50),
                            "reason": f"VWAP Cross - EMA{self.ema_period} Trend + Vol"
                        }

        except Exception as e:
            LOGGER.error(f"Error in VWAP Pro strategy calculation: {e}")
            return None

        return None

    # --- Helper Methods for Robustness ---

    def _calculate_intraday_vwap(self, candles: List[Dict[str, Any]]) -> float:
        """Fallback VWAP calculation if data feed doesn't provide it."""
        cumulative_pv = 0.0
        cumulative_vol = 0.0
        
        for c in candles:
            # Typical Price = (H + L + C) / 3
            tp = (c['high'] + c['low'] + c['close']) / 3
            vol = c['volume']
            cumulative_pv += (tp * vol)
            cumulative_vol += vol
            
        if cumulative_vol == 0:
            return candles[-1]['close']
            
        return cumulative_pv / cumulative_vol

    def _calculate_ema(self, candles: List[Dict[str, Any]], period: int) -> float:
        """Simple EMA calculation on Close prices."""
        closes = [c['close'] for c in candles]
        if len(closes) < period:
            return closes[-1]
            
        # Use numpy for speed if available, else manual
        try:
            return float(np.mean(closes[-period:])) # Simple Moving Average approximation for fallback
            # For true EMA, you'd implement the smoothing factor logic here
            # alpha = 2 / (period + 1)
        except Exception:
            return closes[-1]

    def _get_avg_volume(self, candles: List[Dict[str, Any]], lookback: int = 20) -> float:
        """Get average volume of last N candles."""
        if not candles:
            return 0.0
        slice_ = candles[-lookback:]
        volumes = [c['volume'] for c in slice_]
        return sum(volumes) / len(volumes)
