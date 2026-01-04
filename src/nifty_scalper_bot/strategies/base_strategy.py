"""
BASE STRATEGY: The "Gatekeeper" for World-Class Option Safety.
Refactored for Push-Based Architecture and strict Null-Safety.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import logging

class BaseStrategy(ABC):
    def __init__(self, config: Dict[str, Any], indicator_engine: Any):
        self.config = config
        self.ie = indicator_engine
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # ⚙️ PRODUCTION SETTINGS: Use safe defaults with strict types
        self.min_oi = int(self.config.get("MIN_OPEN_INTEREST", 50000))
        self.max_spread_pct = float(self.config.get("MAX_BID_ASK_SPREAD", 5.0))
        self.min_delta = float(self.config.get("MIN_OPTION_DELTA", 0.30))
        self.max_iv_percentile = float(self.config.get("MAX_IV_PERCENTILE", 80.0))

    @abstractmethod
    def generate_signal(self, tick_data: Any) -> Optional[Dict]:
        pass

    def validate_option_health(self, symbol: str, direction: str) -> bool:
        """🛡️ GATEKEEPER: Validates Liquidity, Spread, and Greeks."""
        # ✅ IMPROVEMENT: Fetch all data in one 'snapshot' if engine supports it
        # This prevents price/greek desync
        quote = self.ie.get_quote(symbol) or {}
        greeks = self.ie.get_greeks(symbol) or {}
        
        if not quote:
            self.logger.warning(f"⛔ {symbol}: Missing Quote. Skipping.")
            return False

        # 💧 LIQUIDITY CHECK
        oi = quote.get('oi', 0)
        if oi < self.min_oi:
            self.logger.info(f"⛔ {symbol}: Low OI ({oi} < {self.min_oi})")
            return False

        # 📉 SPREAD CHECK
        bid = float(quote.get('bid', 0))
        ask = float(quote.get('ask', 0))
        
        # ✅ FIX: Handle zero bid or ask properly
        if bid <= 0 or ask <= 0:
            self.logger.warning(f"⛔ {symbol}: Invalid Bid/Ask ({bid}/{ask})")
            return False
            
        spread_pct = ((ask - bid) / bid) * 100
        if spread_pct > self.max_spread_pct:
            self.logger.info(f"⛔ {symbol}: Wide Spread ({spread_pct:.2f}%)")
            return False

        # 📐 GREEKS CHECK
        if greeks:
            # DELTA (Momentum)
            delta = abs(float(greeks.get('delta', 0)))
            if delta < self.min_delta:
                self.logger.info(f"⛔ {symbol}: Weak Delta ({delta:.2f})")
                return False

            # THETA (Decay)
            theta = float(greeks.get('theta', 0))
            if direction == "BUY" and theta < -15.0: 
                self.logger.info(f"⛔ {symbol}: High Decay ({theta})")
                return False
                
            # IV (Value)
            iv_p = float(greeks.get('iv_percentile', 0))
            if direction == "BUY" and iv_p > self.max_iv_percentile:
                self.logger.info(f"⛔ {symbol}: IV Overpriced ({iv_p})")
                return False

        return True

    def calculate_option_rr(self, premium: float, side: str = "BUY") -> Tuple[float, float]:
        """💰 RISK LOGIC: SL/TP based on Premium %."""
        # ✅ IMPROVEMENT: Pull SL/TP percentages from config
        SL_PCT = float(self.config.get("OPTION_SL_PCT", 0.15))
        TP_PCT = float(self.config.get("OPTION_TP_PCT", 0.30))

        if side == "BUY":
            sl_price = round(premium * (1 - SL_PCT), 1)
            tp_price = round(premium * (1 + TP_PCT), 1)
        else:
            sl_price = round(premium * (1 + SL_PCT), 1)
            tp_price = round(premium * (1 - TP_PCT), 1)
            
        # ✅ FIX: Ensure SL/TP never falls below minimum tick size (0.05)
        return max(0.05, sl_price), max(0.05, tp_price)
