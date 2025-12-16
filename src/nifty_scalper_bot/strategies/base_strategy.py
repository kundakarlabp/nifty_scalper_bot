"""
BASE STRATEGY: The "Gatekeeper" for World-Class Option Safety.
Enforces Liquidity, Greeks, and Spread checks on ALL trades.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import logging

class BaseStrategy(ABC):
    def __init__(self, config: Dict[str, Any], indicator_engine: Any):
        self.config = config
        self.ie = indicator_engine
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # ⚙️ PRODUCTION SETTINGS (Load from .env or use safe defaults)
        self.min_oi = self.config.get("MIN_OPEN_INTEREST", 50000)
        self.max_spread_pct = self.config.get("MAX_BID_ASK_SPREAD", 5.0)
        self.min_delta = self.config.get("MIN_OPTION_DELTA", 0.30)
        self.max_iv_percentile = self.config.get("MAX_IV_PERCENTILE", 80.0)

    @abstractmethod
    def generate_signal(self, tick_data: Any) -> Optional[Dict]:
        """Child strategies (ORB, RSI) must implement this."""
        pass

    def validate_option_health(self, symbol: str, direction: str) -> bool:
        """
        🛡️ GATEKEEPER: Stops the bot from buying "Garbage Options".
        Checks: Liquidity, Spread, Greeks.
        """
        # 1. FETCH DATA
        quote = self.ie.get_quote(symbol)
        greeks = self.ie.get_greeks(symbol)
        
        if not quote:
            self.logger.warning(f"⛔ {symbol}: No Quote Data. Skipping.")
            return False

        # 2. 💧 LIQUIDITY CHECK (Don't trade ghost towns)
        oi = quote.get('oi', 0)
        if oi < self.min_oi:
            self.logger.info(f"⛔ {symbol}: Low Liquidity (OI: {oi} < {self.min_oi}). Skip.")
            return False

        # 3. 📉 SPREAD CHECK (Don't pay spread tax)
        bid = quote.get('bid', 0)
        ask = quote.get('ask', 0)
        if bid > 0:
            spread_pct = ((ask - bid) / bid) * 100
            if spread_pct > self.max_spread_pct:
                self.logger.info(f"⛔ {symbol}: Spread too wide ({spread_pct:.2f}%). Skip.")
                return False

        # 4. 📐 GREEKS CHECK (Only if enabled)
        if greeks:
            # A. DELTA (Momentum): Don't buy deep OTM options
            delta = abs(greeks.get('delta', 0))
            if delta < self.min_delta:
                self.logger.info(f"⛔ {symbol}: Weak Delta ({delta:.2f}). Option won't move.")
                return False

            # B. THETA (Decay): Don't hold melting ice
            theta = greeks.get('theta', 0)
            if theta < -15.0: # Burning >15 points/day
                self.logger.info(f"⛔ {symbol}: High Theta Burn ({theta}). Risk of decay.")
                return False
                
            # C. IV (Value): Don't buy expensive tops
            iv_p = greeks.get('iv_percentile', 0)
            if iv_p > self.max_iv_percentile:
                self.logger.info(f"⛔ {symbol}: IV Too High ({iv_p}). Option is overpriced.")
                return False

        return True

    def calculate_option_rr(self, premium: float, side: str = "BUY") -> Tuple[float, float]:
        """
        💰 RISK LOGIC: Calculates SL/TP based on Premium % (Not Spot Price).
        """
        # Risk Settings (e.g., Risk 15% of premium to make 30%)
        SL_PCT = 0.15 
        TP_PCT = 0.30

        if side == "BUY":
            sl_price = round(premium * (1 - SL_PCT), 1)
            tp_price = round(premium * (1 + TP_PCT), 1)
        else:
            # Selling logic inverted
            sl_price = round(premium * (1 + SL_PCT), 1)
            tp_price = round(premium * (1 - TP_PCT), 1)
            
        return sl_price, tp_price
