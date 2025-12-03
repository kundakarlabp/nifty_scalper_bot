from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseStrategy(ABC):
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.allowed_regimes = config.get("allowed_regimes", ["TRENDING", "RANGE"])
        self.timeframe_min = config.get("timeframe", 5)
        self.last_candle_time = None

    def can_trade(self, regime: dict) -> bool:
        """
        Gatekeeper: Checks if the current market regime favors this strategy.
        """
        if not regime:
            return True # Default to allow if no regime data
        
        current_regime = regime.get("label", "UNKNOWN")
        
        # 1. Regime Check
        if current_regime not in self.allowed_regimes:
            return False
            
        # 2. Volatility Check (Optional)
        volatility = regime.get("volatility", 0)
        if volatility < self.config.get("min_volatility", 0):
            return False
            
        return True

    @abstractmethod
    def calculate_signal(self, tick: dict, candles: list, regime: dict) -> Optional[dict]:
        """
        Core Logic. Must return dict with {side, quantity, price} or None.
        """
        pass
