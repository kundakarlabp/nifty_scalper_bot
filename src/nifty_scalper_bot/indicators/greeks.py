"""Black-Scholes Greeks for Nifty Options"""
import math
from dataclasses import dataclass

try:
    from scipy.stats import norm
except ImportError:
    # Fallback for lightweight environments
    import numpy as np
    class norm:
        @staticmethod
        def cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        @staticmethod
        def pdf(x):
            return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

from nifty_scalper_bot.utils.logging import get_logger
LOGGER = get_logger(__name__)

@dataclass
class OptionGreeks:
    delta: float
    gamma: float
    theta: float  # Daily
    vega: float
    rho: float

class BlackScholesCalculator:
    def __init__(self, risk_free_rate: float = 0.065):
        self.r = risk_free_rate

    def calculate_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        implied_vol: float,
        option_type: str
    ) -> OptionGreeks:
        """Calculate all Greeks."""
        if time_to_expiry_years <= 0:
            return self._expired_greeks(spot, strike, option_type)

        d1 = (math.log(spot / strike) + (self.r + 0.5 * implied_vol**2) * time_to_expiry_years) / (implied_vol * math.sqrt(time_to_expiry_years))
        d2 = d1 - implied_vol * math.sqrt(time_to_expiry_years)

        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)

        if option_type.upper() == 'CE':
            delta = N_d1
            theta_annual = (-(spot * n_d1 * implied_vol) / (2 * math.sqrt(time_to_expiry_years)) - self.r * strike * math.exp(-self.r * time_to_expiry_years) * N_d2)
        else:  # PE
            delta = N_d1 - 1
            theta_annual = (-(spot * n_d1 * implied_vol) / (2 * math.sqrt(time_to_expiry_years)) + self.r * strike * math.exp(-self.r * time_to_expiry_years) * norm.cdf(-d2))

        gamma = n_d1 / (spot * implied_vol * math.sqrt(time_to_expiry_years))
        vega = spot * n_d1 * math.sqrt(time_to_expiry_years) / 100
        rho = 0 # Simplified

        return OptionGreeks(delta=delta, gamma=gamma, theta=theta_annual / 365, vega=vega, rho=rho)

    def _expired_greeks(self, spot, strike, option_type) -> OptionGreeks:
        delta = 1.0 if (option_type == 'CE' and spot > strike) else 0.0
        return OptionGreeks(delta=delta, gamma=0, theta=0, vega=0, rho=0)
