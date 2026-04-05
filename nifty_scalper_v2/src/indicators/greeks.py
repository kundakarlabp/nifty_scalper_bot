"""Black-Scholes Greeks for NSE options. Uses scipy for N(d) functions."""

from __future__ import annotations

import math

from scipy.stats import norm  # type: ignore[import]


def _d1_d2(spot: float, strike: float, tte: float, iv: float, rate: float) -> tuple[float, float]:
    """Compute d1, d2. tte = time to expiry in years."""
    log_sk = math.log(spot / strike)
    d1 = (log_sk + (rate + 0.5 * iv ** 2) * tte) / (iv * math.sqrt(tte))
    d2 = d1 - iv * math.sqrt(tte)
    return d1, d2


def calc_delta(spot: float, strike: float, tte: float, iv: float, rate: float = 0.065, option_type: str = "CE") -> float | None:
    if tte <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return None
    try:
        d1, _ = _d1_d2(spot, strike, tte, iv, rate)
        if option_type.upper() == "CE":
            return float(norm.cdf(d1))
        return float(norm.cdf(d1) - 1.0)
    except Exception:
        return None


def calc_gamma(spot: float, strike: float, tte: float, iv: float, rate: float = 0.065) -> float | None:
    if tte <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return None
    try:
        d1, _ = _d1_d2(spot, strike, tte, iv, rate)
        return float(norm.pdf(d1) / (spot * iv * math.sqrt(tte)))
    except Exception:
        return None


def calc_theta(spot: float, strike: float, tte: float, iv: float, rate: float = 0.065, option_type: str = "CE") -> float | None:
    """Per calendar day (divide annual theta by 365)."""
    if tte <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return None
    try:
        d1, d2 = _d1_d2(spot, strike, tte, iv, rate)
        term1 = -(spot * norm.pdf(d1) * iv) / (2 * math.sqrt(tte))
        if option_type.upper() == "CE":
            term2 = rate * strike * math.exp(-rate * tte) * norm.cdf(d2)
            return float((term1 - term2) / 365.0)
        term2 = rate * strike * math.exp(-rate * tte) * norm.cdf(-d2)
        return float((term1 + term2) / 365.0)
    except Exception:
        return None


def calc_vega(spot: float, strike: float, tte: float, iv: float, rate: float = 0.065) -> float | None:
    """Vega per 1% change in IV."""
    if tte <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return None
    try:
        d1, _ = _d1_d2(spot, strike, tte, iv, rate)
        return float(spot * norm.pdf(d1) * math.sqrt(tte) * 0.01)
    except Exception:
        return None


def calc_iv(option_price: float, spot: float, strike: float, tte: float, rate: float = 0.065, option_type: str = "CE") -> float | None:
    """Newton-Raphson implied volatility solver."""
    if tte <= 0 or option_price <= 0 or spot <= 0 or strike <= 0:
        return None
    try:
        iv = 0.3  # initial guess
        for _ in range(100):
            d1, d2 = _d1_d2(spot, strike, tte, iv, rate)
            if option_type.upper() == "CE":
                price = spot * norm.cdf(d1) - strike * math.exp(-rate * tte) * norm.cdf(d2)
            else:
                price = strike * math.exp(-rate * tte) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            vega_raw = spot * norm.pdf(d1) * math.sqrt(tte)
            if vega_raw < 1e-10:
                break
            diff = option_price - price
            iv = iv + diff / vega_raw
            if abs(diff) < 1e-6:
                return max(0.001, float(iv))
        return max(0.001, float(iv)) if 0.001 <= iv <= 5.0 else None
    except Exception:
        return None
