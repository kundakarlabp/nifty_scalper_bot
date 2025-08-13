# src/utils/greeks.py
from __future__ import annotations
"""
Lightweight Black–Scholes Greeks utilities for options on indices/stocks.

- Works with CE/PE (call/put)
- Inputs:
    spot: current underlying price (S)
    strike: option strike (K)
    days_to_expiry: calendar days to expiry (float); internally converted to years
    iv: implied volatility as a decimal (e.g. 0.20 = 20%)
    r: risk-free rate (annualized, decimal). Default 0.0 (or set ~0.06 for INR).
    q: dividend yield (annualized, decimal). Default 0.0.

Key functions:
    bs_price(...)
    bs_delta(...), bs_gamma(...), bs_theta(...), bs_vega(...)
    implied_vol_bisection(...)
    estimate_delta(...)            # convenience wrapper used by ranking
    estimate_all_greeks(...)       # returns a dict of greeks

All functions are defensive: inputs are clamped to safe minimums to avoid NaNs.
"""

import math
from typing import Literal, Tuple, Dict

OptType = Literal["CE", "PE", "CALL", "PUT"]


# --------------------------- Normal distribution --------------------------- #

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    # Stable CDF via error function
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------- Core Black–Scholes ---------------------------- #

def _clamp_inputs(
    spot: float, strike: float, days_to_expiry: float, iv: float, r: float, q: float
) -> Tuple[float, float, float, float, float, float]:
    S = max(1e-9, float(spot))
    K = max(1e-9, float(strike))
    # Convert to years; ensure strictly positive to avoid division by zero
    T = max(1e-6, float(days_to_expiry) / 365.0)
    sigma = max(1e-6, float(iv))
    r = float(r)
    q = float(q)
    return S, K, T, sigma, r, q


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float]:
    # d1, d2 per Black–Scholes with continuous dividend yield q
    num = math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T
    den = sigma * math.sqrt(T)
    d1 = num / den
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(
    spot: float,
    strike: float,
    days_to_expiry: float,
    iv: float,
    *,
    opt_type: OptType,
    r: float = 0.0,
    q: float = 0.0,
) -> float:
    """Black–Scholes price for European options (continuous dividend yield)."""
    S, K, T, sigma, r, q = _clamp_inputs(spot, strike, days_to_expiry, iv, r, q)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    t = str(opt_type).upper()
    if t in ("CE", "CALL"):
        return disc_q * S * _norm_cdf(d1) - disc_r * K * _norm_cdf(d2)
    else:  # PE/PUT
        return disc_r * K * _norm_cdf(-d2) - disc_q * S * _norm_cdf(-d1)


def bs_delta(
    spot: float,
    strike: float,
    days_to_expiry: float,
    iv: float,
    *,
    opt_type: OptType,
    r: float = 0.0,
    q: float = 0.0,
) -> float:
    """Delta in [-1, 1]."""
    S, K, T, sigma, r, q = _clamp_inputs(spot, strike, days_to_expiry, iv, r, q)
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)
    t = str(opt_type).upper()
    if t in ("CE", "CALL"):
        return disc_q * _norm_cdf(d1)
    else:
        return disc_q * (_norm_cdf(d1) - 1.0)


def bs_gamma(
    spot: float,
    strike: float,
    days_to_expiry: float,
    iv: float,
    *,
    r: float = 0.0,
    q: float = 0.0,
) -> float:
    """Gamma (per 1 unit of underlying)."""
    S, K, T, sigma, r, q = _clamp_inputs(spot, strike, days_to_expiry, iv, r, q)
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)
    return disc_q * _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_theta(
    spot: float,
    strike: float,
    days_to_expiry: float,
    iv: float,
    *,
    opt_type: OptType,
    r: float = 0.0,
    q: float = 0.0,
    per_day: bool = True,
) -> float:
    """Theta (time decay). By default returns *per day*."""
    S, K, T, sigma, r, q = _clamp_inputs(spot, strike, days_to_expiry, iv, r, q)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    first = -disc_q * S * _norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
    t = str(opt_type).upper()
    if t in ("CE", "CALL"):
        second = q * disc_q * S * _norm_cdf(d1)
        third = -r * disc_r * K * _norm_cdf(d2)
    else:
        second = -q * disc_q * S * _norm_cdf(-d1)
        third = r * disc_r * K * _norm_cdf(-d2)

    theta_annual = first + second + third
    if per_day:
        return theta_annual / 365.0
    return theta_annual


def bs_vega(
    spot: float,
    strike: float,
    days_to_expiry: float,
    iv: float,
    *,
    r: float = 0.0,
    q: float = 0.0,
    per_1pct: bool = True,
) -> float:
    """Vega: sensitivity to volatility. If per_1pct=True, returns change per +1% vol."""
    S, K, T, sigma, r, q = _clamp_inputs(spot, strike, days_to_expiry, iv, r, q)
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)
    vega = disc_q * S * _norm_pdf(d1) * math.sqrt(T)  # per 1.00 volatility
    return vega / 100.0 if per_1pct else vega


# --------------------------- Implied volatility ---------------------------- #

def implied_vol_bisection(
    target_price: float,
    spot: float,
    strike: float,
    days_to_expiry: float,
    *,
    opt_type: OptType,
    r: float = 0.0,
    q: float = 0.0,
    low: float = 1e-4,
    high: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Robust IV solver by bisection. Returns IV as decimal (e.g., 0.20).
    If target is out of theoretical bounds, clamps to [low, high].
    """
    target = max(0.0, float(target_price))
    lo, hi = float(low), float(high)

    # Check bounds
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price = bs_price(spot, strike, days_to_expiry, mid, opt_type=opt_type, r=r, q=q)
        if abs(price - target) <= tol:
            return max(low, min(high, mid))
        if price > target:
            hi = mid
        else:
            lo = mid
    return max(low, min(high, 0.5 * (lo + hi)))


# -------------------------- Convenience wrappers --------------------------- #

def estimate_delta(
    spot: float, strike: float, days_to_expiry: float, iv: float, opt_type: OptType
) -> float:
    """
    Convenience wrapper used by strike ranking. Assumes r=q=0 for simplicity.
    """
    try:
        return float(bs_delta(spot, strike, days_to_expiry, iv, opt_type=opt_type, r=0.0, q=0.0))
    except Exception:
        return 0.0


def estimate_all_greeks(
    spot: float,
    strike: float,
    days_to_expiry: float,
    iv: float,
    *,
    opt_type: OptType,
    r: float = 0.0,
    q: float = 0.0,
) -> Dict[str, float]:
    """
    Returns a dict with price and Greeks. Theta is per-day; Vega is per +1% vol.
    """
    try:
        return {
            "price": bs_price(spot, strike, days_to_expiry, iv, opt_type=opt_type, r=r, q=q),
            "delta": bs_delta(spot, strike, days_to_expiry, iv, opt_type=opt_type, r=r, q=q),
            "gamma": bs_gamma(spot, strike, days_to_expiry, iv, r=r, q=q),
            "theta_per_day": bs_theta(spot, strike, days_to_expiry, iv, opt_type=opt_type, r=r, q=q, per_day=True),
            "vega_per_1pct": bs_vega(spot, strike, days_to_expiry, iv, r=r, q=q, per_1pct=True),
        }
    except Exception:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta_per_day": 0.0, "vega_per_1pct": 0.0}


# ------------------------------- Quick demo -------------------------------- #

if __name__ == "__main__":
    # Example: NIFTY-like numbers
    S = 24600.0
    K = 24600.0
    dte = 7.0          # days
    iv = 0.20          # 20%
    r = 0.0            # set ~0.06 if you prefer INR risk-free
    q = 0.0

    for typ in ("CE", "PE"):
        print(f"\n{typ} @ S={S}, K={K}, dte={dte}d, iv={iv*100:.1f}%")
        print("price:", bs_price(S, K, dte, iv, opt_type=typ, r=r, q=q))
        print("delta:", bs_delta(S, K, dte, iv, opt_type=typ, r=r, q=q))
        print("gamma:", bs_gamma(S, K, dte, iv, r=r, q=q))
        print("theta/day:", bs_theta(S, K, dte, iv, opt_type=typ, r=r, q=q, per_day=True))
        print("vega/1%:", bs_vega(S, K, dte, iv, r=r, q=q, per_1pct=True))