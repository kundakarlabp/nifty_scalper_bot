# src/utils/greeks.py
from __future__ import annotations

import math
from typing import Optional


# ---------- Black–Scholes utilities (simplified) ----------

def _norm_cdf(x: float) -> float:
    # Using error function for CDF of standard normal
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _bs_d1(spot: float, strike: float, t: float, vol: float, r: float = 0.0) -> float:
    return (math.log((spot + 1e-12) / (strike + 1e-12)) + (r + 0.5 * vol * vol) * t) / (vol * math.sqrt(t) + 1e-12)


def _bs_d2(d1: float, vol: float, t: float) -> float:
    return d1 - vol * math.sqrt(t)


# ---------- Public estimates ----------

def estimate_delta(
    spot: float,
    strike: float,
    days_to_expiry: float,
    iv: float,
    call: bool = True,
    r: float = 0.0,
) -> float:
    """
    Black–Scholes delta (European) with simple assumptions.
    """
    t = max(days_to_expiry, 0.01) / 365.0
    vol = max(iv, 0.0001)
    d1 = _bs_d1(spot, strike, t, vol, r)
    if call:
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0  # put


def estimate_price(
    spot: float,
    strike: float,
    days_to_expiry: float,
    iv: float,
    call: bool = True,
    r: float = 0.0,
) -> float:
    """
    Black–Scholes price (no dividends).
    """
    t = max(days_to_expiry, 0.01) / 365.0
    vol = max(iv, 0.0001)
    d1 = _bs_d1(spot, strike, t, vol, r)
    d2 = _bs_d2(d1, vol, t)
    if call:
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    return strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def estimate_iv(
    spot: float,
    option_price: float,
    strike: float,
    days_to_expiry: float,
    call: bool = True,
    r: float = 0.0,
    initial_iv: float = 0.20,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """
    Very basic IV inversion via bisection. Returns annualized vol in decimal (e.g., 0.20).
    """
    if option_price <= 0 or spot <= 0 or strike <= 0:
        return max(initial_iv, 0.01)

    t = max(days_to_expiry, 0.01) / 365.0

    # bounds: [1%, 500%] annualized
    lo, hi = 0.01, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price_mid = estimate_price(spot, strike, days_to_expiry, mid, call, r)
        if abs(price_mid - option_price) < tol:
            return float(mid)
        if price_mid > option_price:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))
