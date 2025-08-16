# src/utils/greeks.py
from __future__ import annotations
"""
Lightweight Black–Scholes + Greeks utilities for index/stock options.

Public API (stable):
- bs_price(...)
- bs_delta(...), bs_gamma(...), bs_theta(...), bs_vega(...)
- implied_vol_bisection(...)
- estimate_delta(...)
- estimate_all_greeks(...)

Design notes:
- Handles CE/PE (call/put) via tolerant type normalization.
- Inputs clamped to safe minimums (no NaNs, no div-by-zero).
- IV solver uses robust bracketing (auto-widens upper bound).
"""

import math
from typing import Literal, Tuple, Dict

OptType = Literal["CE", "PE", "CALL", "PUT"]

__all__ = [
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_theta",
    "bs_vega",
    "implied_vol_bisection",
    "estimate_delta",
    "estimate_all_greeks",
]


# --------------------------- helpers & clamps --------------------------- #

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    # Stable CDF via error function
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_opt_type(opt_type: OptType) -> str:
    t = str(opt_type or "").strip().upper()
    return "CE" if t in ("CE", "CALL") else "PE"


def _clamp_inputs(
    spot: float, strike: float, days_to_expiry: float, iv: float, r: float, q: float
) -> Tuple[float, float, float, float, float, float]:
    S = max(1e-9, float(spot))
    K = max(1e-9, float(strike))
    T = max(1e-6, float(days_to_expiry) / 365.0)  # days → years
    sigma = max(1e-6, float(iv))
    r = float(r)
    q = float(q)
    return S, K, T, sigma, r, q


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float]:
    num = math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T
    den = sigma * math.sqrt(T)
    d1 = num / den
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def _price_bounds(S: float, K: float, T: float, r: float, q: float, opt_type: str) -> Tuple[float, float]:
    """Simple theoretical bounds based on put–call parity (lower, upper)."""
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    if opt_type == "CE":
        lower = max(0.0, disc_q * S - disc_r * K)
        upper = disc_q * S  # loose
    else:
        lower = max(0.0, disc_r * K - disc_q * S)
        upper = disc_r * K  # loose
    return lower, upper


# --------------------------- Black–Scholes core --------------------------- #

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
    """Black–Scholes price (continuous dividend yield)."""
    S, K, T, sigma, r, q = _clamp_inputs(spot, strike, days_to_expiry, iv, r, q)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    t = _norm_opt_type(opt_type)

    if t == "CE":
        return disc_q * S * _norm_cdf(d1) - disc_r * K * _norm_cdf(d2)
    else:
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
    t = _norm_opt_type(opt_type)
    if t == "CE":
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
    """
    Theta (time decay).
    Returns per-day if per_day=True, else annualised.
    """
    S, K, T, sigma, r, q = _clamp_inputs(spot, strike, days_to_expiry, iv, r, q)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    first = -disc_q * S * _norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
    t = _norm_opt_type(opt_type)
    if t == "CE":
        second = q * disc_q * S * _norm_cdf(d1)
        third = -r * disc_r * K * _norm_cdf(d2)
    else:
        second = -q * disc_q * S * _norm_cdf(-d1)
        third = r * disc_r * K * _norm_cdf(-d2)

    theta_annual = first + second + third
    return theta_annual / 365.0 if per_day else theta_annual


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
    """
    Vega: sensitivity to volatility.
    If per_1pct=True, returns change per +1% vol (i.e., divide by 100).
    """
    S, K, T, sigma, r, q = _clamp_inputs(spot, strike, days_to_expiry, iv, r, q)
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)
    vega = disc_q * S * _norm_pdf(d1) * math.sqrt(T)  # per 1.00 volatility
    return vega / 100.0 if per_1pct else vega


# --------------------------- Implied volatility --------------------------- #

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
    Robust IV solver (bisection).
    - Ensures target is bracketed: widens `high` if needed up to ~10.0 vol.
    - Clamps to [low, high] if still not bracketed (e.g., absurd targets).
    Returns IV as decimal (e.g., 0.20).
    """
    target = max(0.0, float(target_price))
    dte_days = float(days_to_expiry)
    S, K, T, _, r, q = _clamp_inputs(spot, strike, dte_days, max(low, 1e-4), r, q)
    t = _norm_opt_type(opt_type)

    # Parity sanity (lower bound on price)
    lb, _ub = _price_bounds(S, K, T, r, q, t)
    target = max(target, lb)

    lo = max(1e-6, float(low))
    hi = max(lo * 2.0, float(high))

    price_lo = bs_price(S, K, dte_days, lo, opt_type=t, r=r, q=q)
    price_hi = bs_price(S, K, dte_days, hi, opt_type=t, r=r, q=q)

    # If target below price at lo (near-zero vol), return lo (degenerate case)
    if target <= price_lo:
        return lo

    # Widen hi until we bracket the target or hit safety cap
    widen_cap = 10.0
    tries = 0
    while price_hi < target and hi < widen_cap:
        hi *= 1.5
        price_hi = bs_price(S, K, dte_days, hi, opt_type=t, r=r, q=q)
        tries += 1
        if tries > 20:
            break

    # If still not bracketed, clamp to hi
    if price_hi < target:
        return max(lo, min(widen_cap, hi))

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price_mid = bs_price(S, K, dte_days, mid, opt_type=t, r=r, q=q)
        diff = price_mid - target
        if abs(diff) <= tol:
            return max(low, min(hi, mid))
        if diff > 0.0:
            hi = mid
        else:
            lo = mid

    return max(low, min(hi, 0.5 * (lo + hi)))


# -------------------------- Convenience wrappers ------------------------- #

def estimate_delta(
    spot: float, strike: float, days_to_expiry: float, iv: float, opt_type: OptType
) -> float:
    """Convenience wrapper used by strike ranking (assumes r=q=0)."""
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
    Returns a dict with price and Greeks.
    Theta is per-day; Vega is per +1% vol.
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


# ------------------------------- quick demo ------------------------------ #

if __name__ == "__main__":
    S = 24600.0
    K = 24600.0
    dte = 7.0
    iv = 0.20
    r = 0.0
    q = 0.0
    for typ in ("CE", "PE"):
        print(f"\n{typ} @ S={S}, K={K}, dte={dte}d, iv={iv*100:.1f}%")
        print("price:", bs_price(S, K, dte, iv, opt_type=typ, r=r, q=q))
        print("delta:", bs_delta(S, K, dte, iv, opt_type=typ, r=r, q=q))
        print("gamma:", bs_gamma(S, K, dte, iv, r=r, q=q))
        print("theta/day:", bs_theta(S, K, dte, iv, opt_type=typ, r=r, q=q, per_day=True))
        print("vega/1%:", bs_vega(S, K, dte, iv, r=r, q=q, per_1pct=True))