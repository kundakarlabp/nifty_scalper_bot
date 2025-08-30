"""Black-Scholes greeks utilities with minimal dependencies.

This module implements standard greeks (price, delta, gamma) for European
options, a simple implied volatility solver, and helpers for NSE weekly index
expiry handling. It avoids external libraries to remain lightweight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Literal, Optional
from zoneinfo import ZoneInfo

OptionType = Literal["CE", "PE"]


def _phi(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _nprime(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price_delta_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    opt: OptionType,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return Black-Scholes price, delta and gamma.

    Returns ``(price, delta, gamma)`` for the given inputs. ``None`` is
    returned for all components when inputs are nonsensical (e.g. non-positive
    spot, strike or maturity).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None, None
    fwd = S * math.exp((r - q) * T)
    vol = sigma * math.sqrt(T)
    d1 = (math.log(fwd / K) / vol) + 0.5 * vol
    d2 = d1 - vol
    if opt == "CE":
        price = math.exp(-r * T) * (fwd * _phi(d1) - K * _phi(d2))
        delta = math.exp(-q * T) * _phi(d1)
    else:
        price = math.exp(-r * T) * (K * _phi(-d2) - fwd * _phi(-d1))
        delta = -math.exp(-q * T) * _phi(-d1)
    gamma = (math.exp(-q * T) * _nprime(d1)) / (S * vol)
    return price, delta, gamma


def implied_vol_newton(
    mid: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    opt: OptionType,
    *,
    guess: float = 0.20,
    tol: float = 1e-4,
    max_iter: int = 20,
) -> Optional[float]:
    """Estimate implied volatility via Newton-Raphson iterations."""
    if mid <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    sigma = max(0.05, min(0.95, guess))
    for _ in range(max_iter):
        px, _, _ = bs_price_delta_gamma(S, K, T, r, q, sigma, opt)
        if px is None:
            return None
        fwd = S * math.exp((r - q) * T)
        vol = sigma * math.sqrt(T)
        d1 = (math.log(fwd / K) / vol) + 0.5 * vol
        vega = S * math.exp(-q * T) * _nprime(d1) * math.sqrt(T)
        diff = px - mid
        if abs(diff) < tol:
            return sigma
        if vega <= 1e-8:
            break
        sigma = max(0.05, min(1.0, sigma - diff / vega))
    return None


def next_weekly_expiry_ist(now: datetime, tz: str = "Asia/Kolkata") -> datetime:
    """Return the next NSE weekly index expiry in IST."""
    z = ZoneInfo(tz)
    now = now.astimezone(z)
    target_wd, target_t = 3, time(15, 30)  # Thursday 15:30 IST
    days_ahead = (target_wd - now.weekday()) % 7
    expiry = (now + timedelta(days=days_ahead)).replace(
        hour=target_t.hour, minute=target_t.minute, second=0, microsecond=0
    )
    if expiry <= now:
        expiry = expiry + timedelta(days=7)
    return expiry


@dataclass
class GreekEstimate:
    """Container for estimated option greeks."""

    ok: bool
    sigma: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    T_years: float
    source: str  # "iv", "atr_proxy", "guess"


def estimate_greeks_from_mid(
    S: float,
    K: float,
    mid: float,
    opt: OptionType,
    now: datetime,
    *,
    r: float = 0.065,
    q: float = 0.0,
    tz: str = "Asia/Kolkata",
    atr_pct: Optional[float] = None,
) -> GreekEstimate:
    """Estimate greeks for an option from its mid price.

    Attempts to back out implied volatility first. If unsuccessful, falls back
    to an ATR-based proxy or a constant guess.
    """
    expiry = next_weekly_expiry_ist(now, tz)
    T = max(1e-9, (expiry - now.astimezone(ZoneInfo(tz))).total_seconds() / (365.0 * 24 * 3600.0))
    iv = implied_vol_newton(mid, S, K, T, r, q, opt)
    if iv:
        _, d, g = bs_price_delta_gamma(S, K, T, r, q, iv, opt)
        return GreekEstimate(True, iv, d, g, T, "iv")
    if atr_pct and atr_pct > 0:
        sigma_annual = max(0.08, min(0.50, (atr_pct / 100.0) * 1.6 * math.sqrt(252.0)))
        _, d, g = bs_price_delta_gamma(S, K, T, r, q, sigma_annual, opt)
        return GreekEstimate(True, sigma_annual, d, g, T, "atr_proxy")
    sigma_guess = 0.22
    _, d, g = bs_price_delta_gamma(S, K, T, r, q, sigma_guess, opt)
    return GreekEstimate(True, sigma_guess, d, g, T, "guess")
