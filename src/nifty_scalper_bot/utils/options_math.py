"""Black-Scholes utilities for option pricing and greeks."""

from __future__ import annotations

from dataclasses import dataclass
import math

SQRT_2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    """Return standard normal CDF using ``erf``."""

    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def _norm_pdf(x: float) -> float:
    """Return standard normal PDF."""

    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass(slots=True)
class Greeks:
    """Container for option greeks."""

    delta: float
    theta: float
    vega: float


def _d1(spot: float, strike: float, ttm: float, rate: float, vol: float) -> float:
    if vol <= 0 or ttm <= 0:
        return 0.0
    return (math.log(spot / strike) + (rate + 0.5 * vol * vol) * ttm) / (
        vol * math.sqrt(ttm)
    )


def _d2(d1: float, vol: float, ttm: float) -> float:
    return d1 - vol * math.sqrt(ttm)


def black_scholes_price(
    spot: float,
    strike: float,
    ttm: float,
    rate: float,
    vol: float,
    *,
    is_call: bool,
) -> float:
    """Return option price using the Black-Scholes formula."""

    if ttm <= 0:
        intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
        return intrinsic
    d1_val = _d1(spot, strike, ttm, rate, vol)
    d2_val = _d2(d1_val, vol, ttm)
    if is_call:
        return spot * _norm_cdf(d1_val) - strike * math.exp(-rate * ttm) * _norm_cdf(
            d2_val
        )
    return strike * math.exp(-rate * ttm) * _norm_cdf(-d2_val) - spot * _norm_cdf(
        -d1_val
    )


def black_scholes_greeks(
    spot: float,
    strike: float,
    ttm: float,
    rate: float,
    vol: float,
    *,
    is_call: bool,
) -> dict[str, float]:
    """Return Delta, Theta, and Vega for the option."""

    if ttm <= 0:
        delta = (
            1.0
            if (is_call and spot > strike)
            else -1.0 if (not is_call and strike > spot) else 0.0
        )
        return {"delta": delta, "theta": 0.0, "vega": 0.0}
    d1_val = _d1(spot, strike, ttm, rate, vol)
    d2_val = _d2(d1_val, vol, ttm)
    pdf = _norm_pdf(d1_val)
    if is_call:
        delta = _norm_cdf(d1_val)
        theta = -spot * pdf * vol / (2.0 * math.sqrt(ttm)) - rate * strike * math.exp(
            -rate * ttm
        ) * _norm_cdf(d2_val)
    else:
        delta = _norm_cdf(d1_val) - 1.0
        theta = -spot * pdf * vol / (2.0 * math.sqrt(ttm)) + rate * strike * math.exp(
            -rate * ttm
        ) * _norm_cdf(-d2_val)
    vega = spot * pdf * math.sqrt(ttm)
    return {"delta": delta, "theta": theta, "vega": vega}


def implied_volatility(
    spot: float,
    strike: float,
    ttm: float,
    rate: float,
    option_price: float,
    *,
    is_call: bool,
    tolerance: float = 1e-5,
    max_iterations: int = 100,
) -> float:
    """Infer implied volatility via Newton-Raphson."""

    if option_price <= 0:
        raise ValueError("option_price must be positive")
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if ttm <= 0:
        raise ValueError("ttm must be positive")
    vol = 0.2
    for _ in range(max_iterations):
        price = black_scholes_price(spot, strike, ttm, rate, vol, is_call=is_call)
        diff = price - option_price
        if abs(diff) < tolerance:
            return vol
        greeks = black_scholes_greeks(spot, strike, ttm, rate, vol, is_call=is_call)
        vega = greeks["vega"]
        if vega <= 1e-8:
            break
        vol -= diff / vega
        if vol <= 1e-6:
            vol = 1e-6
        if vol > 5.0:
            vol = 5.0
    raise ValueError("Implied volatility did not converge")


__all__ = [
    "black_scholes_price",
    "black_scholes_greeks",
    "implied_volatility",
    "Greeks",
]
