from __future__ import annotations

import math

from nifty_scalper_bot.brokers.instrument_lookup import Instrument, InstrumentLookup

InstrumentResolver = InstrumentLookup


def atm_strike_for_spot(spot: float, step: int = 50) -> int:
    """Return nearest ATM strike. Args: spot, step. Returns: int. Raises: ValueError."""
    if spot <= 0 or step <= 0:
        raise ValueError("spot and step must be positive")
    return int(math.floor((float(spot) / float(step)) + 0.5) * int(step))


def strike_for_delta(
    spot: float,
    iv: float,
    ttm: float,
    target_delta: float,
    *,
    call: bool = True,
    step: int = 50,
    risk_free_rate: float = 0.06,
) -> int:
    """Return strike for target delta."""
    from nifty_scalper_bot.utils.options_math import black_scholes_greeks

    if spot <= 0:
        raise ValueError("spot must be positive")
    if iv <= 0:
        raise ValueError("iv must be positive")
    if ttm <= 0:
        raise ValueError("ttm must be positive")
    if step <= 0:
        raise ValueError("step must be positive")

    atm = atm_strike_for_spot(spot, step)
    candidates = range(atm - 20 * step, atm + 20 * step + step, step)

    def score(strike: int) -> float:
        greeks = black_scholes_greeks(
            float(spot),
            float(strike),
            float(ttm),
            float(risk_free_rate),
            float(iv),
            is_call=call,
        )
        delta = float(greeks.get("delta", 0.0))
        value = delta if call else abs(delta)
        return abs(value - float(target_delta))

    return int(min(candidates, key=score))


__all__ = [
    "Instrument",
    "InstrumentResolver",
    "InstrumentLookup",
    "atm_strike_for_spot",
    "strike_for_delta",
]
