"""OI-weighted max pain strike calculation."""

from __future__ import annotations


def calc_max_pain(chain_data: dict[float, dict]) -> float | None:
    """Find the strike that minimises total pain (OI × distance).

    Args:
        chain_data: {strike: {"ce_oi": int, "pe_oi": int}}

    Returns:
        The max pain strike, or None if data is insufficient.
    """
    if not chain_data or len(chain_data) < 3:
        return None

    strikes = sorted(chain_data.keys())
    min_pain = float("inf")
    max_pain_strike = strikes[0]

    for target_strike in strikes:
        total_pain = 0.0
        for strike, data in chain_data.items():
            ce_oi = data.get("ce_oi", 0)
            pe_oi = data.get("pe_oi", 0)
            # CE writers lose when spot > strike
            if target_strike > strike:
                total_pain += ce_oi * (target_strike - strike)
            # PE writers lose when spot < strike
            if target_strike < strike:
                total_pain += pe_oi * (strike - target_strike)

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = target_strike

    return float(max_pain_strike)
