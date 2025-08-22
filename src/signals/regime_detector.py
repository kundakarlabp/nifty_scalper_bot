from __future__ import annotations

from typing import Dict, Sequence


def _slope(vals: Sequence[float]) -> float:
    if not vals or len(vals) < 2:
        return 0.0
    return float(vals[-1] - vals[0]) / max(1, len(vals) - 1)


def detect_market_regime(spot_ohlc) -> Dict[str, object]:
    """
    Very lightweight regime detector:
    - Uses simple price slope (close values) and rolling range to classify.
    - Returns dict: {"regime": "trend"|"range", "strength": float}
    """
    try:
        closes = [float(x["close"]) for x in spot_ohlc][-50:]
    except Exception:
        closes = []
    if len(closes) < 10:
        return {"regime": "auto", "strength": 0.0}

    sl = _slope(closes)
    rng = max(closes) - min(closes)
    strength = abs(sl) / (rng / max(1, len(closes))) if rng > 0 else 0.0

    if strength > 0.35:
        return {"regime": "trend", "strength": round(strength, 3)}
    else:
        return {"regime": "range", "strength": round(strength, 3)}