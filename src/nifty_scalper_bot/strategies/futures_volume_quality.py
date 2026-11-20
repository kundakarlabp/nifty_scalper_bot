"""Purpose: Verify futures volume with completed one-minute bar evidence.

Responsibilities: Compute a same-session, prior-bar volume baseline and expose
source, count and readiness separately from cumulative quote-counter deltas.
Safety: Missing candles remain unknown; never authorizes a new trade.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")


def _bar_time(value: Any) -> datetime | None:
    """Normalize an exchange bar timestamp for session and order checks."""
    try:
        if isinstance(value, datetime):
            ts = value
        elif isinstance(value, (int, float)):
            epoch = float(value)
            ts = datetime.fromtimestamp(
                epoch / 1000 if epoch > 1e11 else epoch, tz=timezone.utc
            )
        else:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_IST)
        return ts.astimezone(_IST)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def analyze_completed_futures_volume(
    bars: Any, *, min_baseline_bars: int = 5, baseline_bars: int = 20
) -> dict[str, Any]:
    """Compute a causal bar-volume ratio, never an entry confirmation vote."""
    missing: dict[str, Any] = {
        "source": "completed_futures_1m_bars",
        "ready": False,
        "ratio": None,
        "latest_volume": None,
        "baseline_mean": None,
        "baseline_count": 0,
        "baseline_min": None,
        "baseline_max": None,
        "extreme_low_relative_volume": False,
        "bar_timestamp": None,
        "reason": "insufficient_verified_bars",
    }
    if not isinstance(bars, Sequence) or isinstance(bars, (str, bytes)):
        return missing
    clean: list[tuple[datetime, float]] = []
    for bar in bars:
        if not isinstance(bar, Mapping):
            continue
        if bar.get("is_complete") is not True or bar.get("is_provisional") is True:
            continue
        if bar.get("volume_delta_untrusted") is True:
            continue
        stamp = _bar_time(
            bar.get("timestamp") or bar.get("datetime") or bar.get("time")
        )
        try:
            raw_volume = bar.get("volume")
            if raw_volume is None:
                continue
            volume = float(raw_volume)
        except (TypeError, ValueError, OverflowError):
            continue
        if stamp is None or not math.isfinite(volume) or volume < 0:
            continue
        clean.append((stamp, volume))
    clean.sort(key=lambda pair: pair[0])
    if not clean:
        return missing
    # De-duplicate replayed updates for the same completed one-minute candle.
    unique: dict[datetime, float] = dict(clean)
    chronological = sorted(unique.items())
    latest_ts, latest_vol = chronological[-1]
    session = [pair for pair in chronological if pair[0].date() == latest_ts.date()]
    baseline = [vol for _, vol in session[:-1][-baseline_bars:] if vol > 0]
    if len(baseline) < min_baseline_bars:
        return {
            **missing,
            "latest_volume": latest_vol,
            "baseline_count": len(baseline),
            "bar_timestamp": latest_ts.isoformat(),
        }
    mean = sum(baseline) / len(baseline)
    if mean <= 0:
        return missing
    ratio = latest_vol / mean
    return {
        "source": "completed_futures_1m_bars",
        "ready": True,
        "ratio": round(ratio, 5),
        "latest_volume": latest_vol,
        "baseline_mean": round(mean, 5),
        "baseline_count": len(baseline),
        "baseline_min": min(baseline),
        "baseline_max": max(baseline),
        # Diagnostic only: this cannot authorize or override a trade gate.
        # Genuine quiet trading and data-feed discontinuities require source
        # reconciliation before classification; either may produce this flag.
        "extreme_low_relative_volume": ratio < 0.05,
        "bar_timestamp": latest_ts.isoformat(),
        "reason": "verified",
    }
