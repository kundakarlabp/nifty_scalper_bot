"""Timestamp utilities."""
from datetime import datetime, timezone

def is_fresh_ts_ms(ts_ms: int | float | None, threshold_ms: float = 3000.0) -> bool:
    """Check if a millisecond timestamp is fresh (within threshold)."""
    if not ts_ms:
        return False
    
    try:
        now_ms = datetime.now(timezone.utc).timestamp() * 1000
        age = now_ms - float(ts_ms)
        return 0 <= age <= threshold_ms
    except Exception:
        return False
