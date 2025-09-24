"""Utilities for strategy warm-up checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WarmupInfo:
    """Outcome of a warm-up evaluation."""

    required_bars: int
    have_bars: int
    ok: bool
    reasons: list[str]


def compute_required_bars(
    cfg: Any, *, default_min: int, atr_period: int | None = None
) -> int:
    """Return the number of bars needed before signals are valid."""

    floors = [int(default_min)]
    floors.append(int(getattr(cfg, "min_bars_required", default_min)))
    if atr_period is not None:
        floors.append(int(atr_period) + 5)
    if hasattr(cfg, "warmup_bars"):
        try:
            floors.append(int(cfg.warmup_bars))
        except Exception:  # pragma: no cover - defensive
            pass
    return max(floors)


def warmup_status(have: int, need: int) -> WarmupInfo:
    """Return a ``WarmupInfo`` describing whether warm-up is satisfied."""

    ok = have >= need
    reasons: list[str] = []
    if not ok:
        reasons.append(f"need_bars>={need},have_bars={have}")
    return WarmupInfo(required_bars=need, have_bars=have, ok=ok, reasons=reasons)


# ---------------------------------------------------------------------------
# Backwards-compatible helpers used by existing code/tests
# ---------------------------------------------------------------------------


def required_bars(cfg: Any) -> int:
    """Compatibility wrapper returning ``compute_required_bars`` value."""

    default_min = int(getattr(cfg, "min_bars_required", 0))
    atr_p = int(getattr(cfg, "atr_period", 0)) if hasattr(cfg, "atr_period") else None
    return compute_required_bars(cfg, default_min=default_min, atr_period=atr_p)


def check(cfg: Any, have_bars: int) -> WarmupInfo:
    """Compatibility wrapper returning ``warmup_status`` for ``have_bars``."""

    need = required_bars(cfg)
    return warmup_status(have_bars, need)
