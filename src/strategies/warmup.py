from __future__ import annotations

"""Warm-up bar calculations and checks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WarmupInfo:
    """Outcome of the warm-up bar check."""

    required_bars: int
    have_bars: int
    ok: bool
    reasons: list[str]


def _cfg_int(cfg: object, name: str, default: int) -> int:
    """Return ``cfg.name`` as ``int`` with ``default`` fallback."""

    return int(getattr(cfg, name, default))


def required_bars(cfg: object) -> int:
    """Compute the number of bars required for strategy warm-up."""

    pad = _cfg_int(cfg, "warmup_pad", 2)
    warm_min = _cfg_int(cfg, "warmup_bars_min", 20)
    atr_period = _cfg_int(cfg, "atr_period", 14)
    ema_slow = _cfg_int(cfg, "ema_slow", 21)
    regime_min = _cfg_int(cfg, "regime_min_bars", warm_min)
    feat_min = _cfg_int(cfg, "features_min_bars", warm_min)
    return max(warm_min, atr_period + pad, ema_slow + pad, regime_min, feat_min)


def check(cfg: object, have_bars: int) -> WarmupInfo:
    """Check if ``have_bars`` satisfies warm-up requirements."""

    need = required_bars(cfg)
    ok = have_bars >= need
    reasons: list[str] = [] if ok else [f"need_bars>={need}", f"have_bars={have_bars}"]
    return WarmupInfo(required_bars=need, have_bars=have_bars, ok=ok, reasons=reasons)
