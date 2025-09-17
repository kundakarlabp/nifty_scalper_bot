"""Helpers for strategy hyper-parameters.

This module provides a typed container for the dynamic parameters used by the
scalping strategy together with a parameter space implementation that knows
how to sample, clamp and validate candidates against the existing
``StrategySettings`` schema.  The tuner in :mod:`src.backtesting.tuning`
relies on these helpers to explore the configuration space while respecting
the guard rails already enforced by the live configuration models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from pydantic import ValidationError

from src.config import settings
from src.config import StrategySettings


@dataclass(frozen=True)
class StrategyParameters:
    """Minimal parameter set used by the scalping strategy."""

    ema_fast: int
    ema_slow: int
    atr_period: int
    confidence_threshold: float
    min_signal_score: int
    atr_sl_multiplier: float
    atr_tp_multiplier: float

    @classmethod
    def from_settings(
        cls,
        strategy_settings: StrategySettings | None = None,
    ) -> "StrategyParameters":
        """Create a parameter set from an existing :class:`StrategySettings`."""

        cfg = strategy_settings or settings.strategy
        return cls(
            ema_fast=int(cfg.ema_fast),
            ema_slow=int(cfg.ema_slow),
            atr_period=int(cfg.atr_period),
            confidence_threshold=float(cfg.confidence_threshold),
            min_signal_score=int(cfg.min_signal_score),
            atr_sl_multiplier=float(cfg.atr_sl_multiplier),
            atr_tp_multiplier=float(cfg.atr_tp_multiplier),
        )

    @classmethod
    def from_settings_model(cls, cfg: StrategySettings) -> "StrategyParameters":
        """Alternate constructor for convenience inside validators."""

        return cls.from_settings(cfg)

    def as_settings_update(self) -> Dict[str, float | int]:
        """Return a dictionary that can update ``StrategySettings`` fields."""

        return {
            "ema_fast": int(self.ema_fast),
            "ema_slow": int(self.ema_slow),
            "atr_period": int(self.atr_period),
            "confidence_threshold": float(self.confidence_threshold),
            "min_signal_score": int(self.min_signal_score),
            "atr_sl_multiplier": float(self.atr_sl_multiplier),
            "atr_tp_multiplier": float(self.atr_tp_multiplier),
        }

    def merge_into(self, base: StrategySettings) -> StrategySettings:
        """Return a new :class:`StrategySettings` with this parameter set applied."""

        data = base.model_dump()
        data.update(self.as_settings_update())
        return StrategySettings(**data)


@dataclass(frozen=True)
class ParameterBound:
    """Continuous or integer bound used by the tuner."""

    name: str
    low: float
    high: float
    is_int: bool = False

    def sample(self, rng: np.random.Generator) -> float:
        value = float(rng.uniform(self.low, self.high))
        if self.is_int:
            value = round(value)
            return float(int(np.clip(value, self.low, self.high)))
        return float(np.clip(value, self.low, self.high))

    def normalize(self, value: float) -> float:
        width = self.high - self.low
        if width <= 0:
            return 0.0
        return float(np.clip((value - self.low) / width, 0.0, 1.0))

    def denormalize(self, frac: float) -> float:
        frac = float(np.clip(frac, 0.0, 1.0))
        value = self.low + frac * (self.high - self.low)
        if self.is_int:
            value = round(value)
        return float(np.clip(value, self.low, self.high))


class StrategyParameterSpace:
    """Sampling/validation helpers for ``StrategyParameters``."""

    def __init__(
        self,
        bounds: Sequence[ParameterBound],
        base_settings: StrategySettings | None = None,
    ) -> None:
        self._bounds = list(bounds)
        self._order = [b.name for b in self._bounds]
        self._base = base_settings or settings.strategy
        self._base_dump = self._base.model_dump()
        self._rng = np.random.default_rng()
        self._bound_map = {b.name: b for b in self._bounds}

    @property
    def bounds(self) -> Sequence[ParameterBound]:
        return list(self._bounds)

    @property
    def order(self) -> Sequence[str]:
        return list(self._order)

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, object] | None = None,
        *,
        base_settings: StrategySettings | None = None,
    ) -> "StrategyParameterSpace":
        """Build a space using optional overrides from a config mapping."""

        base = base_settings or settings.strategy
        raw_ranges = {}
        if cfg and isinstance(cfg, Mapping):
            raw_ranges = {
                k: v for k, v in cfg.items() if isinstance(v, Mapping)
            }
            if "ranges" in raw_ranges and isinstance(raw_ranges["ranges"], Mapping):
                raw_ranges = dict(raw_ranges["ranges"])

        def _range(name: str, default_low: float, default_high: float, is_int: bool) -> ParameterBound:
            rng_cfg = raw_ranges.get(name, {}) if isinstance(raw_ranges, Mapping) else {}
            low = float(rng_cfg.get("min", default_low))
            high = float(rng_cfg.get("max", default_high))
            if low > high:
                low, high = high, low
            return ParameterBound(name=name, low=low, high=high, is_int=is_int)

        bounds = [
            _range(
                "ema_fast",
                max(2, base.ema_fast - 6),
                base.ema_fast + 6,
                True,
            ),
            _range(
                "ema_slow",
                max(base.ema_fast + 4, base.ema_slow - 10),
                base.ema_slow + 20,
                True,
            ),
            _range("atr_period", max(5, base.atr_period - 7), base.atr_period + 10, True),
            _range(
                "confidence_threshold",
                max(20.0, base.confidence_threshold - 20.0),
                min(95.0, base.confidence_threshold + 20.0),
                False,
            ),
            _range("min_signal_score", 2, max(6, base.min_signal_score + 2), True),
            _range("atr_sl_multiplier", max(0.4, base.atr_sl_multiplier - 0.6), base.atr_sl_multiplier + 0.8, False),
            _range("atr_tp_multiplier", base.atr_sl_multiplier + 0.2, base.atr_tp_multiplier + 1.2, False),
        ]
        return cls(bounds=bounds, base_settings=base)

    def reseed(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(self) -> StrategyParameters:
        candidate = {}
        for bound in self._bounds:
            val = bound.sample(self._rng)
            candidate[bound.name] = int(val) if bound.is_int else val
        return self.ensure_valid(StrategyParameters(**candidate))

    def ensure_valid(self, params: StrategyParameters) -> StrategyParameters:
        data = dict(self._base_dump)
        data.update(params.as_settings_update())
        try:
            validated = StrategySettings(**data)
        except ValidationError:
            clipped: Dict[str, float | int] = {}
            for bound in self._bounds:
                raw = getattr(params, bound.name)
                val = float(np.clip(raw, bound.low, bound.high))
                if bound.is_int:
                    val = int(round(val))
                clipped[bound.name] = val
            fast = clipped.get("ema_fast")
            slow = clipped.get("ema_slow")
            fast_bound = self._bound_map.get("ema_fast")
            slow_bound = self._bound_map.get("ema_slow")
            if isinstance(fast, (int, float)) and isinstance(slow, (int, float)):
                if fast_bound is not None:
                    fast = max(fast, fast_bound.low)
                    fast = min(fast, fast_bound.high)
                if slow_bound is not None:
                    slow = max(slow, slow_bound.low)
                    slow = min(slow, slow_bound.high)
                if fast >= slow:
                    fast = min(slow - 1, fast if fast_bound is None else fast_bound.high)
                    if fast_bound is not None:
                        fast = max(fast, fast_bound.low)
                    if fast >= slow:
                        slow = fast + 1
                        if slow_bound is not None:
                            slow = min(slow, slow_bound.high)
                            slow = max(slow, slow_bound.low)
                clipped["ema_fast"] = int(round(fast))
                clipped["ema_slow"] = int(round(slow))
            tp = clipped.get("atr_tp_multiplier")
            sl = clipped.get("atr_sl_multiplier")
            tp_bound = self._bound_map.get("atr_tp_multiplier")
            if isinstance(tp, (int, float)) and isinstance(sl, (int, float)):
                if tp <= sl:
                    tp = sl + 0.1
                if tp_bound is not None:
                    tp = min(max(tp, tp_bound.low), tp_bound.high)
                clipped["atr_tp_multiplier"] = float(tp)
            safe_kwargs = {}
            for bound in self._bounds:
                val = clipped.get(bound.name, getattr(params, bound.name))
                safe_kwargs[bound.name] = int(val) if bound.is_int else float(val)
            safe = StrategyParameters(**safe_kwargs)
            data.update(safe.as_settings_update())
            validated = StrategySettings(**data)
        return StrategyParameters.from_settings(validated)

    def to_vector(self, params: StrategyParameters) -> np.ndarray:
        values = []
        for bound in self._bounds:
            value = getattr(params, bound.name)
            values.append(bound.normalize(float(value)))
        return np.asarray(values, dtype=float)

    def from_vector(self, vec: Iterable[float]) -> StrategyParameters:
        values: Dict[str, float | int] = {}
        for bound, frac in zip(self._bounds, vec):
            denorm = bound.denormalize(float(frac))
            values[bound.name] = int(denorm) if bound.is_int else denorm
        return self.ensure_valid(StrategyParameters(**values))

