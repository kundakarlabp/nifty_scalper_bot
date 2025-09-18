from __future__ import annotations

"""Runtime monkey patches for strategy helpers.

This module adjusts ATR band checks so that the lower band is clamped to
``atr_min_pct`` without mutating configuration objects. It also ensures
ATM upkeep helpers on data sources run during strategy evaluation.
"""

import logging
from math import isfinite
from importlib import import_module
from typing import Any, Callable


def _resolve_min_atr_pct() -> float:
    """Resolve the minimum ATR percentage from config or runner."""

    try:  # pragma: no cover - best effort
        from .runner import StrategyRunner

        runner = StrategyRunner.get_singleton()
    except Exception:  # pragma: no cover
        runner = None

    cfg = getattr(runner, "strategy_cfg", None)
    if cfg is None:
        try:  # pragma: no cover - runtime fallback
            from .strategy_config import StrategyConfig, resolve_config_path

            cfg = StrategyConfig.load(resolve_config_path())
        except Exception:  # pragma: no cover
            return 0.0

    sym = getattr(runner, "under_symbol", "")
    if "BANK" in str(sym).upper():
        return float(
            getattr(cfg, "min_atr_pct_banknifty", getattr(cfg, "atr_min", 0.0))
        )
    return float(getattr(cfg, "min_atr_pct_nifty", getattr(cfg, "atr_min", 0.0)))


def _wrap_gate(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return a wrapper that clamps ``band_low`` via call arguments."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        band_arg_index = 1 if len(args) > 1 else None
        band_low_original: float | None = None
        resolved_min: float | None = None
        args_out = args

        if "band_low" in kwargs:
            try:
                band_low_original = float(kwargs["band_low"])
            except (TypeError, ValueError):
                band_low_original = None
        elif band_arg_index is not None:
            try:
                band_low_original = float(args[band_arg_index])
            except (TypeError, ValueError):
                band_low_original = None

        if band_low_original is not None:
            resolved_min = float(_resolve_min_atr_pct())
            if resolved_min > 0:
                new_band_low = max(band_low_original, resolved_min)
                if "band_low" in kwargs:
                    kwargs["band_low"] = new_band_low
                elif band_arg_index is not None:
                    args_out = (
                        *args[:band_arg_index],
                        new_band_low,
                        *args[band_arg_index + 1 :],
                    )

        result = fn(*args_out, **kwargs)

        if resolved_min is None:
            resolved_min = float(_resolve_min_atr_pct())

        if not isinstance(result, tuple) or len(result) < 4 or resolved_min <= 0:
            return result

        try:
            current_min = float(result[2])
            current_max = float(result[3])
        except (TypeError, ValueError):
            return result

        new_min = max(current_min, resolved_min)
        if new_min == current_min:
            return result

        try:
            atr_value = float(args[0])
        except (IndexError, TypeError, ValueError):
            atr_value = new_min

        effective_max = float("inf") if current_max <= 0 else max(current_max, new_min)
        ok = bool(result[0]) and (new_min <= atr_value <= effective_max)
        reason = result[1]
        if not ok:
            if atr_value < new_min:
                reason = f"atr_out_of_band: atr={atr_value:.4f} < min={new_min}"
            else:
                bound = effective_max if isfinite(effective_max) else current_max
                reason = f"atr_out_of_band: atr={atr_value:.4f} > max={bound}"

        updated = (ok, reason, new_min, current_max)
        if len(result) > 4:
            updated += result[4:]
        return updated

    return wrapped


def _patch_atr_band() -> None:
    """Wrap the first available ATR gate/check function."""

    candidates = ["gate_atr_band", "check_atr_band", "check_atr"]
    modules = [
        "src.strategies.runner",
        "src.strategies.scalping_strategy",
        "src.strategies.atr_gate",
        "src.diagnostics.checks",
    ]
    for name in candidates:
        for mod_name in modules:
            try:  # pragma: no cover - best effort
                mod = import_module(mod_name)
            except Exception:
                continue
            fn = getattr(mod, name, None)
            if callable(fn):
                setattr(mod, name, _wrap_gate(fn))
                return


def _patch_runner_step() -> None:
    try:  # pragma: no cover
        from . import runner as _runner
    except Exception:  # pragma: no cover
        return

    def _wrap(cls: Any) -> None:
        if not hasattr(cls, "step"):
            return
        orig = cls.step

        def step(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - thin wrapper
            ds = getattr(self, "data_source", None)
            if ds:
                for name in ("auto_resubscribe_atm", "ensure_atm_tokens"):
                    fn = getattr(ds, name, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            logging.getLogger(__name__).warning(
                                "%s failed", name, exc_info=True
                            )
            return orig(self, *a, **k)

        cls.step = step  # type: ignore[assignment]

    for cls_name in ("Orchestrator", "StrategyRunner"):
        cls = getattr(_runner, cls_name, None)
        if cls:
            _wrap(cls)


_patch_atr_band()
_patch_runner_step()


__all__ = ["_patch_atr_band", "_resolve_min_atr_pct"]
