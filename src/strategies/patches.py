from __future__ import annotations

"""Runtime monkey patches for strategy helpers.

This module adjusts ATR band checks so that the lower band is clamped to
``atr_min_pct`` without mutating configuration objects. It also ensures
ATM upkeep helpers on data sources run during strategy evaluation.
"""

import logging
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
        """Clamp the ``band_low`` argument while preserving ATR inputs."""

        has_kwarg = "band_low" in kwargs
        band_arg = kwargs.get("band_low") if has_kwarg else (args[1] if len(args) > 1 else None)
        resolved_min: float | None = None
        if band_arg is not None or has_kwarg or len(args) > 1:
            resolved_min = float(_resolve_min_atr_pct())

        new_band: float | None = None
        if band_arg is not None:
            try:
                band_val = float(band_arg)
            except (TypeError, ValueError):  # pragma: no cover - defensive fallback
                band_val = None
            if band_val is not None:
                if resolved_min and resolved_min > 0:
                    new_band = max(band_val, resolved_min)
                else:
                    new_band = band_val
            elif resolved_min and resolved_min > 0:
                new_band = resolved_min
        elif resolved_min and resolved_min > 0:
            new_band = resolved_min

        if new_band is not None:
            if has_kwarg:
                kwargs["band_low"] = new_band
            elif len(args) > 1:
                args = (*args[:1], new_band, *args[2:])
        return fn(*args, **kwargs)

    return wrapped


def _patch_atr_band() -> None:
    """Wrap the first available ATR gate/check function."""

    candidates = ["gate_atr_band", "check_atr_band", "check_atr"]
    modules = [
        "src.strategies.runner",
        "src.strategies.scalping_strategy",
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
