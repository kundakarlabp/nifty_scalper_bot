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
        if args or "band_low" in kwargs:
            band_low = kwargs.get("band_low", args[0] if args else None)
            if band_low is not None:
                # ``gate_atr_band`` style helpers expect ``band_low`` to reflect the
                # effective minimum ATR pct.  The configuration might provide a
                # higher or lower raw bound, but we always want the guard to
                # evaluate against the resolved minimum derived from runtime
                # context (instrument, overrides, etc.).  When the resolved
                # minimum is zero we fall back to the provided band so that
                # environments with no lower limit behave unchanged.
                resolved = float(_resolve_min_atr_pct())
                clamped = float(band_low) if resolved <= 0 else resolved
                if "band_low" in kwargs:
                    kwargs["band_low"] = clamped
                else:
                    args = (clamped, *args[1:])
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
