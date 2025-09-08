from __future__ import annotations

"""Monkey patches for runtime tweaks.

This module clamps the lower ATR band to ``atr_min_pct`` to avoid false
``atr_out_of_band`` blocks when ATR hovers around the minimum, and it
ensures ATM option upkeep is invoked during strategy evaluation even if
callers forget to do so.
"""

import logging
from typing import Any


def _patch_atr_band() -> None:
    try:  # pragma: no cover - best effort import
        from .scalping_strategy import EnhancedScalpingStrategy as _Strat
    except Exception:  # pragma: no cover
        from .scalping_strategy import ScalpingStrategy as _Strat  # type: ignore

    orig = _Strat.generate_signal

    def wrapped(self: Any, df, current_tick=None, current_price=None, spot_df=None):
        runner = getattr(self, "runner", None)
        cfg = getattr(runner, "strategy_cfg", None)
        if cfg is None:
            from .strategy_config import StrategyConfig, resolve_config_path

            cfg = StrategyConfig.load(resolve_config_path())
        sym = getattr(runner, "under_symbol", "")
        atr_min_pct = getattr(cfg, "min_atr_pct_banknifty", cfg.atr_min)
        if "BANK" not in str(sym).upper():
            atr_min_pct = getattr(cfg, "min_atr_pct_nifty", cfg.atr_min)
        orig_band = float(cfg.atr_min)
        band_low = min(orig_band, float(atr_min_pct))
        try:
            cfg.atr_min = band_low
            return orig(self, df, current_tick=current_tick, current_price=current_price, spot_df=spot_df)
        finally:
            cfg.atr_min = orig_band

    _Strat.generate_signal = wrapped  # type: ignore[assignment]


def _patch_runner_step() -> None:
    try:  # pragma: no cover
        from . import runner as _runner
    except Exception:  # pragma: no cover
        return

    def _wrap(cls):
        if not hasattr(cls, "step"):
            return
        orig = cls.step

        def step(self, *a, **k):
            ds = getattr(self, "data_source", None)
            if ds:
                for name in ("auto_resubscribe_atm", "ensure_atm_tokens"):
                    fn = getattr(ds, name, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            logging.getLogger(__name__).warning("%s failed", name, exc_info=True)
            return orig(self, *a, **k)

        cls.step = step  # type: ignore[assignment]

    for cls_name in ("Orchestrator", "StrategyRunner"):
        cls = getattr(_runner, cls_name, None)
        if cls:
            _wrap(cls)


_patch_atr_band()
_patch_runner_step()
