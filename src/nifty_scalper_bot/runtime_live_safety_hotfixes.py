"""Runtime live-safety hotfixes for reconciliation, arming and candle guards.

This module is intentionally small and import-hook based because several of the
affected modules are imported late by the runtime.  The fixes are defensive and
fail-closed:

- core.app receives a ``time`` proxy that remains callable as ``datetime.time``
  while also supporting ``time.time()``, ``time.monotonic()`` and friends.  This
  prevents reconciliation/app closures from crashing with
  ``type object 'datetime.time' has no attribute 'time'`` without breaking
  existing ``time(15, 24)`` calls.
- CandleEngine drops stale out-of-order finalized candles and resets only the
  affected symbol stream instead of raising a runner-level critical exception.
- StrategyRunner refuses to schedule signal preparation while LIVE order arming
  is false, so the order path is not entered when global execution is blocked.
- Repeated non-incremental fill warnings are de-duplicated at the logger layer.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import time as _datetime_time
import importlib.abc
import importlib.machinery
import logging
import os
import re
import sys
import threading
import time as _time_module
from types import ModuleType
from typing import Any

_PATCHED_MODULES: set[str] = set()
_HOOK_INSTALLED = False
_LOCK = threading.RLock()
_TARGETS = {
    "nifty_scalper_bot.core.app",
    "nifty_scalper_bot.data.candle_engine",
    "nifty_scalper_bot.strategies.runner",
}


class _DatetimeTimeProxy:
    """Callable datetime.time constructor plus selected stdlib time module APIs."""

    def __call__(self, *args: Any, **kwargs: Any) -> _datetime_time:
        return _datetime_time(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(_time_module, name)

    @property
    def min(self) -> _datetime_time:
        return _datetime_time.min

    @property
    def max(self) -> _datetime_time:
        return _datetime_time.max

    @property
    def resolution(self) -> _datetime_time:
        return _datetime_time.resolution


_TIME_PROXY = _DatetimeTimeProxy()


class _PostImportPatchLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, wrapped: importlib.abc.Loader) -> None:
        self.fullname = fullname
        self.wrapped = wrapped

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        create = getattr(self.wrapped, "create_module", None)
        if callable(create):
            return create(spec)  # type: ignore[misc]
        return None

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)  # type: ignore[attr-defined]
        _patch_module(module.__name__, module)


class _PostImportPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if isinstance(spec.loader, _PostImportPatchLoader):
            return spec
        spec.loader = _PostImportPatchLoader(fullname, spec.loader)  # type: ignore[arg-type]
        return spec


class _NonIncrementalFillDedupFilter(logging.Filter):
    """Suppress alert storms from identical cumulative fill replays."""

    _pattern = re.compile(r"Ignoring non-incremental fill for order\s+(\S+)")

    def __init__(self, interval_s: float) -> None:
        super().__init__()
        self.interval_s = max(float(interval_s), 1.0)
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        match = self._pattern.search(message)
        if match is None:
            return True
        order_id = match.group(1)
        now = _time_module.monotonic()
        with self._lock:
            last = float(self._last_seen.get(order_id, 0.0) or 0.0)
            if now - last < self.interval_s:
                return False
            self._last_seen[order_id] = now
        setattr(record, "deduped_non_incremental_fill", True)
        return True


_DEDUP_FILTER = _NonIncrementalFillDedupFilter(
    float(os.getenv("NON_INCREMENTAL_FILL_LOG_DEDUP_SECONDS", "120") or "120")
)


def _patch_core_app(module: ModuleType) -> None:
    current = getattr(module, "time", None)
    # Preserve ``time(15, 24)`` semantics while repairing ``time.time()`` calls.
    if current is _datetime_time or not hasattr(current, "time"):
        setattr(module, "time", _TIME_PROXY)
    if not hasattr(module, "time_module"):
        setattr(module, "time_module", _time_module)


def _patch_candle_engine(module: ModuleType) -> None:
    engine_cls = getattr(module, "CandleEngine", None)
    original = getattr(engine_cls, "_finalize_current_candle", None)
    if engine_cls is None or not callable(original):
        return
    if getattr(engine_cls, "_live_safety_monotonic_patch", False):
        return

    data_integrity_error = getattr(module, "DataIntegrityError", Exception)
    sanitize = getattr(module, "sanitize", None)
    logger = getattr(module, "LOGGER", logging.getLogger("nifty_scalper_bot.data.candle_engine"))
    log_throttled = getattr(module, "log_throttled", None)

    def _patched_finalize(self: Any) -> dict[str, Any] | None:
        try:
            return original(self)
        except data_integrity_error as exc:  # type: ignore[misc]
            if "monotonic" not in str(exc).lower():
                raise
            symbol = getattr(self, "symbol", None) or "symbol_unset"
            with suppress(Exception):
                if callable(sanitize):
                    self.df = sanitize(getattr(self, "df", None)).tail(
                        int(getattr(self, "max_bars", 500) or 500)
                    ).reset_index(drop=True)
            self.current_candle = None
            if callable(log_throttled):
                log_throttled(
                    logger,
                    f"candle_monotonic_reset:{symbol}",
                    "CANDLE_ENGINE_SYMBOL_RESET symbol=%s reason=out_of_order_candle_dropped source=runtime_live_safety_hotfix",
                    symbol,
                    interval_sec=30.0,
                    level=logging.WARNING,
                    extra={
                        "event": "CANDLE_ENGINE_SYMBOL_RESET",
                        "symbol": symbol,
                        "reason": "out_of_order_candle_dropped",
                        "source": "runtime_live_safety_hotfix",
                    },
                )
            else:
                logger.warning(
                    "CANDLE_ENGINE_SYMBOL_RESET symbol=%s reason=out_of_order_candle_dropped source=runtime_live_safety_hotfix",
                    symbol,
                    extra={
                        "event": "CANDLE_ENGINE_SYMBOL_RESET",
                        "symbol": symbol,
                        "reason": "out_of_order_candle_dropped",
                        "source": "runtime_live_safety_hotfix",
                    },
                )
            return None

    setattr(engine_cls, "_finalize_current_candle", _patched_finalize)
    setattr(engine_cls, "_live_safety_monotonic_patch", True)


def _live_mode_enabled() -> bool:
    mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
    live_flag = str(
        os.getenv("ENABLE_LIVE") or os.getenv("ENABLE_LIVE_TRADING") or "false"
    ).strip().lower() in {"1", "true", "yes", "on", "live"}
    paper_or_shadow = str(os.getenv("PAPER_MODE", "false")).lower() in {"1", "true", "yes", "on"} or str(
        os.getenv("SHADOW_MODE", "false")
    ).lower() in {"1", "true", "yes", "on"}
    return mode == "LIVE" and live_flag and not paper_or_shadow


def _patch_strategy_runner(module: ModuleType) -> None:
    runner_cls = getattr(module, "StrategyRunner", None)
    original = getattr(runner_cls, "_schedule_signal_preparation", None)
    if runner_cls is None or not callable(original):
        return
    if getattr(runner_cls, "_live_orders_armed_preflight_patch", False):
        return

    def _patched_schedule_signal_preparation(self: Any, signal: Any, *args: Any, **kwargs: Any) -> Any:
        if _live_mode_enabled() and not bool(getattr(self, "_runtime_live_orders_armed", False)):
            symbol = str(getattr(signal, "symbol", "") or kwargs.get("symbol") or "")
            reason = str(getattr(self, "_runtime_readiness_reason", "") or "execution_not_armed")
            logger = getattr(self, "_logger", logging.getLogger("nifty_scalper_bot.strategies.runner"))
            logger.info(
                "RUNNER_SIGNAL_PREP_BLOCKED_UNARMED symbol=%s reason=%s broker_attempted=False",
                symbol,
                reason,
                extra={
                    "event": "RUNNER_SIGNAL_PREP_BLOCKED_UNARMED",
                    "symbol": symbol,
                    "reason": reason,
                    "live_orders_armed": False,
                    "broker_attempted": False,
                    "source": "runtime_live_safety_hotfix",
                },
            )
            return False, f"execution_not_armed:{reason}"
        return original(self, signal, *args, **kwargs)

    setattr(runner_cls, "_schedule_signal_preparation", _patched_schedule_signal_preparation)
    setattr(runner_cls, "_live_orders_armed_preflight_patch", True)


def _install_fill_dedup_filter() -> None:
    logger = logging.getLogger("nifty_scalper_bot.execution.position_manager")
    if not any(isinstance(item, _NonIncrementalFillDedupFilter) for item in logger.filters):
        logger.addFilter(_DEDUP_FILTER)


def _patch_module(fullname: str, module: ModuleType) -> None:
    with _LOCK:
        if fullname in _PATCHED_MODULES:
            return
        if fullname == "nifty_scalper_bot.core.app":
            _patch_core_app(module)
        elif fullname == "nifty_scalper_bot.data.candle_engine":
            _patch_candle_engine(module)
        elif fullname == "nifty_scalper_bot.strategies.runner":
            _patch_strategy_runner(module)
        _PATCHED_MODULES.add(fullname)


def install_live_safety_hotfixes() -> None:
    """Install import hooks and apply patches for already-loaded modules."""

    global _HOOK_INSTALLED
    with _LOCK:
        if not _HOOK_INSTALLED:
            if not any(isinstance(item, _PostImportPatchFinder) for item in sys.meta_path):
                sys.meta_path.insert(0, _PostImportPatchFinder())
            _HOOK_INSTALLED = True
        _install_fill_dedup_filter()
        for fullname in tuple(_TARGETS):
            module = sys.modules.get(fullname)
            if module is not None:
                _patch_module(fullname, module)


__all__ = ["install_live_safety_hotfixes"]
