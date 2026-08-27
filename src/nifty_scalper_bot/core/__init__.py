"""Core orchestration utilities for the trading bot."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from functools import wraps
from types import ModuleType
from typing import Any

from nifty_scalper_bot.utils.logging import get_logger
# TODO: remove compatibility shim once downstream modules are updated.
from nifty_scalper_bot.utils.pricing import canonical_price_source  # compat

_CORE_TRUTHY = {"1", "true", "yes", "y", "on", "live"}
_APP_MODULE_NAME = "nifty_scalper_bot.core.app"
_APP_IMPORT_HOOK_ATTR = "_nifty_scalper_core_app_patch_hook"
_RUNTIME_HARDENING_REQUIRED = (
    "market_data_hardening",
    "dynamic_universe",
    "live_ws_tick_receipts",
    "runtime_reliability",
    "runner_candle_cache",
    "strategy_context_fast_path",
    "off_market_controller",
    "off_market_app",
    "session_boundary",
    "boot_readiness",
    "polling_failover",
)


def _core_env_true(name: str) -> bool:
    return str(os.getenv(name, "") or "").strip().lower() in _CORE_TRUTHY


def _real_live_mode_requested() -> bool:
    mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
    live_enabled = _core_env_true("ENABLE_LIVE") or _core_env_true("ENABLE_LIVE_TRADING")
    paper_shadow = (
        _core_env_true("PAPER_MODE")
        or _core_env_true("PAPER__ENABLED")
        or _core_env_true("SHADOW_MODE")
    )
    return mode == "LIVE" and live_enabled and not paper_shadow


try:
    from nifty_scalper_bot.core.strategy_live_safety import (
        apply_patches as _apply_strategy_live_safety,
    )

    _apply_strategy_live_safety()
except Exception as exc:  # noqa: BLE001 - non-live tooling imports should remain usable
    get_logger(__name__).error(
        "STRATEGY_LIVE_SAFETY_PATCH_FAILED error=%s",
        exc,
        extra={"event": "STRATEGY_LIVE_SAFETY_PATCH_FAILED", "error_type": type(exc).__name__},
    )
    if _real_live_mode_requested():
        raise RuntimeError("strategy_live_safety_patch_failed") from exc

try:
    from nifty_scalper_bot.core.strategy_exit_score_diagnostics import (
        apply_patches as _apply_strategy_exit_score_diagnostics,
    )

    _apply_strategy_exit_score_diagnostics()
except Exception as exc:  # noqa: BLE001 - diagnostics must not disable tooling imports
    get_logger(__name__).error(
        "STRATEGY_EXIT_SCORE_DIAGNOSTIC_PATCH_FAILED error=%s",
        exc,
        extra={
            "event": "STRATEGY_EXIT_SCORE_DIAGNOSTIC_PATCH_FAILED",
            "error_type": type(exc).__name__,
        },
    )

try:
    from nifty_scalper_bot.core.strategy_setup_score_gate import (
        apply_patches as _apply_strategy_setup_score_gate,
    )

    _apply_strategy_setup_score_gate()
except Exception as exc:  # noqa: BLE001 - non-live tooling imports should remain usable
    get_logger(__name__).error(
        "STRATEGY_SETUP_SCORE_GATE_PATCH_FAILED error=%s",
        exc,
        extra={"event": "STRATEGY_SETUP_SCORE_GATE_PATCH_FAILED", "error_type": type(exc).__name__},
    )
    if _real_live_mode_requested():
        raise RuntimeError("strategy_setup_score_gate_patch_failed") from exc

try:
    from nifty_scalper_bot.core.boot_log_safety import (
        apply_filters as _apply_boot_log_rate_controls,
    )

    _apply_boot_log_rate_controls()
except Exception as exc:  # noqa: BLE001 - core package import must not crash tooling
    get_logger(__name__).error(
        "BOOT_LOG_RATE_CONTROL_FAILED error=%s",
        exc,
        extra={"event": "BOOT_LOG_RATE_CONTROL_FAILED", "error_type": type(exc).__name__},
    )

__all__ = ["NiftyScalperApp", "canonical_price_source", "install_runtime_hardening"]

_LOGGER = get_logger(__name__)


def _install_market_data_runtime_hardening() -> dict[str, bool]:
    """Install candle/manager/WebSocket hardening in the real app import path.

    This is deliberately called after ``core.app`` has completed importing, so
    settings and production classes are fully initialized. Live startup fails
    closed if any required market-data layer cannot be installed.
    """
    from nifty_scalper_bot.core.market_data_hardening_bootstrap import (
        install_market_data_hardening_or_raise,
    )

    try:
        state = install_market_data_hardening_or_raise(_LOGGER)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error(
            "MARKET_DATA_RUNTIME_HARDENING_FAILED error=%s",
            exc,
            exc_info=True,
            extra={
                "event": "MARKET_DATA_RUNTIME_HARDENING_FAILED",
                "error_type": type(exc).__name__,
            },
        )
        if _real_live_mode_requested():
            raise RuntimeError("market_data_runtime_hardening_failed") from exc
        return {
            "candle": False,
            "clock_flush": False,
            "mdm": False,
            "websocket": False,
        }
    return state


def _install_runner_candle_engine_cache_patch() -> bool:
    """Avoid reacquiring two shared registry locks for every steady-state tick.

    MDM owns one stable CandleEngine object per canonical symbol. Runner only
    mirrors that object for compatibility, so after first resolution the same
    reference can be returned directly. Concurrent first use still delegates to
    the original locked MDM/Runner path and therefore preserves SSOT identity.
    """
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    attr = "_candle_engine_mirror_cache_patch_installed"
    if bool(getattr(StrategyRunner, attr, False)):
        return True
    original = StrategyRunner._mirror_authoritative_candle_engine

    @wraps(original)
    def _mirror_authoritative_candle_engine(self: Any, symbol: str) -> Any | None:
        normalized = self._normalize_symbol(symbol)
        cached = getattr(self, "_candle_engines", {}).get(normalized)
        if cached is not None:
            return cached
        return original(self, symbol)

    StrategyRunner._mirror_authoritative_candle_engine = (  # type: ignore[method-assign]
        _mirror_authoritative_candle_engine
    )
    setattr(StrategyRunner, attr, True)
    return True


def _marked_callable(owner: Any, name: str, marker: str) -> bool:
    value = getattr(owner, name, None)
    return callable(value) and bool(getattr(value, marker, False))


def _boot_readiness_installation_complete(app_module: Any) -> bool:
    """Verify every class/function adapter installed by boot_readiness_safety."""
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager
    from nifty_scalper_bot.strategies.indicators import IndicatorEngine
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    checks = (
        _marked_callable(
            app_module,
            "compute_live_readiness",
            "_session_readiness_adapted",
        ),
        _marked_callable(
            app_module,
            "_replay_latest_mdm_ticks_to_bus",
            "_inactive_bus_replay_guarded",
        ),
        _marked_callable(
            app_module,
            "_register_and_subscribe_live_symbol",
            "_runner_activation_gated",
        ),
        _marked_callable(
            app_module,
            "_wire_and_start_message_bus",
            "_direct_mdm_bus_detach_adapted",
        ),
        _marked_callable(
            StrategyRunner,
            "sync_history_from_mdm",
            "_history_role_corrected",
        ),
        _marked_callable(
            MarketDataManager,
            "_update_pipeline_overload_locked",
            "_active_drain_overload_recovery_guarded",
        ),
        _marked_callable(
            IndicatorEngine,
            "get_history",
            "_missing_history_single_log_adapted",
        ),
        _marked_callable(
            IndicatorEngine,
            "get_indicators",
            "_option_direction_context_authority_adapted",
        ),
    )
    return all(checks)


def _runtime_hardening_install_proof(
    app_module: Any,
    *,
    runtime_reliability_state: Any,
) -> dict[str, bool]:
    """Return explicit proof that replay and LIVE share all runtime adapters."""
    from nifty_scalper_bot.core.strategy_manager import StrategyManager
    from nifty_scalper_bot.core.universe_controller import UniverseController
    from nifty_scalper_bot.data.candle_engine import CandleEngine
    from nifty_scalper_bot.data.data_hub import DataHub
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager
    from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    reliability = (
        isinstance(runtime_reliability_state, dict)
        and bool(runtime_reliability_state)
        and all(bool(value) for value in runtime_reliability_state.values())
    )
    market_data = all(
        (
            bool(getattr(CandleEngine, "_candle_state_hardening_installed", False)),
            bool(
                getattr(
                    MarketDataManager,
                    "_candle_clock_flush_hardening_installed",
                    False,
                )
            ),
            bool(getattr(DataHub, "_active_basket_subscription_hardening_installed", False)),
            bool(getattr(MarketDataManager, "_freshness_hardening_installed", False)),
            bool(getattr(WebSocketManager, "_market_data_hardening_installed", False)),
        )
    )
    return {
        "market_data_hardening": market_data,
        "dynamic_universe": bool(
            getattr(StrategyRunner, "_dynamic_universe_safety_installed", False)
        ),
        "live_ws_tick_receipts": bool(
            getattr(MarketDataManager, "_live_ws_tick_receipt_patch_installed", False)
        ),
        "runtime_reliability": reliability,
        "runner_candle_cache": bool(
            getattr(StrategyRunner, "_candle_engine_mirror_cache_patch_installed", False)
        ),
        "strategy_context_fast_path": bool(
            getattr(StrategyManager, "_context_only_fast_path_installed", False)
        ),
        "off_market_controller": bool(
            getattr(UniverseController, "_off_market_basket_safety_installed", False)
        ),
        "off_market_app": bool(
            getattr(app_module, "_off_market_basket_commit_safety_installed", False)
        ),
        "session_boundary": bool(
            getattr(app_module, "_session_boundary_rearm_installed", False)
        ),
        "boot_readiness": _boot_readiness_installation_complete(app_module),
        "polling_failover": bool(
            getattr(app_module, "_polling_failover_runtime_patch_installed", False)
        ),
    }


def _apply_app_runtime_patches(app_module: Any) -> dict[str, bool]:
    """Install and verify the single runtime-hardening contract idempotently."""
    _install_market_data_runtime_hardening()

    from nifty_scalper_bot.core.boot_readiness_safety import apply_app_patch as _ready_adapter
    from nifty_scalper_bot.core.live_ws_tick_receipts import apply_patch as _live_ws_receipt_adapter
    from nifty_scalper_bot.core.off_market_basket_safety import (
        apply_app_patch as _off_market_app_adapter,
        apply_patches as _off_market_controller_adapter,
    )
    from nifty_scalper_bot.core.polling_failover_runtime import apply_app_patch as _polling_adapter
    from nifty_scalper_bot.core.runtime_reliability_hardening import (
        apply_patches as _runtime_reliability_adapter,
    )
    from nifty_scalper_bot.core.session_boundary_rearm import (
        apply_app_patch as _session_boundary_adapter,
    )
    from nifty_scalper_bot.core.strategy_context_fast_path import (
        apply_patches as _strategy_context_fast_path_adapter,
    )
    from nifty_scalper_bot.core.strategy_runner_dynamic_universe_safety import (
        apply_patches as _dynamic_universe_adapter,
    )

    _dynamic_universe_adapter()
    _live_ws_receipt_adapter()
    runtime_reliability_state = _runtime_reliability_adapter()
    _install_runner_candle_engine_cache_patch()
    _strategy_context_fast_path_adapter()
    _off_market_controller_adapter()
    _off_market_app_adapter(app_module)
    _session_boundary_adapter(app_module)
    _ready_adapter(app_module)
    _polling_adapter(app_module)

    proof = _runtime_hardening_install_proof(
        app_module,
        runtime_reliability_state=runtime_reliability_state,
    )
    missing = [name for name in _RUNTIME_HARDENING_REQUIRED if not proof.get(name)]
    if missing:
        _LOGGER.error(
            "RUNTIME_HARDENING_INCOMPLETE missing=%s proof=%s",
            missing,
            proof,
            extra={
                "event": "RUNTIME_HARDENING_INCOMPLETE",
                "missing": missing,
                "proof": proof,
            },
        )
        raise RuntimeError(
            f"runtime_hardening_incomplete missing={missing} proof={proof}"
        )
    setattr(app_module, "_runtime_hardening_install_proof", dict(proof))
    return proof


def install_runtime_hardening() -> dict[str, bool]:
    """Install the exact production runtime adapters for LIVE, replay and tests.

    The public contract is intentionally explicit: validation code must not rely
    on whether an import hook happened to run. Missing hardening is an error, not
    a silently degraded replay mode.
    """
    from . import app as _app_module

    return _apply_app_runtime_patches(_app_module)


class _CoreAppPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        create = getattr(self._wrapped, "create_module", None)
        if callable(create):
            return create(spec)
        return None

    def exec_module(self, module: ModuleType) -> None:
        exec_module = getattr(self._wrapped, "exec_module", None)
        if callable(exec_module):
            exec_module(module)
        else:
            load_module = getattr(self._wrapped, "load_module", None)
            if callable(load_module):
                loaded = load_module(module.__name__)  # pragma: no cover - legacy loader path
                if loaded is not module:
                    module.__dict__.update(getattr(loaded, "__dict__", {}))
        _apply_app_runtime_patches(module)


class _CoreAppPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != _APP_MODULE_NAME:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or isinstance(spec.loader, _CoreAppPatchLoader):
            return spec
        spec.loader = _CoreAppPatchLoader(spec.loader)
        return spec


def _install_core_app_patch_import_hook() -> None:
    if any(getattr(finder, _APP_IMPORT_HOOK_ATTR, False) for finder in sys.meta_path):
        return
    finder = _CoreAppPatchFinder()
    setattr(finder, _APP_IMPORT_HOOK_ATTR, True)
    sys.meta_path.insert(0, finder)


_install_core_app_patch_import_hook()


def __getattr__(name: str) -> Any:
    """Return lazily imported core entry points."""
    _LOGGER.debug(
        "Entered core.__getattr__",
        extra={"event": "core_init_getattr_enter", "attribute": name},
    )
    if name in {"NiftyScalperApp", "app"}:
        try:
            from . import app as _app_module

            _apply_app_runtime_patches(_app_module)
            if name == "app":
                return _app_module
            _App = _app_module.NiftyScalperApp
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error(
                "Failure in core.__getattr__: %s",
                exc,
                extra={"event": "core_init_getattr_error", "attribute": name},
            )
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from exc
        _LOGGER.info(
            "Condition met: resolved NiftyScalperApp",
            extra={"event": "core_init_getattr_resolved", "attribute": name},
        )
        return _App
    _LOGGER.info(
        "Condition met: attribute not exported",
        extra={"event": "core_init_getattr_missing", "attribute": name},
    )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
