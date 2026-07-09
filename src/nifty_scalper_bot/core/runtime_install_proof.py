"""Runtime install-proof snapshot for market-data hardening patches.

This module is observability only. It reports whether the runtime process has the
expected hardening/adapter hooks installed without mutating trading state.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

_CORE_APP_HOOK_ATTR = "_nifty_scalper_core_app_patch_hook"
_DATAHUB_HOOK_ATTR = "_nifty_scalper_datahub_synthetic_guard_hook"


def _hook_count(attr: str) -> int:
    return sum(1 for finder in sys.meta_path if bool(getattr(finder, attr, False)))


def _module(name: str) -> ModuleType | None:
    value = sys.modules.get(name)
    return value if isinstance(value, ModuleType) else None


def _class_attr(module_name: str, class_name: str, attr: str) -> bool:
    module = _module(module_name)
    cls = getattr(module, class_name, None) if module is not None else None
    return bool(getattr(cls, attr, False))


def _module_attr(module_name: str, attr: str) -> bool:
    module = _module(module_name)
    return bool(getattr(module, attr, False)) if module is not None else False


def build_runtime_install_proof(ctx: Any | None = None) -> dict[str, Any]:
    """Return compact runtime install-proof status.

    Args:
        ctx: Optional live bot context. When supplied, instance-level references
            are inspected in addition to class/module-level patch markers.

    Returns:
        dict[str, Any]: JSON-safe install proof snapshot.
    """

    mdm = getattr(ctx, "market_data_manager", None) if ctx is not None else None
    datahub = getattr(ctx, "data_hub", None) if ctx is not None else None
    ws_manager = getattr(ctx, "websocket_manager", None) if ctx is not None else None
    mdm_cls = type(mdm) if mdm is not None else None
    datahub_cls = type(datahub) if datahub is not None else None
    ws_cls = type(ws_manager) if ws_manager is not None else None
    core_hook_count = _hook_count(_CORE_APP_HOOK_ATTR)
    datahub_hook_count = _hook_count(_DATAHUB_HOOK_ATTR)

    market_data_manager_hardened = bool(
        getattr(mdm_cls, "_freshness_hardening_installed", False)
        or _class_attr(
            "nifty_scalper_bot.data.market_data_manager",
            "MarketDataManager",
            "_freshness_hardening_installed",
        )
    )
    websocket_hardened = bool(
        getattr(ws_cls, "_market_data_hardening_installed", False)
        or _class_attr(
            "nifty_scalper_bot.streaming.websocket_manager",
            "WebSocketManager",
            "_market_data_hardening_installed",
        )
    )
    datahub_guarded = bool(
        getattr(datahub_cls, "_synthetic_timestamp_guard_installed", False)
        or _class_attr(
            "nifty_scalper_bot.data.data_hub",
            "DataHub",
            "_synthetic_timestamp_guard_installed",
        )
    )
    polling_failover_patched = bool(
        _module_attr("nifty_scalper_bot.core.app", "_polling_failover_runtime_patch_installed")
    )

    return {
        "market_data_manager_hardened": market_data_manager_hardened,
        "websocket_hardened": websocket_hardened,
        "datahub_synthetic_guard_installed": datahub_guarded,
        "polling_failover_runtime_patch_installed": polling_failover_patched,
        "core_app_import_hook_installed": core_hook_count == 1,
        "datahub_import_hook_installed": datahub_hook_count == 1,
        "import_hook_counts": {
            "core_app": core_hook_count,
            "datahub": datahub_hook_count,
        },
        "all_required_installed": bool(
            market_data_manager_hardened
            and websocket_hardened
            and datahub_guarded
            and polling_failover_patched
            and core_hook_count == 1
            and datahub_hook_count == 1
        ),
    }


__all__ = ["build_runtime_install_proof"]
