"""Data layer exports used across the runtime.

Hardening note: MarketDataManager hardening is installed explicitly in
``market_data_manager.py`` and the IST time adapter in ``source.py`` — at
their definition sites. DataHub quote timestamp-quality guarding and required
MarketDataManager tick-backlog bounding are installed by narrow import hooks so
the package import does not eagerly load either runtime module.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType

from nifty_scalper_bot.brokers.instrument_lookup import Instrument
from nifty_scalper_bot.data.instrument_resolver import InstrumentResolver
from nifty_scalper_bot.data.instrument_loader import (
    InstrumentUniverseStatus,
    ensure_sqlite,
    load_rows_for_resolver,
    parse_kite_csv,
    refresh_from_csv,
    sync_instrument_csv_from_broker,
    upsert_instruments,
    write_instrument_rows_to_csv,
)

_DATAHUB_MODULE_NAME = "nifty_scalper_bot.data.data_hub"
_MDM_MODULE_NAME = "nifty_scalper_bot.data.market_data_manager"
_DATAHUB_IMPORT_HOOK_ATTR = "_nifty_scalper_datahub_synthetic_guard_hook"
_MDM_IMPORT_HOOK_ATTR = "_nifty_scalper_mdm_required_tick_backlog_hook"
Installer = Callable[[ModuleType], None]


def _install_datahub_guard(module: ModuleType) -> None:
    datahub_cls = getattr(module, "DataHub", None)
    if datahub_cls is None:
        return
    from nifty_scalper_bot.data.data_hub_synthetic_guard import (
        install_datahub_synthetic_timestamp_guard,
    )

    install_datahub_synthetic_timestamp_guard(datahub_cls)


def _install_mdm_guard(module: ModuleType) -> None:
    manager_cls = getattr(module, "MarketDataManager", None)
    if manager_cls is None:
        return
    from nifty_scalper_bot.data.required_tick_backlog_hardening import (
        install_required_tick_backlog_hardening,
    )

    install_required_tick_backlog_hardening(manager_cls)


class _GuardLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, installer: Installer) -> None:
        self._wrapped = wrapped
        self._installer = installer

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
        self._installer(module)


class _GuardFinder(importlib.abc.MetaPathFinder):
    def __init__(self, module_name: str, installer: Installer) -> None:
        self._module_name = module_name
        self._installer = installer

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != self._module_name:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or isinstance(spec.loader, _GuardLoader):
            return spec
        spec.loader = _GuardLoader(spec.loader, self._installer)
        return spec


def _install_guard_import_hook(
    *, module_name: str, marker_attr: str, installer: Installer
) -> None:
    module = sys.modules.get(module_name)
    if isinstance(module, ModuleType):
        installer(module)
        return
    if any(getattr(finder, marker_attr, False) for finder in sys.meta_path):
        return
    finder = _GuardFinder(module_name, installer)
    setattr(finder, marker_attr, True)
    sys.meta_path.insert(0, finder)


_install_guard_import_hook(
    module_name=_DATAHUB_MODULE_NAME,
    marker_attr=_DATAHUB_IMPORT_HOOK_ATTR,
    installer=_install_datahub_guard,
)
_install_guard_import_hook(
    module_name=_MDM_MODULE_NAME,
    marker_attr=_MDM_IMPORT_HOOK_ATTR,
    installer=_install_mdm_guard,
)


def __getattr__(name: str) -> ModuleType:
    if name == "rest":
        module = importlib.import_module("nifty_scalper_bot.data.rest")
        setattr(sys.modules[__name__], name, module)
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Instrument",
    "InstrumentResolver",
    "InstrumentUniverseStatus",
    "ensure_sqlite",
    "load_rows_for_resolver",
    "parse_kite_csv",
    "refresh_from_csv",
    "sync_instrument_csv_from_broker",
    "upsert_instruments",
    "write_instrument_rows_to_csv",
]
