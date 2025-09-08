from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Protocol, Any

# ---- Protocols (no imports from your code here; keep it decoupled) ----

class StrategyLike(Protocol):
    def name(self) -> str: ...
    # evaluate(...) signature stays whatever you have today

class DataProviderLike(Protocol):
    @property
    def name(self) -> str: ...
    def health(self) -> dict: ...  # {status: "OK"/"DEGRADED"/"DOWN", details:{...}}

class OrderConnectorLike(Protocol):
    @property
    def name(self) -> str: ...
    def health(self) -> dict: ...

# ---- Registries --------------------------------------------------------

@dataclass
class _Entry:
    factory: Callable[[], Any]
    desc: str

class _BaseRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, _Entry] = {}

    def register(self, key: str, factory: Callable[[], Any], desc: str = "") -> None:
        k = key.strip().lower()
        if not k:
            raise ValueError("empty registry key")
        self._items[k] = _Entry(factory=factory, desc=desc or key)

    def create(self, key: str) -> Any:
        k = (key or "").strip().lower()
        if k not in self._items:
            raise KeyError(f"Unknown key: {key!r}; available={list(self._items.keys())}")
        return self._items[k].factory()

    def keys(self) -> list[str]:
        return sorted(self._items.keys())

    def describe(self) -> dict:
        return {k: v.desc for k, v in self._items.items()}

class StrategyRegistry(_BaseRegistry): ...
class DataProviderRegistry(_BaseRegistry): ...
class OrderConnectorRegistry(_BaseRegistry): ...

# ---- Adapters (wrap your existing classes without changing them) -------

def make_data_provider_adapter(obj: Any, name: str) -> DataProviderLike:
    class _Adapter(DataProviderLike):  # type: ignore[misc]
        @property
        def name(self) -> str:
            return name
        def health(self) -> dict:
            # Try to ask the underlying object; else return a minimal OK.
            try:
                if hasattr(obj, "health"):
                    return obj.health()
            except Exception as e:
                return {"status": "DOWN", "details": {"error": str(e)}}
            return {"status": "OK", "details": {}}
        # Expose original for callers that expect the concrete object
        def __getattr__(self, item):  # delegate
            return getattr(obj, item)
    return _Adapter()

def make_order_connector_adapter(obj: Any, name: str) -> OrderConnectorLike:
    class _Adapter(OrderConnectorLike):  # type: ignore[misc]
        @property
        def name(self) -> str:
            return name
        def health(self) -> dict:
            try:
                if hasattr(obj, "health"):
                    return obj.health()
            except Exception as e:
                return {"status": "DOWN", "details": {"error": str(e)}}
            return {"status": "OK", "details": {}}
        def __getattr__(self, item):
            return getattr(obj, item)
    return _Adapter()

# ---- Bootstrap helpers -------------------------------------------------

@dataclass
class ActiveComponents:
    strategy: StrategyLike
    data_provider: DataProviderLike
    order_connector: OrderConnectorLike
    names: dict  # {"strategy": "...", "data_provider": "...", "order_connector": "..."}

def init_default_registries(settings, *, make_strategy, make_data_kite, make_connector_kite, make_connector_shadow) -> ActiveComponents:
    """
    settings: your config object (must expose ACTIVE_STRATEGY/ACTIVE_DATA_PROVIDER/ACTIVE_CONNECTOR or env-backed).
    make_*: callables returning your existing instances. We wrap them with adapters.
    """
    sreg = StrategyRegistry()
    dreg = DataProviderRegistry()
    oreg = OrderConnectorRegistry()

    # Register strategy(ies)
    sreg.register("scalping", lambda: make_strategy(), "NIFTY scalping strategy (current)")
    # add more here later: sreg.register("meanrev", ...)

    # Register data providers
    dreg.register("kite", lambda: make_data_provider_adapter(make_data_kite(), "kite"), "Broker minute OHLC (primary)")
    dreg.register("auto", lambda: make_data_provider_adapter(make_data_kite(), "auto"), "Auto-select (currently kite)")

    # Register order connectors
    oreg.register("kite", lambda: make_order_connector_adapter(make_connector_kite(), "kite"), "Zerodha Kite connector")
    oreg.register("shadow", lambda: make_order_connector_adapter(make_connector_shadow(), "shadow"), "Shadow (paper) connector")

    # Resolve names from settings/env (non-fatal fallbacks)
    s_name = (getattr(settings, "ACTIVE_STRATEGY", None) or "scalping").lower()
    d_name = (getattr(settings, "ACTIVE_DATA_PROVIDER", None) or "auto").lower()
    o_name = (getattr(settings, "ACTIVE_CONNECTOR", None) or "kite").lower()

    try:
        strategy = sreg.create(s_name)
    except Exception:
        strategy = sreg.create("scalping")
        s_name = "scalping"

    try:
        data_provider = dreg.create(d_name)
    except Exception:
        data_provider = dreg.create("kite")
        d_name = "kite"

    try:
        order_connector = oreg.create(o_name)
    except Exception:
        order_connector = oreg.create("kite")
        o_name = "kite"

    return ActiveComponents(
        strategy=strategy,
        data_provider=data_provider,
        order_connector=order_connector,
        names={"strategy": s_name, "data_provider": d_name, "order_connector": o_name},
    )
