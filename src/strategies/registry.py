from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Type


class StrategyRegistry:
    """Registry for strategy classes keyed by name."""

    _strategies: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[Any]) -> None:
        cls._strategies[name] = strategy_cls

    @classmethod
    def get(cls, name: str) -> Type[Any]:
        if name not in cls._strategies:
            raise KeyError(f"strategy '{name}' not registered")
        return cls._strategies[name]


class DataProviderRegistry:
    """Registry for data source providers with optional health scoring."""

    _providers: Dict[str, Tuple[Type[Any], Callable[[], float]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        provider_cls: Type[Any],
        health_fn: Callable[[], float] | None = None,
    ) -> None:
        cls._providers[name] = (provider_cls, health_fn or (lambda: 0.0))

    @classmethod
    def get(cls, name: str) -> Type[Any]:
        if name not in cls._providers:
            raise KeyError(f"data provider '{name}' not registered")
        return cls._providers[name][0]

    @classmethod
    def best(cls) -> Tuple[str, Type[Any]]:
        if not cls._providers:
            raise RuntimeError("no data providers registered")
        name, (provider, _) = max(
            cls._providers.items(), key=lambda item: item[1][1]()
        )
        return name, provider


class OrderConnectorRegistry:
    """Registry for order connectors."""

    _connectors: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: str, connector_cls: Type[Any]) -> None:
        cls._connectors[name] = connector_cls

    @classmethod
    def get(cls, name: str) -> Type[Any]:
        if name not in cls._connectors:
            raise KeyError(f"order connector '{name}' not registered")
        return cls._connectors[name]
