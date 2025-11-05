"""Typed configuration loader for the order execution hub."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml  # type: ignore[import]

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class OrderQueueConfig(BaseModel):
    """Queue sizing and prioritisation controls."""

    model_config = {"extra": "forbid"}

    max_size: PositiveInt = Field(default=100)
    exit_priority: PositiveInt = Field(default=10)
    entry_priority: PositiveInt = Field(default=3)


class ValidationConfig(BaseModel):
    """Preflight validation thresholds and guardrails."""

    model_config = {"extra": "forbid"}

    quote_max_age_ms: PositiveInt = Field(default=2000)
    spread_max_pct: float = Field(default=1.5, ge=0.0)
    depth_min_lots: PositiveInt = Field(default=1)
    margin_buffer_pct: float = Field(default=20.0, ge=0.0)


class RoutingConfig(BaseModel):
    """Execution routing parameters including retry policy."""

    model_config = {"extra": "forbid"}

    mode: str = Field(default="SHADOW")
    retry_attempts: PositiveInt = Field(default=3)
    retry_delay_ms: PositiveInt = Field(default=500)
    partial_fill_tolerance_pct: float = Field(default=5.0, ge=0.0)

    @field_validator("mode")
    @classmethod
    def _normalise_mode(cls, value: str) -> str:
        """Normalise routing mode tokens."""

        return value.strip().upper() or "SHADOW"


class LifecycleGammaConfig(BaseModel):
    """Gamma mode overrides for lifecycle management."""

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=True)
    tuesday_after: str | None = Field(default="14:45")
    tp2_R_override: float = Field(default=1.4, ge=0.0)


class LifecycleSection(BaseModel):
    """Lifecycle risk-reward configuration."""

    model_config = {"extra": "forbid"}

    tp1_R: float = Field(default=1.0, ge=0.0)
    tp1_partial: float = Field(default=0.6, ge=0.0, le=1.0)
    tp2_R_trend: float = Field(default=1.8, ge=0.0)
    tp2_R_range: float = Field(default=1.4, ge=0.0)
    trail_atr_mult: float = Field(default=0.8, ge=0.0)
    time_stop_min: PositiveInt = Field(default=12)
    gamma: LifecycleGammaConfig = Field(default_factory=LifecycleGammaConfig)


class ReconciliationConfig(BaseModel):
    """Reconciliation cadence and behaviour toggles."""

    model_config = {"extra": "forbid"}

    interval_sec: PositiveInt = Field(default=30)
    alert_on_mismatch: bool = Field(default=True)
    broker_is_truth: bool = Field(default=True)


class OrderExecutionConfig(BaseSettings):
    """Aggregate configuration for the order execution hub."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    queue: OrderQueueConfig = Field(default_factory=OrderQueueConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    lifecycle: LifecycleSection = Field(default_factory=LifecycleSection)
    reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)


@dataclass(slots=True)
class OrderExecutionConfigBundle:
    """Bundle returning resolved configuration and source path."""

    config: OrderExecutionConfig
    source_path: Path


def _load_yaml_payload(path: Path) -> dict[str, Any]:
    """Load YAML configuration from ``path`` for order execution settings.

    Args:
        path: Path to the configuration file.

    Returns:
        dict[str, Any]: Parsed configuration data or an empty dictionary.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered _load_yaml_payload",
        extra={"event": "order_execution_config_yaml", "path": str(path)},
    )
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        LOGGER.error(
            "Failure in _load_yaml_payload: missing file",
            extra={"event": "order_execution_config_missing", "path": str(path)},
        )
        return {}
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _load_yaml_payload: %s",
            exc,
            extra={"event": "order_execution_config_error", "path": str(path)},
            exc_info=exc,
        )
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def load_order_execution_config(
    path: str | Path | None = None,
) -> OrderExecutionConfigBundle:
    """Resolve and validate order execution configuration.

    Args:
        path: Optional override path for the YAML configuration file.

    Returns:
        OrderExecutionConfigBundle: Bundle containing the config and source path.

    Raises:
        ValueError: If configuration validation fails.
    """

    LOGGER.debug(
        "Entered load_order_execution_config",
        extra={"event": "order_execution_config_load", "path": str(path)},
    )
    config_path = Path(path or "config/order_execution.yaml").resolve()
    payload = _load_yaml_payload(config_path)
    try:
        config = OrderExecutionConfig(**payload)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in load_order_execution_config: %s",
            exc,
            extra={
                "event": "order_execution_config_validate_error",
                "path": str(config_path),
            },
            exc_info=exc,
        )
        raise ValueError("Invalid order execution configuration") from exc
    return OrderExecutionConfigBundle(config=config, source_path=config_path)


__all__ = [
    "OrderExecutionConfig",
    "OrderExecutionConfigBundle",
    "OrderQueueConfig",
    "ValidationConfig",
    "RoutingConfig",
    "LifecycleGammaConfig",
    "LifecycleSection",
    "ReconciliationConfig",
    "load_order_execution_config",
]
