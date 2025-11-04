"""Deterministic helpers for environment variable parsing."""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Iterable, TypeVar, cast

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

T = TypeVar("T")


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY


def _first_env(names: tuple[str, ...]) -> tuple[str | None, str | None]:
    for name in names:
        val = os.getenv(name)
        if val is not None:
            return name, val
    return None, None


def _normalize_args(values: tuple[Any, ...], default: T) -> tuple[tuple[str, ...], T]:
    fallback: T = default
    raw: Iterable[Any] = values
    if values and not isinstance(values[-1], str):
        fallback = cast(T, values[-1])
        raw = values[:-1]
    names = tuple(str(name) for name in raw if isinstance(name, str))
    return names, fallback


def get_bool(*names: Any, default: bool = False) -> bool:
    resolved_names, fallback = _normalize_args(names, default)
    if not resolved_names:
        return bool(fallback)
    _name, raw = _first_env(resolved_names)
    if raw is None:
        return bool(fallback)
    normalized = raw.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    return bool(fallback)


def get_str(*names: Any, default: str = "") -> str:
    resolved_names, fallback = _normalize_args(names, default)
    if not resolved_names:
        return fallback
    _name, raw = _first_env(resolved_names)
    if raw is None:
        return fallback
    return raw


def _coerce(
    converter: Callable[[str], float | int],
    names: tuple[str, ...],
    default: float | int,
) -> float | int:
    if not names:
        return default
    _name, raw = _first_env(names)
    if raw is None or not raw.strip():
        return default
    try:
        return converter(raw)
    except Exception:
        return default


def get_int(*names: Any, default: int = 0) -> int:
    resolved_names, fallback = _normalize_args(names, default)
    return int(_coerce(int, resolved_names, default=fallback))


def get_float(*names: Any, default: float = 0.0) -> float:
    resolved_names, fallback = _normalize_args(names, default)
    return float(_coerce(float, resolved_names, default=fallback))


def get_csv(*names: Any) -> list[str]:
    """Return sanitized tokens from comma and newline separated env values.

    Args:
        *names: Candidate environment variable names.

    Returns:
        A list of non-empty, trimmed tokens without stray key/value pairs.
    """

    resolved_names, _ = _normalize_args(names, "")
    raw = get_str(*resolved_names, default="") if resolved_names else ""
    if not raw:
        return []

    parts = re.split(r"[,\r\n]+", raw)
    tokens: list[str] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
            token = token[1:-1].strip()
        if "=" in token:
            continue
        if token:
            tokens.append(token)
    return tokens


def coalesce_str(*names: Any, default: str | None = None) -> str:
    resolved_names, fallback = _normalize_args(names, default or "")
    return get_str(*resolved_names, default=fallback)


def coalesce_bool(*names: Any, default: bool | None = None) -> bool:
    resolved_names, fallback = _normalize_args(
        names, default if default is not None else False
    )
    if not resolved_names:
        return bool(fallback) if default is not None else False
    _name, raw = _first_env(resolved_names)
    if raw is None:
        return bool(fallback) if default is not None else False
    normalized = raw.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    return bool(fallback) if default is not None else False


def coalesce_int(*names: Any, default: int | None = None) -> int:
    resolved_names, fallback = _normalize_args(names, default or 0)
    if not resolved_names:
        return int(fallback)
    _name, raw = _first_env(resolved_names)
    if raw is None:
        return int(fallback)
    try:
        return int(raw)
    except Exception:
        return int(fallback)


def coalesce_float(*names: Any, default: float | None = None) -> float:
    resolved_names, fallback = _normalize_args(names, default or 0.0)
    if not resolved_names:
        return float(fallback)
    _name, raw = _first_env(resolved_names)
    if raw is None:
        return float(fallback)
    try:
        return float(raw)
    except Exception:
        return float(fallback)


def normalize_path(p: str | None) -> str | None:
    if not p:
        return p
    return p.rstrip(" .")


__all__ = [
    "get_bool",
    "get_str",
    "get_int",
    "get_float",
    "get_csv",
    "coalesce_str",
    "coalesce_bool",
    "coalesce_int",
    "coalesce_float",
    "normalize_path",
]
