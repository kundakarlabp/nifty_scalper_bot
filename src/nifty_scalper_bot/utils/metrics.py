"""Tiny metrics facade with safe fallbacks for missing Prometheus support."""

# ruff: noqa: I001

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Sequence, Tuple, cast

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

_PCounter: Any | None = None
_PGauge: Any | None = None
_PHistogram: Any | None = None

_DEFAULT_DIR = Path("/tmp/prometheus")

_PROM_MODULE: ModuleType | None = None
spec = importlib.util.find_spec("prometheus_client")
if spec is not None:
    try:
        _PROM_MODULE = importlib.import_module("prometheus_client")
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.info(
            "Prometheus client import failed; metrics disabled",
            extra={"event": "metrics_import_failure", "error": str(exc)},
        )
        _PROM_MODULE = None

if _PROM_MODULE is not None:
    _PCounter = cast(
        Any,
        getattr(_PROM_MODULE, "Counter", None),
    )
    _PGauge = cast(
        Any,
        getattr(_PROM_MODULE, "Gauge", None),
    )
    _PHistogram = cast(
        Any,
        getattr(_PROM_MODULE, "Histogram", None),
    )


_METRIC_CACHE: Dict[tuple[Any, ...], Any] = {}
_PROMETHEUS_WARNING_LOGGED = False


def ensure_multiproc_dir(clear_stale: bool = False) -> Path | None:
    """Ensure the Prometheus multiprocess directory exists and is writable.

    Args:
        clear_stale: When ``True`` delete stale ``*.db`` files from the
            multiprocess directory.

    Returns:
        Path | None: The directory path prepared for Prometheus multiprocess
            metrics, or ``None`` when preparation fails.

    Raises:
        None. Exceptions are caught and logged before returning ``None``.
    """

    dir_str = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    target = Path(dir_str) if dir_str else _DEFAULT_DIR
    try:
        target.mkdir(parents=True, exist_ok=True)
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(target)
        if clear_stale:
            for db_file in target.glob("*.db"):
                try:
                    db_file.unlink(missing_ok=True)
                except OSError:
                    pass
        return target
    except Exception as exc:  # pragma: no cover - defensive fallback
        fallback = _DEFAULT_DIR / f"mp-{os.getpid()}"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(fallback)
            LOGGER.warning(
                "Prometheus multiprocess dir fallback engaged",
                extra={"error": str(exc), "fallback": str(fallback)},
            )
            if clear_stale:
                for db_file in fallback.glob("*.db"):
                    try:
                        db_file.unlink(missing_ok=True)
                    except OSError:
                        pass
            return fallback
        except Exception as inner_exc:  # pragma: no cover - defensive fallback
            LOGGER.warning(
                "Prometheus multiprocess dir prep failed; disabling metrics",
                extra={"dir": str(target), "error": f"{exc}; fallback={inner_exc}"},
            )
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            return None


# Prepare directory as early as possible to avoid race conditions.
ensure_multiproc_dir(clear_stale=False)


class _Noop:
    """Simple no-op metric used when Prometheus is unavailable."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self._labels: Dict[Tuple[Any, ...], "_Noop"] = {}

    def labels(self, *args: Any, **kwargs: Any) -> "_Noop":
        key = tuple(args)
        if kwargs:
            key += tuple(sorted(kwargs.items()))
        return self._labels.setdefault(key, _Noop())

    def inc(self, *_: Any, **__: Any) -> None:  # noqa: D401 - intentionally empty
        """No-op."""

    def dec(self, *_: Any, **__: Any) -> None:  # noqa: D401 - intentionally empty
        """No-op."""

    def set(self, *_: Any, **__: Any) -> None:  # noqa: D401 - intentionally empty
        """No-op."""

    def observe(self, *_: Any, **__: Any) -> None:  # noqa: D401 - intentionally empty
        """No-op."""


def Counter(name: str, doc: str, labels: Iterable[str] | None = None) -> Any:
    """Return a Prometheus ``Counter`` or a no-op replacement.

    Args:
        name: Metric identifier shared across all processes.
        doc: Human-readable description of the metric.
        labels: Optional sequence of label names for the metric series.

    Returns:
        Any: Concrete Prometheus ``Counter`` instance or an inert fallback.

    Raises:
        None. Initialization errors fall back to a no-op metric and are logged.
    """

    if _PCounter is None:
        return _Noop()
    label_key = tuple(labels or ())
    cache_key = ("counter", name, label_key)
    metric = _METRIC_CACHE.get(cache_key)
    if metric is None:
        ensure_multiproc_dir()
        try:
            metric = _PCounter(name, doc, list(label_key))
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning(
                "Prometheus Counter init failed; using no-op",
                extra={"metric_name": name, "error": str(exc)},
            )
            metric = _Noop()
        _METRIC_CACHE[cache_key] = metric
    return metric


def Gauge(name: str, doc: str, labels: Iterable[str] | None = None) -> Any:
    """Return a Prometheus ``Gauge`` or a no-op replacement.

    Args:
        name: Metric identifier shared across all processes.
        doc: Human-readable description of the metric.
        labels: Optional sequence of label names for the metric series.

    Returns:
        Any: Concrete Prometheus ``Gauge`` instance or an inert fallback.

    Raises:
        None. Initialization errors fall back to a no-op metric and are logged.
    """

    if _PGauge is None:
        return _Noop()
    label_key = tuple(labels or ())
    cache_key = ('gauge', name, label_key)
    metric = _METRIC_CACHE.get(cache_key)
    if metric is None:
        ensure_multiproc_dir()
        try:
            multiproc_mode = (
                'livesum'
                if os.environ.get('PROMETHEUS_MULTIPROC_DIR')
                else None
            )
            if multiproc_mode:
                try:
                    metric = _PGauge(
                        name,
                        doc,
                        list(label_key),
                        multiprocess_mode=multiproc_mode,
                    )
                except TypeError:
                    metric = _PGauge(name, doc, list(label_key))
            else:
                metric = _PGauge(name, doc, list(label_key))
        except Exception as exc:  # pragma: no cover - defensive fallback
            global _PROMETHEUS_WARNING_LOGGED
            if not _PROMETHEUS_WARNING_LOGGED:
                LOGGER.warning(
                    'Prometheus disabled; using no-op metrics',
                    extra={'metric_name': name, 'error': str(exc)},
                )
                _PROMETHEUS_WARNING_LOGGED = True
            metric = _Noop()
        _METRIC_CACHE[cache_key] = metric
    return metric


def Histogram(
    name: str,
    doc: str,
    labels: Iterable[str] | None = None,
    *,
    buckets: Sequence[float] | None = None,
) -> Any:
    """Return a Prometheus ``Histogram`` or a no-op replacement.

    Args:
        name: Metric identifier shared across all processes.
        doc: Human-readable description of the metric.
        labels: Optional sequence of label names for the metric series.
        buckets: Optional list of bucket boundaries for histogram buckets.

    Returns:
        Any: Concrete Prometheus ``Histogram`` instance or an inert fallback.

    Raises:
        None. Initialization errors fall back to a no-op metric and are logged.
    """

    if _PHistogram is None:
        return _Noop()
    label_key = tuple(labels or ())
    bucket_key = tuple(buckets) if buckets is not None else None
    cache_key = ("histogram", name, label_key, bucket_key)
    metric = _METRIC_CACHE.get(cache_key)
    if metric is None:
        ensure_multiproc_dir()
        try:
            if buckets is None:
                metric = _PHistogram(name, doc, list(label_key))
            else:
                metric = _PHistogram(name, doc, list(label_key), buckets=buckets)
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning(
                "Prometheus Histogram init failed; using no-op",
                extra={"metric_name": name, "error": str(exc)},
            )
            metric = _Noop()
        _METRIC_CACHE[cache_key] = metric
    return metric


reference_price_source_total: Any
strategy_skips_total: Any

if "reference_price_source_total" not in globals():
    reference_price_source_total = Counter(
        "reference_price_source_total", "Ref price source used", ["source"]
    )

if "strategy_skips_total" not in globals():
    strategy_skips_total = Counter(
        "strategy_skips_total", "Strategy skips by reason", ["reason"]
    )


__all__ = ["Counter", "Gauge", "Histogram", "ensure_multiproc_dir"]
