from __future__ import annotations

from typing import Any

from src.nifty_scalper_bot.utils import metrics


def test_gauge_uses_multiprocess_mode_when_env_set(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """Test multiprocess Gauge config. Args: None. Returns: None. Raises: None."""
    calls: dict[str, Any] = {}

    class DummyGauge:
        def __init__(self, *_: Any, **kwargs: Any) -> None:
            """Store call kwargs. Args: None. Returns: None. Raises: None."""
            calls.update(kwargs)

    monkeypatch.setenv('PROMETHEUS_MULTIPROC_DIR', str(tmp_path))
    monkeypatch.setattr(metrics, '_PGauge', DummyGauge)
    metrics._METRIC_CACHE.clear()

    metrics.Gauge('test_gauge', 'desc', ['label'])

    assert calls.get('multiprocess_mode') == 'livesum'
