"""Tests for ATR gate logging helpers."""

from __future__ import annotations

import importlib

import pytest

from src.strategies import atr_gate


@pytest.mark.parametrize(
    "atr_values",
    [
        (1.21, 1.24),
        (0.91, 0.94),
    ],
)
def test_normalise_log_state_buckets_ok_values(monkeypatch, atr_values) -> None:
    """ATR readings within the bucket should normalise to the same state."""

    monkeypatch.setenv("ATR_LOG_OK_BUCKET", "0.5")
    module = importlib.reload(atr_gate)

    first = module._normalise_log_state(
        "BANKNIFTY",
        atr_value=atr_values[0],
        min_val=0.8,
        max_bound=2.0,
        ok=True,
    )
    second = module._normalise_log_state(
        "BANKNIFTY",
        atr_value=atr_values[1],
        min_val=0.8,
        max_bound=2.0,
        ok=True,
    )
    assert first == second

    monkeypatch.delenv("ATR_LOG_OK_BUCKET", raising=False)
    importlib.reload(module)


def test_normalise_log_state_preserves_blocked_precision(monkeypatch) -> None:
    """Out-of-band readings retain the full precision for diagnostics."""

    monkeypatch.setenv("ATR_LOG_OK_BUCKET", "0.5")
    module = importlib.reload(atr_gate)

    _, state = module._normalise_log_state(
        "FINNIFTY",
        atr_value=2.3456,
        min_val=0.8,
        max_bound=2.0,
        ok=False,
    )
    assert state[-1] == pytest.approx(2.3456, rel=0, abs=1e-4)

    monkeypatch.delenv("ATR_LOG_OK_BUCKET", raising=False)
    importlib.reload(module)
