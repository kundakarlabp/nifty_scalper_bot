import smoke  # noqa: F401
import logging
from types import SimpleNamespace

import pytest

from src.signals.patches import resolve_atr_band
from src.strategies import atr_gate
from src.strategies.atr_gate import check_atr


def _make_cfg(
    *,
    min_nifty: float | None = None,
    min_banknifty: float | None = None,
    gate_min: float | None = None,
    gate_max: float | None = None,
):
    raw: dict[str, object] = {}
    thresholds: dict[str, float] = {}
    if min_nifty is not None:
        thresholds["min_atr_pct_nifty"] = min_nifty
    if min_banknifty is not None:
        thresholds["min_atr_pct_banknifty"] = min_banknifty
    if thresholds:
        raw["thresholds"] = thresholds
    gates: dict[str, float] = {}
    if gate_min is not None:
        gates["atr_pct_min"] = gate_min
    if gate_max is not None:
        gates["atr_pct_max"] = gate_max
    if gates:
        raw["gates"] = gates
    return SimpleNamespace(raw=raw)


def test_check_atr_allows_minimum_threshold():
    """ATR exactly on the configured minimum should pass the gate."""

    cfg = _make_cfg(min_nifty=0.05, gate_min=0.02, gate_max=0.9)
    band = resolve_atr_band(cfg, "NIFTY")
    assert band[0] == pytest.approx(0.05)
    assert band[1] == pytest.approx(0.9)
    ok, reason, min_val, max_val = check_atr(0.05, cfg, "NIFTY")
    assert ok
    assert reason is None
    assert min_val == 0.05
    assert max_val == 0.9


def test_check_atr_prefers_threshold_over_gate():
    """Threshold minimums must override lower gate bounds."""

    cfg = _make_cfg(min_nifty=0.05, gate_min=0.02, gate_max=0.9)
    band = resolve_atr_band(cfg, "NIFTY")
    assert band[0] == pytest.approx(0.05)
    assert band[1] == pytest.approx(0.9)
    ok, reason, min_val, _ = check_atr(0.04, cfg, "NIFTY")
    assert not ok
    assert min_val == 0.05
    assert isinstance(reason, str) and "< min=0.05" in reason


def test_resolve_atr_band_uses_gate_when_threshold_missing():
    """Gate minimums should apply when thresholds are absent."""

    cfg = _make_cfg(min_nifty=None, gate_min=0.03, gate_max=0.8)
    band = resolve_atr_band(cfg, "NIFTY")
    assert band[0] == pytest.approx(0.03)
    assert band[1] == pytest.approx(0.8)


def test_check_atr_throttles_repeated_info_logs(caplog):
    """Repeated ATR values should emit info logs only once."""

    atr_gate._reset_log_throttle_state()
    cfg = _make_cfg(gate_min=0.02, gate_max=0.9)

    with caplog.at_level(logging.DEBUG, logger="src.strategies.atr_gate"):
        check_atr(0.05, cfg, "NIFTY")
        check_atr(0.05, cfg, "NIFTY")

    info_logs = [record for record in caplog.records if record.levelno == logging.INFO]
    debug_logs = [record for record in caplog.records if record.levelno == logging.DEBUG]
    assert len(info_logs) == 1
    assert len(debug_logs) == 1


def test_check_atr_logs_when_value_changes(caplog):
    """Changing ATR values should produce a fresh info log."""

    atr_gate._reset_log_throttle_state()
    cfg = _make_cfg(gate_min=0.02, gate_max=0.9)

    with caplog.at_level(logging.INFO, logger="src.strategies.atr_gate"):
        check_atr(0.05, cfg, "NIFTY")
        check_atr(0.051, cfg, "NIFTY")

    info_logs = [record for record in caplog.records if record.levelno == logging.INFO]
    assert len(info_logs) == 2


def test_check_atr_out_of_band_always_logs(caplog):
    """Out-of-band readings should not be throttled."""

    atr_gate._reset_log_throttle_state()
    cfg = _make_cfg(gate_min=0.02, gate_max=0.9)

    with caplog.at_level(logging.INFO, logger="src.strategies.atr_gate"):
        check_atr(0.005, cfg, "NIFTY")
        check_atr(0.005, cfg, "NIFTY")

    info_logs = [record for record in caplog.records if record.levelno == logging.INFO]
    assert len(info_logs) == 2
