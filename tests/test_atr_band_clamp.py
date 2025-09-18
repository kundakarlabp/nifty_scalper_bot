import smoke  # noqa: F401
from types import SimpleNamespace

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
    ok, reason, min_val, max_val = check_atr(0.05, cfg, "NIFTY")
    assert ok
    assert reason is None
    assert min_val == 0.05
    assert max_val == 0.9


def test_check_atr_prefers_threshold_over_gate():
    """Threshold minimums must override lower gate bounds."""

    cfg = _make_cfg(min_nifty=0.05, gate_min=0.02, gate_max=0.9)
    ok, reason, min_val, _ = check_atr(0.04, cfg, "NIFTY")
    assert not ok
    assert min_val == 0.05
    assert isinstance(reason, str) and "< min=0.05" in reason
