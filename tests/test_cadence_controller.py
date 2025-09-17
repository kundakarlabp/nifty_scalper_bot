import pytest

from src.strategies.runner import CadenceController


def test_cadence_responds_to_atr() -> None:
    cc = CadenceController()
    fast = cc.update(0.8, 0.1, False)
    slow = cc.update(0.05, 0.1, False)
    assert fast < slow
    aged = cc.update(0.8, 2.0, False)
    assert aged >= fast


def test_cadence_step_configurable() -> None:
    cc = CadenceController(step=0.5)
    baseline = cc.update(0.8, 0.1, False)
    widened = cc.update(0.8, 2.0, False)
    assert widened > baseline
    assert widened - baseline == pytest.approx(0.5)
