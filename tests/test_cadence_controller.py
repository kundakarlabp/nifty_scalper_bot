from src.strategies.runner import CadenceController


def test_cadence_responds_to_atr() -> None:
    cc = CadenceController()
    fast = cc.update(0.8, 0.1, False)
    slow = cc.update(0.05, 0.1, False)
    assert fast < slow
    aged = cc.update(0.8, 2.0, False)
    assert aged >= fast
