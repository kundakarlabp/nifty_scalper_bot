from src.notifications.telegram_controller import _kpis
import pytest


def test_summary_kpis() -> None:
    trades = [
        {"pnl_R": 1.0},
        {"pnl_R": -0.5},
        {"pnl_R": 2.0},
        {"pnl_R": -1.5},
    ]
    k = _kpis(trades)
    assert k["trades"] == 4
    assert k["PF"] == pytest.approx(1.5)
    assert k["Win%"] == pytest.approx(50.0)
    assert k["MaxDD_R"] == pytest.approx(1.5)
