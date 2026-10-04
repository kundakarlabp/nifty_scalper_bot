"""Development-only ORB ranking must freeze five candidates before OOS testing."""

from scripts.research_orb_multiyear import select_exploratory_top5


def _row(name: str, slip: float, expectancy: float, trades: int) -> dict:
    return {
        "candidate": name,
        "slippage_bps_per_side": slip,
        "metrics": {"expectancy": expectancy, "trade_count": trades},
        "data_quality": {"unresolved_exit_count": 0},
    }


def test_select_exploratory_top5_uses_development_only_10bps_ranking():
    rows = []
    for index, expectancy in enumerate((8, 7, 6, 5, 4, 3), start=1):
        name = f"candidate_{index}"
        rows.extend(
            [
                _row(name, 10.0, expectancy, 100 + index),
                _row(name, 25.0, expectancy - 2, 50),
                _row(name, 50.0, expectancy - 4, 20),
            ]
        )
    assert select_exploratory_top5(rows) == [
        "candidate_1",
        "candidate_2",
        "candidate_3",
        "candidate_4",
        "candidate_5",
    ]


def test_select_exploratory_top5_requires_100_resolved_base_trades():
    rows = [
        _row("thin", 10.0, 100.0, 99),
        _row("thin", 25.0, 90.0, 50),
        _row("thin", 50.0, 80.0, 20),
        _row("eligible", 10.0, 1.0, 100),
        _row("eligible", 25.0, 0.0, 50),
        _row("eligible", 50.0, -1.0, 20),
    ]
    assert select_exploratory_top5(rows) == ["eligible"]
