from src.utils.strike_selector import (
    select_strike,
    needs_reatm,
    set_option_info_fetcher,
)


def test_liquidity_guard(monkeypatch):
    def fetcher(strike: int):
        if strike == 19500:
            return {"oi": 400000, "spreads": [0.4]}
        return {"oi": 600000, "spreads": [0.2]}

    set_option_info_fetcher(fetcher)
    assert select_strike(19510, score=8) is None
    info = select_strike(19510, score=10)
    assert info is not None and info.strike != 19500


def test_needs_reatm():
    assert needs_reatm(100.0, 100.5) is True
    assert needs_reatm(100.0, 100.2) is False
