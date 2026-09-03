from __future__ import annotations

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


BarIdentity = tuple[object, float, float, float, float, float]


def _bar(ts: int, close: float) -> BarIdentity:
    return (ts, close - 1.0, close + 1.0, close - 2.0, close, 10.0)


def test_projection_rollover_accepts_matching_retained_overlap() -> None:
    previous = [_bar(1, 100.0), _bar(2, 101.0), _bar(3, 102.0)]
    canonical = [_bar(2, 101.0), _bar(3, 102.0), _bar(4, 103.0)]

    assert MarketDataManager._projection_matches_canonical_slice(previous, canonical)


def test_projection_rollover_rejects_ohlcv_mismatch_in_retained_overlap() -> None:
    previous = [_bar(1, 100.0), _bar(2, 999.0), _bar(3, 102.0)]
    canonical = [_bar(2, 101.0), _bar(3, 102.0), _bar(4, 103.0)]

    assert not MarketDataManager._projection_matches_canonical_slice(previous, canonical)


def test_projection_rollover_rejects_disjoint_windows() -> None:
    previous = [_bar(1, 100.0), _bar(2, 101.0)]
    canonical = [_bar(3, 102.0), _bar(4, 103.0)]

    assert not MarketDataManager._projection_matches_canonical_slice(previous, canonical)
