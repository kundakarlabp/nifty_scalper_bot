from nifty_scalper_bot.core.strategy_manager import _enrich_smc_pre_strategy


def _bar(open_, high, low, close, volume=100):
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def test_rolling_pivot_detects_latest_confirmed_without_current_lookahead():
    bars = [_bar(100, 101 + (i % 3), 99 - (i % 2), 100 + (i % 4) * 0.1) for i in range(80)]
    pivot_idx = len(bars) - 1 - 5 - 3
    bars[pivot_idx] = _bar(100, 150, 80, 110)
    bars[-1] = _bar(120, 999, 119, 121)  # unfinished/current bar must not become pivot

    enriched = _enrich_smc_pre_strategy("NFO:NIFTY26MAY24000CE", {}, bars)

    assert enriched["swing_high"] == 150
    assert enriched["swing_low"] == 80
    assert enriched["swing_high"] != 999
    assert enriched["smc_enrichment_lookahead_safe"] is True


def test_premium_domain_fields_use_option_premium_values():
    bars = [_bar(100, 106, 99, 103) for _ in range(25)]
    bars[-2] = _bar(103, 106, 102, 104)
    bars[-1] = _bar(104, 111, 103, 110)

    enriched = _enrich_smc_pre_strategy(
        "NFO:NIFTY26MAY24000CE",
        {"vwap": 105},
        bars,
    )

    assert enriched["premium_current"] == 110
    assert enriched["premium_prev_close"] == 104
    assert enriched["premium_vwap"] == 105
    assert enriched["premium_reclaim"] is True


def test_feature_completeness_counts_false_booleans_as_present():
    bars = [_bar(100, 102, 99, 101) for _ in range(25)]
    enriched = _enrich_smc_pre_strategy(
        "NFO:NIFTY26MAY24000CE",
        {"bos_confirmed": False},
        bars,
    )

    assert enriched["bos_confirmed"] is False
    assert enriched["feature_completeness"] > 0
    assert enriched["feature_completeness"] == 1.0
