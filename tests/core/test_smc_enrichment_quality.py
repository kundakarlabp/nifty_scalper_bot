from nifty_scalper_bot.core.strategy_manager import _enrich_smc_pre_strategy


def test_vwap_touch_is_context_not_smc_structural_retest() -> None:
    bars = []
    for index in range(25):
        base = 100.0 + index * 0.1
        bars.append(
            {
                "open": base,
                "high": base + 1.0,
                "low": base - 0.1,
                "close": base + 0.2,
                "volume": 1000.0,
            }
        )
    bars[10] = {
        "open": 104.0,
        "high": 120.0,
        "low": 103.5,
        "close": 104.5,
        "volume": 1000.0,
    }
    for index in range(20, 25):
        bars[index]["low"] = 99.95

    enriched = _enrich_smc_pre_strategy(
        "NFO:NIFTY26SEP23100CE",
        {"direction_bias": "CE", "vwap": 100.0},
        bars,
    )

    assert enriched["vwap_retest_context"] is True
    assert enriched["structure_retest_confirmed"] is False
    assert enriched["retest_confirmed"] is False
