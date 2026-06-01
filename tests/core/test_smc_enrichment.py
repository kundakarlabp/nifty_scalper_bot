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


def test_normalise_ohlcv_bars_handles_dataframe_and_zero_values():
    from nifty_scalper_bot.core.strategy_manager import _normalise_ohlcv_bars

    class _Frame:
        def __init__(self, rows):
            self._rows = rows

        def tail(self, _limit):
            return self

        def to_dict(self, orient):
            assert orient == "records"
            return self._rows

    rows = _Frame([
        {"open": 0, "o": 999, "high": 0, "h": 999, "low": 0, "l": 999, "close": 0, "c": 999, "volume": 0, "v": 999},
        {"o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 0},
        "bad-row",
        {"open": None, "high": 2, "low": 1, "close": 1.5},
    ])

    normalised = _normalise_ohlcv_bars(rows)

    assert normalised == [
        {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0},
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 0.0},
    ]


def test_extract_option_strike_accepts_4_to_6_digits():
    from nifty_scalper_bot.core.strategy_manager import _extract_option_strike

    assert _extract_option_strike("NFO:NIFTY26MAY9500CE") == 9500
    assert _extract_option_strike("NFO:NIFTY26MAY24000PE") == 24000
    assert _extract_option_strike("NFO:NIFTY26MAY124000CE") == 124000


def test_smc_enrichment_separates_presence_from_positive_signal_count():
    bars = [_bar(100, 102, 99, 101) for _ in range(25)]
    enriched = _enrich_smc_pre_strategy(
        "NFO:NIFTY26MAY24000CE",
        {
            "premium_reclaim": False,
            "bullish_reversal": False,
            "choch_confirmed": False,
            "bos_confirmed": False,
            "retest_confirmed": False,
        },
        bars,
    )

    assert enriched["feature_completeness"] == 1.0
    assert enriched["smc_feature_presence_ratio"] == 1.0
    assert enriched["smc_positive_feature_count"] == 0
    assert enriched["smc_positive_features"] == []


def test_runtime_strategy_manager_class_is_core_manager_or_enrichment_hook_executes(caplog):
    import logging
    import typing as t
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    symbol = "NFO:NIFTY26MAY24000CE"

    class _Strategy:
        name = "SMC"
        last = {}

        def get_required_indicators(self):
            return []

        def generate_signal(self, symbol: str, indicators: t.Mapping[str, t.Any], current_price: float, position: t.Any):
            self.last = dict(indicators)
            return None

    class _IE:
        def get_history(self, _symbol):
            return [_bar(100, 102, 99, 101) for _ in range(30)]

        def get_indicators(self, _symbol, _names):
            return {"vwap": 100, "exchange_vwap": 100, "volume": 100, "avg_volume": 100}

    class _PM:
        def get_position(self, _symbol):
            return None

    st = _Strategy()
    with caplog.at_level(logging.INFO):
        manager = StrategyManager([st], _IE(), _PM())
        manager.generate_signal(symbol, 101)

    assert manager.__class__.__module__ == "nifty_scalper_bot.core.strategy_manager"
    assert st.last["smc_enrichment_source"] == "strategy_manager_pre_strategy"
    assert "RUNTIME_STRATEGY_MANAGER_CLASS" in caplog.text


def test_history_provider_exception_does_not_crash_generate_signal(caplog):
    import logging
    import typing as t
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    symbol = "NFO:NIFTY26MAY24000CE"

    class _Strategy:
        name = "SMC"
        seen = False

        def get_required_indicators(self):
            return []

        def generate_signal(self, symbol: str, indicators: t.Mapping[str, t.Any], current_price: float, position: t.Any):
            self.seen = True
            return None

    class _Hub:
        indicators_ready = True

        def get_ohlc_bars(self, *_args, **_kwargs):
            raise RuntimeError("history provider down")

        def get_quote(self, *_args, **_kwargs):
            return {}

    class _IE:
        def get_history(self, _symbol):
            return [_bar(100, 102, 99, 101) for _ in range(30)]

        def get_indicators(self, _symbol, _names):
            return {"vwap": 100, "exchange_vwap": 100, "volume": 100, "avg_volume": 100}

    class _PM:
        def get_position(self, _symbol):
            return None

    st = _Strategy()
    manager = StrategyManager([st], _IE(), _PM(), data_hub=_Hub())
    with caplog.at_level(logging.WARNING):
        manager.generate_signal(symbol, 101)

    assert st.seen is True
    assert "SMC_HISTORY_PROVIDER_FAILED" in caplog.text
