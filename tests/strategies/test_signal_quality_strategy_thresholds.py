from nifty_scalper_bot.strategies.signal_quality import score_signal_quality


def test_strategy_threshold_aliases_live(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")

    for strategy_name in ("RSIDivergence", "CPRBreakout", "BBSqueeze", "OrderFlow"):
        context = score_signal_quality(
            direction_score=10,
            strategy_score=10,
            option_score=10,
            data_score=10,
            rr_score=10,
            strategy_name=strategy_name,
        )
        assert context.components["threshold"] == 8.0
        assert not context.allowed
        assert "context_only_strategy" in context.reasons

    unk = score_signal_quality(
        direction_score=8,
        strategy_score=8,
        option_score=8,
        data_score=8,
        rr_score=7,
        strategy_name="unknown",
    )
    assert unk.components["threshold"] == 8.0
    assert not (unk.final_score < 8.0 and unk.allowed)

    premium = score_signal_quality(
        direction_score=7.5,
        strategy_score=7.5,
        option_score=7.5,
        data_score=7.5,
        rr_score=7.0,
        strategy_name="premium_momentum_squeeze",
    )
    assert premium.components["threshold"] == 7.4
    assert premium.allowed

    smc = score_signal_quality(
        direction_score=7.2,
        strategy_score=7.2,
        option_score=7.2,
        data_score=7.2,
        rr_score=7.2,
        strategy_name="smc",
    )
    assert smc.components["threshold"] == 7.0
    assert smc.allowed

    orb = score_signal_quality(
        direction_score=7.5,
        strategy_score=7.5,
        option_score=7.5,
        data_score=7.5,
        rr_score=7.5,
        strategy_name="ORBPro",
    )
    assert orb.components["threshold"] == 7.4
    assert orb.allowed


def test_new_global_score_setting_uses_exact_zero_to_ten_units(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("GLOBAL_MIN_SIGNAL_SCORE", "8.6")
    monkeypatch.setenv("GLOBAL_MIN_SIGNAL_CONFIDENCE", "0.1")

    score = score_signal_quality(
        direction_score=9,
        strategy_score=9,
        option_score=9,
        data_score=9,
        rr_score=9,
        strategy_name="VWAPPro",
    )

    assert score.components["threshold"] == 8.6


def test_legacy_global_confidence_preserves_fraction_compatibility(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.delenv("GLOBAL_MIN_SIGNAL_SCORE", raising=False)
    monkeypatch.setenv("GLOBAL_MIN_SIGNAL_CONFIDENCE", "0.82")

    score = score_signal_quality(
        direction_score=9,
        strategy_score=9,
        option_score=9,
        data_score=9,
        rr_score=9,
        strategy_name="VWAPPro",
    )

    assert score.components["threshold"] == 8.2


def test_canonical_trigger_setting_does_not_reinterpret_fraction(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("TRIGGER_VWAP_PRO_LIVE_MIN_SCORE", "0.75")
    monkeypatch.setenv("TRIGGER_VWAP_PRO_LIVE_MIN", "9.5")

    score = score_signal_quality(
        direction_score=9,
        strategy_score=9,
        option_score=9,
        data_score=9,
        rr_score=9,
        strategy_name="VWAPPro",
    )

    assert score.components["threshold"] == 0.75
