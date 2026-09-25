from nifty_scalper_bot.strategies.signal_quality import (
    score_signal_metadata,
    score_signal_quality,
)


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


def test_smc_runner_uses_independent_setup_score(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    quality = score_signal_metadata(
        {
            "direction_score": 6.5,
            "strategy_score": 7.5,
            "independent_setup_score": 6.0,
            "option_score": 10.0,
            "data_score": 10.0,
            "rr_score": 10.0,
        },
        strategy_name="SMC",
    )

    assert quality.direction_score == 6.5
    assert quality.strategy_score == 6.0
    assert quality.components["threshold"] == 7.0
    assert quality.allowed is True
