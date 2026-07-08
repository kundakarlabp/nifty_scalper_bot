from nifty_scalper_bot.execution.protective_order_intent_patch import _normalise_protective_intent_kwargs


def test_exit_tag_with_risk_bypass_gets_explicit_exit_intent():
    kwargs = _normalise_protective_intent_kwargs(
        {
            "symbol": "NFO:NIFTY24JUL24000CE",
            "side": "SELL",
            "quantity": 75,
            "tag": "exit_HAR_abc123",
            "check_risk": False,
        }
    )

    assert kwargs["intent"] == "EXIT"
    assert kwargs["strategy_name"] == "protective_exit"


def test_flatten_tag_gets_explicit_reduce_intent():
    kwargs = _normalise_protective_intent_kwargs(
        {
            "symbol": "NFO:NIFTY24JUL24000CE",
            "side": "SELL",
            "quantity": 75,
            "tag": "FLATTEN_TELEGRAM",
            "check_risk": False,
        }
    )

    assert kwargs["intent"] == "REDUCE"
    assert kwargs["strategy_name"] == "operator_flatten"


def test_normal_entry_is_not_relabelled_from_tag_text():
    kwargs = _normalise_protective_intent_kwargs(
        {
            "symbol": "NFO:NIFTY24JUL24000CE",
            "side": "BUY",
            "quantity": 75,
            "tag": "runner",
            "check_risk": True,
        }
    )

    assert "intent" not in kwargs
