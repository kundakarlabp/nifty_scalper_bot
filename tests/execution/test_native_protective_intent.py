from nifty_scalper_bot.execution.runtime_order_manager import (
    RuntimeOrderManager,
    _bind_place_order,
    _normalise_protective_intent_kwargs,
)


def test_protective_intent_is_owned_by_runtime_order_manager() -> None:
    assert RuntimeOrderManager.place_order.__module__ == (
        "nifty_scalper_bot.execution.runtime_order_manager"
    )
    assert not hasattr(RuntimeOrderManager, "_protective_order_intent_patch")


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


def test_explicit_intent_remains_authoritative_and_is_normalised():
    kwargs = _normalise_protective_intent_kwargs(
        {
            "tag": "FLATTEN_TELEGRAM",
            "check_risk": False,
            "intent": "exit",
            "strategy_name": "bracket",
        }
    )

    assert kwargs["intent"] == "EXIT"
    assert kwargs["strategy_name"] == "bracket"


def test_positional_order_fields_are_bound_before_intent_inference():
    values = _bind_place_order(
        ("NFO:NIFTY24JUL24000CE", "SELL", 75),
        {"tag": "SL_HAR_abc123", "check_risk": False},
    )

    assert values is not None
    kwargs = _normalise_protective_intent_kwargs(values)
    assert kwargs["symbol"] == "NFO:NIFTY24JUL24000CE"
    assert kwargs["side"] == "SELL"
    assert kwargs["quantity"] == 75
    assert kwargs["intent"] == "EXIT"
