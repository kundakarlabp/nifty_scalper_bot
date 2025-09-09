from src.utils.broker_errors import (
    AUTH,
    classify_broker_error,
    SUBSCRIPTION,
    THROTTLE,
    UNKNOWN,
)


def test_generic_string_unknown() -> None:
    msg = "Incorrect 'api_key' or 'access_token'"
    assert classify_broker_error(msg) == UNKNOWN
    assert classify_broker_error(msg, 400) == UNKNOWN


def test_auth_classification() -> None:
    assert classify_broker_error("Incorrect 'api_key' or 'access_token'", 401) == AUTH
    assert classify_broker_error("Session expired", 403) == AUTH
    assert classify_broker_error("Invalid session token") == AUTH
    assert classify_broker_error("Not a valid access token") == AUTH


def test_throttle_and_subscription_classification() -> None:
    assert classify_broker_error("Rate limit exceeded", 429) == THROTTLE
    assert classify_broker_error("Too many requests") == THROTTLE
    assert (
        classify_broker_error("No subscription for this instrument")
        == SUBSCRIPTION
    )
