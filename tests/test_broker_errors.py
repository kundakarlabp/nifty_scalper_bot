from src.utils.broker_errors import classify_broker_error, AUTH, UNKNOWN


def test_generic_string_unknown() -> None:
    msg = "Incorrect 'api_key' or 'access_token'"
    assert classify_broker_error(msg) == UNKNOWN
    assert classify_broker_error(msg, 400) == UNKNOWN


def test_auth_classification() -> None:
    assert classify_broker_error("Incorrect 'api_key' or 'access_token'", 401) == AUTH
    assert classify_broker_error("Session expired", 403) == AUTH
    assert classify_broker_error("Invalid session token") == AUTH
    assert classify_broker_error("Not a valid access token") == AUTH
