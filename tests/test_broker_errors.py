from src.utils.broker_errors import classify_broker_error, AUTH, UNKNOWN


def test_generic_string_unknown():
    msg = "Incorrect 'api_key' or 'access_token'"
    assert classify_broker_error(msg) == UNKNOWN


class DummyExc(Exception):
    def __init__(self, status: int):
        super().__init__("boom")
        self.status = status


def test_status_401_auth():
    err = DummyExc(401)
    assert classify_broker_error(err) == AUTH
