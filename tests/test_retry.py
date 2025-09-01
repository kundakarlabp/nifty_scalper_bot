from execution import broker_retry


class Transient(Exception):
    pass


def test_retry_succeeds_after_transient_errors() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("timeout")
        return "ok"

    out = broker_retry.call(flaky)
    assert out == "ok"
    assert calls["n"] == 3
