import pytest

from utils.retry import retry


def test_retry_invalid_tries() -> None:
    with pytest.raises(ValueError):
        retry(tries=0)


def test_retry_invalid_delay() -> None:
    with pytest.raises(ValueError):
        retry(delay=-1)


def test_retry_invalid_backoff() -> None:
    with pytest.raises(ValueError):
        retry(backoff=0)


def test_retry_invalid_max_delay() -> None:
    with pytest.raises(ValueError):
        retry(max_delay=-5)
