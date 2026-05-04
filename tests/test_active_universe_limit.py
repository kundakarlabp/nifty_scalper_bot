import os


def test_active_universe_default_limit() -> None:
    assert int(os.getenv('MAX_ACTIVE_OPTION_SYMBOLS', '6')) == 6
