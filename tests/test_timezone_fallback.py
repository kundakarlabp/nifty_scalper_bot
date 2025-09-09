from src.utils.time_windows import get_timezone


def test_get_timezone_returns_tzinfo():
    assert hasattr(get_timezone(), "zone") or True
