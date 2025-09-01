from src.utils.env import env_flag


def test_env_flag_true_variants(monkeypatch):
    for val in ["1", "true", "TRUE", "yes", "on", "Yes "]:
        monkeypatch.setenv("FLAG", val)
        assert env_flag("FLAG") is True


def test_env_flag_false_variants(monkeypatch):
    for val in ["0", "false", "FALSE", "no", "off", "No "]:
        monkeypatch.setenv("FLAG", val)
        assert env_flag("FLAG") is False


def test_env_flag_defaults_when_missing(monkeypatch):
    monkeypatch.delenv("FLAG", raising=False)
    assert env_flag("FLAG") is True


def test_env_flag_defaults_when_unparseable(monkeypatch):
    monkeypatch.setenv("FLAG", "maybe")
    assert env_flag("FLAG") is True
