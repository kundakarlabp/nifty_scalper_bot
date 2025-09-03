from src.strategies.strategy_config import resolve_config_path


def test_resolve_config_path_fallback(monkeypatch):
    monkeypatch.setenv("STRATEGY_CONFIG_FILE", "/config/strategy.yml")
    assert resolve_config_path() == "config/strategy.yaml"
