import pytest
from src.strategies.strategy_config import StrategyConfig
from src.utils.events import load_calendar


def test_strategy_config_invalid_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo: [\n")
    with pytest.raises(ValueError, match="bad.yaml"):
        StrategyConfig.load(str(bad))


def test_events_calendar_invalid_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo: [\n")
    with pytest.raises(ValueError, match="bad.yaml"):
        load_calendar(str(bad))
