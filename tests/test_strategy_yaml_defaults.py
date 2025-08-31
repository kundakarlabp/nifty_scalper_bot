import yaml
from pathlib import Path
from src.strategies.strategy_config import StrategyConfig


def test_strategy_yaml_defaults(tmp_path):
    cfg = StrategyConfig.load("config/strategy.yaml")
    assert cfg.atr_min == 0.20
    assert cfg.gamma_after.hour == 14

    data = yaml.safe_load(Path("config/strategy.yaml").read_text())
    data["gates"]["atr_pct_min"] = 0.25
    new_path = tmp_path / "strategy.yaml"
    new_path.write_text(yaml.safe_dump(data))
    cfg2 = StrategyConfig.load(str(new_path))
    assert cfg2.atr_min == 0.25
