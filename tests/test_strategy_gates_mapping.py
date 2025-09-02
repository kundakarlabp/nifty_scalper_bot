import yaml
from pathlib import Path
from src.strategies.strategy_config import StrategyConfig


def test_strategy_min_score_mapping(tmp_path):
    data = yaml.safe_load(Path("config/strategy.yaml").read_text())
    data["strategy"]["min_score"] = 0.35
    cfg_path = tmp_path / "strategy.yaml"
    cfg_path.write_text(yaml.safe_dump(data))
    cfg = StrategyConfig.load(str(cfg_path))
    need = cfg.raw.get("strategy", {}).get("min_score")
    assert need == 0.35
