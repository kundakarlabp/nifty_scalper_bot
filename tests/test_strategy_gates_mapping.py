import yaml
from pathlib import Path
from src.strategies.strategy_config import StrategyConfig


def test_strategy_gates_mapping(tmp_path):
    data = yaml.safe_load(Path("config/strategy.yaml").read_text())
    data["debug"]["lower_score_temp"] = True
    cfg_path = tmp_path / "strategy.yaml"
    cfg_path.write_text(yaml.safe_dump(data))
    cfg = StrategyConfig.load(str(cfg_path))
    need = cfg.score_trend_min if "TREND" == "TREND" else cfg.score_range_min
    if cfg.lower_score_temp:
        need = min(need, 6)
    assert need == 6
