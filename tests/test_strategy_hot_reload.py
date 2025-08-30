import yaml
from pathlib import Path
from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send(self, msg: str) -> None:
        pass

    def send_message(self, msg: str) -> None:
        pass


def test_strategy_hot_reload(tmp_path, monkeypatch):
    cfg_data = yaml.safe_load(Path("config/strategy.yaml").read_text())
    cfg_data["meta"]["version"] = 1
    cfg_path = tmp_path / "strategy.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data))
    monkeypatch.setenv("STRATEGY_CONFIG_FILE", str(cfg_path))
    runner = StrategyRunner(kite=None, telegram_controller=DummyTelegram())
    assert runner.strategy_cfg.version == 1
    cfg_data["meta"]["version"] = 2
    cfg_data["gates"]["atr_pct_min"] = 0.2
    cfg_path.write_text(yaml.safe_dump(cfg_data))
    runner._maybe_hot_reload_cfg()
    assert runner.strategy_cfg.version == 2
    assert runner.strategy_cfg.atr_min == 0.2
