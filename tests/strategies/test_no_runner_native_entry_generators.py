from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.strategies import runner

ROOT = Path(__file__).resolve().parents[2]


def test_runner_has_no_native_entry_generators_outside_strategy_manager() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")

    assert "RUNNER_ENABLE_PREMIUM_SQUEEZE" not in source
    assert "RUNNER_ENABLE_LEGACY_VWAP_CROSSOVER" not in source
    assert "def _maybe_generate_premium_squeeze_signal(" not in source
    assert "_premium_squeeze_last_signal_ts" not in source
    assert "premium_momentum_squeeze" not in source
    assert "vwap_crossover_up" not in source
    assert "vwap_crossover_down" not in source
    assert "_force_signal_enabled" not in source
    assert "_final_quality_approved_counter" not in source


def test_runtime_config_has_no_retired_signal_scoring_knobs() -> None:
    settings_source = (ROOT / "src/nifty_scalper_bot/config/settings.py").read_text(
        encoding="utf-8"
    )
    env_source = (ROOT / ".env.example").read_text(encoding="utf-8")
    strategy_profile = (ROOT / "config/strategy.yaml").read_text(encoding="utf-8")

    assert "SIGNAL_WEIGHT_" not in settings_source
    assert "SIGNAL_WEIGHT_" not in env_source
    assert "min_score" not in strategy_profile
    assert "lower_score_temp" not in strategy_profile
