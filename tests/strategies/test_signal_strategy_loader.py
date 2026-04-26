from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.strategies.elite_strategies.config import ELITE_STRATEGY_MODULES
from nifty_scalper_bot.strategies.signal import _load_enabled_strategy_modules


def test_loader_empty_config_falls_back_to_default_registry(
    monkeypatch,
) -> None:
    """Args: monkeypatch. Returns: None. Raises: AssertionError."""

    def _fake_open(self: Path, *_args, **_kwargs):  # noqa: ANN001
        from io import StringIO

        return StringIO('enabled_strategies: []\n')

    monkeypatch.setattr(Path, 'open', _fake_open)
    loaded = _load_enabled_strategy_modules()
    assert loaded == list(ELITE_STRATEGY_MODULES)
